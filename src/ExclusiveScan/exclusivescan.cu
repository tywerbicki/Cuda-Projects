#include "debug.h"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>
#include <type_traits>

namespace cg = cooperative_groups;


namespace
{
    template<typename T>
    concept IsoAddable = requires(T a, T b)
    {
        { a + b } -> std::convertible_to<T>;
    };

    constexpr unsigned int WarpSize = 4;

    using Warp = cg::thread_block_tile<WarpSize, cg::thread_block>;


    template<typename T, typename F>
        requires std::is_trivially_copyable_v<T> &&
                 std::is_copy_assignable_v<T>    &&
                 (sizeof(T) <= 32)               &&
                 std::is_invocable_v<F, T, T>
    __device__ void ExclusiveScan_block_kernel(const cg::thread_block& block,
                                               T* const                pData,
                                               const unsigned int      len,
                                               F&&                     op,
                                               T* const                pBlockResult)
    {
        assert(pData != nullptr);
        assert(len   <= block.num_threads());

        const Warp warp = cg::tiled_partition<Warp::num_threads(), cg::thread_block>(block);

        // 1. Allocate shared memory to hold intermediate results obtained from each warp in the thread block.
        // 2. The allocation size is conservative to make it static - the exact calculation is `numWritesToShared`,
        //    which can only be performed dynamically.
        // 3. This allocation size ensures we can scan the intermediate results with only 1 warp.
        __shared__ T shared[warp.num_threads()];

        const unsigned int numWritesToShared = (len + warp.num_threads() - 1) / warp.num_threads();

        assert(numWritesToShared <= warp.num_threads());

        T originalVal = {};
        T scannedVal  = {};
        T warpResult  = {};

        if (block.thread_rank() < len)
        {
            assert(warp.meta_group_rank() < numWritesToShared);

            const cg::coalesced_group active = cg::coalesced_threads();

            originalVal = pData[block.thread_rank()];
            scannedVal  = cg::exclusive_scan(active, originalVal, op);

            if (active.thread_rank() + 1 == active.num_threads())
            {
                warpResult = op(scannedVal, originalVal);
                shared[warp.meta_group_rank()] = warpResult;
            }
        }

        block.sync();

        if ((block.thread_rank() + 1 == len) && pBlockResult)
        {
            assert(warp.meta_group_rank() + 1 == numWritesToShared);

            *pBlockResult = warpResult;
        }

        if (block.thread_rank() < numWritesToShared)
        {
            assert(warp.meta_group_rank() == 0);

            const cg::coalesced_group active = cg::coalesced_threads();

            shared[active.thread_rank()] = cg::exclusive_scan(active, shared[active.thread_rank()], op);
        }

        block.sync();

        if (block.thread_rank() < len)
        {
            pData[block.thread_rank()] = op(shared[warp.meta_group_rank()], scannedVal);
        }
    }


    template<typename T>
    __global__ void ExclusiveScan_Add_grid_kernel(T* const pData, const size_t len, T* const pBlockSums)
    {
        assert(pData != nullptr);

        const cg::grid_group   grid  = cg::this_grid();
        const cg::thread_block block = cg::this_thread_block();

        assert(len <= grid.num_threads());

        const size_t blockBaseGlobalThrIdx = static_cast<size_t>(grid.block_rank()) *
                                             static_cast<size_t>(block.num_threads());

        // This ensures that too many blocks weren't launched.
        assert(len > blockBaseGlobalThrIdx);

        T* const           pBlockData = pData + blockBaseGlobalThrIdx;
        const unsigned int blockLen   = static_cast<unsigned int>(
            min(static_cast<size_t>(block.num_threads()), len - blockBaseGlobalThrIdx)
        );
        T* const           pBlockSum  = pBlockSums ? pBlockSums + grid.block_rank() : nullptr;

        ExclusiveScan_block_kernel(block, pBlockData, blockLen, cg::plus<T>(), pBlockSum);
    }


    template<typename T>
        requires IsoAddable<T> &&
                 std::is_copy_assignable_v<T>
    __global__ void DistributeBlockSums_kernel(T* const pData, const size_t len, const T* const pBlockSums)
    {
        assert(pData      != nullptr);
        assert(pBlockSums != nullptr);

        const cg::grid_group grid = cg::this_grid();

        assert(len <= grid.num_threads());

        if (grid.thread_rank() < len)
        {
            pData[grid.thread_rank()] += pBlockSums[grid.block_rank()];
        }
    }
}


template<typename T>
cudaError_t DeviceLaunchAsync(T* const           pDataDevice,
                              const size_t       len,
                              const cudaStream_t stream)
{
    cudaError_t result = cudaSuccess;

    const unsigned int blockNumThreads = WarpSize;
    const unsigned int gridNumBlocks   = static_cast<unsigned int>((len + blockNumThreads - 1) / blockNumThreads);

    // This ensures that the block sums can be scanned by one block.
#ifdef _DEBUG
    int device       = -1;
    int maxBlockDimX = -1;

    result = cudaGetDevice(&device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    if (gridNumBlocks > static_cast<unsigned int>(maxBlockDimX))
    {
        DBG_MSG_STD_ERR("Launch parameters require blockDim.x be ", gridNumBlocks,
                        ": max supported is ", maxBlockDimX);
        return cudaErrorInvalidValue;
    }
#endif // _DEBUG

    T* pBlockSumsDevice = nullptr;
    result = cudaMallocAsync(&pBlockSumsDevice, gridNumBlocks * sizeof(T), stream);

    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    ExclusiveScan_Add_grid_kernel<<<gridNumBlocks, blockNumThreads, 0, stream>>>(pDataDevice,
                                                                                 len,
                                                                                 pBlockSumsDevice);

    ExclusiveScan_Add_grid_kernel<<<1, gridNumBlocks, 0, stream>>>(pBlockSumsDevice,
                                                                   static_cast<size_t>(gridNumBlocks),
                                                                   static_cast<T*>(nullptr));

    DistributeBlockSums_kernel<<<gridNumBlocks, blockNumThreads, 0, stream>>>(pDataDevice,
                                                                              len,
                                                                              pBlockSumsDevice);

    result = cudaFreeAsync(pBlockSumsDevice, stream);
    DBG_PRINT_ON_CUDA_ERROR(result);

    return result;
}


int main()
{
    constexpr size_t len = 17;
    unsigned int data[len] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    unsigned int* pDataDevice = nullptr;
    cudaMalloc(&pDataDevice, sizeof(data));

    cudaMemcpy(pDataDevice, data, sizeof(data), cudaMemcpyHostToDevice);

    DeviceLaunchAsync(pDataDevice, len, (cudaStream_t)0);

    cudaMemcpy(data, pDataDevice, sizeof(data), cudaMemcpyDeviceToHost);

    for (const auto i : data)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
