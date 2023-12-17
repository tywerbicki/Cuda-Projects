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

        // 1. Allocate shared memory to hold intermediate results obtained from each warp in the thread block.
        // 2. The allocation size is conservative - the exact calculation is block.num_threads() / WarpSize.
        __shared__ T shared[WarpSize];

        const cg::thread_block_tile<WarpSize, cg::thread_block> warp = cg::tiled_partition<WarpSize, cg::thread_block>(block);

        // This ensures we can scan the intermediate results with only 1 warp.
        assert(warp.meta_group_rank() < WarpSize);

        T originalVal = {};
        T scannedVal  = {};

        if (block.thread_rank() < len)
        {
            const cg::coalesced_group active = cg::coalesced_threads();

            originalVal = pData[block.thread_rank()];
            scannedVal  = cg::exclusive_scan(active, originalVal, op);

            printf("Scanned val: %u\n", scannedVal);

            if (active.thread_rank() + 1 == active.num_threads())
            {
                shared[warp.meta_group_rank()] = op(scannedVal, originalVal);
            }
        }

        block.sync();

        if ((block.thread_rank() + 1 == len) && pBlockResult)
        {
            *pBlockResult = op(scannedVal, originalVal);
        }

        if (warp.meta_group_rank() == 0)
        {
            const unsigned int numWritesToShared = (len + WarpSize - 1) / WarpSize;

            if (warp.thread_rank() < numWritesToShared)
            {
                const cg::coalesced_group active = cg::coalesced_threads();

                shared[active.thread_rank()] = cg::exclusive_scan(active, shared[active.thread_rank()], op);
            }
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
        const unsigned int blockLen   = static_cast<unsigned int>( min(static_cast<size_t>(block.num_threads()),
                                                                       len - blockBaseGlobalThrIdx) );
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


int main()
{
    unsigned int data[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    unsigned int blockSums[4];

    unsigned int* pDataDevice = nullptr;
    unsigned int* pBlockSums  = nullptr;
    cudaMalloc(&pDataDevice, sizeof(data));
    cudaMalloc(&pBlockSums, sizeof(blockSums));

    cudaMemcpy(pDataDevice, data, sizeof(data), cudaMemcpyHostToDevice);

    ExclusiveScan_Add_grid_kernel<<<4, 4>>>(pDataDevice, 13, pBlockSums);
    ExclusiveScan_Add_grid_kernel<<<1, 4>>>(pBlockSums, 4, (unsigned int*)0);
    DistributeBlockSums_kernel<<<4, 4>>>(pDataDevice, 13, pBlockSums);

    cudaMemcpy(data, pDataDevice, sizeof(data), cudaMemcpyDeviceToHost);
    cudaMemcpy(blockSums, pBlockSums, sizeof(blockSums), cudaMemcpyDeviceToHost);

    for (const auto i : data)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    for (const auto i : blockSums)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
