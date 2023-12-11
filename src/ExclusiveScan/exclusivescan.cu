#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>

namespace cg = cooperative_groups;


namespace
{
    constexpr unsigned int WarpSize         = 4;
    constexpr unsigned int MaxBlockSize     = 16;
    constexpr unsigned int MaxWarpsPerBlock = MaxBlockSize / WarpSize;

    // This assumption is implictly made in kernels below.
    static_assert(MaxWarpsPerBlock <= WarpSize);


    template<typename TyVal, typename TyFn>
    __device__ void ExclusiveScan(const cg::thread_block& block, TyVal* const pData, TyFn&& op)
    {
        // 1. Allocate shared memory to hold intermediate results obtained from each warp in the thread block.
        // 2. The allocation size is conservative - the exact calculation is:
        //        block.num_threads() / WarpSize
        __shared__ TyVal shared[MaxWarpsPerBlock];

        // Partition thread block into warps.
        const cg::thread_block_tile<WarpSize, cg::thread_block> warp = cg::tiled_partition<WarpSize, cg::thread_block>(block);
        // Calculate the location of the thread in the parent thread block.
        const unsigned int threadIdxInBlock = (warp.meta_group_rank() * warp.num_threads()) + warp.thread_rank();
        // Load value to be used in the warp-local exclusive scan.
        const TyVal originalVal = pData[threadIdxInBlock];

        const TyVal scannedVal = cg::exclusive_scan(warp, originalVal, op);

        // The highest-ranked thread in the warp stores its scanned value + original value in shared memory
        // as an intermediate result.
        if (warp.thread_rank() + 1 == warp.num_threads())
        {
            shared[warp.meta_group_rank()] = scannedVal + originalVal;
        }

        block.sync();

        // The first warp in the thread block is used to scan the intermediate results.
        if (warp.meta_group_rank() == 0)
        {
            shared[warp.thread_rank()] = cg::exclusive_scan(warp, shared[warp.thread_rank()], op);
        }

        block.sync();

        // Add the scanned intermediate result to the original scanned value to complete the block-wide scan.
        pData[threadIdxInBlock] = shared[warp.meta_group_rank()] + scannedVal;
    }


    __global__ void kernel(unsigned int* const pData)
    {
        ExclusiveScan(cg::this_thread_block(), pData, cg::plus<unsigned int>());
    }
}


int main()
{
    unsigned int data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    unsigned int* pDataDevice = nullptr;
    cudaMalloc(&pDataDevice, sizeof(data));

    cudaMemcpy(pDataDevice, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, 12>>>(pDataDevice);

    cudaMemcpy(data, pDataDevice, sizeof(data), cudaMemcpyDeviceToHost);

    for (const auto i : data)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
