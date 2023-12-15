#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>

namespace cg = cooperative_groups;


namespace
{
    constexpr unsigned int WarpSize = 4;


    template<typename T, typename F>
    __device__ void ExclusiveScan_block_kernel(const cg::thread_block& block, T* const pData, F&& op, T* const pBlockSum)
    {
        // 1. Allocate shared memory to hold intermediate results obtained from each warp in the thread block.
        // 2. The allocation size is conservative - the exact calculation is block.num_threads() / WarpSize.
        __shared__ T shared[WarpSize];

        const cg::thread_block_tile<WarpSize, cg::thread_block> warp = cg::tiled_partition<WarpSize, cg::thread_block>(block);

        // This ensures we can scan the intermediate results with only 1 warp.
        assert(warp.meta_group_rank() < WarpSize);

        const T originalVal = pData[block.thread_rank()];
        const T scannedVal  = cg::exclusive_scan(warp, originalVal, op);

        printf("Scanned val: %u\n", scannedVal);

        if (warp.thread_rank() + 1 == warp.num_threads())
        {
            shared[warp.meta_group_rank()] = scannedVal + originalVal;
        }

        block.sync();

        if ((block.thread_rank() + 1 == block.num_threads()) && pBlockSum)
        {
            *pBlockSum = scannedVal + originalVal;
        }

        if (warp.meta_group_rank() == 0)
        {
            shared[warp.thread_rank()] = cg::exclusive_scan(warp, shared[warp.thread_rank()], op);
        }

        block.sync();

        pData[block.thread_rank()] = shared[warp.meta_group_rank()] + scannedVal;
    }


    template<typename T>
    __global__ void ExclusiveScan_Add_grid_kernel(T* const pData, T* const pBlockSums)
    {
        const cg::grid_group   grid  = cg::this_grid();
        const cg::thread_block block = cg::this_thread_block();

        ExclusiveScan_block_kernel(block,
                                   &pData[grid.block_rank() * block.num_threads()],
                                   cg::plus<T>(),
                                   pBlockSums ? &pBlockSums[grid.block_rank()] : pBlockSums);
    }
}


int main()
{
    unsigned int data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    unsigned int blockSums[3];

    unsigned int* pDataDevice = nullptr;
    unsigned int* pBlockSums  = nullptr;
    cudaMalloc(&pDataDevice, sizeof(data));
    cudaMalloc(&pBlockSums, sizeof(blockSums));

    cudaMemcpy(pDataDevice, data, sizeof(data), cudaMemcpyHostToDevice);

    ExclusiveScan_Add_grid_kernel<<<3, 4>>>(pDataDevice, pBlockSums);
    ExclusiveScan_Add_grid_kernel<<<1, 4>>>(pBlockSums, (unsigned int*)0);

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
