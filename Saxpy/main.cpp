#include <stdint.h>

#include "cuda_runtime.h"

#include "debug.h"

#include <iostream>
int main()
{
    cudaError_t result   = cudaSuccess;
    int32_t     nDevices = 0;
        
    result = cudaGetDeviceCount(&nDevices);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    std::cout << "Num devices: " << nDevices << std::endl;

    return cudaSuccess;
}