#include <algorithm>
#include <stdint.h>

#include "cuda_runtime.h"

#include "debug.h"
#include "device.h"
#include "saxpy.h"


int main()
{
    cudaError_t    result                   = cudaSuccess;
    int32_t        deviceCount              = 0;
    int32_t        selectedDevice           = -1;
    cudaDeviceProp selectedDeviceProperties = {};
    cudaStream_t   saxpyStream              = nullptr;
    cudaEvent_t    saxpyComplete            = nullptr;

    result = cudaGetDeviceCount(&deviceCount);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    if (deviceCount < 1)
    {
        MSG_STD_OUT("No CUDA-enabled device was detected");
        return cudaSuccess;
    }

    result = device::GetMostMultiProcessors(deviceCount, selectedDevice);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaSetDevice(selectedDevice);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaGetDeviceProperties(&selectedDeviceProperties, selectedDevice);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaStreamCreateWithFlags(&saxpyStream, cudaStreamNonBlocking);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaEventCreate(&saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    const float      a           = 3.0;
    constexpr size_t size        = 10;
    constexpr size_t sizeInBytes = size * sizeof(float);
    float*           pXHost      = nullptr;
    float*           pYHost      = nullptr;
    float*           pXDevice    = nullptr;
    float*           pYDevice    = nullptr;

    if (selectedDeviceProperties.integrated && selectedDeviceProperties.canMapHostMemory)
    {
        // If the device is integrated and supports mapped host memory, then we use pinned
        // mapped allocations to entirely remove any copying from host to device and vice versa.

        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocMapped);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::fill(pXHost, pXHost + size, 3.0f);
        std::fill(pYHost, pYHost + size, 1.5f);

        result = cudaHostGetDevicePointer(&pXDevice, pXHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostGetDevicePointer(&pYDevice, pYHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(saxpyStream, size, a, pXDevice, pYDevice);
    }
    else
    {
        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::fill(pXHost, pXHost + size, 2.0f);
        std::fill(pYHost, pYHost + size, 1.5f);

        result = cudaMalloc(&pXDevice, sizeInBytes);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMalloc(&pYDevice, sizeInBytes);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMemcpyAsync(pXDevice, pXHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMemcpyAsync(pYDevice, pYHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(saxpyStream, size, a, pXDevice, pYDevice);

        result = cudaMemcpyAsync(pYHost, pYDevice, sizeInBytes, cudaMemcpyDeviceToHost, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    }

    result = cudaEventRecord(saxpyComplete, saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaEventSynchronize(saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    for (size_t i = 0; i < size; i++)
    {
        std::cout << pYHost[i] << " ";
    }
    std::cout << std::endl;

    return cudaSuccess;
}