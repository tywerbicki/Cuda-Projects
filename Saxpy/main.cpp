#include <algorithm>
#include <span>
#include <stdint.h>

#include "cuda_runtime.h"

#include "debug.h"
#include "device.h"
#include "saxpy.h"


void FillXY(const std::span<float> x, const std::span<float> y)
{
    std::fill(x.begin(), x.end(), 1.0f);
    std::fill(y.begin(), y.end(), 2.0f);
}


int main()
{
    cudaError_t    result                   = cudaSuccess;
    int32_t        deviceCount              = 0;
    int32_t        selectedDevice           = -1;
    cudaDeviceProp selectedDeviceProperties = {};
    cudaStream_t   stream                   = nullptr;

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

    result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
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
        result = cudaHostAlloc(&pXHost,
                               sizeInBytes,
                               cudaHostAllocMapped | cudaHostAllocWriteCombined);

        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost,
                               sizeInBytes,
                               cudaHostAllocMapped);

        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    }
    else
    {
        result = cudaHostAlloc(&pXHost,
                               sizeInBytes,
                               cudaHostAllocDefault);

        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost,
                               sizeInBytes,
                               cudaHostAllocDefault);

        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        FillXY({pXHost, size}, {pYHost, size});

        result = cudaMalloc(&pXDevice, sizeInBytes);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMalloc(&pYDevice, sizeInBytes);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMemcpyAsync(pXDevice, pXHost, sizeInBytes, cudaMemcpyHostToDevice, stream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMemcpyAsync(pYDevice, pYHost, sizeInBytes, cudaMemcpyHostToDevice, stream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    }

    

    return cudaSuccess;
}