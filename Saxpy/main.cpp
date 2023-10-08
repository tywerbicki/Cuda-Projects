#include <algorithm>
#include <stdint.h>

#include "cuda_runtime.h"

#include "debug.h"
#include "device.h"
#include "saxpy.h"
#include "settings.h"


int main()
{
    cudaError_t result         = cudaSuccess;
    int32_t     deviceCount    = -1;
    int32_t     selectedDevice = -1;

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

    int32_t deviceIsIntegrated     = -1;
    int32_t deviceCanMapHostMemory = -1;

    result = cudaDeviceGetAttribute(&deviceIsIntegrated, cudaDevAttrIntegrated, selectedDevice);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaDeviceGetAttribute(&deviceCanMapHostMemory, cudaDevAttrCanMapHostMemory, selectedDevice);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    cudaStream_t saxpyStream   = nullptr;
    cudaEvent_t  saxpyComplete = nullptr;

    result = cudaStreamCreateWithFlags(&saxpyStream, cudaStreamNonBlocking);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaEventCreate(&saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    const float      a           = 3.0;
    constexpr size_t size        = 1000;
    constexpr size_t sizeInBytes = size * sizeof(float);
    float*           pXHost      = nullptr;
    float*           pYHost      = nullptr;
    float*           pXDevice    = nullptr;
    float*           pYDevice    = nullptr;

    if (saxpy::settings::memoryStrategy == saxpy::MemoryStrategy::forceMapped ||
        (deviceIsIntegrated && deviceCanMapHostMemory)                          )
    {
        // If the device is integrated and supports mapped host memory, then we use pinned
        // mapped allocations to entirely remove any copying from host to device and vice versa.
        DBG_MSG_STD_OUT("Mapped memory allocation strategy chosen");

        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocMapped);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::fill(pXHost, pXHost + size, 2.0f);
        std::fill(pYHost, pYHost + size, 1.5f);

        result = cudaHostGetDevicePointer(&pXDevice, pXHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostGetDevicePointer(&pYDevice, pYHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(saxpyStream, size, a, pXDevice, pYDevice);
    }
    else
    {
        // If the device is discrete, we will do everything asynchronously with respect to the host.
        DBG_MSG_STD_OUT("Async memory allocation strategy chosen");

        result = cudaMallocAsync(&pXDevice, sizeInBytes, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::fill(pXHost, pXHost + size, 4.0f);

        result = cudaMemcpyAsync(pXDevice, pXHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMallocAsync(&pYDevice, sizeInBytes, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::fill(pYHost, pYHost + size, 1.5f);

        result = cudaMemcpyAsync(pYDevice, pYHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(saxpyStream, size, a, pXDevice, pYDevice);

        result = cudaMemcpyAsync(pYHost, pYDevice, sizeInBytes, cudaMemcpyDeviceToHost, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaFreeAsync(pXDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
        result = cudaFreeAsync(pYDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    }

    result = cudaEventRecord(saxpyComplete, saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    // Here is where we synchronize the host with the saxpy device operations.
    result = cudaEventSynchronize(saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    // TODO: replace this with a std::equal
    for (size_t i = 0; i < size; i++)
    {
        std::cout << pYHost[i] << " ";
    }
    std::cout << std::endl;

    result = cudaFreeHost(pXHost);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    result = cudaFreeHost(pYHost);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaEventDestroy(saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    result = cudaStreamDestroy(saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    return cudaSuccess;
}