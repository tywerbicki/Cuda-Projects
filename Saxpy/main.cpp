#include "debug.h"
#include "device.h"
#include "saxpy.h"
#include "settings.h"

#include "cuda_runtime.h"

#include <stdlib.h>

#include <algorithm>
#include <vector>


int main()
{
    cudaError_t result         = cudaSuccess;
    int         deviceCount    = -1;
    int         selectedDevice = -1;

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

    int deviceIsIntegrated     = -1;
    int deviceCanMapHostMemory = -1;

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

    std::srand(10);
    const auto randFloat = []() { return static_cast<float>(std::rand()); };

    const float  a           = 2.75f;
    float*       pXHost      = nullptr;
    float*       pYHost      = nullptr;
    float*       pZHost      = nullptr;
    float*       pXDevice    = nullptr;
    float*       pYDevice    = nullptr;
    float*       pZDevice    = nullptr;
    const size_t size        = 1000000;
    const size_t sizeInBytes = size * sizeof(float);

    std::vector<float> solution(size);

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

        // saxpy::DeviceExecute(saxpyStream, size, a, pXDevice, pYDevice);
    }
    else
    {
        // If the device is discrete, we will do everything asynchronously with respect to the host.
        DBG_MSG_STD_OUT("Async memory allocation strategy chosen");

        result = cudaMallocAsync(&pXDevice, sizeInBytes, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::generate(pXHost, pXHost + size, randFloat);

        result = cudaMemcpyAsync(pXDevice, pXHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMallocAsync(&pYDevice, sizeInBytes, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::generate(pYHost, pYHost + size, randFloat);

        result = cudaMemcpyAsync(pYDevice, pYHost, sizeInBytes, cudaMemcpyHostToDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMallocAsync(&pZDevice, sizeInBytes, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(a, pXDevice, pYDevice, pZDevice, size, saxpyStream);

        result = cudaHostAlloc(&pZHost, sizeInBytes, cudaHostAllocDefault);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaMemcpyAsync(pZHost, pZDevice, sizeInBytes, cudaMemcpyDeviceToHost, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaFreeAsync(pXDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
        result = cudaFreeAsync(pYDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
        result = cudaFreeAsync(pZDevice, saxpyStream);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::HostExecute(a, pXHost, pYHost, solution.data(), size);
    }

    result = cudaEventRecord(saxpyComplete, saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    // Here is where we synchronize the host with the saxpy device operations.
    result = cudaEventSynchronize(saxpyComplete);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    if (std::equal(solution.cbegin(), solution.cend(), pZHost))
    {
        MSG_STD_OUT("Host and device saxpy execution results are equal");
    }
    else
    {
        MSG_STD_ERR("Host and device saxpy execution results are not equal");
    }

    result = cudaFreeHost(pXHost);
    DBG_PRINT_ON_CUDA_ERROR(result);
    result = cudaFreeHost(pYHost);
    DBG_PRINT_ON_CUDA_ERROR(result);
    result = cudaFreeHost(pZHost);
    DBG_PRINT_ON_CUDA_ERROR(result);

    result = cudaEventDestroy(saxpyComplete);
    DBG_PRINT_ON_CUDA_ERROR(result);
    result = cudaStreamDestroy(saxpyStream);
    DBG_PRINT_ON_CUDA_ERROR(result);

    result = cudaDeviceReset();
    DBG_PRINT_ON_CUDA_ERROR(result);

    return cudaSuccess;
}