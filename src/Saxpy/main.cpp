#include "debug.h"
#include "device.h"
#include "saxpy.h"
#include "settings.h"
#include "streamtimer.h"

#include "cuda_runtime.h"

#include <stdlib.h>

#include <algorithm>
#include <chrono>
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

    StreamTimer saxpyStreamTimer(saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(saxpyStreamTimer.GetStatus());

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
#ifdef _DEBUG
        result = debug::DisplayUnifiedMemoryCapabilities(selectedDevice);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
#endif // _DEBUG

        result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostAlloc(&pZHost, sizeInBytes, cudaHostAllocMapped);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        std::generate(pXHost, pXHost + size, randFloat);
        std::generate(pYHost, pYHost + size, randFloat);

        result = cudaHostGetDevicePointer(&pXDevice, pXHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostGetDevicePointer(&pYDevice, pYHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaHostGetDevicePointer(&pZDevice, pZHost, 0);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = saxpyStreamTimer.Start();
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(a, pXDevice, pYDevice, pZDevice, size, saxpyStream);

        result = saxpyStreamTimer.Stop();
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
    }
    else
    {
        // If the device is discrete, we will do everything asynchronously with respect to the host.
#ifdef _DEBUG
        result = debug::DisplayAsyncCapabilities(selectedDevice);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);
#endif // _DEBUG

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

        result = saxpyStreamTimer.Start();
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        saxpy::DeviceExecute(a, pXDevice, pYDevice, pZDevice, size, saxpyStream);

        result = saxpyStreamTimer.Stop();
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

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
    }

    result = cudaEventRecord(saxpyComplete, saxpyStream);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    const auto hostSaxpyExecStart = std::chrono::steady_clock::now();

    saxpy::HostExecute(a, pXHost, pYHost, solution.data(), size);

    const auto hostSaxpyExecStop = std::chrono::steady_clock::now();

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

    float deviceSaxpyExecTimeInMs = 0.0f;

    result = saxpyStreamTimer.GetElapsedTimeInMs(deviceSaxpyExecTimeInMs);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    MSG_STD_OUT("Saxpy execution times:\n",
                "   Host:   ", std::chrono::duration_cast<std::chrono::microseconds>(hostSaxpyExecStop - hostSaxpyExecStart), "\n",
                "   Device: ", deviceSaxpyExecTimeInMs * 1000, "us");

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

    return cudaSuccess;
}