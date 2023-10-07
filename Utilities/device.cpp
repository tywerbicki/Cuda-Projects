#include "debug.h"
#include "device.h"


cudaError_t device::GetMostMultiProcessors(const int32_t deviceCount,
                                           int32_t&      selectedDevice)
{
    cudaError_t result                 = cudaSuccess;
    int32_t     maxMultiProcessorCount = 0;

    for (int32_t device = 0; device < deviceCount; device++)
    {
        int multiProcessorCount = 0;
        result                  = cudaDeviceGetAttribute(&multiProcessorCount,
                                                         cudaDevAttrMultiProcessorCount,
                                                         device);

        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        if (multiProcessorCount > maxMultiProcessorCount)
        {
            maxMultiProcessorCount = multiProcessorCount;
            selectedDevice         = device;
        }
    }

    return result;
}