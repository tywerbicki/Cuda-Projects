#include "debug.h"
#include "device.h"


cudaError_t device::GetMostMultiProcessors(const int deviceCount,
                                           int&      selectedDevice)
{
    cudaError_t result                 = cudaSuccess;
    int         maxMultiProcessorCount = 0;

    for (int device = 0; device < deviceCount; device++)
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