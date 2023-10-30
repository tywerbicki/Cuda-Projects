#ifndef UTILITIES_DEVICE_H
#define UTILITIES_DEVICE_H

#include "cuda_runtime.h"


namespace device
{
    [[nodiscard]] cudaError_t GetMostMultiProcessors(int  deviceCount,
                                                     int& selectedDevice);
}


#endif // UTILITIES_DEVICE_H