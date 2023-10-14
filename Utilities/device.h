#pragma once

#include "cuda_runtime.h"


namespace device
{
    [[nodiscard]] cudaError_t GetMostMultiProcessors(int  deviceCount,
                                                     int& selectedDevice);
}