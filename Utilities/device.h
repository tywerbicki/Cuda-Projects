#pragma once

#include <stdint.h>

#include "cuda_runtime.h"


namespace device
{
    cudaError_t GetMostMultiProcessors(int32_t  deviceCount,
                                       int32_t& selectedDevice);
}