#pragma once

#include "cuda_runtime.h"


namespace saxpy
{
    void DeviceExecute(      float        a,
                       const float*       pXDevice,
                       const float*       pYDevice,
                             float*       pZDevice,
                             size_t       len,
                             cudaStream_t stream);
}