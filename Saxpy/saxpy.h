#pragma once

#include "cuda_runtime.h"


namespace saxpy
{
    void DeviceExecute(cudaStream_t stream,
                       size_t       len,
                       float        a,
                       const float* pXDevice,
                       float*       pYDevice);
}