#ifndef SAXPY_SAXPY_H
#define SAXPY_SAXPY_H

#include "cuda_runtime.h"


namespace saxpy
{
    [[nodiscard]] cudaError_t DeviceLaunchAsync(float        a,
                                                const float* pXDevice,
                                                const float* pYDevice,
                                                float*       pZDevice,
                                                size_t       len,
                                                cudaStream_t stream);

    void HostExec(float        a,
                  const float* pXHost,
                  const float* pYHost,
                  float*       pZHost,
                  size_t       len);
}


#endif // SAXPY_SAXPY_H