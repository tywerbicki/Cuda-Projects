#include "debug.h"
#include "saxpy.h"

#include "device_launch_parameters.h"

#include <algorithm>


__global__ void Saxpy(const float                     a,
                      const float* const __restrict__ pXDevice,
                      const float* const __restrict__ pYDevice,
                            float* const __restrict__ pZDevice,
                      const size_t                    len)
{
    unsigned int gridThreadIdx = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int gridSize      = gridDim.x * blockDim.x;

    for (size_t idx = gridThreadIdx; idx < len; idx += gridSize)
    {
        pZDevice[idx] = (a * pXDevice[idx]) + pYDevice[idx];
    }
}


void saxpy::DeviceExecute(const float        a,
                          const float* const pXDevice,
                          const float* const pYDevice,
                                float* const pZDevice,
                          const size_t       len,
                          const cudaStream_t stream)
{
    if (len > 0)
    {
        // NOTE: these values require hw-specific tuning.
        const unsigned int tpb           = 128;
        const size_t       maxAllowedBpg = 256;
        const size_t       maxNeededBpg  = (len + tpb - 1) / tpb;
        const unsigned int bpg           = static_cast<unsigned int>(std::min(maxAllowedBpg, maxNeededBpg));

        DBG_MSG_STD_OUT("Saxpy launch parameters:\n\tLen: ", len, "\n\tTPB: ", tpb, "\n\tBPG: ", bpg);

        Saxpy<<<bpg, tpb, 0, stream>>>(a, pXDevice, pYDevice, pZDevice, len);
    }
}


void saxpy::HostExecute(const float        a,
                        const float* const pXHost,
                        const float* const pYHost,
                              float* const pZHost,
                        const size_t len)
{
    std::transform(pXHost, pXHost + len,
                   pYHost,
                   pZHost,
                   [=](float x, float y) { return (a * x) + y; });
}