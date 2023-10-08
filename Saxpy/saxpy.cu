#include <stdint.h>

#include "device_launch_parameters.h"

#include "debug.h"
#include "saxpy.h"


__global__ void SaxpyCuKernel(const size_t       len,
                              const float        a,
                              const float* const pXDevice,
                              float*       const pYdevice)
{
    unsigned int gridThreadIdx  = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int gridSize       = gridDim.x * blockDim.x;

    for (size_t idx = gridThreadIdx; idx < len; idx += gridSize)
    {
        pYdevice[idx] += a * pXDevice[idx];
    }
}


void saxpy::DeviceExecute(const cudaStream_t stream,
                          const size_t       len,
                          const float        a,
                          const float* const pXDevice,
                          float*       const pYDevice)
{
    if (len > 0)
    {
        // NOTE: these values require hw-specific tuning.
        const uint32_t tpb           = 128;
        const uint32_t maxAllowedBpg = 256;

        const size_t   maxNeededBpg = (len + tpb - 1) / tpb;
        const uint32_t bpg          = static_cast<uint32_t>(std::min(static_cast<size_t>(maxAllowedBpg),
                                                                     maxNeededBpg));

        DBG_MSG_STD_OUT("Saxpy launch parameters:\n\tTPB: ", tpb, "\n\tBPG: ", bpg);

        SaxpyCuKernel<<<bpg, tpb, 0, stream>>>(len, a, pXDevice, pYDevice);
    }
}