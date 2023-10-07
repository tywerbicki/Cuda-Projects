#include "device_launch_parameters.h"

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
    // TODO: properly select thread grid size.

    SaxpyCuKernel<<<256, 5, 0, stream>>>(len, a, pXDevice, pYDevice);
}