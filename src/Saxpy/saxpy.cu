#include "debug.h"
#include "saxpy.h"

#include "device_launch_parameters.h"

#include <algorithm>


namespace
{
    template<typename T>
    concept Axpyable = requires(T a, T b)
    {
        { a + b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
    };

    template<Axpyable T>
    __global__ void Axpy(const T                     a,
                         const T* const __restrict__ pXDevice,
                         const T* const __restrict__ pYDevice,
                         T* const       __restrict__ pZDevice,
                         const size_t                    len)
    {
        unsigned int globThrIdxX = (blockDim.x * blockIdx.x) + threadIdx.x;

        if (globThrIdxX < len)
        {
            pZDevice[globThrIdxX] = (a * pXDevice[globThrIdxX]) + pYDevice[globThrIdxX];
        }
    }
}


cudaError_t saxpy::DeviceLaunchAsync(const float        a,
                                     const float* const pXDevice,
                                     const float* const pYDevice,
                                     float* const       pZDevice,
                                     const size_t       len,
                                     const cudaStream_t stream)
{
    cudaError_t result = cudaSuccess;

    if (len > 0)
    {
        int device   = -1;
        int warpSize = -1;

        result = cudaGetDevice(&device);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        result = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        // NOTE: the multiple of the warp size should be tuned per-hardware.
        const unsigned int blockDimX = static_cast<unsigned int>(warpSize) * 4;

#ifdef _DEBUG
        int maxGridDimX = -1;

        result = cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, device);
        DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

        if (static_cast<size_t>(maxGridDimX) * static_cast<size_t>(blockDimX) < len)
        {
            DBG_MSG_STD_ERR("Saxpy `len` exceeds maximum supported launch parameters: ", len);
            return cudaErrorInvalidValue;
        }
#endif // _DEBUG

        const unsigned int gridDimX = static_cast<unsigned int>((len + blockDimX - 1) / blockDimX);

        Axpy<<<gridDimX, blockDimX, 0, stream>>>(a, pXDevice, pYDevice, pZDevice, len);
    }

    return result;
}


void saxpy::HostExec(const float        a,
                     const float* const pXHost,
                     const float* const pYHost,
                     float* const       pZHost,
                     const size_t       len)
{
    std::transform(pXHost, pXHost + len,
                   pYHost,
                   pZHost,
                   [=](float x, float y) { return (a * x) + y; });
}
