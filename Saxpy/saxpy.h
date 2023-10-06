#pragma once

#include <span>

#include "cuda_runtime.h"


namespace saxpy
{
    cudaError_t HostExecute(cudaStream_t           stream,
                            float                  a,
                            std::span<const float> x,
                            std::span<float>       y);
}