#pragma once

#include <stdint.h>
#include <string_view>

#include "cuda_runtime.h"


#ifdef _DEBUG

#define DBG_PRINT_RETURN_ON_CUDA_ERROR(result) if ((result) != cudaSuccess) \
                                               { \
                                                   debug::DisplayCudaError(result, __FILE__, __FUNCTION__, __LINE__); \
                                                   return result; \
                                               }

#else // _DEBUG

#define DBG_PRINT_RETURN_ON_CUDA_ERROR(result) if ((result) != cudaSuccess) return result

#endif // _DEBUG


namespace debug
{
    void DisplayCudaError(cudaError_t      error,
                          std::string_view fileName,
                          std::string_view callerName,
                          uint32_t         lineNumber);
}