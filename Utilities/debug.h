#pragma once

#include <iostream>
#include <stdint.h>
#include <string_view>

#include "cuda_runtime.h"


#define MSG_STD_OUT(...) debug::DisplayMessage(std::cout, "MESSAGE", __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)


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
    template<typename... Args>
    inline void DisplayMessage(std::ostream&          oStream,
                               const std::string_view title,
                               const std::string_view fileName,
                               const std::string_view callerName,
                               const uint32_t         lineNumber,
                               Args&&...              args)
    {
        oStream << "\n" << title << "\n";
    
        oStream << "File: "     << fileName   << "\n";
        oStream << "Function: " << callerName << "\n";
        oStream << "Line: "     << lineNumber << "\n";
    
        (oStream << ... << args) << "\n";
    }

    void DisplayCudaError(cudaError_t      error,
                          std::string_view fileName,
                          std::string_view callerName,
                          uint32_t         lineNumber);
}