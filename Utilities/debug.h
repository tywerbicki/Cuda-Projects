#pragma once

#include "cuda_runtime.h"

#include <iostream>
#include <string_view>


#define MSG_STD_OUT(...) debug::DisplayMessage(std::cout, "MESSAGE", __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define MSG_STD_ERR(...) debug::DisplayMessage(std::cerr, "ERROR"  , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)


#ifdef _DEBUG

#define DBG_MSG_STD_OUT(...) debug::DisplayMessage(std::cout, "DEBUG MESSAGE", __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define DBG_MSG_STD_ERR(...) debug::DisplayMessage(std::cerr, "DEBUG ERROR"  , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define DBG_PRINT_ON_CUDA_ERROR(result) if ((result) != cudaSuccess) debug::DisplayCudaError(result, __FILE__, __FUNCTION__, __LINE__)

#define DBG_PRINT_RETURN_ON_CUDA_ERROR(result) if ((result) != cudaSuccess) \
                                               { \
                                                   debug::DisplayCudaError(result, __FILE__, __FUNCTION__, __LINE__); \
                                                   return result; \
                                               }

#else // _DEBUG

#define DBG_MSG_STD_OUT(...)
#define DBG_MSG_STD_ERR(...)

#define DBG_PRINT_ON_CUDA_ERROR(result)
#define DBG_PRINT_RETURN_ON_CUDA_ERROR(result) if ((result) != cudaSuccess) return result

#endif // _DEBUG


namespace debug
{
    template<typename... Args>
    inline void DisplayMessage(std::ostream&          oStream,
                               const std::string_view title,
                               const std::string_view fileName,
                               const std::string_view callerName,
                               const unsigned int     lineNumber,
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
                          unsigned int     lineNumber);
}