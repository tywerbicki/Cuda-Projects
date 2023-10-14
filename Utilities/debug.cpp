#include "debug.h"


void debug::DisplayCudaError(const cudaError_t      error,
                             const std::string_view fileName,
                             const std::string_view callerName,
                             const unsigned int     lineNumber)
{
    std::cerr << "\nCUDA ERROR\n";

    std::cerr << "File: "       << fileName   << "\n";
    std::cerr << "Function: "   << callerName << "\n";
    std::cerr << "Line: "       << lineNumber << "\n";
    std::cerr << "Error code: " << error      << "\n";
}


cudaError_t debug::DisplayAsyncCapabilities(int device)
{
    cudaError_t result                    = cudaSuccess;
    int         supportsGpuOverlap        = -1;
    int         supportsConcurrentKernels = -1;

    result = cudaDeviceGetAttribute(&supportsGpuOverlap, cudaDevAttrGpuOverlap, device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    result = cudaDeviceGetAttribute(&supportsConcurrentKernels, cudaDevAttrConcurrentKernels, device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    DBG_MSG_STD_OUT("Device ", device, " async capabilities:\n",
                    "   GPU overlap: ", supportsGpuOverlap, "\n",
                    "   Concurrent kernels: ", supportsConcurrentKernels);

    return result;
}