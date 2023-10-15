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


cudaError_t debug::DisplayUnifiedMemoryCapabilities(const int device)
{
    cudaError_t    result           = cudaSuccess;
    cudaDeviceProp deviceProperties = {};

    result = cudaGetDeviceProperties_v2(&deviceProperties, device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    DBG_MSG_STD_OUT("Device ", device, " unified memory capabilities:\n",
                    "   Is integrated: ", deviceProperties.integrated, "\n",
                    "   Can map host memory: ", deviceProperties.canMapHostMemory, "\n",
                    "   Has unified addressing: ", deviceProperties.unifiedAddressing, "\n",
                    "   Supports managed memory: ", deviceProperties.managedMemory, "\n",
                    "   Supports direct managed memory access from host: ", deviceProperties.directManagedMemAccessFromHost, "\n",
                    "   Supports host register: ", deviceProperties.hostRegisterSupported, "\n",
                    "   Supports read-only host register flag: ", deviceProperties.hostRegisterReadOnlySupported);

    return result;
}


cudaError_t debug::DisplayAsyncCapabilities(const int device)
{
    cudaError_t    result           = cudaSuccess;
    cudaDeviceProp deviceProperties = {};

    result = cudaGetDeviceProperties_v2(&deviceProperties, device);
    DBG_PRINT_RETURN_ON_CUDA_ERROR(result);

    DBG_MSG_STD_OUT("Device ", device, " async capabilities:\n",
                    "   Concurrent kernels: ", deviceProperties.concurrentKernels, "\n",
                    "   Async engine count: ", deviceProperties.asyncEngineCount, "\n",
                    "   Supports stream priorities: ", deviceProperties.streamPrioritiesSupported);

    return result;
}