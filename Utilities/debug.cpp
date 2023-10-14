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