cmake_minimum_required(VERSION 3.12)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_SUPPRESS_REGENERATION ON)

project(CudaProjects LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
add_subdirectory(include)