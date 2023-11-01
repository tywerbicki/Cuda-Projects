cmake_minimum_required(VERSION 3.12)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_SUPPRESS_REGENERATION ON)

project(CudaProjects LANGUAGES CUDA CXX)

add_library(Defaults INTERFACE)

# set_target_properties(Defaults PROPERTIES
#                          CXX_STANDARD 20
#                          CXX_STANDARD_REQUIRED ON
#                          CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_STANDARD 20)

target_compile_features(Defaults INTERFACE cxx_std_20)

add_subdirectory(src)
add_subdirectory(include)