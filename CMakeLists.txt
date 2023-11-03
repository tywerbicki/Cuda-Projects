cmake_minimum_required(VERSION 3.12)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_SUPPRESS_REGENERATION ON)

project(CudaProjects LANGUAGES CUDA CXX)

# Target defaults that are shared between CUDA targets and non-CUDA targets.
add_library(SharedDefaults INTERFACE)
# Target defaults specific to non-CUDA targets.
add_library(HostDefaults INTERFACE)
# Target defaults specific to CUDA targets.
add_library(CudaDefaults INTERFACE)

target_compile_features(SharedDefaults INTERFACE cxx_std_20)

target_compile_definitions(SharedDefaults INTERFACE
                              $<$<CONFIG:Debug>:_DEBUG>
                              $<$<CONFIG:Release>:_RELEASE>)

if (MSVC)
    target_compile_definitions(HostDefaults INTERFACE
                                  $<$<CONFIG:Debug>:_ALLOW_RTCc_IN_STL>)

    target_compile_options(HostDefaults INTERFACE
                              /W4 /EHsc
                              $<$<CONFIG:Debug>:/MDd /Od /Ob0 /RTC1 /RTCc>
                              $<$<CONFIG:Release>:/MD /O2 /Ob2 /Oi>)
endif()

add_subdirectory(src)
add_subdirectory(include)