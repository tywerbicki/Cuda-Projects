cmake_minimum_required(VERSION 3.18)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_SUPPRESS_REGENERATION ON)

set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

project(CudaProjects LANGUAGES CUDA CXX)

# Target defaults that are shared between CUDA targets and non-CUDA targets.
add_library(SharedDefaults INTERFACE)
# Target defaults specific to CUDA targets.
add_library(CudaDefaults INTERFACE)

set_target_properties(CudaDefaults PROPERTIES
                         CUDA_RUNTIME_LIBRARY Static)

target_compile_features(SharedDefaults INTERFACE cxx_std_20)

target_compile_definitions(SharedDefaults INTERFACE
                              $<$<CONFIG:Debug>:_DEBUG>
                              $<$<CONFIG:Release>:_RELEASE>)

if (MSVC)
    target_compile_definitions(SharedDefaults INTERFACE
                                  $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:_ALLOW_RTCc_IN_STL>>)

    target_compile_options(SharedDefaults INTERFACE
                              $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc>
                              $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:/MDd /Od /Ob0 /RTC1 /RTCc>>
                              $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:/MD /O2 /Ob2 /Oi>>)
else()
    # TODO: fully support more compilers.
    target_compile_options(SharedDefaults INTERFACE
                              $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

add_subdirectory(src)
add_subdirectory(include)