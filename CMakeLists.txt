cmake_minimum_required(VERSION 3.28)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_SUPPRESS_REGENERATION ON)
set(CMAKE_SKIP_INSTALL_RULES ON)

set(CMAKE_CUDA_ARCHITECTURES 60 90)

project(CudaProjects LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# Target defaults that are shared between CUDA targets and non-CUDA targets.
add_library(SharedDefaults INTERFACE)
# Target defaults specific to CUDA targets.
add_library(CudaDefaults INTERFACE)

target_compile_features(SharedDefaults INTERFACE cxx_std_20)

target_compile_definitions(SharedDefaults INTERFACE
                              $<$<CONFIG:Debug>:_DEBUG>
                              $<$<CONFIG:Release>:_RELEASE _OPT_FOR_SPEED>)

target_link_libraries(CudaDefaults INTERFACE
                         CUDA::cudart_static)

if (MSVC)
    target_compile_definitions(SharedDefaults INTERFACE
                                  $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:_ALLOW_RTCc_IN_STL>>)

    target_compile_options(SharedDefaults INTERFACE
                              $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc>
                              $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:/MDd /Od /Ob0 /RTC1 /RTCc>>
                              $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:/MD /O2 /Ob2 /Oi>>)

    # 1. Always ignore the static C++ runtime libraries.
    # 2. Conditionally ignore the incorrect DLL runtime library.
    target_link_options(CudaDefaults INTERFACE
                           $<HOST_LINK:/NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib>
                           $<$<CONFIG:Debug>:$<HOST_LINK:/NODEFAULTLIB:msvcrt.lib>>
                           $<$<CONFIG:Release>:$<HOST_LINK:/NODEFAULTLIB:msvcrtd.lib>>)
else()
    # TODO: fully support more compilers and linkers.
    target_compile_options(SharedDefaults INTERFACE
                              $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

include(FetchContent)
include(GoogleTest)

FetchContent_Declare(googletest
                     GIT_REPOSITORY https://github.com/google/googletest.git
                     GIT_TAG b10fad38c4026a29ea6561ab15fc4818170d1c10
                     EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Tests)

target_link_libraries(Tests PRIVATE
                         CUDA::cudart_static
                         GTest::gtest_main
                         SharedDefaults)

add_subdirectory(src)
add_subdirectory(include)

gtest_discover_tests(Tests)
