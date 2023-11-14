add_library(Utilities STATIC
                debug.cpp
                device.cpp
                streamtimer.cpp)

target_link_libraries(Utilities PRIVATE
                         CUDA::cudart_static
                         SharedDefaults)

target_link_libraries(Tests PRIVATE
                          Utilities)
