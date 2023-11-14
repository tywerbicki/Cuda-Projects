add_library(Saxpy STATIC
                saxpy.cu
                settings.h)

target_link_libraries(Saxpy PRIVATE
                         CudaDefaults
                         SharedDefaults
                         Utilities)

target_sources(Tests PRIVATE
                   saxpy.test.cpp)

target_link_libraries(Tests PRIVATE
                         Saxpy)
