add_executable(Saxpy
                   main.cpp
                   saxpy.cu
                   settings.h)

target_link_libraries(Saxpy PRIVATE
                         CudaDefaults
                         SharedDefaults
                         Utilities)