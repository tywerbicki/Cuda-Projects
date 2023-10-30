add_executable(Saxpy
                   main.cpp
                   saxpy.cu)

target_link_libraries(Saxpy PRIVATE Utilities)