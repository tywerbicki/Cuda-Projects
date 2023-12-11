add_executable(ExclusiveScan
                   exclusivescan.cu)

target_link_libraries(ExclusiveScan PRIVATE
                         CudaDefaults
                         SharedDefaults)
