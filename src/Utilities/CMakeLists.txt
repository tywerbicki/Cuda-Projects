add_library(Utilities STATIC
                debug.cpp
                device.cpp
                streamtimer.cpp)

target_link_libraries(Utilities PRIVATE
                         HostDefaults
                         SharedDefaults)