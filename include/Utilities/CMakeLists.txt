target_sources(Utilities PUBLIC
                  FILE_SET utilitiesPublicHeaders
                  TYPE HEADERS
                  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                  FILES
                      debug.h
                      device.h
                      streamtimer.h)