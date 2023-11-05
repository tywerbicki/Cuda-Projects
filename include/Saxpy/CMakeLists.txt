target_sources(Saxpy PUBLIC
                  FILE_SET saxpyPublicHeaders
                  TYPE HEADERS
                  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                  FILES
                      saxpy.h)