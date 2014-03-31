# - Try to find Yeppp library
# Once done, this will define
#  YEPPP_FOUND
#  YEPPP_INCLUDE_DIRS
#  YEPPP_LIBRARY_DIRS
#  YEPPP_LIBRARIES

message(STATUS "Finding YEPPP library")

execute_process(
  COMMAND locate "library/headers/yepBuiltin.h"
  COMMAND sed "s/.library.headers.yepBuiltin.h//"
  OUTPUT_VARIABLE YEPPPROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(YEPPPROOT)
  message(STATUS "  found YEPPP library at ${YEPPPROOT}")
  set(YEPPP_FOUND true)

  set(YEPPP_INCLUDE_DIRS ${YEPPPROOT}/library/headers)
  set(YEPPP_LIBRARY_DIRS ${YEPPPROOT}/binaries/linux/x86_64)

  set(YEPPP_LIBRARIES
    yeppp
    m
  )

else()
  message(STATUS "  missing YEPPP library")
  set(YEPPP_FOUND false)
endif()
