# - Try to find AMD Math Kernel Library (LibM)
# Once done, this will define
#  AMD_LIBM_FOUND
#  AMD_LIBM_INCLUDE_DIRS
#  AMD_LIBM_LIBRARY_DIRS
#  AMD_LIBM_LIBRARIES

message(STATUS "Finding AMD LibM library")

execute_process(
  COMMAND locate "include/amdlibm.h"
  COMMAND sed "s/.include.amdlibm.h//"
  OUTPUT_VARIABLE LIBM_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(LIBM_ROOT)
  message(STATUS "  found AMD LibM at ${LIBM_ROOT}")
  set(AMD_LIBM_FOUND true)

  set(AMD_LIBM_INCLUDE_DIRS ${LIBM_ROOT}/include)
  set(AMD_LIBM_LIBRARY_DIRS ${LIBM_ROOT}/lib/static)

  set(AMD_LIBM_LIBRARIES
    amdlibm
  )

else()
  message(STATUS "  missing AMD LibM")
  set(AMD_LIBM_FOUND false)
endif()
