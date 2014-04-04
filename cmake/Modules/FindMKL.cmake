# - Try to find Intel Math Kernel Library (MKL)
# Once done, this will define
#  MKL_FOUND
#  MKL_INCLUDE_DIRS
#  MKL_LIBRARY_DIRS
#  MKL_LIBRARIES

message(STATUS "Finding Intel MKL library")

execute_process(
  COMMAND locate mkl_vml.h
  COMMAND grep -o ".*\\<mkl\\>"
  OUTPUT_VARIABLE MKL_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(MKL_ROOT)
  message(STATUS "  found Intel MKL at ${MKL_ROOT}")
  set(MKL_FOUND true)

  set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include)
  set(MKL_LIBRARY_DIRS ${MKL_ROOT}/lib/intel64)

  set(MKL_LIBRARIES
    mkl_intel_ilp64
    mkl_sequential
    mkl_core
    pthread
    m
  )

else()
  message(STATUS "  missing Intel MKL")
  set(MKL_FOUND false)
endif()
