cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(distributions)
set(DISTRIBUTIONS_SHARED_LIBS distributions_shared m)
set(DISTRIBUTIONS_STATIC_LIBS distributions_static m)

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  # for anaconda builds
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7")
  # clang complains about register
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter -Wno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mfpmath=sse -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funsafe-math-optimizations")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math -ffinite-math-only -fvect-cost-model -mrecip")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx") # on sandy bridge and later
if(DEFINED ENV{CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS}") # for testing
endif()
message("CXX_FLAGS =${CMAKE_CXX_FLAGS}")

if(DEFINED EXTRA_INCLUDE_PATH)
  include_directories(${EXTRA_INCLUDE_PATH})
endif()
if(DEFINED EXTRA_LIBRARY_PATH)
  link_directories(${EXTRA_LIBRARY_PATH})
endif()

find_package(Yeppp)
if(YEPPP_FOUND)
  message(STATUS "Using YEPPP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_YEPPP")
  include_directories(${YEPPP_INCLUDE_DIRS})
  link_directories(${YEPPP_LIBRARY_DIRS})
  #set(DISTRIBUTIONS_SHARED_LIBS ${DISTRIBUTIONS_SHARED_LIBS} ${YEPPP_LIBRARIES})
  #set(DISTRIBUTIONS_STATIC_LIBS ${DISTRIBUTIONS_STATIC_LIBS} ${YEPPP_LIBRARIES})
endif()

find_package(LibM)
if(AMD_LIBM_FOUND)
  message(STATUS "Using AMD LibM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_AMD_LIBM")
  include_directories(${AMD_LIBM_INCLUDE_DIRS})
  link_directories(${AMD_LIBM_LIBRARY_DIRS})
  set(DISTRIBUTIONS_SHARED_LIBS ${DISTRIBUTIONS_SHARED_LIBS} ${AMD_LIBM_LIBRARIES})
  set(DISTRIBUTIONS_STATIC_LIBS ${DISTRIBUTIONS_STATIC_LIBS} ${AMD_LIBM_LIBRARIES})
endif()

find_package(MKL)
if(MKL_FOUND)
  message(STATUS "Using Intel MKL")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_INTEL_MKL")
  include_directories(${MKL_INCLUDE_DIRS})
  link_directories(${MKL_LIBRARY_DIRS})
  set(DISTRIBUTIONS_SHARED_LIBS ${DISTRIBUTIONS_SHARED_LIBS} ${MKL_LIBRARIES})
  set(DISTRIBUTIONS_STATIC_LIBS ${DISTRIBUTIONS_STATIC_LIBS} ${MKL_LIBRARIES})
endif()

if (DEFINED ENV{DISTRIBUTIONS_USE_PROTOBUF})
  find_package(Protobuf)
  if(NOT PROTOBUF_FOUND)
    message(FATAL_ERROR
      "DISTRIBUTIONS_USE_PROTOBUF specified but protobuf not found")
  endif()
  message(STATUS "Using Google Protocol Buffers")
  set(DISTRIBUTIONS_SHARED_LIBS ${DISTRIBUTIONS_SHARED_LIBS} protobuf)
  set(DISTRIBUTIONS_STATIC_LIBS ${DISTRIBUTIONS_STATIC_LIBS} protobuf)
endif()

enable_testing()
include(CTest)

add_subdirectory(src)
add_subdirectory(benchmarks)

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "distributions")
include(CPack)
