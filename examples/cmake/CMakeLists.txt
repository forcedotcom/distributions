cmake_minimum_required(VERSION 2.8)
project(cmake_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


#-----------------------------------------------------------------------------
# BEGIN generic section
# germ of a future FindDistributions.cmake
find_path(DISTRIBUTIONS_INCLUDE_DIR distributions/random.hpp)
if(NOT DISTRIBUTIONS_INCLUDE_DIR)
  message(FATAL_ERROR
    "distributions include dir not found, try setting CMAKE_PREFIX_PATH")
else()
  message(STATUS
    "using distributions include dir ${DISTRIBUTIONS_INCLUDE_DIR}")
endif()
find_library(DISTRIBUTIONS_LIBRARIES distributions_shared)
if(NOT DISTRIBUTIONS_LIBRARIES)
  message(FATAL_ERROR
    "distributions libraries not found, try setting CMAKE_PREFIX_PATH")
endif()
if(DEFINED ENV{DISTRIBUTIONS_USE_PROTOBUF})
  find_package(Protobuf)
  if(NOT PROTOBUF_FOUND)
    message(FATAL_ERROR
      "DISTRIBUTIONS_USE_PROTOBUF specified but protobuf not found")
  endif()
  set(DISTRIBUTIONS_LIBRARIES ${DISTRIBUTIONS_LIBRARIES} protobuf)
endif()
# END generic section
#-----------------------------------------------------------------------------


add_executable(foo foo.cc)
include_directories(${DISTRIBUTIONS_INCLUDE_DIR})
target_link_libraries(foo
  ${DISTRIBUTIONS_LIBRARIES}
)
