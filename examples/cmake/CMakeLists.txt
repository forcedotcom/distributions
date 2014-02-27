cmake_minimum_required(VERSION 2.8)
project(cmake_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")

add_subdirectory(../../src distributions)
include_directories(../../include)

add_executable(foo foo.cc)

target_link_libraries(foo
  distributions
)
