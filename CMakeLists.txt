cmake_minimum_required(VERSION 2.6)
project(distributions)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")

enable_testing()
include(CTest)

add_subdirectory(src)
