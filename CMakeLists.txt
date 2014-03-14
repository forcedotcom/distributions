cmake_minimum_required(VERSION 2.6)
project(distributions)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -Wno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mfpmath=sse -msse4.1 -ffast-math -funsafe-math-optimizations")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx") # on sandy bridge and later
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math -ffinite-math-only -fvect-cost-model -mrecip")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native") # for testing

enable_testing()
include(CTest)

add_subdirectory(src)
add_subdirectory(benchmarks)
