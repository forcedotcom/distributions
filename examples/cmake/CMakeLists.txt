cmake_minimum_required(VERSION 2.8)
project(cmake_example)

add_subdirectory(../../src distributions)
include_directories(../../include)

add_executable(foo foo.cc)

target_link_libraries(foo
  distributions
)
