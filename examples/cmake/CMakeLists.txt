cmake_minimum_required(VERSION 2.8)
project(cmake_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")



#-----------------------------------------------------------------------------
# BEGIN generic section
#
# To build with local copy of distributions, first build distributions, and
# then set the environment variable
# export DISTRIBUTIONS_PATH=/path/to/distributions
# for example in ~/.bashrc or $VIRTUAL_ENV/bin/postactivate

include(ExternalProject)
if(DEFINED ENV{DISTRIBUTIONS_PATH})
  ExternalProject_Add(distributions
    PREFIX distributions
    DOWNLOAD_COMMAND ""
    SOURCE_DIR $ENV{DISTRIBUTIONS_PATH}
    BINARY_DIR $ENV{DISTRIBUTIONS_PATH}/build
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
else()
  ExternalProject_Add(distributions
    PREFIX distributions
    GIT_REPOSITORY https://github.com/forcedotcom/distributions.git
    INSTALL_COMMAND ""
  )
endif()
ExternalProject_Get_Property(distributions SOURCE_DIR)
ExternalProject_Get_Property(distributions BINARY_DIR)
include_directories("${SOURCE_DIR}/include")
link_directories("${BINARY_DIR}/src")

# END generic section
#-----------------------------------------------------------------------------



add_executable(foo foo.cc)

target_link_libraries(foo
  distributions_shared
)
