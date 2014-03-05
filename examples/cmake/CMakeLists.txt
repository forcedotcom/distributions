cmake_minimum_required(VERSION 2.8)
project(cmake_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")



#-----------------------------------------------------------------------------
# BEGIN generic section

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
    GIT_REPOSITORY git@github.com:forcedotcom/distributions.git
    GIT_TAG 2.0
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
