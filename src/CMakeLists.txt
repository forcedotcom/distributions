include_directories(../include)

set(DISTRIBUTIONS_SOURCE_FILES
  common.cc
  special.cc
  random.cc
  std_wrapper.cc
  vector_math.cc
  clustering.cc
  models/nich.cc
  models/gp.cc
)

add_library(distributions_shared SHARED ${DISTRIBUTIONS_SOURCE_FILES})
add_library(distributions_static STATIC ${DISTRIBUTIONS_SOURCE_FILES})
install(TARGETS distributions_shared LIBRARY DESTINATION lib)
install(TARGETS distributions_static ARCHIVE DESTINATION lib)

add_executable(test_headers_static test_headers.cc)
add_test(test_headers_static test_headers_static)
target_link_libraries(test_headers_static
  ${MKL_LIBRARIES}
  distributions_static
)

add_executable(test_headers_shared test_headers.cc)
add_test(test_headers_shared test_headers_static)
target_link_libraries(test_headers_shared
  ${MKL_LIBRARIES}
  distributions_shared
)
