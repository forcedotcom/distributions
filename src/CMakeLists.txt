include_directories(../include)

add_library(distributions_shared SHARED
  common.cc
  special.cc
  random.cc
  std_wrapper.cc
  vector_math.cc
)
install(TARGETS distributions_shared LIBRARY DESTINATION lib)

add_library(distributions_static STATIC
  common.cc
  special.cc
  random.cc
  std_wrapper.cc
  vector_math.cc
)
install(TARGETS distributions_static ARCHIVE DESTINATION lib)

add_executable(test_models
  test_models.cc
)
add_test(test_models
  test_models
)
target_link_libraries(test_models
  distributions_shared
)
