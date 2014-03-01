include_directories(../include)

add_library(distributions
  common.cc
  special.cc
  random.cc
)

install(TARGETS distributions ARCHIVE DESTINATION lib)

add_executable(test_models
  test_models.cc
)
add_test(test_models
  test_models
)
target_link_libraries(test_models
  distributions
)
