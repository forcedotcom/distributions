include_directories(../include)

set(DISTRIBUTIONS_SOURCE_FILES
  common.cc
  special.cc
  random.cc
  std_wrapper.cc
  vector_math.cc
  clustering.cc
)

add_library(distributions_shared SHARED ${DISTRIBUTIONS_SOURCE_FILES})
add_library(distributions_static STATIC ${DISTRIBUTIONS_SOURCE_FILES})
install(TARGETS distributions_shared LIBRARY DESTINATION lib)
install(TARGETS distributions_static ARCHIVE DESTINATION lib)

add_executable(test_models_static test_models.cc)
add_test(test_models_static test_models_static)
target_link_libraries(test_models_static distributions_static)

add_executable(test_models_shared test_models.cc)
add_test(test_models_shared test_models_static)
target_link_libraries(test_models_shared distributions_shared)
