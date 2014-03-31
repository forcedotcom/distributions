include_directories(../include)

add_executable(sample_from_scores sample_from_scores.cc)
target_link_libraries(sample_from_scores
  ${DISTRIBUTIONS_SHARED_LIBS}
)

add_executable(sample_assignment_from_py sample_assignment_from_py.cc)
target_link_libraries(sample_assignment_from_py
  ${DISTRIBUTIONS_SHARED_LIBS}
)

add_executable(special special.cc)
target_link_libraries(special
  ${DISTRIBUTIONS_SHARED_LIBS}
)

add_executable(classifier classifier.cc)
target_link_libraries(classifier
  ${DISTRIBUTIONS_SHARED_LIBS}
)
