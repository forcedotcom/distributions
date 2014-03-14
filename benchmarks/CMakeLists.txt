include_directories(../include)

add_executable(sample_from_scores sample_from_scores.cc)
target_link_libraries(sample_from_scores
     distributions_shared
)

add_executable(sample_assignment_from_py sample_assignment_from_py.cc)
target_link_libraries(sample_assignment_from_py
     distributions_shared
)
