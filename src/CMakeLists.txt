include_directories(../include)

set(DISTRIBUTIONS_SOURCE_FILES
  common.cc
  special.cc
  random.cc
  vector_math.cc
  clustering.cc
  models/nich.cc
  models/gp.cc
  models/niw.cc
)

install(DIRECTORY ../include/ DESTINATION include
  FILES_MATCHING PATTERN "*.h*")

if(PROTOBUF_FOUND)
  set(DISTRIBUTIONS_SOURCE_FILES io/schema.pb.cc ${DISTRIBUTIONS_SOURCE_FILES})
  install(DIRECTORY ../distributions/ DESTINATION include/distributions
    FILES_MATCHING PATTERN "*.proto")
  install(DIRECTORY ../include/ DESTINATION include
    FILES_MATCHING PATTERN "*.cc")
endif()

add_library(distributions_shared SHARED ${DISTRIBUTIONS_SOURCE_FILES})
target_link_libraries(distributions_shared ${DISTRIBUTIONS_SHARED_LIBS})
install(TARGETS distributions_shared LIBRARY DESTINATION lib)

add_executable(test_headers_shared test_headers.cc)
add_test(test_headers_shared test_headers_shared)
target_link_libraries(test_headers_shared distributions_shared)

if(PROTOBUF_FOUND)
  add_executable(test_protobuf_shared test_protobuf.cc)
  add_test(test_protobuf_shared test_protobuf_shared)
  target_link_libraries(test_protobuf_shared distributions_shared)
endif()
