include_directories(../include)

set(DISTRIBUTIONS_SOURCE_FILES
  common.cc
  special.cc
  random.cc
  vector_math.cc
  clustering.cc
  models/nich.cc
  models/gp.cc
)

if(PROTOBUF_FOUND)
  set(DISTRIBUTIONS_SOURCE_FILES io/schema.pb.cc ${DISTRIBUTIONS_SOURCE_FILES})
endif()

add_library(distributions_shared SHARED ${DISTRIBUTIONS_SOURCE_FILES})
add_library(distributions_static STATIC ${DISTRIBUTIONS_SOURCE_FILES})
install(TARGETS distributions_shared LIBRARY DESTINATION lib)
install(TARGETS distributions_static ARCHIVE DESTINATION lib)

add_executable(test_headers_static test_headers.cc)
add_test(test_headers_static test_headers_static)
target_link_libraries(test_headers_static ${DISTRIBUTIONS_STATIC_LIBS})

add_executable(test_headers_shared test_headers.cc)
add_test(test_headers_shared test_headers_shared)
target_link_libraries(test_headers_shared ${DISTRIBUTIONS_SHARED_LIBS})

if(PROTOBUF_FOUND)
  add_executable(test_protobuf_static test_protobuf.cc)
  add_test(test_protobuf_static test_protobuf_static)
  target_link_libraries(test_protobuf_static ${DISTRIBUTIONS_STATIC_LIBS})

  add_executable(test_protobuf_shared test_protobuf.cc)
  add_test(test_protobuf_shared test_protobuf_shared)
  target_link_libraries(test_protobuf_shared ${DISTRIBUTIONS_SHARED_LIBS})
endif()
