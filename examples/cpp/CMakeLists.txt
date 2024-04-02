# examples
add_executable(example_search src/example_search.cpp)
target_include_directories(example_search PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_epsilon_search src/example_epsilon_search.cpp)
target_include_directories(example_epsilon_search PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_multivector_search src/example_multivector_search.cpp)
target_include_directories(example_multivector_search PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_filter src/example_filter.cpp)
target_include_directories(example_filter PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_replace_deleted src/example_replace_deleted.cpp)
target_include_directories(example_replace_deleted PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_mt_search src/example_mt_search.cpp)
target_include_directories(example_mt_search PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_mt_filter src/example_mt_filter.cpp)
target_include_directories(example_mt_filter PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(example_mt_replace_deleted src/example_mt_replace_deleted.cpp)
target_include_directories(example_mt_replace_deleted PUBLIC ${CMAKE_SOURCE_DIR})