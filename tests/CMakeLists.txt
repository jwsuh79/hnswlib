add_executable(hnswalg_test unittests/hnswalg_test.cpp)
target_include_directories(hnswalg_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(hnswalg_test GTest::gtest_main)

add_executable(multivector_search_test cpp/multivector_search_test.cpp)
target_include_directories(multivector_search_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(multivector_search_test GTest::gtest_main)

add_executable(epsilon_search_test cpp/epsilon_search_test.cpp)
target_include_directories(epsilon_search_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(epsilon_search_test GTest::gtest_main)

add_executable(multiThread_replace_test cpp/multiThread_replace_test.cpp)
target_include_directories(multiThread_replace_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(multiThread_replace_test GTest::gtest_main)

add_executable(multiThreadLoad_test cpp/multiThreadLoad_test.cpp)
target_include_directories(multiThreadLoad_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(multiThreadLoad_test GTest::gtest_main)

# add_executable(updates_test cpp/updates_test.cpp)
# target_include_directories(updates_test PUBLIC ${CMAKE_SOURCE_DIR})
# target_link_libraries(updates_test GTest::gtest_main)

add_executable(searchKnnCloserFirst_test cpp/searchKnnCloserFirst_test.cpp)
target_include_directories(searchKnnCloserFirst_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(searchKnnCloserFirst_test GTest::gtest_main)

add_executable(searchKnnWithFilter_test cpp/searchKnnWithFilter_test.cpp)
target_include_directories(searchKnnWithFilter_test PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(searchKnnWithFilter_test GTest::gtest_main)

# add_executable(searchKnnWithFilter_test cpp/searchKnnWithFilter_test.cpp)
# target_include_directories(searchKnnWithFilter_test PUBLIC ${CMAKE_SOURCE_DIR})
# target_link_libraries(searchKnnWithFilter_test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hnswalg_test)
gtest_discover_tests(multivector_search_test)
gtest_discover_tests(epsilon_search_test)
gtest_discover_tests(multiThread_replace_test)
gtest_discover_tests(multiThreadLoad_test)
#gtest_discover_tests(updates_test)
gtest_discover_tests(searchKnnCloserFirst_test)
gtest_discover_tests(searchKnnWithFilter_test)
