cmake_minimum_required(VERSION 3.20)

project(hnswlib_examples LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 11)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
  if(COMPILER_SUPPORT_NATIVE_FLAG)
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
    message("set -march=native flag")
  else()
    check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
    if(COMPILER_SUPPORT_M1_FLAG)
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
      message("set -mcpu=apple-m1 flag")
    endif()
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
endif()

# examples
add_executable(example_search src/example_search.cpp)
target_include_directories(example_search PUBLIC "../../")

add_executable(example_epsilon_search src/example_epsilon_search.cpp)
target_include_directories(example_epsilon_search PUBLIC "../../")

add_executable(example_multivector_search src/example_multivector_search.cpp)
target_include_directories(example_multivector_search PUBLIC "../../")

add_executable(example_filter src/example_filter.cpp)
target_include_directories(example_filter PUBLIC "../../")

add_executable(example_replace_deleted src/example_replace_deleted.cpp)
target_include_directories(example_replace_deleted PUBLIC "../../")

add_executable(example_mt_search src/example_mt_search.cpp)
target_include_directories(example_mt_search PUBLIC "../../")

add_executable(example_mt_filter src/example_mt_filter.cpp)
target_include_directories(example_mt_filter PUBLIC "../../")

add_executable(example_mt_replace_deleted src/example_mt_replace_deleted.cpp)
target_include_directories(example_mt_replace_deleted PUBLIC "../../")