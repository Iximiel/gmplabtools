cmake_minimum_required(VERSION 3.16)

project(clusteringPPExec LANGUAGES CXX)

# require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(base libpamm++.cpp)
target_include_directories(base
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )


add_executable(pamm++ pamm++.cxx)
target_link_libraries(pamm++ base)