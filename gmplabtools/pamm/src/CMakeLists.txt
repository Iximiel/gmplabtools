cmake_minimum_required(VERSION 2.4)
project(libpamm)

add_library(base libpamm++.cpp)
target_include_directories(base
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )


add_executable(pamm++ pamm++.cxx)
target_link_libraries(pamm++ base)