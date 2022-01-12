cmake_minimum_required(VERSION 2.4)
project(clusteringPP)

add_library(base libpamm++.cpp)

add_executable(pamm++ pamm++.cxx)
target_link_libraries(pamm++ base)