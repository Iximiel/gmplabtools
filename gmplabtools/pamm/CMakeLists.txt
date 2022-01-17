cmake_minimum_required(VERSION 3.16)

project(clusteringPP LANGUAGES CXX)

# require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#the code subdir
add_subdirectory(src)

#the tests subdir
add_subdirectory(tests)