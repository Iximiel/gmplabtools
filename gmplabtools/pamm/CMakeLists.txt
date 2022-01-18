cmake_minimum_required(VERSION 3.16)

project(clusteringPP LANGUAGES CXX)

# require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(OPT_DEBUGWARNINGS "Activates all the warning in compilation -for developers-" OFF)
#the code subdir
add_subdirectory(src)

#the tests subdir
add_subdirectory(tests)