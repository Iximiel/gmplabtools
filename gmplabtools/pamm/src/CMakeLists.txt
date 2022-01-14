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

#include(FetchContent)
#
## declare OpenBLAS
#FetchContent_Declare(OpenBLAS                           # name of the conte
#  GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git  # the repository
#  GIT_TAG        v0.3.19                           # the tag
#  )
#
## make available
#FetchContent_MakeAvailable(OpenBLAS)
#
#  # enable FetchContent to get BLASpp
#
##find_package(BLAS)
## declare BLASpp
#FetchContent_Declare(BLASpp                              # name of the content
#  GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git  # the repository
#  GIT_TAG        2021.04.01                             # the tag
#  )
#
## make available
#FetchContent_MakeAvailable(BLASpp)

add_executable(pamm++ pamm++.cxx)
target_link_libraries(pamm++ base)