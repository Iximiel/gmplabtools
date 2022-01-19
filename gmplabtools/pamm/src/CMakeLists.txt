cmake_minimum_required(VERSION 3.16)

project(clusteringPPExec LANGUAGES CXX)


# require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(${OPT_DEBUGWARNINGS})
    if(MSVC)
      add_compile_options(/W4 /WX)
    else()
      add_compile_options(
          -Wall
          -Wextra
          -Wshadow
          -Wnon-virtual-dtor
          -Wold-style-cast
          -Wcast-align
          -Wunused
          -Woverloaded-virtual
          -Wpedantic
          -Wconversion
          -Wsign-conversion
          -Wnull-dereference
          -Wdouble-promotion
          -Wformat=2
          -Wduplicated-cond
          -Wduplicated-branches
          -Wlogical-op
          -Wuseless-cast
          #-Wlifetime
          )
    endif()
endif()

include(FetchContent)

#FetchContent_Declare(gcem
#    GIT_REPOSITORY https://github.com/kthohr/gcem.git
#    GIT_TAG v1.13.1
#)
#FetchContent_MakeAvailable(gcem)

FetchContent_Declare(DynamicMatrices
    GIT_REPOSITORY https://github.com/Iximiel/dynamicMatrices.git
    #GIT_TAG V0.1
    GIT_TAG 733da9934159b8a191aefce6eb6cdcd76fabc819
)
# make available
FetchContent_MakeAvailable(DynamicMatrices)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

add_library(base libpamm++.cpp libPammObjects.cpp)
target_link_libraries(base DynamicMatrices::DynamicMatrices Eigen3::Eigen)
target_include_directories(base
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
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
