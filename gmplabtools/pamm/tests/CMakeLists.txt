cmake_minimum_required(VERSION 3.16)

project(clusteringPPTests LANGUAGES CXX)

# require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# enable FetchContent to get Catch2
include(FetchContent)

# declare Catch2
FetchContent_Declare(Catch2                              # name of the content
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git  # the repository
  GIT_TAG        v2.13.4                                 # the tag
  )

# make available
FetchContent_MakeAvailable(Catch2)

# testing binary
add_executable(pamm_test test.cxx)
target_link_libraries(pamm_test PRIVATE base Catch2::Catch2)
target_include_directories(pamm_test
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )



# enable testing functionality
enable_testing()

# define test
add_test(
  NAME SOAP
  COMMAND $<TARGET_FILE:pamm_test> --success
  )