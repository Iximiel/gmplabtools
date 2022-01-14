cmake_minimum_required(VERSION 2.4)
project(clusteringPPTests)

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



# enable testing functionality
enable_testing()

# define test
add_test(
  NAME pamm_test
  COMMAND $<TARGET_FILE:pamm_test> --success
  )