cmake_minimum_required(VERSION 3.10)
project(distributed_perception)

set(CMAKE_CXX_STANDARD 11)

ADD_SUBDIRECTORY(libgp)

include_directories(
  include
)

add_executable(distributed_perception main.cpp)

TARGET_LINK_LIBRARIES(distributed_perception gp)
