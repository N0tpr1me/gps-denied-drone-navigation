# Gazebo Harmonic (gz-sim 8) world system: StructureHoldWind.
# Built inside the SITL image (sim2real/docker/Dockerfile); on a workstation with
# libgz-sim8-dev:  cmake -S . -B build && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(structure_hold_gz_plugins CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(StructureHoldWind SHARED WindAdapter.cc)
target_link_libraries(StructureHoldWind PRIVATE gz-sim8::gz-sim8 gz-plugin2::register)

# No Gazebo dependency: the parity test builds this one with plain g++ as well.
add_executable(disturbance_parity parity_main.cc)
