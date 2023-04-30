cmake_minimum_required(VERSION 3.8)

project(Oobleck)

find_package(pybind11 REQUIRED)

add_subdirectory(modules/cppcoro)
add_subdirectory(planning)