cmake_minimum_required(VERSION 3.13)

project(pipeline_template)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(TBB REQUIRED)

pybind11_add_module(pipeline_template pipeline_template.cpp bind.cpp)

target_compile_definitions(pipeline_template PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_options(pipeline_template PRIVATE -fcoroutines)
target_link_libraries(pipeline_template PRIVATE cppcoro::cppcoro TBB::tbb)
target_include_directories(pipeline_template
PRIVATE ${CMAKE_SOURCE_DIR}/modules/json/single_include
PRIVATE ${CMAKE_SOURCE_DIR}/modules/cppcoro/include
PRIVATE $ENV{CONDA_PREFIX}/include)

add_executable(pipeline_template_test pipeline_template.cpp pipeline_template_test.cpp)
target_compile_options(pipeline_template_test PRIVATE -fcoroutines)
target_link_libraries(pipeline_template_test PRIVATE cppcoro::cppcoro TBB::tbb)
target_include_directories(pipeline_template_test
PRIVATE ${CMAKE_SOURCE_DIR}/modules/json/single_include
PRIVATE ${CMAKE_SOURCE_DIR}/modules/cppcoro/include
PRIVATE $ENV{CONDA_PREFIX}/include)
