cmake_minimum_required(VERSION 3.13)

project(pipeline_template)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(pipeline_template pipeline_template.cpp)

# set(VERSION_INFO 0.1.0)
target_compile_definitions(pipeline_template PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_options(pipeline_template PRIVATE -fcoroutines)
target_link_libraries(pipeline_template PRIVATE cppcoro::cppcoro)
target_include_directories(pipeline_template
PRIVATE ${CMAKE_SOURCE_DIR}/modules/json/single_include
PRIVATE ${CMAKE_SOURCE_DIR}/modules/concurrentqueue
PRIVATE ${CMAKE_SOURCE_DIR}/modules/parallel-hashmap
PRIVATE ${CMAKE_SOURCE_DIR}/modules/cppcoro/include)