cmake_minimum_required(VERSION 3.13)

project(cplanning)

pybind11_add_module(cplanning gpu_stage_mapping.cpp)

# set(VERSION_INFO 0.1.0)
target_compile_definitions(cplanning PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_options(cplanning PRIVATE -fcoroutines)
target_link_libraries(cplanning PRIVATE cppcoro::cppcoro)
target_include_directories(cplanning
PRIVATE ${CMAKE_SOURCE_DIR}/modules/json/single_include
PRIVATE ${CMAKE_SOURCE_DIR}/modules/concurrentqueue
PRIVATE ${CMAKE_SOURCE_DIR}/modules/parallel-hashmap
PRIVATE ${CMAKE_SOURCE_DIR}/modules/cppcoro/include)