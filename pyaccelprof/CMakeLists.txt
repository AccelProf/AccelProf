cmake_minimum_required(VERSION 3.16.3)
project(pyaccelprof)

# pybind11
set(PYBIND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11)
add_subdirectory(${PYBIND_DIR} ${CMAKE_BINARY_DIR}/pybind11)
pybind11_add_module(pyaccelprof pyaccelprof.cpp)

# compute_sanitizer
set(COMPUTE_SANITIZER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nv-compute/lib)
set(COMPUTE_SANITIZER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nv-compute/include)
target_include_directories(pyaccelprof PRIVATE ${COMPUTE_SANITIZER_INCLUDE_DIR})
target_link_directories(pyaccelprof PRIVATE ${COMPUTE_SANITIZER_LIB_DIR})
target_link_libraries(pyaccelprof PRIVATE compute_sanitizer)
