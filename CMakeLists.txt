cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# Include the CUDA extensions if possible
include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
  gpu_ops 
  ${CMAKE_CURRENT_LIST_DIR}/lib/cuda_examples.cu 
  ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
install(TARGETS gpu_ops LIBRARY DESTINATION .)
