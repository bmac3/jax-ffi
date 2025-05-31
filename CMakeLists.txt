cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)


find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)
# this works
set(CMAKE_CUDA_ARCHITECTURES 90)
# this doesn't work
# set(CMAKE_CUDA_ARCHITECTURES 90a)
find_package(CUDAToolkit REQUIRED)

add_library(_cuda_examples SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/jax_ffi_example/cuda_examples.cu")
set_target_properties(_cuda_examples PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CUDA_STANDARD 17
                                                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/jax_ffi_example")
target_include_directories(_cuda_examples PUBLIC ${XLA_DIR})
install(TARGETS _cuda_examples LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

