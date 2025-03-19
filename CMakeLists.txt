cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(integrate-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 52)


# Separate host and device compilation: note that main.cpp is host-only.
add_executable(integrate-cuda utils.cpp utils.hpp main.cu integrate.cu integrate.cuh)


# Link CUDA explicitly (if necessary)
target_link_libraries(integrate-cuda cuda)

# Enable CUDA separable compilation
set_target_properties(integrate-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
