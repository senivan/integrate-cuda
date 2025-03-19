cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(integrate-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(integrate-cuda integrate.cu integrate.cuh main.cu)

# Enable CUDA separable compilation
set_target_properties(integrate-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
