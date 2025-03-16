cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(integrate-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(integrate-cuda main.cpp kernels/integrand.cu)

# Enable CUDA separable compilation
set_target_properties(integrate-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
