# integrate-cuda
Done by: Sen Ivan, Viktor Paholok and Lidiia Semsichko

## Description
In this repo we have a program that leverages NVIDIA CUDA technology, to accelerate
compuration of numerical integration of a function. The function is defined in the
`integrate.cu` file. The program is able to compute the integral of a function
using the trapezoidal rule.
This is project for the course "Architecture of computer systems" at the Ukranian Catholic University.

## How to compile
This should work out of the box with cmake:
```bash
mkdir -p build
cd build
cmake ..
make
```

However, WARNING. Cmake file and program code was tailored to work on NVDIA Titan X with supports CUDA Compute Capability 5.2. Your experience may differ.
