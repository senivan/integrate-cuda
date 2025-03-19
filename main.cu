#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include <iomanip>
#include "utils.hpp"
#include "integrate.cuh"

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Function to calculate absolute error
double absError(double result, double trueValue) {
    return fabs(result - trueValue);
}



// Function to calculate relative error
double relError(double result, double trueValue) {
    if (trueValue == 0.0f) {
        return (result == 0.0f) ? 0.0f : INFINITY;
    }
    return fabs((result - trueValue) / trueValue);
}

int main(int arg_c, char *argv[]) {
    if (arg_c != 3) {
        std::cerr << "Wrong number of arguments" << std::endl;
        return 1;
    }

    const int function_num = std::atoi(argv[1]);
    if (function_num < 1 && function_num > 3) {
        std::cerr << "Invalid function number" << std::endl;
        return 1;
    }
    conf_file_t conf_file = parse_conf_file(argv[2]);

    const double absErrorThreshold = conf_file.abs_err;
    const double relErrorThreshold = conf_file.rel_err;

    const double x_start = conf_file.x_start;
    const double x_end = conf_file.x_end;

    const double y_start = conf_file.y_start;
    const double y_end = conf_file.y_end;

    const int maxIterations = conf_file.max_iter;

    int steps_x = conf_file.init_steps_x-1;
    int steps_y = conf_file.init_steps_y-1;

    double *d_result, h_result = 0.0f;

    double prev_result = 0.0;


    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice));
    std::cout << std::setprecision(16);
    bool success = false;
    int iteration = 0;
    double absErr = std::numeric_limits<double>::max();
    double relErr = std::numeric_limits<double>::max();
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    while (iteration < maxIterations) {
        
        double zero = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_result, &zero, sizeof(double), cudaMemcpyHostToDevice));

        dim3 blockSize(16, 16);
        dim3 gridSize((steps_x + blockSize.x - 1) / blockSize.x, (steps_y + blockSize.y - 1) / blockSize.y);

        integrateKernel<<<gridSize, blockSize, 0, stream>>>(x_start, x_end, y_start, y_end, steps_x, steps_y, d_result, function_num);
        CHECK_CUDA_ERROR(cudaGetLastError());  
        CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        if (iteration > 0){
            absErr = h_result - prev_result;
            relErr = absErr / h_result;
        }

        std::cout << "Iteration " << iteration + 1 << ": " 
                  << "Absolute Error = " << absErr << ", "
                  << "Relative Error = " << relErr << std::endl;

        if (absErr <= absErrorThreshold && relErr <= relErrorThreshold) {
            success = true;
            break; 
        }
        steps_x *= 2;
        steps_y *= 2;
        prev_result = h_result;
        iteration++;
    }

    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    if (success) {
        std::cout << "Integral result: " << h_result << std::endl;
    } else {
        std::cerr << "Error did not converge within " << maxIterations << " iterations." << std::endl;
    }

    return 0;
}
