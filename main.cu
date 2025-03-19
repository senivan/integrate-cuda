#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include <iomanip>
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

int main() {
    double x_start = -50, x_end = 50;
    double y_start = -50, y_end = 50;

    double *d_result, h_result = 0.0f;

    int maxIterations = 20;
    double absErrorThreshold = 0.000005;
    double relErrorThreshold = 0.0002;
    double prev_result = 0.0;

    int steps_x = 100;
    int steps_y = 100;

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

        integrateKernel<<<gridSize, blockSize, 0, stream>>>(x_start, x_end, y_start, y_end, steps_x, steps_y, d_result, 1);
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
