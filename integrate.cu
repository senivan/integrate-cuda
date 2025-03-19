#include "integrate.cuh"
#include <cuda_runtime.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                         __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double function_1(const double x, const double y) {
    double sum_res = 0.0;
    for (int i = -2; i < 3; i++){
        for (int j = -2; j < 3; j++){
            sum_res += 1/(5*(i+2) + j + 3 + powf((x - 16*j), 6) + powf((y - 16*i), 6));
        }
    }

    sum_res += 0.002;
    return 1/sum_res;
}

__device__ double function_2(const double x, const double y){
    constexpr int a = 20;
    constexpr double b = 0.2;
    constexpr double c = 2 * M_PI;

    return -a *expf(-b * sqrtf((static_cast<double>(1)/2)*(powf(x, 2) + powf(y, 2))))
    - expf((static_cast<double>(1)/2) * (cosf(c * x) + cosf(c * y))) + a + expf(1);
}

__device__ double function_3(const double x, const double y){
    constexpr int m = 5;
    const int a1[] = {1, 2, 1, 1, 5};
    const int a2[] = {4, 5, 1, 2, 4};
    const int c[] = {2, 1, 4, 7, 2};

    double res = 0;

    for (size_t i = 0; i < m; i++) {
        res += c[i] * expf(-M_1_PI*(powf(x - a1[i], 2) + powf(y - a2[i], 2))) *
        cosf(M_PI*(powf(x - a1[i], 2) + powf(y - a2[i], 2)));
    }

    return -res;
}

__global__ void integrateKernel(double x_start, double x_end, double y_start, double y_end, int steps_x, int steps_y, double* result, int function_id){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double shared_result[256];

    double dx = (x_end - x_start) / steps_x;
    double dy = (y_end - y_start) / steps_y;

    double x = x_start + idx * dx;
    double y = y_start + idy * dy;

    double res = 0.0;
    
    if (idx < steps_x && idy < steps_y){
        switch (function_id){
            case 1:
                res = function_1(x, y) * dx * dy;
                break;
            case 2:
                res = function_2(x, y) * dx * dy;
                break;
            case 3:
                res = function_3(x, y) * dx * dy;
                break;
        }
    }
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    shared_result[tid] = res;

    __syncthreads();
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1){
        if (tid < s){
            shared_result[tid] += shared_result[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        myAtomicAdd(result, shared_result[0]);
    }
    // myAtomicAdd(result, res * dx * dy);
}