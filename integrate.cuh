#ifndef KERNEL_INCLUDE
#define KERNEL_INCLUDE
__device__ double function_1(const double x, const double y);
__device__ double function_2(const double x, const double y);
__device__ double function_3(const double x, const double y);

__global__ void integrateKernel(double x_start, double x_end, double y_start, double y_end, int steps_x, int steps_y, double *result, int function_id);
#endif
