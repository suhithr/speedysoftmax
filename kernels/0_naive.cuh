#ifndef NAIVE_CUH
#define NAIVE_CUH

#include <cstdio>
#include <math.h>
#include <cuda_runtime.h>

__global__ void softmax_naive(const float *matrix, float *result, int M, int N)
{
    float x_max = -INFINITY;
    float norm_val = 0;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M)
    {
        for (int i = 0; i < N; i++) {
            int loc = row * N + i;
            x_max = max(x_max, matrix[loc]);
        }
        for (int i = 0; i < N; i++) {
            int loc = row * N + i;
            norm_val += exp(matrix[loc] - x_max);
        }
        for (int i = 0; i < N; i++) {
            int loc = row * N + i;
            result[loc] = exp(matrix[loc] - x_max) / norm_val;
        }
    }
}

void run_kernel_0(const float* __restrict__ matd, float* __restrict__ resd, int M, int N, int repeat_times) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    for (int i = 0; i < repeat_times; i++)
    {
        softmax_naive<<<grid_size, block_size>>>(matd, resd, M, N);
    }
}
#endif