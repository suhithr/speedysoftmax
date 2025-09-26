#ifndef ONLINE_SHARED_CUH
#define ONLINE_SHARED_CUH

#include <cstdio>
#include <math.h>
#include <cuda_runtime.h>

template <const uint BLOCK_SIZE>
__global__ void softmax_online_shared(const float *matrix, float *result, int M, int N)
{
    float x_max = -INFINITY;
    float local_norm_val = 0;
    __shared__ float smem_max[BLOCK_SIZE];
    __shared__ float smem_norm[BLOCK_SIZE];
    int row = blockIdx.x * N;
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x)
    {
        float curr = matrix[row + i];
        if (curr > x_max) {
            local_norm_val = (local_norm_val*exp(x_max-curr)) + 1;
            x_max = max(x_max, curr);
        } else {
            local_norm_val += exp(curr - x_max);
        }
    }
    smem_max[tid] = x_max;
    __syncthreads();

    // reduce the max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            if (smem_max[tid] < smem_max[tid+stride])
            {

            }
            int max_val = max(smem_max[tid], smem_max[tid+stride]);
            smem_norm[tid] = (
                (smem_norm[tid] * exp(smem_max[tid] - max_val)) + (smem_norm[tid+stride] * exp(smem_max[tid+stride] - max_val))
            );
            smem_max[tid] = max_val;
        }
        // sync before next iteration
        __syncthreads();
    }
    float global_max = smem_max[0]; // global max
    float global_norm = smem_norm[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        float curr = matrix[row + i];
        result[row + i] = exp(curr - global_max) / global_norm;
    }
}

void run_kernel_2(const float* __restrict__ matd, float* __restrict__ resd, int M, int N, int repeat_times) {
    // grid size and block size for this kernel
    // change as necessary
    const uint BLOCK_SIZE = 256;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    for (int i = 0; i < repeat_times; i++)
    {
        softmax_online_shared<BLOCK_SIZE><<<grid_size, block_size>>>(matd, resd, M, N);
    }
}

#endif