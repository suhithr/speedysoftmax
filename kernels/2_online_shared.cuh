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
    int row = blockIdx.x;
    if (row >= M)
        return;

    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x)
    {
        float curr = matrix[row * N + i];
        if (curr > x_max)
        {
            local_norm_val *= exp(x_max - curr);
            x_max = curr;
        }
        local_norm_val += exp(curr - x_max);
    }
    __syncthreads();

    smem_max[tid] = x_max;
    __syncthreads();

    // reduce the max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            smem_max[tid] = max(smem_max[tid], smem_max[tid + stride]);
        }
        // sync before next iteration
        __syncthreads();
    }
    float global_max = smem_max[0]; // global max

    // Re-normalizes according to global maximum p/b property of exponentials
    smem_norm[tid] = local_norm_val * exp(smem_max[tid] - global_max);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            smem_norm[tid] += smem_norm[tid + stride];
        }
        __syncthreads();
    }
    float global_norm = smem_norm[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x)
    {
        float curr = matrix[row * N + i];
        result[row * N + i] = exp(curr - global_max) / global_norm;
    }
}

void run_kernel_2(const float *__restrict__ matd, float *__restrict__ resd, int M, int N, int repeat_times)
{
    // grid size and block size for this kernel
    // change as necessary
    const uint BLOCK_SIZE = 1024;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(M);

    for (int i = 0; i < repeat_times; i++)
    {
        softmax_online_shared<BLOCK_SIZE><<<grid_size, block_size>>>(matd, resd, M, N);
    }
}

#endif