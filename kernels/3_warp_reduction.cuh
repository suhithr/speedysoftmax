#ifndef WARP_REDUCTION_CUH
#define WARP_REDUCTION_CUH

#include <cstdio>
#include <math.h>
#include <cuda_runtime.h>

__global__ void softmax_warp_reduction(const float *__restrict__ matrix, float *result, int M, int N)
{
    float local_max = -INFINITY;
    float local_norm_val = 0;
    __shared__ float smem[32];

    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int warpid = tid / warpSize, warplane = tid % warpSize;

    for (int i = tid; i < N; i += blockDim.x)
    {
        float curr = matrix[row * N + i];
        if (curr > local_max)
        {
            local_norm_val *= expf(local_max - curr);
            local_max = curr;
        }
        local_norm_val += expf(curr - local_max);
    }
    __syncthreads();


    // the local warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    // only if we have more than 32 threads in the block
    if (blockDim.x > warpSize)
    {
        if (warplane == 0)
        {
            smem[warpid] = local_max;
        }
        __syncthreads();

        if (warpid == 0)
        {
            float val = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid] : -INFINITY; // handle case where num threads is less than warp
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            }
            if (tid == 0)
            {
                smem[tid] = val;
            }
        }
    }
    else{
        if (tid == 0) smem[0] = local_max;
    }
    __syncthreads();
    // need to handle case where numthreads < 32
    float global_max = smem[0];
    __syncthreads();

    local_norm_val = local_norm_val * expf(local_max - global_max);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        local_norm_val += __shfl_down_sync(0xFFFFFFFF, local_norm_val, offset);
    }

    if (blockDim.x > warpSize)
    {
        if (warplane == 0)
        {
            smem[warpid] = local_norm_val;
        }
        __syncthreads();
        if (warpid == 0)
        {
            float val = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid] : 0;
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            if (tid == 0)
            {
                smem[tid] = val;
            }
        }
    }
    else{
        if (tid == 0) smem[0] = local_norm_val;
    }
    __syncthreads();

    float global_norm = smem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x)
    {
        float curr = matrix[row * N + i];
        result[row * N + i] = expf(curr-global_max) / global_norm;
    }
    
}

void run_kernel_3(const float *__restrict__ matd, float *__restrict__ resd, int M, int N, int repeat_times)
{
    const uint BLOCK_SIZE = 1024;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(M);

    for (int i = 0; i < repeat_times; i++)
    {
        softmax_warp_reduction<<<grid_size, block_size>>>(matd, resd, M, N);
    }
}
#endif