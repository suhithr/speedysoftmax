#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include "kernel_utils.cuh"
#include "kernels.cuh"

// move this to header file/util file later
void randomize_matrix(float *mat, int N) {
    struct timeval time{};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i=0; i < N; ++i)
    {
        float tmp = (float)(rand() % 10);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.0);
        mat[i] = tmp;
    }
}

void benchmark_kernel_softmax(int M, int N) {
    int matrix_size = M * N;
    int total_size = matrix_size * sizeof(float);

    printf("------------ Running CUDA softmax benchmark for MxN = (%d, %d) -------------\n", M, N);
    float *matrix = nullptr, *result = nullptr;
    matrix = (float*)malloc(total_size);
    result = (float*)malloc(total_size);

    randomize_matrix(matrix, matrix_size);
    randomize_matrix(result, matrix_size);

    float *matd, *resd;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&matd, total_size));
    CUDA_CHECK(cudaMalloc(&resd, total_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(matd, matrix, total_size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // run softmax kernel
    ms = run_kernel_0(matd, resd, M, N);

    printf("%d %f\n", M, ms);


    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(result, resd, total_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    free(matrix);
    free(result);
    cudaFree(matd);
    cudaFree(resd);

}

int main() {
    benchmark_kernel_softmax(1024, 32768);
}