#include <cuda_runtime.h>
#include <iostream>
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

void run_kernel(int kernel_num, const float* __restrict__ matd, float* __restrict__ resd, int M, int N, int repeat_times) {
    switch (kernel_num)
    {
        case 0:
            run_kernel_0(matd, resd, M, N, repeat_times);
            break;
        case 1:
            run_kernel_1(matd, resd, M, N, repeat_times);
            break;
        case 2:
            run_kernel_2(matd, resd, M, N, repeat_times);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}

void benchmark_kernel_softmax(int M, int N, int kernel_num) {
    int REPEAT_TIMES = 10;
    int matrix_size = M * N;
    int total_size = matrix_size * sizeof(float);

    printf("------------ Running CUDA softmax benchmark for MxN = (%d, %d) -------------\n", M, N);
    float *matrix = nullptr, *result = nullptr, *result_ref = nullptr;
    matrix = (float*)malloc(total_size);
    result = (float*)malloc(total_size);

    randomize_matrix(matrix, matrix_size);
    randomize_matrix(result, matrix_size);

    float *matd, *resd, *resd_ref;
    float ms = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&matd, total_size));
    CUDA_CHECK(cudaMalloc(&resd, total_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    result_ref = (float*)malloc(total_size);
    CUDA_CHECK(cudaMalloc(&resd_ref, total_size));

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(matd, matrix, total_size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // verify correctness and avoid cold start
    if (kernel_num > 0)
    {
        run_kernel_0(matd, resd, M, N, 1);
        run_kernel(kernel_num, matd, resd_ref, M, N, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(result, resd, total_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result_ref, resd_ref, total_size, cudaMemcpyDeviceToHost));

        if (!verify_matrix(result, result_ref, matrix_size))
        {
            throw std::runtime_error("Kernel outputs do not match");
        } else {
            std::cout << "Verified correctness for selected kernel\n";
        }
    } 


    // run softmax kernel
    cudaEventRecord(start);
    run_kernel(kernel_num, matd, resd, M, N, REPEAT_TIMES);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms / REPEAT_TIMES);

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

int main(int argc, char **argv) {
    if (argc < 2)
    {
        std::cerr << "Select a kernel (range 0 - 2)" << std::endl;
        exit(EXIT_FAILURE);
    }
    // read kernel number
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 2)
    {
        std::cerr << "Select a kernel (range 0 - 2)" << std::endl;
        exit(EXIT_FAILURE);
    }

    CudaDeviceInfo();
    benchmark_kernel_softmax(1024, 32768, kernel_num);
}