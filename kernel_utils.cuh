#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

void CudaDeviceInfo()
{
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
      Name: %s\n\
      Compute Capability: %d.%d\n\
      memoryBusWidth: %d\n\
      maxThreadsPerBlock: %d\n\
      maxThreadsPerMultiProcessor: %d\n\
      maxRegsPerBlock: %d\n\
      maxRegsPerMultiProcessor: %d\n\
      totalGlobalMem: %zuMB\n\
      sharedMemPerBlock: %zuKB\n\
      sharedMemPerMultiprocessor: %zuKB\n\
      totalConstMem: %zuKB\n\
      multiProcessorCount: %d\n\
      Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};


bool verify_matrix(float *D_ref, float *D, int SIZE)
{
  double diff = 0.0;
  for (int i = 0; i < SIZE; i++)
  {
    diff = std::fabs(D[i] - D_ref[i]);
    if (diff > 0.01)
    {
      printf("Value %6.2f is %6.2f :: diff %6.2f at %d \n", D_ref[i], D[i],
             diff, i);
      fflush(stdout);
      return false;
    }
  }
  return true;
}

#endif