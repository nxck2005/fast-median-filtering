#include <cuda_runtime.h>
#include <stdio.h> // For fprintf

// --- Add this error checking macro ---
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
// ------------------------------------

__global__ void addKernel(int *c, const int *a, const int *b) {
    *c = *a + *b;
}

int main() {
    int a = 10;
    int b = 25;
    int c = 0;

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // --- Wrap all CUDA calls in the macro ---
    gpuErrchk(cudaMalloc((void**)&dev_a, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_b, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_c, sizeof(int)));

    gpuErrchk(cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice));

    addKernel<<<1, 1>>>(dev_c, dev_a, dev_b);
    
    // Check for errors from the asynchronous kernel launch
    gpuErrchk(cudaGetLastError());

    // Synchronize and check for any lingering errors
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Hello from the GPU! The result is %d\n", c);

    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));
    
    return 0;
}