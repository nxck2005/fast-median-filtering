#include <cuda_runtime.h>
#include <stdio.h>
// __global__ defines a function (a "kernel") that runs on the GPU
// It can be called from the CPU (the host)
__global__ void addKernel(int *c, const int *a, const int *b) {
    *c = *a + *b;
}

int main() {
    // DEFINE HOST DATA
    // these variables live in CPU's ram.
    int a = 10;
    int b = 25;
    int c = 0; // this holds the result

    // ALLOCATE MEM TO GPU
    // pointers to memory locations on the GPU's vram
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // allocate memory on device (gpu)
    cudaMalloc((void**)&dev_a, sizeof(int));
    cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_c, sizeof(int));

    // COPY DATA FROM HOST TO DEVICE
    // move values of a and b from host ram to gpu vram
    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // LAUNCH KERNEL ON GPU
    // <<<1, 1>>> means we are launching 1 block of 1 thread
    addKernel<<<1, 1>>>(dev_c, dev_a, dev_b);

    // Wait for GPU to finish the kernel before proceeding to avoid a race condition lol
    cudaDeviceSynchronize();

    // COPY RESULT FROM DEVICE TO HOST
    // bring result from dev_c on the GPU back to c on CPU
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // PRINT AND CLEAN UP
    // print and clean, obviously
    printf("Hello from the GPU! THe result is: %d\n", c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}