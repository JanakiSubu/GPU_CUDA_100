#include <stdio.h>
#include <cuda_runtime.h>

// __global__ indicates this function runs on the GPU.
__global__ void helloFromGPU() {
    if (threadIdx.x == 0) {
        printf("Hello from the GPU!\n");
    }
}

int main() {
    // Launching the kernel with 1 block of 1 thread.
    helloFromGPU<<<1, 1>>>();

    // Synchronized to ensure the GPU finishes before exiting.
    cudaDeviceSynchronize();

    printf("Hello from the CPU!\n");
    return 0;
}