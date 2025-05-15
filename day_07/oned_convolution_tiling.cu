#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int KERNEL_RADIUS = 2;
constexpr int KERNEL_WIDTH  = 2 * KERNEL_RADIUS + 1;

// Convolution kernel in constant memory
__constant__ float d_Kernel[KERNEL_WIDTH];

// Helper macro for CUDA error checking
#define CUDA_SAFE(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                      << " – " << cudaGetErrorString(err) << std::endl;               \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

// 1D tiled convolution kernel
__global__ void convolutionTiledKernel(const float* in, float* out, int N) {
    extern __shared__ float s_data[];  // size = blockDim.x + 2*KERNEL_RADIUS

    int tid      = threadIdx.x;
    int globalX  = blockIdx.x * blockDim.x + tid;
    int sharedX  = tid + KERNEL_RADIUS;

    // Load center region
    if (globalX < N) {
        s_data[sharedX] = in[globalX];
    }

    // Load left halo
    if (tid < KERNEL_RADIUS) {
        int leftIdx = blockIdx.x * blockDim.x + tid - KERNEL_RADIUS;
        s_data[tid] = (leftIdx >= 0 ? in[leftIdx] : 0.0f);
    }

    // Load right halo
    if (tid < KERNEL_RADIUS) {
        int rightIdx = blockIdx.x * blockDim.x + blockDim.x + tid;
        s_data[sharedX + blockDim.x] = (rightIdx < N ? in[rightIdx] : 0.0f);
    }

    __syncthreads();

    // Perform convolution
    if (globalX < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < KERNEL_WIDTH; ++k) {
            acc += s_data[tid + k] * d_Kernel[k];
        }
        out[globalX] = acc;
    }
}

int main() {
    const int N = 10;
    std::vector<float> h_in(N), h_out(N), h_kernel(KERNEL_WIDTH);

    // Initialize host data
    for (int i = 0; i < N; ++i)        h_in[i]     = static_cast<float>(i);
    for (int i = 0; i < KERNEL_WIDTH; ++i) h_kernel[i] = static_cast<float>(i);

    // Allocate device buffers
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_SAFE(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_SAFE(cudaMalloc(&d_out, N * sizeof(float)));

    // Copy input and kernel to GPU
    CUDA_SAFE(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpyToSymbol(d_Kernel, h_kernel.data(), KERNEL_WIDTH * sizeof(float)));

    // Launch parameters
    const int BLOCK_SIZE = 32;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemBytes = (BLOCK_SIZE + 2 * KERNEL_RADIUS) * sizeof(float);

    convolutionTiledKernel<<<gridSize, BLOCK_SIZE, sharedMemBytes>>>(d_in, d_out, N);
    CUDA_SAFE(cudaGetLastError());
    CUDA_SAFE(cudaDeviceSynchronize());

    // Copy result back
    CUDA_SAFE(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_SAFE(cudaFree(d_in));
    CUDA_SAFE(cudaFree(d_out));

    // Print results
    std::cout << "Input:\n";
    for (auto v : h_in)  std::cout << v << " ";
    std::cout << "\n\nKernel:\n";
    for (auto v : h_kernel)  std::cout << v << " ";
    std::cout << "\n\nOutput:\n";
    for (auto v : h_out) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
