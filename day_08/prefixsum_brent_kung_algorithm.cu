#define LOAD_SIZE 64   // now 2×blockDim.x when blockDim.x=32
#include <iostream>
#include <cuda_runtime.h>

inline void checkCudaError(const char *message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << message << "): "
                  << cudaGetErrorString(err) << "\n";
        std::exit(-1);
    }
}

// Brent–Kung inclusive scan per-block
__global__ void prefixsum_kernel(const float* __restrict__ A,
                                 float* __restrict__ C,
                                 int N)
{
    const int tid   = threadIdx.x;
    const int base  = 2 * blockDim.x * blockIdx.x;
    __shared__ float S[LOAD_SIZE];

    // === load ===
    int i = base + tid;
    if (i < N) {
        S[tid] = A[i];
    } else {
        S[tid] = 0.0f;
    }
    int j = base + blockDim.x + tid;
    if (j < N) {
        S[blockDim.x + tid] = A[j];
    } else {
        S[blockDim.x + tid] = 0.0f;
    }
    __syncthreads();

    // === upsweep (reduce) ===
    for (int stride = 1; stride < blockDim.x * 2; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < LOAD_SIZE) {
            S[idx] += S[idx - stride];
        }
    }

    // === downsweep ===
    for (int stride = blockDim.x; stride >= 1; stride >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx + stride < LOAD_SIZE) {
            S[idx + stride] += S[idx];
        }
    }
    __syncthreads();

    // === store ===
    if (i < N) {
        C[i] = S[tid];
    }
    if (j < N) {
        C[j] = S[blockDim.x + tid];
    }
}

int main() {
    const int N = 10;
    float h_A[N], h_C[N];
    for (int i = 0; i < N; ++i) h_A[i] = float(i + 1);

    float *d_A = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    checkCudaError("alloc");

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("copy H2D");

    dim3 block(32);
    dim3 grid((N + 2*block.x - 1) / (2*block.x));
    prefixsum_kernel<<<grid, block>>>(d_A, d_C, N);
    checkCudaError("launch");
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("copy D2H");

    cudaFree(d_A);
    cudaFree(d_C);

    // print
    std::cout << "A:\n";
    for (float v : h_A) std::cout << v << " ";
    std::cout << "\nC (prefix sum):\n";
    for (float v : h_C) std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}
