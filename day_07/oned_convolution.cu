#include <iostream>
#include <cuda_runtime.h>

#define MASK_WIDTH 5
#define HALF_MASK (MASK_WIDTH / 2)

// Constant memory for the convolution mask
__constant__ float d_Mask[MASK_WIDTH];

// CUDA error‐checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << err << " \"" << cudaGetErrorString(err)     \
                      << "\"" << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// 1D convolution kernel (no tiling)
__global__ void oned_convolution_kernel(const float* input,
                                        float* output,
                                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;

    // Print the mask offsets with a space between each, then newline
    for (int k = -HALF_MASK; k <= HALF_MASK; ++k) {
        printf("%d ", k);
        int in_idx = idx + k;
        if (in_idx >= 0 && in_idx < n) {
            sum += input[in_idx] * d_Mask[k + HALF_MASK];
        }
    }
    printf("\n");              // ← new: separate each thread's debug line

    output[idx] = sum;
}

int main()
{
    const int N = 10;
    float h_Input[N], h_Output[N], h_Mask[MASK_WIDTH];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_Input[i] = static_cast<float>(i);
    }
    for (int i = 0; i < MASK_WIDTH; ++i) {
        h_Mask[i] = static_cast<float>(i);
    }

    // Device pointers
    float *d_Input = nullptr, *d_Output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Output, N * sizeof(float)));

    // Copy input and mask to device
    CUDA_CHECK(cudaMemcpy(d_Input, h_Input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Mask, h_Mask, MASK_WIDTH * sizeof(float)));

    // Launch kernel
    const int BLOCK_SIZE = 32;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    oned_convolution_kernel<<<gridSize, BLOCK_SIZE>>>(d_Input, d_Output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_Output, d_Output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_Input));
    CUDA_CHECK(cudaFree(d_Output));

    // Print results with two‐decimal formatting
    std::cout << "Input:\n";
    for (int i = 0; i < N; ++i) {
        std::printf("%.2f ", h_Input[i]);
    }
    std::cout << "\n\nMask:\n";
    for (int i = 0; i < MASK_WIDTH; ++i) {
        std::printf("%.2f ", h_Mask[i]);
    }
    std::cout << "\n\nOutput:\n";
    for (int i = 0; i < N; ++i) {
        std::printf("%.2f ", h_Output[i]);
    }
    std::cout << std::endl;

    return 0;
}
