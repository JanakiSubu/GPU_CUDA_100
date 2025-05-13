#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Utility: check the result of any CUDA call
inline void cudaCheck(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "Error: " << msg 
                  << " (" << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// Kernel: transpose a width×height matrix
__global__ void transposeKernel(const float* src, float* dst, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // read element at (row, col) and write it to (col, row)
        dst[col * height + row] = src[row * width + col];
    }
}

int main() {
    const int WIDTH  = 1024;
    const int HEIGHT = 1024;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);

    // 1. Prepare host data
    std::vector<float> h_src(WIDTH * HEIGHT), h_dst(WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        h_src[i] = static_cast<float>(i);  // just fill with 0,1,2,...
    }

    // 2. Allocate device memory
    float *d_src = nullptr, *d_dst = nullptr;
    cudaCheck(cudaMalloc(&d_src, bytes), "allocating d_src");
    cudaCheck(cudaMalloc(&d_dst, bytes), "allocating d_dst");

    // 3. Copy input matrix up to the GPU
    cudaCheck(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice),
              "copying to device");

    // 4. Configure and launch the transpose kernel
    dim3 block(32, 32);
    dim3 grid( (WIDTH  + block.x - 1) / block.x,
               (HEIGHT + block.y - 1) / block.y );
    transposeKernel<<<grid, block>>>(d_src, d_dst, WIDTH, HEIGHT);
    cudaCheck(cudaGetLastError(), "launching kernel");
    cudaCheck(cudaDeviceSynchronize(), "waiting for kernel");

    // 5. Retrieve result
    cudaCheck(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost),
              "copying back to host");

    // 6. Validate
    bool ok = true;
    for (int r = 0; r < HEIGHT && ok; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            if (h_dst[c * HEIGHT + r] != h_src[r * WIDTH + c]) {
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok 
        ? "Success: matrix transposed correctly.\n"
        : "Failure: discrepancy found in transposition.\n");

    // 7. Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
