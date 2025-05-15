#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int KERNEL_RADIUS = 2;
constexpr int KERNEL_DIAMETER = 2 * KERNEL_RADIUS + 1;

// Mask stored in constant memory for fast access
__constant__ float d_Mask[KERNEL_DIAMETER * KERNEL_DIAMETER];

// Macro to wrap CUDA calls with error checks
#define CUDA_CHECK(expr)                                          \
    do {                                                         \
        cudaError_t err = expr;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)  \
                      << " at " << __FILE__ << ":" << __LINE__  \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                            \
        }                                                        \
    } while (0)

// 2D convolution with tiling and halo in shared memory
__global__ void conv2DTiled(const float* input, float* output, int N) {
    extern __shared__ float sharedTile[];  // (blockDim.x + 2*R) * (blockDim.y + 2*R)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int tileWidth  = blockDim.x + 2 * KERNEL_RADIUS;
    int tileHeight = blockDim.y + 2 * KERNEL_RADIUS;

    // Compute indices into shared tile
    int sharedX = tx + KERNEL_RADIUS;
    int sharedY = ty + KERNEL_RADIUS;

    // Load center
    if (row < N && col < N) {
        sharedTile[sharedY * tileWidth + sharedX] = input[row * N + col];
    }

    // Load halos
    // Left
    if (tx < KERNEL_RADIUS) {
        int c = col - KERNEL_RADIUS;
        sharedTile[sharedY * tileWidth + tx] = (c >= 0 && row < N) ? input[row * N + c] : 0.0f;
    }
    // Right
    if (tx < KERNEL_RADIUS) {
        int c = col + blockDim.x;
        sharedTile[sharedY * tileWidth + (sharedX + blockDim.x)] = (c < N && row < N) ? input[row * N + c] : 0.0f;
    }
    // Top
    if (ty < KERNEL_RADIUS) {
        int r = row - KERNEL_RADIUS;
        sharedTile[ty * tileWidth + sharedX] = (r >= 0 && col < N) ? input[r * N + col] : 0.0f;
    }
    // Bottom
    if (ty < KERNEL_RADIUS) {
        int r = row + blockDim.y;
        sharedTile[(sharedY + blockDim.y) * tileWidth + sharedX] = (r < N && col < N) ? input[r * N + col] : 0.0f;
    }

    __syncthreads();

    // Apply convolution
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int dy = 0; dy < KERNEL_DIAMETER; ++dy) {
            for (int dx = 0; dx < KERNEL_DIAMETER; ++dx) {
                sum += sharedTile[(ty + dy) * tileWidth + (tx + dx)]
                     * d_Mask[dy * KERNEL_DIAMETER + dx];
            }
        }
        output[row * N + col] = sum;
    }
}

int main() {
    const int N = 10;
    std::vector<float> h_input(N * N, 3.0f);
    std::vector<float> h_output(N * N);
    std::vector<float> h_mask(KERNEL_DIAMETER * KERNEL_DIAMETER, 5.0f);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Mask, h_mask.data(), h_mask.size() * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    size_t sharedMemSize = (block.x + 2 * KERNEL_RADIUS)
                         * (block.y + 2 * KERNEL_RADIUS)
                         * sizeof(float);

    conv2DTiled<<<grid, block, sharedMemSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "2D Convolution Output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_output[i * N + j] << ' ';
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
