// tiled_matmul.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE    32      
#define N_DEFAULT    256    

// Error‐checking macro
#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr,                                                      \
              "CUDA Error: %s (%s:%d)\n",                                  \
              cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Kernel: tiled multiplication of N×N matrices
__global__ void tiled_matmul(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[])
{
    int N = (argc > 1) ? std::atoi(argv[1]) : N_DEFAULT;
    size_t bytes = size_t(N) * N * sizeof(float);

    // Host allocations
    float *h_A = (float*)std::malloc(bytes),
          *h_B = (float*)std::malloc(bytes),
          *h_C = (float*)std::malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    // Timing setup
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("tiled_matmul: N=%d, TILE_SIZE=%d → %.3f ms\n", N, TILE_SIZE, ms);

    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    
    float expected = 2.0f * N;  // sum of 1×2 over N terms
    bool ok = (std::fabs(h_C[0] - expected) < 1e-3f)
           && (std::fabs(h_C[N*N - 1] - expected) < 1e-3f);
    printf("Result %s (C[0]=%f, C[last]=%f, expected=%f)\n",
           ok ? "OK" : "FAIL",
           h_C[0], h_C[N*N - 1], expected);

   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
