#include <stdio.h>
#include <cuda.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void partialSumKernel(const int *input, int *output, int n) {
    extern __shared__ int sdata[];
    int tid   = threadIdx.x;
    int base  = blockIdx.x * blockDim.x * 2;
    int idx1  = base + tid;
    int idx2  = base + tid + blockDim.x;

    // Guarded load (handle out-of-bounds)
    int v1 = (idx1 < n) ? input[idx1] : 0;
    int v2 = (idx2 < n) ? input[idx2] : 0;
    sdata[tid] = v1 + v2;
    __syncthreads();

    // In-place inclusive scan (tree-based)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int t = 0;
        if (tid >= stride) t = sdata[tid - stride];
        __syncthreads();
        sdata[tid] += t;
        __syncthreads();
    }

    // Write back to both output positions
    if (idx1 < n) output[idx1] = sdata[tid];
    if (idx2 < n) output[idx2] = sdata[tid];
}

int main() {
    const int N = 16;
    const int blockSize = 8;
    const int gridSize = (N + blockSize*2 - 1) / (blockSize*2);

    int h_in[N]  = {1, 2, 3, 4, 5, 6, 7, 8,
                    9,10,11,12,13,14,15,16};
    int h_out[N] = {0};

    int *d_in, *d_out;
    size_t bytes = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    partialSumKernel
        <<<gridSize, blockSize, blockSize * sizeof(int)>>>
        (d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; ++i) printf("%2d ", h_in[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; ++i) printf("%2d ", h_out[i]);
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
