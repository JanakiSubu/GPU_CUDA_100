#include <stdio.h>
#include <cuda_runtime.h>

// Integer-safe replacements for min and max
__device__ __host__ inline int my_max(int a, int b) {
    return (a > b) ? a : b;
}

__device__ __host__ inline int my_min(int a, int b) {
    return (a < b) ? a : b;
}

// Error checking macro
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n",                        \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Merge path partition: binary search to find (i,j) split for thread `diag`
__device__ void merge_path_partition(int diag, const int* A, const int* B,
                                     int A_len, int B_len, int& i_out, int& j_out) {
    int low = my_max(0, diag - B_len);
    int high = my_min(diag, A_len);

    while (low <= high) {
        int i = (low + high) / 2;
        int j = diag - i;

        if (i < A_len && j > 0 && B[j - 1] > A[i]) {
            low = i + 1;
        } else if (j < B_len && i > 0 && A[i - 1] > B[j]) {
            high = i - 1;
        } else {
            i_out = i;
            j_out = j;
            return;
        }
    }
}

// Main merge kernel: each thread writes one output element to C
__global__ void merge_path_kernel(const int* A, const int* B, int* C, int A_len, int B_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A_len + B_len;

    if (tid < total) {
        int i, j;
        merge_path_partition(tid, A, B, A_len, B_len, i, j);

        // 💡 Debugging / Explanative Output
        printf("Thread %2d → diag=%2d → A[%2d], B[%2d] → ", tid, tid, i, j);

        if (i < A_len && (j >= B_len || A[i] <= B[j])) {
            printf("C[%2d] = A[%2d] = %2d\n", tid, i, A[i]);
            C[tid] = A[i];
        } else {
            printf("C[%2d] = B[%2d] = %2d\n", tid, j, B[j]);
            C[tid] = B[j];
        }
    }
}

int main() {
    const int N = 8, M = 8;
    int A[N] = {1, 3, 5, 7, 9, 11, 13, 15};
    int B[M] = {2, 4, 6, 8, 10, 12, 14, 16};
    int C[N + M];

    printf("\n Input Array A: ");
    for (int i = 0; i < N; ++i) printf("%d ", A[i]);

    printf("\n Input Array B: ");
    for (int i = 0; i < M; ++i) printf("%d ", B[i]);

    printf("\n\n Launching Merge Path Kernel...\n\n");

    // Device memory allocation
    int *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B, M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C, (N + M) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, M * sizeof(int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int totalThreads = N + M;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    merge_path_kernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C, d_C, (N + M) * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\n Final Merged Result:\n");
    for (int i = 0; i < N + M; ++i)
        printf("C[%2d] = %2d\n", i, C[i]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}