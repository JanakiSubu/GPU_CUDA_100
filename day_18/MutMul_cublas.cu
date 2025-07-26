#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    // Create a cuBLAS handle to manage cuBLAS library context
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define matrix dimensions: A is MxK, B is KxN, so C is MxN
    int M = 2, N = 3, K = 4;

    // Allocate host memory for matrices A, B, and C
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(M * K * sizeof(float));  // Host matrix A (MxK)
    h_B = (float *)malloc(K * N * sizeof(float));  // Host matrix B (KxN)
    h_C = (float *)malloc(M * N * sizeof(float));  // Host matrix C (MxN)

    // Initialize matrix A with values: A[i][j] = i + j
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = i + j;

    // Initialize matrix B with values: B[i][j] = i + j
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = i + j;

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication using cuBLAS: C = alpha * A * B + beta * C
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for A or B
                M, N, K,                   // Dimensions of the matrices
                &alpha,                   // Scalar multiplier for A * B
                d_A, M,                   // Device matrix A and its leading dimension
                d_B, K,                   // Device matrix B and its leading dimension
                &beta,                    // Scalar multiplier for existing C
                d_C, M);                  // Output matrix C and its leading dimension

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Print result matrix C = A * B
    // cuBLAS uses column-major layout, so index = i + j * M
    printf("Matrix C = A * B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_C[i + j * M]);
        }
        printf("\n");
    }

    // Free host and device memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    return 0;
}
