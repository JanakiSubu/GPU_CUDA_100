// Compile with: nvcc vec_cublas.cu -o vec_cublas -lstdc++ -lcublas

#include <iostream>
#include <cublas_v2.h>

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input vectors A and B with sample values
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Create cuBLAS context handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for vectors A and B
    float *d_a, *d_b;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    // Transfer vectors A and B from host to device
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set the scalar multiplier for the AXPY operation
    const float alpha = 1.0f;

    // Perform vector addition on GPU: d_b = alpha * d_a + d_b
    cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);

    // Copy result vector from device to host (result is in d_b)
    cudaMemcpy(C, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result vector C
    for(int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory and destroy cuBLAS handle
    cudaFree(d_a);
    cudaFree(d_b);
    cublasDestroy(handle);

    return 0;
}
