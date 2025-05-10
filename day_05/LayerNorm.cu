#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA Kernel function to perform Layer Normalization on each row of matrix A
__global__ void layerNormKernel(const float* input, float* output, int rows, int cols) {
    // Calculate the row index for each thread
    int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // If the thread is responsible for processing a valid row
    if (rowIndex < rows) {
        // Use shared memory to store the row data for faster access
        extern __shared__ float sharedMemory[];

        // Pointer to shared memory where the row will be temporarily stored
        float* rowData = sharedMemory;

        // Copy the data of the row from global memory to shared memory
        for (int col = threadIdx.y; col < cols; col += blockDim.y) {
            rowData[col] = input[rowIndex * cols + col];
        }

        // Synchronize threads within the block to ensure all data is copied to shared memory
        __syncthreads();

        // Calculate the mean of the row
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += rowData[col];
        }
        mean /= cols;  // Divide by the number of columns to get the mean

        // Calculate the variance of the row (using the mean)
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            variance += (rowData[col] - mean) * (rowData[col] - mean);
        }
        variance /= cols;  // Divide by the number of columns to get the variance
        float stddev = sqrtf(variance + 1e-7);  // Calculate standard deviation (with a small epsilon for stability)

        // Normalize the row by subtracting the mean and dividing by the standard deviation
        for (int col = threadIdx.y; col < cols; col += blockDim.y) {
            output[rowIndex * cols + col] = (rowData[col] - mean) / stddev;
        }
    }
}

int main() {
    // Matrix dimensions
    const int rows = 10;
    const int cols = 10;

    // Pointers for the host memory (CPU)
    float *A, *B;

    // Allocate memory for input (A) and output (B) matrices on the host
    A = (float*)malloc(rows * cols * sizeof(float));
    B = (float*)malloc(rows * cols * sizeof(float));

    // Initialize the input matrix A with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Allocate memory for input (d_A) and output (d_B) matrices on the device (GPU)
    float *d_A, *d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));

    // Copy the input data from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel for layer normalization
    int blockSize = 256;  // Number of threads per block
    int gridSize = (rows + blockSize - 1) / blockSize;  // Number of blocks needed
    size_t sharedMemorySize = cols * sizeof(float);  // Shared memory size required per block

    // Launch the kernel with grid and block configuration
    layerNormKernel<<<gridSize, blockSize, sharedMemorySize>>>(d_A, d_B, rows, cols);

    // Synchronize device to ensure the kernel has finished execution
    cudaDeviceSynchronize();

    // Copy the result matrix B from device to host
    cudaMemcpy(B, d_B, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the input matrix A
    std::cout << "Input Matrix A (Random Values):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << A[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    // Print the output matrix B (Normalized values)
    std::cout << "\nNormalized Matrix B:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << B[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    // Free the allocated memory on both host and device
    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);

    return 0;
}
