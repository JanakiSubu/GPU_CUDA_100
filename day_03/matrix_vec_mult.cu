#include <iostream>
#include <cuda_runtime.h>

// GPU kernel: each thread computes one element of the output vector

__global__ void matrix_vec_mult(const float* matA,
                                const float* vecB,
                                float*       resultC,
                                int          dim)
{
    // Compute the global row index this thread will handle

    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < dim) {

        float acc = 0.0f;


        // Pointer to the start of row 'idx' in the matrix
        const float* rowPtr = matA + idx * dim;


        // Perform dot‑product of rowPtr with vecB
        for (int k = 0; k < dim; ++k) {


            acc += rowPtr[k] * vecB[k];


        }


        // Write the result for this row
        resultC[idx] = acc;
    }
}

int main()
{
    const int dim = 10;  // Matrix and vector size


    // --- Host allocations using new/delete ---

    float* matA    = new float[dim * dim];  // Host matrix
    float* vecB    = new float[dim];        // Host input vector
    float* resultC = new float[dim];        // Host output vector

    // --- Initialize host data ---
    // Fill matrix with 1.0f, vector with 2.0f, and zero out result

    for (int i = 0; i < dim; ++i) {

        vecB[i]    = 2.0f;
        resultC[i] = 0.0f;
        for (int j = 0; j < dim; ++j) {


            matA[i * dim + j] = 1.0f;

        }
    }

    // --- Allocate device buffers ---
    float *d_mat, *d_vec, *d_res;
    cudaMalloc(&d_mat, dim * dim * sizeof(float));  // Matrix on GPU
    cudaMalloc(&d_vec, dim * sizeof(float));        // Input vector on GPU
    cudaMalloc(&d_res, dim * sizeof(float));        // Output vector on GPU

    // --- Copy inputs from host to device ---
    cudaMemcpy(d_mat, matA, dim * dim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vecB, dim * sizeof(float),
               cudaMemcpyHostToDevice);

    // --- Launch kernel ---
    const int threadsPerBlock = 128;
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    matrix_vec_mult<<<blocks, threadsPerBlock>>>(d_mat, d_vec, d_res, dim);

    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();

    // --- Copy result back from device to host ---
    cudaMemcpy(resultC, d_res, dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    // --- Print matrix A ---
    std::cout << "Matrix A:\n";
    for (int i = 0; i < dim; ++i) {

        for (int j = 0; j < dim; ++j) {

            std::cout << matA[i * dim + j] << ' ';
            
        }
        std::cout << '\n';
    }

    // --- Print vector B ---
    std::cout << "Vector B: ";
    for (int i = 0; i < dim; ++i) {

        std::cout << vecB[i] << ' ';
    }
    std::cout << "\n";

    // --- Print result vector C ---
    std::cout << "Result C: ";
    for (int i = 0; i < dim; ++i) {

        std::cout << resultC[i] << ' ';
    }
    std::cout << "\n";

    // --- Cleanup ---
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
    delete[] matA;
    delete[] vecB;
    delete[] resultC;

    return 0;
}
