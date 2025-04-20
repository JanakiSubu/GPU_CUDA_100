#include <cstdlib>
#include <cassert>
#include <iostream>
#include <ctime>         
#include <cuda_runtime.h>

using namespace std;

// simple CUDA kernel: each thread computes C[row,col] = A[row,col] + B[row,col]
__global__ void matrixAdd(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// fill an N×N matrix with random ints in [0,99]
void initMatrix(int *M, int N) {
    for (int i = 0; i < N*N; ++i) {
        M[i] = rand() % 100;
    }
}

// making sure C really is A+B
void verify(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N*N; ++i) {
        assert(C[i] == A[i] + B[i]);
    }
}

int main() {
    // seeding the RNG so we get different values each run
    srand((unsigned)time(nullptr));

    int N = 1 << 10;               // working with a

    // allocate unified memory for A, B, C
    int *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // fill A and B with random data
    initMatrix(A, N);
    initMatrix(B, N);

    // launching a grid of 16×16 blocks covering all N×N elements
    dim3 threads(16,16);
    dim3 blocks((N + threads.x - 1)/threads.x,
                (N + threads.y - 1)/threads.y);
    matrixAdd<<<blocks,threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    //  verification
    verify(A, B, C, N);
    cout << "Matrix addition OK!\n\n";

    // printing just the top-left 8×8 corner 
    int P = 8;
    cout << "A (8×8 corner):\n";
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            cout << A[i*N + j] << ' ';
        }
        cout << '\n';
    }

    cout << "\nB (8×8 corner):\n";
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            cout << B[i*N + j] << ' ';
        }
        cout << '\n';
    }

    cout << "\nC = A + B (8×8 corner):\n";
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            cout << C[i*N + j] << ' ';
        }
        cout << '\n';
    }

    // free up memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
