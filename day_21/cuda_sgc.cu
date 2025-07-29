#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

// Kernel: Compute per-sample predictions and squared losses for linear regression
__global__ void compute_loss(float* X, float* y, float* W, float* b, float* loss, float* y_pred, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Calculate dot-product of feature vector X[idx] and weight vector W
        float y_pred_val = 0.0f;
        for (int i = 0; i < D; i++) {
            y_pred_val += X[idx * D + i] * W[i];
        }
        // Add the scalar bias term
        y_pred_val += *b;
        // Store the predicted value
        y_pred[idx] = y_pred_val;
        // Compute and store squared error loss: (y - y_pred)^2
        float diff = y[idx] - y_pred_val;
        loss[idx] = diff * diff;
    }
}

// Kernel: Compute gradients of weights and bias using mean squared error
__global__ void compute_gradients(float* X, float* loss, float* dW, float* db, int N, int D) {
    // Shared memory to accumulate bias-gradient contributions per block
    __shared__ float db_shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute gradient wrt W[idx] if within weight dimension
    if (idx < D) {
        float gradW = 0.0f;
        // Sum loss * feature X[:, idx] over all samples
        for (int i = 0; i < N; i++) {
            gradW += X[i * D + idx] * loss[i];
        }
        // Apply scaling factor and negate for gradient descent step
        dW[idx] = - (2.0f / N) * gradW;
    }

    // Compute per-thread contribution to bias gradient
    float gradb = 0.0f;
    if (idx < N) {
        gradb = loss[idx];
    }
    db_shared[tid] = gradb;
    __syncthreads();

    // Thread 0 reduces block's contributions and atomically adds to global db
    if (tid == 0) {
        float sum_db = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum_db += db_shared[i];
        }
        atomicAdd(db, - (2.0f / N) * sum_db);
    }
}

// Kernel: Update weights and bias using computed gradients and learning rate
__global__ void update_weights(float* W, float* dW, float* b, float* db, float lr, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        // SGD weight update: W = W - lr * dW
        W[idx] -= lr * dW[idx];
    }
    // Update bias by thread 0 only
    if (idx == 0) {
        *b -= lr * (*db);
    }
}

// Host function: Allocate GPU memory, copy data, run training loop, and retrieve results
void train_sgd(float* h_X, float* h_y, float* h_W, float* h_b, int N, int D, float lr, int epochs) {
    float *d_X, *d_y, *d_W, *d_b, *d_gradW, *d_gradb, *d_loss, *d_y_pred;
    cudaMalloc(&d_X, N * D * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_W, D * sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_gradW, D * sizeof(float));
    cudaMalloc(&d_gradb, sizeof(float));
    cudaMalloc(&d_loss, N * sizeof(float));
    cudaMalloc(&d_y_pred, N * sizeof(float));
    
    // Copy input data and initial parameters to device
    cudaMemcpy(d_X, h_X, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks_data = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_param = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Main SGD training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward and loss computation
        compute_loss<<<blocks_data, BLOCK_SIZE>>>(d_X, d_y, d_W, d_b, d_loss, d_y_pred, N, D);
        cudaDeviceSynchronize();

        // Gradient computation
        cudaMemset(d_gradb, 0, sizeof(float)); // Reset bias gradient accumulator
        compute_gradients<<<blocks_param, BLOCK_SIZE>>>(d_X, d_loss, d_gradW, d_gradb, N, D);
        cudaDeviceSynchronize();

        // Parameter update
        update_weights<<<blocks_param, BLOCK_SIZE>>>(d_W, d_gradW, d_b, d_gradb, lr, D);
        cudaDeviceSynchronize();
    }

    // Copy trained parameters back to host
    cudaMemcpy(h_W, d_W, D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_gradW);
    cudaFree(d_gradb);
    cudaFree(d_loss);
    cudaFree(d_y_pred);
}

int main() {
    int N = 1024;   // Number of samples
    int D = 10;     // Number of features per sample
    float lr = 0.01f;
    int epochs = 1000;

    // Allocate host memory for dataset and parameters
    float *h_X = new float[N * D];
    float *h_y = new float[N];
    float *h_W = new float[D];
    float *h_b = new float[1];

    // Initialize data and parameters with random values
    srand(42);
    for (int i = 0; i < N * D; i++) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        h_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < D; i++) {
        h_W[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    *h_b = static_cast<float>(rand()) / RAND_MAX;

    // Train the model using SGD on GPU
    train_sgd(h_X, h_y, h_W, h_b, N, D, lr, epochs);

    // Output trained parameters
    std::cout << "Trained Weights: ";
    for (int i = 0; i < D; i++) std::cout << h_W[i] << " ";
    std::cout << "\nTrained Bias: " << *h_b << std::endl;

    // Clean up host memory
    delete[] h_X;
    delete[] h_y;
    delete[] h_W;
    delete[] h_b;
    return 0;
}
