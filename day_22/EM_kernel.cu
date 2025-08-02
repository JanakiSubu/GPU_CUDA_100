#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Number of Gaussian components and data size parameters
#define NUM_CLUSTERS 2         // Number of Gaussian clusters in the mixture
#define N 1024                 // Total number of 1D data points
#define THREADS_PER_BLOCK 256  // Threads per CUDA block

// Macro for CUDA API error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// E-step: compute responsibilities for each data point and cluster
__global__ void eStepKernel(
    float* data,               // Input data array
    int N,                     // Number of data points
    float* mu,                 // Cluster means
    float* sigma,              // Cluster standard deviations
    float* pival,              // Cluster mixing coefficients
    float* responsibilities    // Output responsibilities (N x NUM_CLUSTERS)
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        float probs[NUM_CLUSTERS];
        float sum = 0.0f;

        // Compute weighted Gaussian PDF for each cluster
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float diff = x - mu[k];
            float var = sigma[k] * sigma[k];
            float exponent = -0.5f * (diff * diff) / var;
            float gauss = expf(exponent) / (sqrtf(2.0f * M_PI * var));
            probs[k] = pival[k] * gauss;
            sum += probs[k];
        }

        // Normalize to get responsibilities (posterior probabilities)
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            responsibilities[idx * NUM_CLUSTERS + k] = probs[k] / sum;
        }
    }
}

// M-step: accumulate sufficient statistics for parameter updates
__global__ void mStepKernel(
    float* data,                 // Input data array
    int N,                       // Number of data points
    float* responsibilities,     // Responsibilities from E-step
    float* sum_gamma,            // Output: sum of responsibilities per cluster
    float* sum_x,                // Output: weighted sum of x per cluster
    float* sum_x2                // Output: weighted sum of x^2 per cluster
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        // Update accumulators atomically for each cluster
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float gamma = responsibilities[idx * NUM_CLUSTERS + k];
            atomicAdd(&sum_gamma[k], gamma);         // N_k = Σ γ_{ik}
            atomicAdd(&sum_x[k], gamma * x);         // Σ γ_{ik} x_i
            atomicAdd(&sum_x2[k], gamma * x * x);    // Σ γ_{ik} x_i^2
        }
    }
}

int main() {
    // Initialize RNG for synthetic data generation
    srand(static_cast<unsigned>(time(NULL)));

    // Host: generate two clusters of 1D data around 2.0 and 8.0
    float h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (i < N/2)
                  ? 2.0f + static_cast<float>(rand()) / RAND_MAX
                  : 8.0f + static_cast<float>(rand()) / RAND_MAX;
    }

    // Host: initial GMM parameters
    float h_mu[NUM_CLUSTERS]    = {1.0f, 9.0f};  // Initial means
    float h_sigma[NUM_CLUSTERS] = {1.0f, 1.0f};  // Initial standard deviations
    float h_pival[NUM_CLUSTERS] = {0.5f, 0.5f};  // Initial mixture weights

    // Device memory pointers
    float *d_data, *d_mu, *d_sigma, *d_pival;
    float *d_responsibilities, *d_sum_gamma, *d_sum_x, *d_sum_x2;

    // Allocate device buffers
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pival, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_gamma, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_x2, NUM_CLUSTERS * sizeof(float)));

    // Copy input data and initial parameters to GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pival, h_pival, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid configuration
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Host accumulators for M-step results
    float h_sum_gamma[NUM_CLUSTERS];
    float h_sum_x[NUM_CLUSTERS];
    float h_sum_x2[NUM_CLUSTERS];

    // Run EM for fixed number of iterations
    int maxIter = 100;
    for (int iter = 0; iter < maxIter; iter++) {
        // E-step: compute responsibilities on GPU
        eStepKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, N, d_mu, d_sigma, d_pival, d_responsibilities
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset accumulators for M-step
        CUDA_CHECK(cudaMemset(d_sum_gamma, 0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_x,     0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_x2,    0, NUM_CLUSTERS * sizeof(float)));

        // M-step: accumulate sufficient statistics on GPU
        mStepKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, N, d_responsibilities,
            d_sum_gamma, d_sum_x, d_sum_x2
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy accumulators back to host
        CUDA_CHECK(cudaMemcpy(h_sum_gamma, d_sum_gamma, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sum_x,     d_sum_x,     NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sum_x2,    d_sum_x2,    NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));

        // Host: update GMM parameters using closed-form formulas
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            if (h_sum_gamma[k] > 1e-6f) {
                // New mean = Σ γ_{ik} x_i / N_k
                h_mu[k] = h_sum_x[k] / h_sum_gamma[k];
                // New variance = Σ γ_{ik} x_i^2 / N_k - μ_k^2
                float var = h_sum_x2[k] / h_sum_gamma[k] - h_mu[k] * h_mu[k];
                h_sigma[k] = sqrtf(fmaxf(var, 1e-6f));  // Avoid zero stdev
                // New mixing weight = N_k / N
                h_pival[k] = h_sum_gamma[k] / N;
            }
        }

        // Copy updated parameters back to GPU for next iteration
        CUDA_CHECK(cudaMemcpy(d_mu,    h_mu,    NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pival, h_pival, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

        // Periodically print parameters to track convergence
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << ":\n";
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                std::cout << " Cluster " << k
                          << ": mu=" << h_mu[k]
                          << ", sigma=" << h_sigma[k]
                          << ", pi=" << h_pival[k] << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Free all GPU memory
    cudaFree(d_data);
    cudaFree(d_mu);
    cudaFree(d_sigma);
    cudaFree(d_pival);
    cudaFree(d_responsibilities);
    cudaFree(d_sum_gamma);
    cudaFree(d_sum_x);
    cudaFree(d_sum_x2);

    return 0;
}
