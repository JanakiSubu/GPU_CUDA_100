// NaiveBayesTrain.cpp – Host-side launcher for GPU-accelerated Naive Bayes training
#include <cuda_runtime.h>
#include "NaiveBayesTrain.cuh"
#include "NaiveBayesKernel.cuh"

void trainNaiveBayes(
    int* h_Dataset, int* h_priors, int* h_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    // Device memory allocations
    int* d_Dataset;
    int* d_priors;
    int* d_likelihoods;

    // Compute memory sizes
    int datasetSize = numSamples * (numFeatures + 1) * sizeof(int); // +1 for class label per sample
    int priorsSize = numClasses * sizeof(int);
    int likelihoodsSize = numClasses * numFeatures * numFeatureValues * sizeof(int);

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_Dataset, datasetSize);
    cudaMalloc((void**)&d_priors, priorsSize);
    cudaMalloc((void**)&d_likelihoods, likelihoodsSize);

    // Transfer data from host to device
    cudaMemcpy(d_Dataset, h_Dataset, datasetSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_priors, h_priors, priorsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_likelihoods, h_likelihoods, likelihoodsSize, cudaMemcpyHostToDevice);

    // Configure thread launch — one thread per sample
    int threadsPerBlock = 256;
    int numBlocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the parallel training kernel
    computePriorsAndLikelihood<<<numBlocks, threadsPerBlock>>>( 
        d_Dataset, d_priors, d_likelihoods,
        numSamples, numFeatures, numClasses, numFeatureValues
    );

    // Retrieve computed statistics from device to host
    cudaMemcpy(h_priors, d_priors, priorsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_likelihoods, d_likelihoods, likelihoodsSize, cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_Dataset);
    cudaFree(d_priors);
    cudaFree(d_likelihoods);
}
