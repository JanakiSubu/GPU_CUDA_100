// NaiveBayes.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NaiveBayesKernel.cuh"
#include "NaiveBayesTrain.cuh"

#define SHARED_SIZE 20  // Shared memory budget per block (manually tuned based on class/feature size)

// Kernel: Computes class priors and conditional feature likelihoods in parallel
__global__ void computePriorsAndLikelihood(
    int* d_Dataset, int* d_priors, int* d_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory buffers to reduce global memory contention
    __shared__ int local_d_priors[SHARED_SIZE];
    __shared__ int local_d_likelihoods[SHARED_SIZE];

    // Thread-level data processing — each thread processes one sample
    if (threadId < numSamples) {
        // Fetch class label from the last column of the sample
        int classLabel = d_Dataset[threadId * (numFeatures + 1) + numFeatures];

        // Atomically increment class count (used to calculate prior P(Y=c))
        atomicAdd(&local_d_priors[classLabel], 1);

        // Compute and accumulate feature likelihoods conditioned on class
        for (int fIdx = 0; fIdx < numFeatures; ++fIdx) {
            int featureValue = d_Dataset[threadId * (numFeatures + 1) + fIdx];
            int likelihoodIndex = classLabel * numFeatures * numFeatureValues + (fIdx * numFeatureValues) + featureValue;

            // Atomically increment likelihood count for the given feature/class/value combo
            atomicAdd(&local_d_likelihoods[likelihoodIndex], 1);
        }
    }

    // Synchronize threads before committing shared-memory stats to global
    __syncthreads();

    // Single thread writes aggregate shared results back to global memory
    if (threadIdx.x == 0) {
        for (int c = 0; c < numClasses; ++c) {
            atomicAdd(&d_priors[c], local_d_priors[c]);
        }

        for (int l = 0; l < numClasses * numFeatures * numFeatureValues; ++l) {
            atomicAdd(&d_likelihoods[l], local_d_likelihoods[l]);
        }
    }
}
