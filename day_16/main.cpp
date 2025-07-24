// main.cpp – Entry point for CUDA-accelerated Naive Bayes training
#include <stdio.h>
#include "NaiveBayesTrain.cuh"

int main() {
    // Setup: 6 training samples with 2 features each and 1 class label (last column)
    const int numSamples = 6;
    const int numFeatures = 2;
    const int numClasses = 2;
    const int numFeatureValues = 3; // Categorical features: values ∈ {0, 1, 2}

    // Input dataset format: [feature_0, feature_1, class_label]
    int h_Dataset[numSamples][numFeatures + 1] = {
        {0, 1, 1},
        {1, 1, 1},
        {2, 2, 0},
        {1, 0, 1},
        {0, 2, 0},
        {2, 1, 1}
    };

    // Host-side buffers to store model statistics
    int h_priors[numClasses] = {0};  // P(Y = c)
    int h_likelihoods[numClasses * numFeatures * numFeatureValues] = {0};  // P(X_i = v | Y = c)

    // Launch training pipeline — moves data to device and computes stats
    trainNaiveBayes(
        (int*)h_Dataset, h_priors, h_likelihoods,
        numSamples, numFeatures, numClasses, numFeatureValues
    );

    // Output learned priors
    printf("=== Class Priors ===\n");
    for (int c = 0; c < numClasses; ++c)
        printf("Class %d: %.4f\n", c, (float)h_priors[c] / numSamples);

    // Output conditional probabilities for each class-feature-value combination
    printf("\n=== Likelihood Table ===\n");
    for (int c = 0; c < numClasses; ++c) {
        printf("Class %d:\n", c);
        for (int f = 0; f < numFeatures; ++f) {
            for (int v = 0; v < numFeatureValues; ++v) {
                int index = c * numFeatures * numFeatureValues + f * numFeatureValues + v;
                printf("  P(Feature %d = %d | Class %d): %.4f\n", f, v, c, (float)h_likelihoods[index] / h_priors[c]);
            }
        }
        printf("\n");
    }

    return 0;
}
