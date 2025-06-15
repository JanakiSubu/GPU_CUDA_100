// flash_attention_forward.cu
// Day 9 of my #100DaysOfCUDA challenge
// Flash Attention forward pass prototype: tile-based Q×K^T softmax × V
// - Shared-memory tiling (Br × Bc)  
// - Per-row softmax with max & sum tracking  
// - Randomized inputs on N=2, d=2 toy example  

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define SRAM_SIZE           1024    // bytes of shared RAM per block
#define sequence_length       2     // N
#define embed_dimension       2     // d

// tile dims
constexpr int Bc = SRAM_SIZE / (4 * embed_dimension);  
constexpr int Br = std::min(SRAM_SIZE / (4 * embed_dimension), embed_dimension);

static_assert(Bc > 0 && Br > 0, "Invalid tile sizes");

constexpr int Tr = (sequence_length + Br - 1) / Br;  
constexpr int Tc = (sequence_length + Bc - 1) / Bc;  

__global__ void flashAttentionForward(
    const float *Q,        // [N×d]
    const float *K,        // [N×d]
    const float *V,        // [N×d]
    float       *O,        // [N×d], init zero
    float       *rowMax,   // [N]
    float       *rowSum,   // [N]
    float        scale     // = 1/√d
) {
    int t = threadIdx.x;

    // shared tiles
    __shared__ float Qs[Br * embed_dimension];
    __shared__ float Ks[Bc * embed_dimension];
    __shared__ float Vs[Bc * embed_dimension];

    // local buffers
    float scr[Br * Bc];
    float wts[Br * Bc];

    // loop over column-tiles
    for (int cb = 0; cb < Tc; ++cb) {
        // load K,V
        if (t < Bc) {
            int base = cb * Bc * embed_dimension + t * embed_dimension;
            for (int d = 0; d < embed_dimension; ++d) {
                Ks[t*embed_dimension + d] = K[base + d];
                Vs[t*embed_dimension + d] = V[base + d];
            }
        }
        __syncthreads();

        // loop over row-tiles
        for (int rb = 0; rb < Tr; ++rb) {
            // load Q
            if (t < Br) {
                int base = rb * Br * embed_dimension + t * embed_dimension;
                for (int d = 0; d < embed_dimension; ++d)
                    Qs[t*embed_dimension + d] = Q[base + d];
            }
            __syncthreads();

            if (t < Br) {
                int row = rb*Br + t;
                float m = -1e20f;
                // dot Q·K^T
                for (int j = 0; j < Bc; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < embed_dimension; ++d)
                        dot += Qs[t*embed_dimension + d] * Ks[j*embed_dimension + d];
                    float s = dot * scale;
                    scr[t*Bc + j] = s;
                    m = fmaxf(m, s);
                }
                if (row < sequence_length) rowMax[row] = m;

                // softmax
                float sum = 0.0f;
                for (int j = 0; j < Bc; ++j) {
                    float ex = expf(scr[t*Bc + j] - m);
                    wts[t*Bc + j] = ex;
                    sum += ex;
                }
                if (row < sequence_length) rowSum[row] = sum;

                // weighted sum × V → O
                int ob = row * embed_dimension;
                for (int d = 0; d < embed_dimension; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < Bc; ++j)
                        acc += wts[t*Bc + j] * Vs[j*embed_dimension + d];
                    O[ob + d] = (sum > 0 ? acc / sum : 0.0f);
                }
            }
            __syncthreads();
        }
    }
}

int main() {
    // host buffers
    float (*hQ)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*hK)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*hV)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*hO)[embed_dimension] = new float[sequence_length][embed_dimension]();
    float *hMax = new float[sequence_length];
    float *hSum = new float[sequence_length];

    // init
    for (int i = 0; i < sequence_length; ++i) {
        for (int d = 0; d < embed_dimension; ++d) {
            hQ[i][d] = 2.f*rand()/RAND_MAX - 1.f;
            hK[i][d] = 2.f*rand()/RAND_MAX - 1.f;
            hV[i][d] = 2.f*rand()/RAND_MAX - 1.f;
        }
        hMax[i] = -1e20f;
        hSum[i] = 0.f;
    }

    // device alloc
    float *dQ, *dK, *dV, *dO, *dMax, *dSum;
    cudaMalloc(&dQ, sequence_length*embed_dimension*sizeof(float));
    cudaMalloc(&dK, sequence_length*embed_dimension*sizeof(float));
    cudaMalloc(&dV, sequence_length*embed_dimension*sizeof(float));
    cudaMalloc(&dO, sequence_length*embed_dimension*sizeof(float));
    cudaMalloc(&dMax, sequence_length*sizeof(float));
    cudaMalloc(&dSum, sequence_length*sizeof(float));

    // copy → device
    cudaMemcpy(dQ, hQ, sequence_length*embed_dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, sequence_length*embed_dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, sequence_length*embed_dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dO, hO, sequence_length*embed_dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMax,hMax,sequence_length*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSum,hSum,sequence_length*sizeof(float), cudaMemcpyHostToDevice);

    // launch
    float scale = 1.f / sqrtf((float)embed_dimension);
    flashAttentionForward<<<1, Br>>>(dQ,dK,dV,dO,dMax,dSum,scale);
    cudaDeviceSynchronize();

    // copy ← device
    cudaMemcpy(hO, dO, sequence_length*embed_dimension*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hMax,dMax,sequence_length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hSum,dSum,sequence_length*sizeof(float), cudaMemcpyDeviceToHost);

    // print
    std::cout<<"Output:\n";
    for(int i=0;i<sequence_length;++i){
        for(int d=0;d<embed_dimension;++d)
            std::cout<<hO[i][d]<<" ";
        std::cout<<"\n";
    }

    // cleanup
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dMax); cudaFree(dSum);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO;
    delete[] hMax; delete[] hSum;
    return 0;
}
