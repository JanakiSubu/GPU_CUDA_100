// cmpFHD_real_image_ppm_fast.cu
// nvcc -O3 cmpFHD_real_image_ppm_fast.cu -o cmpFHD_ppm_fast
// ./cmpFHD_ppm_fast → writes output.ppm

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846f
#define CHUNK_SIZE 256

// smaller synthetic “image”
const int IMG_W = 256;
const int IMG_H = 256;

// constant‐memory chunk
__constant__ float kx_c[CHUNK_SIZE];
__constant__ float ky_c[CHUNK_SIZE];
__constant__ float kz_c[CHUNK_SIZE];

// FHD kernel
__global__ void cmpFHd(
    float* rPhi, float* iPhi, float* phiMag,
    const float* x, const float* y, const float* z,
    const float* rMu, const float* iMu,
    int chunkSize, int N)
{
    int idx = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    if (idx >= N) return;

    float xn = x[idx], yn = y[idx], zn = z[idx];
    float rA = rPhi[idx], iA = iPhi[idx];

    #pragma unroll 4
    for (int m = 0; m < chunkSize; ++m) {
        float ang = 2.0f*PI*(kx_c[m]*xn + ky_c[m]*yn + kz_c[m]*zn);
        float c = cosf(ang), s = sinf(ang);
        rA += rMu[m]*c - iMu[m]*s;
        iA += iMu[m]*c + rMu[m]*s;
    }

    rPhi[idx]   = rA;
    iPhi[idx]   = iA;
    phiMag[idx] = sqrtf(rA*rA + iA*iA);
}

// write a P5 PPM
void write_ppm(const char* fname, int w, int h, const unsigned char* data) {
    FILE* f = fopen(fname, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, w*h, f);
    fclose(f);
}

int main(){
    const int N = IMG_W * IMG_H;
    const int M = 512;              // total freq samples
    const int numChunks = M/CHUNK_SIZE; // = 2
    const int blocks = (N + FHD_THREADS_PER_BLOCK -1)/FHD_THREADS_PER_BLOCK;

    std::cout<<"Generating synthetic image ("<<IMG_W<<"×"<<IMG_H<<")...\n";

    // host buffers
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_z = new float[N];
    float *h_rMu = new float[M];
    float *h_iMu = new float[M];

    // build a simple radial pattern
    for(int y=0;y<IMG_H;++y) for(int x=0;x<IMG_W;++x){
        int idx=y*IMG_W + x;
        h_x[idx] = float(x)/IMG_W;
        h_y[idx] = float(y)/IMG_H;
        float dx = (x-IMG_W/2)/(float)(IMG_W/2);
        float dy = (y-IMG_H/2)/(float)(IMG_H/2);
        h_z[idx] = 0.5f + 0.5f*cosf(10.0f*sqrtf(dx*dx+dy*dy));
    }

    // random complex weights
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> d(-1,1);
    for(int i=0;i<M;++i){ h_rMu[i]=d(rng); h_iMu[i]=d(rng); }

    std::cout<<"Allocating device memory...\n";
    float *d_x,*d_y,*d_z,*d_rMu,*d_iMu,*d_rPhi,*d_iPhi,*d_phiMag;
    cudaMalloc(&d_x,N*sizeof(float));
    cudaMalloc(&d_y,N*sizeof(float));
    cudaMalloc(&d_z,N*sizeof(float));
    cudaMalloc(&d_rMu,M*sizeof(float));
    cudaMalloc(&d_iMu,M*sizeof(float));
    cudaMalloc(&d_rPhi,N*sizeof(float));
    cudaMalloc(&d_iPhi,N*sizeof(float));
    cudaMalloc(&d_phiMag,N*sizeof(float));

    // copy data
    cudaMemcpy(d_x,h_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_rMu,h_rMu,M*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_iMu,h_iMu,M*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_rPhi,0,N*sizeof(float));
    cudaMemset(d_iPhi,0,N*sizeof(float));

    std::cout<<"Running FHD over "<<numChunks<<" chunks...\n";
    for(int c=0;c<numChunks;++c){
        cudaMemcpyToSymbol(kx_c, h_x + c*CHUNK_SIZE, CHUNK_SIZE*sizeof(float));
        cudaMemcpyToSymbol(ky_c, h_y + c*CHUNK_SIZE, CHUNK_SIZE*sizeof(float));
        cudaMemcpyToSymbol(kz_c, h_z + c*CHUNK_SIZE, CHUNK_SIZE*sizeof(float));
        cmpFHd<<<blocks,FHD_THREADS_PER_BLOCK>>>(
            d_rPhi,d_iPhi,d_phiMag,
            d_x,d_y,d_z,
            d_rMu,d_iMu,
            CHUNK_SIZE, N
        );
        cudaDeviceSynchronize();
    }

    // retrieve magnitude
    float* h_phiMag = new float[N];
    cudaMemcpy(h_phiMag, d_phiMag, N*sizeof(float), cudaMemcpyDeviceToHost);

    // normalize to 0–255
    float minv=1e9f, maxv=-1e9f;
    for(int i=0;i<N;++i){
        if(h_phiMag[i]<minv) minv=h_phiMag[i];
        if(h_phiMag[i]>maxv) maxv=h_phiMag[i];
    }
    unsigned char* img = new unsigned char[N];
    for(int i=0;i<N;++i){
        float v = (h_phiMag[i]-minv)/(maxv-minv);
        img[i] = (unsigned char)(v*255.0f);
    }

    std::cout<<"Writing output.ppm...\n";
    write_ppm("output.ppm", IMG_W, IMG_H, img);

    // cleanup
    delete[] h_x; delete[] h_y; delete[] h_z;
    delete[] h_rMu; delete[] h_iMu; delete[] h_phiMag; delete[] img;
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_rMu); cudaFree(d_iMu);
    cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_phiMag);

    std::cout<<"Done.\n";
    return 0;
}
