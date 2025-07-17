// cnn.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call)                                       \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      fprintf(stderr,                                          \
        "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,          \
        cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while(0)

////////////////////////////////////////////////////////////////////////////////
// 1) im2col and col2im
////////////////////////////////////////////////////////////////////////////////
__global__ void im2col(
    const float* __restrict__ input,
    float*       __restrict__ col,
    int C, int H, int W,
    int K, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = idx / outW;
    for (int c = 0; c < C; ++c)
    for (int ky = 0; ky < K; ++ky)
    for (int kx = 0; kx < K; ++kx) {
        int in_y = oh + ky;
        int in_x = ow + kx;
        int row = c*K*K + ky*K + kx;
        col[row*total + idx] = input[c*H*W + in_y*W + in_x];
    }
}

__global__ void col2im(
    const float* __restrict__ dcol,
    float*       __restrict__ dinput,
    int C, int H, int W,
    int K, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = idx / outW;
    for (int c = 0; c < C; ++c)
    for (int ky = 0; ky < K; ++ky)
    for (int kx = 0; kx < K; ++kx) {
        int in_y = oh + ky;
        int in_x = ow + kx;
        int row = c*K*K + ky*K + kx;
        atomicAdd(&dinput[c*H*W + in_y*W + in_x],
                  dcol[row*total + idx]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// 2) Forward: conv, ReLU, maxpool
////////////////////////////////////////////////////////////////////////////////
__global__ void gemmConv(
    const float* __restrict__ col,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    int outH, int outW, int F, int P)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = outH*outW*F;
    if (idx >= total) return;

    int ow = idx % (outW*outH);
    int f  = idx / (outW*outH);
    float sum = 0.0f;
    for (int i = 0; i < P; ++i)
        sum += col[i*(outH*outW) + ow] * weight[i*F + f];
    output[f*(outH*outW) + ow] = sum;
}

__global__ void reluAct(float* data, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) data[i] = fmaxf(0.0f, data[i]);
}

__global__ void maxpool(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int C, int H, int W,
    int pool, int stride,
    int outH, int outW)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = C*outH*outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int tmp = idx / outW;
    int oh = tmp % outH;
    int c  = tmp / outH;

    float m = -1e20f;
    for (int i = 0; i < pool; ++i)
    for (int j = 0; j < pool; ++j) {
        int y = oh*stride + i;
        int x = ow*stride + j;
        m = fmaxf(m, input[c*H*W + y*W + x]);
    }
    output[c*outH*outW + oh*outW + ow] = m;
}

////////////////////////////////////////////////////////////////////////////////
// 3) Backward: pool grad, conv grads
////////////////////////////////////////////////////////////////////////////////
__global__ void maxpoolGrad(
    const float* __restrict__ input,
    const float* __restrict__ gradOut,
    float*       __restrict__ gradIn,
    int C, int H, int W,
    int pool, int stride,
    int outH, int outW)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = C*outH*outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int tmp = idx / outW;
    int oh = tmp % outH;
    int c  = tmp / outH;

    float m = -1e20f; int mi=0, mj=0;
    for (int i = 0; i < pool; ++i)
    for (int j = 0; j < pool; ++j) {
        int y = oh*stride + i;
        int x = ow*stride + j;
        float v = input[c*H*W + y*W + x];
        if (v > m) { m = v; mi = y; mj = x; }
    }
    atomicAdd(&gradIn[c*H*W + mi*W + mj],
              gradOut[c*outH*outW + oh*outW + ow]);
}

__global__ void gemmDW(
    const float* __restrict__ col,
    const float* __restrict__ gradY,
    float*       __restrict__ gradW,
    int outH, int outW, int P, int F)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = P*F;
    if (idx >= total) return;

    int p = idx % P;
    int f = idx / P;
    float sum = 0.0f;
    for (int i = 0; i < outH*outW; ++i)
        sum += col[p*(outH*outW) + i] * gradY[f*(outH*outW) + i];
    gradW[p*F + f] = sum;
}

__global__ void gemmDX(
    const float* __restrict__ gradY,
    const float* __restrict__ weight,
    float*       __restrict__ gradCol,
    int outH, int outW, int P, int F)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = P*(outH*outW);
    if (idx >= total) return;

    int col_idx = idx % (outH*outW);
    int p       = idx / (outH*outW);
    float sum = 0.0f;
    for (int f = 0; f < F; ++f)
        sum += weight[p*F + f] * gradY[f*(outH*outW) + col_idx];
    gradCol[p*(outH*outW) + col_idx] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// 4) Host wrappers
////////////////////////////////////////////////////////////////////////////////
void convForward(
    const float* input, const float* weight, float* out,
    int C, int H, int W, int K, int F,
    float* colBuf)
{
    int outH = H - K + 1, outW = W - K + 1;
    int P = C*K*K;
    int threads = 256;
    int total = outH*outW, bUn = (total + threads - 1)/threads;
    im2col<<<bUn,threads>>>(input, colBuf, C,H,W,K,outH,outW);
    CHECK_CUDA(cudaGetLastError());

    int totOut = F*outH*outW, bG = (totOut + threads - 1)/threads;
    gemmConv<<<bG,threads>>>(colBuf, weight, out, outH,outW,F,P);
    CHECK_CUDA(cudaGetLastError());
}

void convBackward(
    const float* input, const float* weight,
    const float* gradY, float* gradW, float* gradX,
    int C, int H, int W, int K, int F,
    float* colBuf, float* gradCol)
{
    int outH = H - K + 1, outW = W - K + 1;
    int P = C*K*K, threads = 256;

    CHECK_CUDA(cudaMemset(gradW, 0, sizeof(float)*P*F));
    CHECK_CUDA(cudaMemset(gradX, 0, sizeof(float)*C*H*W));

    int bUn = ((outH*outW) + threads - 1)/threads;
    im2col<<<bUn,threads>>>(input, colBuf, C,H,W,K,outH,outW);
    CHECK_CUDA(cudaGetLastError());

    int bDW = (P*F + threads - 1)/threads;
    gemmDW<<<bDW,threads>>>(colBuf, gradY, gradW, outH,outW,P,F);
    CHECK_CUDA(cudaGetLastError());

    int bDX = (P*(outH*outW) + threads - 1)/threads;
    gemmDX<<<bDX,threads>>>(gradY, weight, gradCol, outH,outW,P,F);
    CHECK_CUDA(cudaGetLastError());

    col2im<<<bUn,threads>>>(gradCol, gradX, C,H,W,K,outH,outW);
    CHECK_CUDA(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////
// 5) main() with verification prints
////////////////////////////////////////////////////////////////////////////////
int main(){
    const int C=1, H=4, W=4, K=3, F=2;
    const int outH = H-K+1, outW = W-K+1;
    size_t inSize   = C*H*W;
    size_t wSize    = F*C*K*K;
    size_t outSize  = F*outH*outW;

    // Host init
    float h_in[inSize], h_w[wSize];
    for(int i=0;i<inSize;i++)   h_in[i] = float(i+1);
    for(int i=0;i<wSize;i++)    h_w[i]  = float((i%3)-1);

    // Device alloc
    float *d_in,*d_w,*d_conv,*d_pool;
    float *d_col,*d_gradCol,*d_gradW,*d_gradX,*d_gradPool;
    CHECK_CUDA(cudaMalloc(&d_in,      inSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w,       wSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv,    outSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pool,    outSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col,     C*K*K*outH*outW*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gradCol, C*K*K*outH*outW*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gradW,   wSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gradX,   inSize*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gradPool,outSize*sizeof(float)));

    // Copy inputs
    CHECK_CUDA(cudaMemcpy(d_in, h_in, inSize*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w,  h_w,  wSize*sizeof(float),  cudaMemcpyHostToDevice));

    // --- Forward conv + ReLU ---
    convForward(d_in, d_w, d_conv, C,H,W,K,F, d_col);
    int threads = 256;
    int bReLU = (outSize + threads - 1)/threads;
    reluAct<<<bReLU,threads>>>(d_conv, outSize);

    // --- Forward maxpool over F channels (not C!) ---
    int pool=2, stride=2;
    int pH = outH/pool, pW = outW/pool;
    int bPool = (F * pH * pW + threads - 1)/threads;   // <-- use F here
    maxpool<<<bPool,threads>>>(d_conv, d_pool, F, outH, outW, pool, stride, pH, pW);

    // --- Backward maxpool ---
    CHECK_CUDA(cudaMemset(d_gradPool,0,outSize*sizeof(float)));
    maxpoolGrad<<<bPool,threads>>>(d_conv, d_pool, d_gradPool, F, outH, outW, pool, stride, pH, pW);

    // --- Backward conv ---
    convBackward(d_in, d_w, d_gradPool, d_gradW, d_gradX, C,H,W,K,F, d_col, d_gradCol);

    // --- Copy back and print ---
    float h_conv[outSize], h_pool[outSize];
    float h_gradW[wSize], h_gradX[inSize];
    CHECK_CUDA(cudaMemcpy(h_conv,    d_conv,     outSize*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_pool,    d_pool,     outSize*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gradW,   d_gradW,    wSize*sizeof(float),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gradX,   d_gradX,    inSize*sizeof(float),   cudaMemcpyDeviceToHost));

    printf("\n=== Forward Conv ===\n");
    for(int f=0; f<F; f++){
      printf("Filter %d:\n", f);
      for(int i=0;i<outH;i++){
        for(int j=0;j<outW;j++)
          printf("%6.1f ", h_conv[f*outH*outW + i*outW + j]);
        printf("\n");
      }
      printf("\n");
    }

    printf("=== After 2×2 Pool ===\n");
    for(int f=0; f<F; f++){
      printf("Filter %d:\n", f);
      for(int i=0;i<pH;i++){
        for(int j=0;j<pW;j++)
          printf("%6.1f ", h_pool[f*pH*pW + i*pW + j]);
        printf("\n");
      }
      printf("\n");
    }

    printf("=== dW ===\n");
    for(int f=0; f<F; f++){
      printf("Filter %d:\n", f);
      for(int p=0;p<C*K*K;p++){
        printf("%6.3f ", h_gradW[p*F + f]);
        if((p+1)%K==0) printf("  ");
        if((p+1)%(K*K)==0) printf("\n");
      }
      printf("\n");
    }

    printf("=== dX ===\n");
    for(int c=0; c<C; c++){
      printf("Channel %d:\n", c);
      for(int i=0;i<H;i++){
        for(int j=0;j<W;j++)
          printf("%6.3f ", h_gradX[c*H*W + i*W + j]);
        printf("\n");
      }
      printf("\n");
    }

    // Cleanup
    cudaFree(d_in);       cudaFree(d_w);
    cudaFree(d_conv);     cudaFree(d_pool);
    cudaFree(d_col);      cudaFree(d_gradCol);
    cudaFree(d_gradW);    cudaFree(d_gradX);
    cudaFree(d_gradPool);

    printf("Done forward/backward pass.\n");
    return 0;
}
