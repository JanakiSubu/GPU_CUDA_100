# GPU 100 Days Learning Journey

A daily log of my hands‑on work in GPU programming alongside learnings from *Parallel Programming and Optimization* (PMPP).

**Inspired By:** [hkproj (https://github.com/hkproj/)  ](https://github.com/hkproj/)  
**100‑Day Challenge Base Repo:** [JanakiSubu/GPU_CUDA_100](https://github.com/JanakiSubu/GPU_CUDA_100/tree/main) 

---

## Day 1 — vectadd.cu

**Project File:** `vectadd.cu`

**What I Did**  
- Wrote a basic Hello World from GPU Code!
- Wrote a CUDA kernel that adds two float arrays element‑by‑element.Launched one thread per element, each computing `C[i] = A[i] + B[i]`.Used `blockIdx.x * blockDim.x + threadIdx.x` to map threads to data. Added an `if (i < N)` guard to prevent out‑of‑bounds writes.


**Key Takeaways**  
- Declared and invoked a `__global__` function on the GPU.  
- Understood the grid–block–thread hierarchy and how to size them.  
- Managed GPU memory lifecycle with `cudaMalloc`, `cudaMemcpy`, `cudaFree`.  
- Synchronized the device using `cudaDeviceSynchronize()` to flush and catch errors.

**What I Read**  
- Finished PMPP Chapter 1: Overview of parallel architectures, CUDA execution model, and fundamentals of GPU programming.  
