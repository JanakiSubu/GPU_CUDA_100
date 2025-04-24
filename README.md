# GPU 100 Days Learning Journey

A daily log of my hands‑on work in GPU programming alongside learnings from *Parallel Programming and Optimization* (PMPP).

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

## Day 2 — matrixadd.cu

**Project File:** `matrixadd.cu`

**What I Did**  
- Wrote a CUDA kernel that adds two N×N matrices element‑by‑element. Launched a 2D grid of 16×16 thread‑blocks, mapping each thread to one output element.
- Added an if (row < N && col < N) guard to prevent out‑of‑bounds writes.
- 2D thread mapping - used a 2D grid of 16×16 thread‑blocks and computed to map each CUDA thread onto a unique matrix element.
  `int row = blockIdx.y*blockDim.y + threadIdx.y;`
  `int col = blockIdx.x*blockDim.x + threadIdx.x;`
- Allocated and initialized matrices with cudaMallocManaged and a simple initMatrix loop.

**Key Takeaways**  
- Mastered mapping 2D data onto the CUDA grid–block–thread hierarchy.
- Saw how a single `__global__` function can process an entire matrix by distributing work.

**What I Read**  
- Finished PMPP Chapter 2: Memory hierarchy and data locality in CUDA, and why coalesced global loads matter even for simple element‑wise kernels.

## Day 3 — matrix_vec_mult.cu
Project File: `matrix_vec_mult.cu`

**What I Did**
- Wrote a CUDA kernel that multiplies an N×N matrix by a length‑N vector, computing one dot‑product per thread.
- Launched a 1D grid of threads `(gridSize = (N+blockSize–1)/blockSize, blockSize = 256)`, mapping `threadIdx.x + blockIdx.x*blockDim.x` → row index.
- Added an if (row < N) guard to prevent out‑of‑bounds accesses.
- Allocated matrix and vectors with cudaMallocManaged, initialized host data in simple loops, then let unified memory handle host↔device transfers.

**Key Takeaways**
- Learned how to map a 1D data structure (rows of a matrix) onto CUDA’s grid–block–thread hierarchy for dot‑product operations.
- Understood the importance of bounds checking when the total thread count may exceed the problem size.
- Saw how a single __global__ kernel can parallelize all row‑wise dot‑products, leveraging unified memory to simplify data management.

## Day 04 — partialsum.cu
**Project File**: `partialsum.cu`

**What I Did**
- Summed two elements per thread into shared memory with guarded, coalesced loads.
- Implemented a tree-based inclusive `scan (O(log blockSize))` using synchronized strides.
- Computed `gridSize = (N + 2*blockSize - 1)/(2*blockSize)` to cover all inputs.
- Added a `CUDA_CHECK` macro to validate every CUDA API call.

Key Takeaways
- Tree-based scan pattern: How to implement an in-place inclusive prefix sum in shared memory with minimal divergence.
- Memory coalescing: Summing two elements per thread maximizes contiguous loads and stores, boosting bandwidth utilization.
- Grid‐block sizing: Planning 2 elements per thread lets you halve the number of blocks needed, improving occupancy.










  
