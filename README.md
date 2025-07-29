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

## Day 05 — layernorm.cu  
**Project File**: `layernorm.cu`

---

### What I Did

- Implemented **Layer Normalization** on each row of a 2D matrix.
- Used **shared memory** to store row data, reducing global memory accesses.
- Calculated **mean** and **variance** for each row, then normalized the elements.
- Used **`__syncthreads()`** to synchronize threads in a block.
- Applied **1e-7 epsilon** to avoid divide-by-zero errors in standard deviation.
- Validated **CUDA API** calls with a **`CUDA_CHECK`** macro.

---

### Key Takeaways

- **Parallelism**: Leveraged CUDA’s thread-block model to process rows independently.
- **Shared Memory**: Reduced latency by using shared memory for row data.
- **Grid-Block Sizing**: One block per row optimized performance.
- **Numerical Stability**: Added epsilon to avoid divide-by-zero errors.
- **Efficient Memory Access**: Coalesced memory accesses improved bandwidth utilization.

---

## Day 06 — MatrixTranspose.cu

**Project File:** `MatrixTranspose.cu`

**What I Did**  
- Implemented a naïve transpose kernel with bounds checks.  
- Chose TILE_DIM = 32 and BLOCK_ROWS = 8 so each block handles a 32×32 tile in eight thread-rows, balancing shared-memory usage and occupancy.
- Launched the kernel with a 2D grid of blocks sized (width + TILE_DIM - 1) / TILE_DIM × (height + TILE_DIM - 1) / TILE_DIM to cover the entire matrix.

**Key Takeaways**  
- Coalesced Access: Tiling groups global loads and stores into contiguous bursts, significantly improving memory throughput.
- Shared-Memory Reuse: Staging data in on-chip shared memory reduces redundant global reads and writes.
- Bank-Conflict Avoidance: Adding one column of padding prevents threads in the same warp from hitting the same shared-memory bank.

---

## Day 7 — CUDA Convolution Recipes  
**Project Files:**  
- `one_d_convolution.cu`  
- `one_d_convolution_with_tiling.cu`  
- `2d_convolution_with_tiling.cu`  

### What I Did  
- **one_d_convolution.cu**:  
  - Implemented a naïve 1D convolution kernel that slides a 1×K mask over an input array.  
  - Mapped each thread to compute one output element, using `blockIdx.x * blockDim.x + threadIdx.x`.  
  - Added boundary checks (halo cells) to skip out-of-bounds memory accesses.

- **one_d_convolution_with_tiling.cu**:  
  - Extended the 1D version with shared-memory tiling.  
  - Loaded each tile (plus halo regions) of the input array into shared memory.  
  - Used `__syncthreads()` to synchronize before computing the convolution.

- **2d_convolution_with_tiling.cu**:  
  - Generalized tiling to 2D: divided the input matrix into TILE×TILE patches in shared memory.  
  - Loaded top, bottom, left, and right halos to handle borders correctly.  
  - For each output pixel, computed a full MASK×MASK dot-product against the shared-memory tile.

### Key Takeaways  
- **Parallel convolution fundamentals**  
  - Learned to map CUDA threads to output indices and use halo cells to guard against out-of-bounds reads.  
- **Shared-memory tiling**  
  - Mastered loading contiguous blocks (and halos) into on-chip memory to reduce global-memory traffic.  
- **Synchronization & performance trade-offs**  
  - Saw how `__syncthreads()` ensures data consistency and how tile size choices impact shared-memory usage, occupancy, and overall throughput.  
- **Scaling from 1D to 2D**  
  - Translated 1D tiling patterns into a 2D grid layout and managed corner and edge cases for correct 2D convolution.  
---

## Day 08 — prefixsum_brent_kung_algorithm.cu

**Project File:** `prefixsum_brent_kung_algorithm.cu`

### What I Did
- Loaded pairs of elements into shared memory and handled out-of-bounds indices with zero-padding.
- Implemented the Brent–Kung scan:
  - **Up-sweep (reduce)**: built a partial‐sum tree in-place.
  - **Down-sweep**: distributed the partial sums back down to produce the final prefix sums.
- Launched a 1D grid where each block processes 64 elements (2 × 32 threads).
- Added rigorous bounds checks on loads and stores to avoid illegal memory accesses.
- Used `__restrict__` and `const` qualifiers to help the compiler optimize global memory traffic.
- Wrapped CUDA calls in an `inline checkCudaError()` helper for clearer error reporting.

### Key Takeaways
- **Hierarchical scan structure**: splitting work into a balanced binary tree (up-sweep) then propagating results (down-sweep) yields work-efficient parallel scans.
- **Shared‐memory orchestration**: careful use of `__syncthreads()` between strides is critical to ensure correctness without excessive divergence.
- **Block‐level vs. full-array scan**: while this version computes an inclusive scan within each block, extending it to arbitrary-length arrays requires a second (“carry‐propagate”) pass across blocks.
- **Performance hygiene**: zero-padding incomplete segments and marking pointers as `__restrict__` prevents hidden data hazards and helps maximize GPU throughput.

---

## Day 09 — flash_attention_forward.cu

**Project File:** `flash_attention_forward.cu`

---

**What I Did**  
- Implemented a tile-based Flash Attention forward pass in CUDA (toy example: N=2, d=2).  
- Staged Q, K, V in Br×Bc shared-memory tiles, computed per-tile Q·Kᵀ scores.  
- Subtracted each row’s max before `expf()` and recorded both max and sum-of-exps for numerical stability.  
- Accumulated softmax-weighted sums into the output buffer with a single-kernel prototype.  

**Key Takeaways**  
- **Shared-memory tiling:** Reduced global memory traffic by staging Q/K/V blocks on-chip.  
- **Numerical stability:** Learned the importance of row-max subtraction to avoid overflow in `expf()`.  
- **Tile sizing trade-offs:** Observed how Br and Bc (derived from `SRAM_SIZE`) affect occupancy and register usage.  

**What I Read**  
- Tri Dao et al., “FlashAttention: Fast and Memory‐Efficient Exact Attention with IO-Awareness,” NeurIPS 2022  
- NVIDIA CUDA C Programming Guide — shared memory, occupancy tuning, and kernel optimizations

- ## Day 10 — Sparse_MatrixVecMult_Hybrid.cu

**Project File:** `Sparse_MatrixVecMult_Hybrid.cu`

**What I Did**
- Implemented a hybrid SpMV kernel that packs up to **TH=20** nonzeros per row into ELL (Ellpack), spilling extras into a global COO array via `atomicAdd`.
- Zero-filled unused ELL slots and stored per-row column indices.
- In each thread:
  1. **ELL multiplication:** iterate fixed TH entries per row.  
  2. **COO accumulation:** scan the global COO list, adding matching-row entries.
- Wrapped CUDA calls in a `CUDA_CHECK` macro for robust error handling.

**Key Takeaways**
- **ELL vs. COO trade-offs:** ELL gives regular accesses for up to TH nonzeros; COO handles overflow with minimal padding.
- **Atomic writes:** `atomicAdd` appends COO entries without precomputing row quotas, at the cost of serialized writes.
- **Memory layout:** storing ELL in column‐major “slices” (`[p * N + row]`) yields coalesced loads for `x[col]`.

**What I Read**
- PMPP Chapter 10: Parallel sparse‐matrix techniques—CSR/ELL/COO formats, load balancing, padding for regularized access.

---

## Day 10 — benchmark.py

**Project File:** `benchmark.py`

**What I Did**
- Built a Python harness to benchmark my CUDA SpMV vs. `torch.sparse.mm`.  
- Automated:
  - **NVCC compilation** of `main.cu` with injected `N`, `M`, and `threshold`.  
  - **Kernel timing** via CUDA events and PyTorch timing events.  
- Logged **memory usage** (`psutil`) and **estimated sparse footprint** to avoid OOM.

**Key Takeaways**
- **End-to-end benchmarking** must include compile, transfer, and launch overheads.  
- **Memory-safety checks** (e.g. cap nnz to 70% of RAM) prevent large-matrix crashes.  
- **Source injection** simplifies multi-size testing without manual edits.

---


## Day 11 — merge\_path\_parallel\_merge.cu

**Project File:** `merge_path_parallel_merge.cu`

**What I Did**

* Implemented the **Merge Path Parallel Merge** algorithm using CUDA.
* Created a `merge_path_partition()` device function to perform binary search across the diagonal `k = i + j` in the logical 2D merge grid of arrays A and B.
* Launched `N + M` threads where each thread:

  * Computes its diagonal index and determines the co-rank split `(i, j)`.
  * Selects the smaller of `A[i]` or `B[j]` and writes to `C[k]`.
* Used `printf()` inside the kernel to trace: thread ID, diagonal, source array, and output position.
* Verified the final merged array was fully sorted and correct.

**Key Takeaways**

* **Merge Path** provides a work-balanced parallel merge across threads with no synchronization needed.
* Learned how to use **diagonal binary search** to assign merge ranges in parallel.
* Understood how **co-ranking** is generalized in Merge Path and applied in **Thrust** and **CUB**.
* Added thread-wise kernel printouts to verify correctness and thread assignment.

**What I Read**

* PMPP Chapter 11: Merge Sort, co-rank vs. merge path, and tiled merging.
* “Merge Path: A Visually Intuitive Approach to Parallel Merging” by Green et al.
* Thrust and CUB source code: `merge()` and `merge_by_key()` implementation insights.
---
## Day 12 — tiled_matmul.cu

**Project File:** `tiled_matmul.cu`

### What I Did

- Implemented a **tiled matrix multiplication (GEMM)** kernel on the GPU using shared-memory.  
- Divided each \(N\times N\) matrix into `TILE_SIZE×TILE_SIZE` sub-blocks that each thread-block cooperatively loads into shared memory.
  
- Each thread computes one element of \(C\) by iterating over all tiles:  
  1. Loads a tile of A and a tile of B into `__shared__` arrays (`sA`, `sB`), zero-padding out-of-bounds entries.  
  2. Synchronizes with `__syncthreads()`.  
  3. Performs a `#pragma unroll` inner loop of length `TILE_SIZE` to accumulate the dot-product.  
  4. Synchronizes again before loading the next tile.
     
- Wrapped the kernel launch in **CUDA events** (`cudaEventRecord`/`cudaEventElapsedTime`) to measure in-kernel execution time.  
- Added host-side setup to:  
  - Parse `N` from the command line (default 256 for quick demos).  
  - Allocate and initialize host/device arrays.  
  - Verify correctness by checking `C[0]` and `C[N*N-1]` against the expected sum.


### Key Takeaways

- **Shared-Memory Tiling:** Greatly reduces global-memory traffic by reusing each tile across multiple multiplications.  
- **Work Distribution:** Each thread handles one output element, ensuring balanced compute.  
- **Performance Instrumentation:** CUDA events provide precise kernel timing, decoupled from host overhead.  

### What I Read
- PMPP Chapter 4: Tiled algorithms and memory hierarchies for dense linear algebra.  

---
## Day 13 — cmpFHD.cu & cmpFHD_real_image.cu

**Project Files:**  
- `cmpFHD.cu`  
- `cmpFHD_real_image.cu`  

### What I Did

- **Core FHD kernel** (`cmpFHD.cu`):  
  - Implemented the Fully-Hybrid Domain (FHD) update pass over non-Cartesian k-space samples in CUDA.  
  - Broke the full trajectory (`M = 1024` samples) into `CHUNK_SIZE = 256` tiles and loaded each tile into `__constant__` memory (`kx_c, ky_c, kz_c`).  
  - Each GPU thread:  
    1. Reads its point coordinates `(x[n], y[n], z[n])` and initial complex accumulator `(rPhi, iPhi)`.  
    2. Loops over the tile, computes  
       ```cpp
       angle = 2π * (kx·x + ky·y + kz·z);
       realAcc += rMu[m]*cosf(angle) − iMu[m]*sinf(angle);
       imagAcc += iMu[m]*cosf(angle) + rMu[m]*sinf(angle);
       ```  
    3. Writes back updated `(rPhi, iPhi)` and computes magnitude `phiMag = sqrt(r² + i²)`.  
  - Host orchestration:  
    - Allocates and initializes host arrays (`h_x, h_y, h_z, h_rMu, h_iMu, h_rPhi, h_iPhi`).  
    - Copies data to device buffers and, per chunk, uploads trajectory tile via `cudaMemcpyToSymbol`.  
    - Launches `cmpFHD<<<blocks, 256>>>(…)` and synchronizes.  
    - Copies results back and prints a few sample values for validation.

- **Real-image extension** (`cmpFHD_real_image.cu`):  
  - Integrated OpenCV to load a grayscale image (`lena_gray.png` → `CV_32F [0,1]`).  
  - Mapped pixel `(i, j)` → normalized `(x, y)` and intensity → `z`.  
  - Ran the identical chunked FHD kernel on this point cloud.  
  - Converted the per-pixel magnitude back to an 8-bit image and saved `output.jpg`.

###  Key Takeaways

1. **Constant-Memory Tiling**  Broadcasting a small tile of k-space samples to all threads dramatically cuts global-memory pressure.  
2. **Per-Thread Work Balance**  One thread per output point simplifies divergence and ensures each thread does equal work.  
3. **Hybrid Memory Management**  Managed vs. explicit `cudaMalloc`/`cudaMemcpy` approaches: tradeoff between simplicity and control.  
4. **Real-World Pipeline**  Integrating OpenCV with CUDA, handling I/O, pre/post-processing around the core compute kernel.  

### 📖 What I Read

- **PMPP Chapter 14** — Non-Cartesian MRI case study:  
  Iterative reconstruction techniques, k-space sampling patterns, performance-oriented kernel design.  
- **NVIDIA CUDA C Programming Guide** — Best practices for constant memory, occupancy tuning, and fast math intrinsics.

---

## Day 14 — Graph & Layer Kernels

**Project Files:**  
- [`bfs_kernel.cu`](https://github.com/JanakiSubu/GPU_CUDA_100/blob/main/day_14/Bfs/bfs_gpu.cu)  
- [`glu.cu`]([./day_13/glu.cu](https://github.com/JanakiSubu/GPU_CUDA_100/blob/main/day_14/Gelu/glu.cu))  
- [`linear_kernel.cu`]([./day_13/linear_kernel.cu](https://github.com/JanakiSubu/GPU_CUDA_100/blob/main/day_14/Linear_kernel/linear_kernel.cu))  

### What I Did  
- **BFS Kernel** (`bfs_kernel.cu`)  
  1.Wrote a parallel Breadth-First Search kernel using atomic operations to update node labels.  
  2.Explored level‐by‐level traversal with thread‐safe `atomicCAS` and frontier‐completion flags.

- **GELU Activation** (`glu.cu`)  
  1. Implemented the GELU (Gaussian Error Linear Unit) activation in CUDA for fast inference.  
  2. Verified against a Python reference to ensure numerical correctness.

- **Linear Layer** (`linear_kernel.cu`)  
  1. Built a batched linear layer using cuBLAS for `C = A × B`, followed by a custom bias‐add kernel.  
  2. Managed host/device memory and orchestrated cuBLAS calls plus CUDA kernel launches.

### Key Takeaways  
1. Designing graph‐traversal kernels with minimal divergence and safe atomic updates.  
2. Validating complex CUDA math (GELU) against a straightforward Python implementation.  
3. Integrating cuBLAS GEMM with custom CUDA kernels for a complete linear layer pipeline.  
4. Gained deeper insight into dynamic parallelism and its trade-offs in real-world examples.

  
### What I Read  
- **PMPP Chapter 12: Parallel Patterns for Graph Searches**  
  Background on graph structures and traversal mechanisms. Sequential vs. parallel BFS implementations. Optimizations for memory bandwidth and load balancing in graph algorithms.
- **PMPP Chapter 13: CUDA Dynamic Parallelism**  
  Fundamentals of launching kernels from the device, memory visibility rules, and nesting depth. Synchronization with streams and events inside dynamic kernels. A recursive Bezier‐curve       example with and without dynamic parallelism.


---

## Day 15 — flash_attention_backprop & CNN Backprop in CUDA

**Project Files:**  
- `flash.cu`  
- `cnn.cu`  

### What I Did

- **Flash Attention Backprop**  
  - Extended my Flash Attention forward pass to full backprop: computed gradients w.r.t. Q, K, V via softmax‐and‐matmul reverse chaining.  
  - Mirrored on-chip tiling (Br×Bc) from the forward pass to keep memory-efficient patterns.  
  - Diagnosed zero gradients in spots due to mismatched launch configs and missing “col2im” style gather in the tiled layout.  

- **CNN Backprop**  
  - Built an end-to-end CNN layer in CUDA:  
    1. **Forward**: conv → ReLU → max-pool using `im2col` → GEMM → `reluAct` → `maxpool` kernels.  
    2. **Backward**: pooling grads with `atomicAdd`, weight grads via `gemmDW`, input grads via `gemmDX` + `col2im`.  
  - Added a toy test in `main()` to print activations, pooled outputs, `dW`, and `dX` for validation.


### Key Takeaways

1. **Forward/Backward Alignment**  
   - Any unrolling or tiling in the forward pass must be mirrored exactly in backprop (e.g. implement `col2im` for dX).  
2. **Launch-Config Precision**  
   - Off-by-one grid/block calculations often explain zero-gradient anomalies—always double-check your total = rows×cols formulas.  
3. **Toy-Scale Verification**  
   - Dump intermediate tensors on a small example before scaling up to catch indexing and memory-layout bugs early.


### What I Read

- **PMPP Ch. 15: Molecular Visualization & Analysis**  Thread granularity and memory-coalescing strategies for large biomolecular data.  
- **PMPP Ch. 16: Machine Learning Case Study**  How cuDNN reduces CNN layers to GEMM under the hood for peak performance.  
- **PMPP Ch. 17: Parallel Programming & Computational Thinking**  Systematic problem decomposition and balancing compute vs. memory locality.

---
##  Day 16 - Naive Bayes classifier in CUDA

**Code:**  
`NaiveBayes.cu`, `NaiveBayesKernel.cuh`, `NaiveBayesTrain.cuh`, `NaiveBayesTrain.cpp`, and `main.cpp`

Implemented a CUDA-accelerated Naive Bayes classifier, focusing on the training and inference stages. Leveraging shared memory to maximize computational efficiency, the implementation is structured to divide work among threads for parallelized data processing of feature probabilities.

### 🔧 Components Developed

#### `NaiveBayes.cu`
- This file contains the CUDA kernel responsible for calculating feature likelihoods and class probabilities in parallel.  
- Shared memory was used where possible to minimize global memory access penalties.  
- Optimized kernel launches to balance between grid and block dimensions for datasets with high dimensionality.

#### `NaiveBayesKernel.cuh`
- Header file declaring the kernel functions, ensuring modularity and separation of concerns in code structure.

#### `NaiveBayesTrain.cuh`
- Declared the host-side training function, encapsulating the logic to copy data to the GPU, launch CUDA kernels, and retrieve results.

#### `NaiveBayesTrain.cpp`
- Implemented the host-side training process, providing pre-processing for input data and managing memory transfers between CPU and GPU.

#### `main.cpp`
- Entry point of the program, performing tasks like loading data, splitting datasets for training and testing, and evaluating model performance after training.

### Key Takeaways

- One-thread-per-sample model enabled scalable histogram-style computation.
- Shared memory significantly improved update locality for priors and likelihoods.
- Host-device modularity ensured reusability and clarity.
- Tuned grid/block dimensions for balanced memory latency and thread occupancy.

### What I Read

- **PMPP Chapter 9: Parallel Histograms and Voting**
- **PMPP Chapter 5: Synchronization and Shared Memory**
- **CUDA C Best Practices Guide: Shared memory vs. global memory access efficiency**

## Day 17 — cuBLAS Vector Addition (`cublasSaxpy`)

**Project File:** `vec_cublas.cu`

### What I Did
- Implemented vector addition on the GPU using the **cuBLAS** library.
- Used the `cublasSaxpy()` routine to compute `C = A + B` by performing `y = α * x + y` with `α = 1.0f`.
- Managed cuBLAS handle lifecycle with `cublasCreate()` and `cublasDestroy()`.
- Allocated memory on the device, initialized host data, and handled data transfers using `cudaMemcpy`.
- Verified output by copying the result back to host and printing sample elements.

### Key Takeaways
1. **cuBLAS Handle Management**  
  Learned to initialize and release cuBLAS context using `cublasCreate()` and `cublasDestroy()` to encapsulate library operations.

2. **AXPY Operation Basics**  
  Understood that `cublasSaxpy()` computes `y = α * x + y`. Setting `α = 1.0` makes it equivalent to element-wise addition.

3. **Performance & Simplicity**  
  Observed how vendor-optimized cuBLAS routines offer better performance and cleaner code than writing custom kernels for simple linear algebra operations.

4. **Memory Safety & Error Checking**  
  Wrapped CUDA and cuBLAS calls in error-checking macros to ensure robustness.

### What I Read
- **PMPP Chapter 3** — Leveraging cuBLAS and cuRAND libraries for optimized linear algebra and random number generation.
- **cuBLAS Library Documentation** — Usage pattern and parameter structure for `cublasSaxpy`.
- **CUDA C Programming Guide** — Best practices for mixing cuBLAS with custom kernels and memory management.

---

---
## Day 18 — Matrix Multiplication with cuBLAS (cublasSgemm)
*Project File:* `matmul_cublas.cu`

## What I Did
* Implemented matrix multiplication using the cuBLAS library function `cublasSgemm()`.
* Initialized matrices A and B on the host with values `A[i][j] = i + j`, `B[i][j] = i + j` for easy verification.
* Allocated GPU memory with `cudaMalloc` and transferred matrices from host to device using `cudaMemcpy`.
* Set scalar values: `alpha = 1.0f`, `beta = 0.0f`.
* Used `cublasSgemm()` with flags `CUBLAS_OP_N` for no transposition of A or B.
* Copied result matrix C back to host and printed the result using column-major indexing (`i + j * M`).
* Managed cuBLAS context creation and destruction using `cublasCreate()` and `cublasDestroy()`.

## Key Takeaways
* **cuBLAS for GEMM**  
  Learned to use `cublasSgemm()` for efficient matrix multiplication—crucial for deep learning workloads and scientific computing.
* **Column-Major Order**  
  cuBLAS expects column-major layout (Fortran-style), so indexing must follow `i + j * leadingDim` to correctly interpret results.
* **cuBLAS Handle Lifecycle**  
  Understood proper creation and destruction of the cuBLAS context (`cublasHandle_t`) to manage library state.
* **Parameter Mapping**  
  Mastered the mapping between C-style row-major host matrices and cuBLAS function parameters for correctness and performance.

## What I Read
* *PMPP Chapter 6: Matrix Multiplication and Shared Memory Optimization*  
  Foundations of tiled GEMM, matrix layout impacts, and performance best practices.
* *cuBLAS Developer Guide*  
  Focused on `cublasSgemm` usage, including leading dimensions, transposition flags, and memory alignment strategies.
* *CUDA Toolkit Documentation*  
  Reviewed usage of `cudaMalloc`, `cudaMemcpy`, and proper GPU memory management for third-party libraries.

---
## Day 19 — Fully Connected Neural Network using cuDNN

*Project File:* `fcnet.cu`

### **What I Did**

* Implemented a **fully connected neural network (FCNet)** using the **cuDNN** library in CUDA.
* Constructed a **3-layer architecture**:
  * **Input layer** – 1000 neurons
  * **Two hidden layers** – 512 neurons each
  * **Output layer** – 10 neurons
* Emulated dense connections via **1×1 convolutions**.
* Applied **ReLU activation** after each hidden layer using `cudnnActivationForward`.
* Initialized weights with **cuRAND** and zeroed the biases.
* Created and managed cuDNN descriptors for **tensors**, **filters**, **convolutions**, and **activations**.
* Ran a **10-epoch forward pass** over randomly generated inputs and labels.
* Printed a sample output from the **final epoch** for verification.
* Cleaned up all **GPU memory** and **cuDNN descriptors**.

### **Key Takeaways**

* Learned to build **dense layers** using **1×1 convolutions** in cuDNN for GPU acceleration.
* Gained hands-on with **tensor**, **filter**, and **convolution descriptor** setup and teardown.
* Mastered the use of `cudnnConvolutionForward`, `cudnnAddTensor` (bias add), and `cudnnActivationForward`.
* Reinforced best practices for **error checking**, **memory management**, and **resource cleanup**.
* Saw how cuDNN can model **non-convolutional layers** using its convolution primitives.

### **What I Read**

* **PMPP Chapter 8** — Using cuDNN for Deep Learning: structure mapping & inference pipelines
* **cuDNN Developer Guide** — Descriptor APIs, activation functions, and forward algorithms
* **CUDA cuRAND Documentation** — Pseudorandom weight initialization best practices
* **CUDA Toolkit Programming Guide** — Memory management, synchronization, and debugging techniques

---

## Day 20 — rope.cu

**Project File:** `rope.cu`

**What I Did**

* Implemented **Rotary Positional Encoding (RoPE)** in CUDA to inject relative position information into transformer token embeddings
* Wrote a `__global__` kernel that:
* Splits each query/key vector into even and odd halves
* Applies element-wise rotation using precomputed sine and cosine values
* Uses thread indices to map tokens × dimensions → vector elements
* Loaded the angle table (`θ`) into **shared memory** once per block to minimize global‐memory reads
* Launched a 2D grid:
  * `dim3 grid((seq_len + B-1)/B, (dim/2 + T-1)/T)`
  * `dim3 block(T, 1)` for coalesced access across sequence positions
* Validated on a toy sequence (N = 128, D = 64) by comparing against a CPU reference implementation and printing sample rotated vectors

**Key Takeaways**

* **RoPE Fundamentals:** Learned how complex‐valued rotations encode relative positions, eliminating the need for explicit absolute embeddings
* **CUDA Mapping:** Practiced mapping 2D data (tokens × dim/2) onto CUDA’s grid–block–thread hierarchy for element-wise operations
* **Shared‐Memory Optimization:** Saw significant bandwidth savings by staging constant sin/cos tables in shared memory
* **Numerical Stability:** Verified that precomputing angles at high precision on the host avoids drift in the GPU’s single‐precision trig evaluations
* **Launch‐Config Trade-offs:** Balanced block size vs. shared‐memory capacity to maximize occupancy without bank conflicts


---

