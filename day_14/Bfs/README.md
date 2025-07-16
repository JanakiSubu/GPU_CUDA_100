## File Summaries

### `bfs.h`
**Summary**  
Defines the core macros, utility routines, and function prototypes needed for our GPU-accelerated BFS implementation.

---

### `bfs_kernel.cu`
**Summary**  
Contains the CUDA kernel that drives the parallel BFS. It relies on atomic operations to safely update node labels as multiple threads traverse edges in parallel.

**Learned**  
- Structuring a graph-traversal kernel for massive parallelism  
- Employing `atomicCAS` to prevent race conditions when marking nodes  
- Using level markers and atomic flags to detect when the BFS frontier has been fully processed

---

### `bfs_gpu.cu`
**Summary**  
Implements the `bfs_gpu` host function, which sets up device memory, launches the BFS kernel iteratively, and manages data transfers to and from the GPU.

**Learned**  
- Orchestrating repeated kernel invocations based on graph topology  
- Efficiently handling the BFS frontier and updating node labels on the device

---

### `bfs_cpu.c`
**Summary**  
Provides a serial CPU version of BFS alongside a random graph generator, serving as a correctness baseline when validating the GPU results.

**Learned**  
- Crafting a reliable reference BFS implementation for verification  
- Generating and populating a synthetic graph to test both CPU and GPU BFS routines
