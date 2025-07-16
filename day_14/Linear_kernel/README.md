# **Linear Layer Kernel**

## Project Structure

1. **`linear_kernel.cu`**  
   - Entry point for the Linear Layer example.  
   - Handles command-line parsing, orchestrates data allocation, and invokes helper routines and CUDA kernels.

2. **`helper_functions.h`**  
   - Declarations for utility routines: memory allocation/deallocation, error checking, and matrix initialization.  
   - Includes prototypes for CUDA memory management and status-reporting functions.

3. **`helper_functions.cpp`**  
   - Definitions for the utilities declared in `helper_functions.h`.  
   - Implements functions such as `initializeRandomMatrix()`, `allocateDeviceMemory()`, and comprehensive error checks.

4. **`cuda_kernels.h`**  
   - Declares the CUDA kernel `addBiasKernel` for parallel bias addition.  
   - Provides the signature for `performLinearLayerOperation()`, which combines GEMM and bias application.

5. **`cuda_kernels.cu`**  
   - Implements the CUDA kernels and their host wrappers:  
     - **`addBiasKernel`**: Parallel bias injection into the output matrix.  
     - **`performLinearLayerOperation()`**: Executes `cublasSgemm` for matrix multiplication, then calls `addBiasKernel`.

---

## **Building the Linear Layer**

Make sure CUDA and cuBLAS are installed, then compile with `nvcc`:

```bash
nvcc linear_kernel.cu helper_functions.cpp cuda_kernels.cu -lcublas -o linear_layer


## **Execution Workflow**
1. **Initialization**:
   - Allocate and initialize input matrices (`host_input`, `host_weights`, `host_bias`) on the host (CPU).  
   - Transfer data to the device (GPU) for computation.  

2. **Matrix Multiplication**:
   - Perform `C = A × B` using cuBLAS.  
   - Input (`A`), weights (`B`), and output (`C`) dimensions are defined by `batch_size`, `input_features`, and `output_features`.

3. **Bias Addition**:
   - Use the `addBiasKernel` CUDA kernel to add bias to each output feature.

4. **Results**:
   - Transfer the results from the GPU back to the CPU for verification and further processing.  

5. **Clean-Up**:
   - Free both host and device memory, and destroy the cuBLAS handle.

---
