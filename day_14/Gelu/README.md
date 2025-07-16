# **GELU Kernel**

## Files

1. **`glu_python.py`**  
   - A Python reference implementation of the GELU (Gaussian Error Linear Unit) activation.  
   - Serves to validate the CUDA version and examine I/O behavior.  
   - Works with input arrays of any shape or dimensionality.

2. **`glu.cu`**  
   - A CUDA-based GELU activation kernel optimized for GPU execution.  
   - Designed for high-throughput tensor processing with efficient memory usage and parallelism.
