#include "bfs.h"
#include "bfs_kernel.cu"

/*
 * For now, we launch bfs_kernel multiple times from the host. This simple approach
 * helps avoid the warp divergence that can occur when trying to fuse everything into
 * a single kernel. In a production setting, I plan to replace this with a more
 * efficient design—using private and global queues—to better suit real-world use.
 */

void bfs_gpu(int source, int num_vertices, int num_edges, int* h_edges, int* h_dest, int* h_labels) {
    int *d_edges, *d_dest, *d_labels, *d_done;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_edges, (num_vertices + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dest, num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, num_vertices * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_done, sizeof(int)));
    
    CHECK_CUDA_ERROR(cudaMemset(d_labels, -1, num_vertices * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edges, h_edges, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dest, h_dest, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    
    int initial_level = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels + source, &initial_level, sizeof(int), cudaMemcpyHostToDevice));
    
    int level = 0;
    int h_done;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (num_vertices + threadsPerBlock - 1) / threadsPerBlock;
    
    do {
        h_done = 1;
        CHECK_CUDA_ERROR(cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice));
        
        bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(level, num_vertices, d_edges, d_dest, d_labels, d_done);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        CHECK_CUDA_ERROR(cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    } while (!h_done && level < num_vertices);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_edges));
    CHECK_CUDA_ERROR(cudaFree(d_dest));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_done));
}