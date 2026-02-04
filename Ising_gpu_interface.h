#ifndef ISING_GPU_INTERFACE_H
#define ISING_GPU_INTERFACE_H
#include <stddef.h> 

// Memory handling for CUDA
void* gpu_alloc(size_t size);
void  gpu_free(void* ptr);
void  gpu_memcpy_to_device(void* dest, const void* src, size_t size);
void  gpu_memcpy_to_host(void* dest, const void* src, size_t size);

// Kernel launches
void launch_setup_rng(void* d_states, unsigned int seed, int L, int row_stride);
void launch_metropolis_step(int* d_lattice, int L, int row_stride, float* d_lookup_probs, int color, void* d_states, int block_size);
void launch_sync_padding_gpu(int* d_lattice, int L, int row_stride);

#endif