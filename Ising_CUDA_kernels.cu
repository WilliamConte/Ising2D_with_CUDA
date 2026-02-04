#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Ising_gpu_interface.h"

// Memory handling

void* gpu_alloc(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void gpu_free(void* ptr) {
    cudaFree(ptr);
}

void gpu_memcpy_to_device(void* dest, const void* src, size_t size) {
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void gpu_memcpy_to_host(void* dest, const void* src, size_t size) {
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

// Kernels

/******************** Setup fot RNGs **********************/
__global__ void setup_rand_kernel(curandState* states, unsigned int seed, int L, int row_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x <= L && y <= L) {
        int index = y * row_stride + x;
        curand_init(seed, index, 0, &states[index]);
    }
}

/******************** Sync padding **********************/
__global__ void sync_padding_kernel(int* lattice, int L, int row_stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (k <= L) {
        lattice[0 * row_stride + k] = lattice[L * row_stride + k];
        lattice[(L + 1) * row_stride + k] = lattice[1 * row_stride + k];
        lattice[k * row_stride + 0] = lattice[k * row_stride + L];
        lattice[k * row_stride + (L + 1)] = lattice[k * row_stride + 1];
    }
}


/******************** Ising update step **********************/
__global__ void Ising_step_kernel(int* lattice, int L, int row_stride, float* lookup_probs, int color, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x <= L && y <= L) {
        if ((x + y) % 2 == color) {
            int spin_index = y * row_stride + x; 
            int current_spin = lattice[spin_index];
            
            int sum_n = lattice[(y - 1) * row_stride + x] + 
                        lattice[(y + 1) * row_stride + x] + 
                        lattice[y * row_stride + (x - 1)] + 
                        lattice[y * row_stride + (x + 1)];
            
            int lookup_index_0 = (current_spin == -1) ? 0 : 1;
            int lookup_index_1 = (sum_n + 4) / 2;

            float rand_val = curand_uniform(&states[spin_index]);

            if (rand_val < lookup_probs[lookup_index_0 * 5 + lookup_index_1]) {
                lattice[spin_index] = -current_spin;
            }
        }
    }
}

/******************** Wrappers (Kernel launch) **********************/

void launch_setup_rng(void* d_states, unsigned int seed, int L, int row_stride, int block_size) {
    curandState* states = (curandState*)d_states;

    // create blocks
    dim3 block(block_size, block_size);
    dim3 grid((L + block.x - 1) / block.x, (L + block.y - 1) / block.y);
    
    setup_rand_kernel<<<grid, block>>>(states, seed, L, row_stride);
}

void launch_sync_padding_gpu(int* d_lattice, int L, int row_stride, int block_size) {
    // Per il padding (che è 1D), usiamo un blocco lineare di block_size * block_size thread
    int threads = block_size * block_size; 
    int grid = (L + threads - 1) / threads;
    
    sync_padding_kernel<<<grid, threads>>>(d_lattice, L, row_stride);
}

void launch_metropolis_step(int* d_lattice, int L, int row_stride, float* d_lookup_probs, int color, void* d_states, int block_size) {
    curandState* states = (curandState*)d_states;

    dim3 block(block_size, block_size);
    dim3 grid((L + block.x - 1) / block.x, (L + block.y - 1) / block.y);
    
    // Launch kernel
    Ising_step_kernel<<<grid, block>>>(d_lattice, L, row_stride, d_lookup_probs, color, states);
}