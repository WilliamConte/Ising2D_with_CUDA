#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdio>
#include "Ising_gpu_interface.h"

// ==========================================
// MEMORY HANDLING
// ==========================================

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

// ==========================================
// HELPER KERNELS (RNG & PADDING)
// ==========================================

/******************** Setup for RNGs **********************/
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
        // Copy top/bottom rows
        lattice[0 * row_stride + k] = lattice[L * row_stride + k];
        lattice[(L + 1) * row_stride + k] = lattice[1 * row_stride + k];
        // Copy left/right columns
        lattice[k * row_stride + 0] = lattice[k * row_stride + L];
        lattice[k * row_stride + (L + 1)] = lattice[k * row_stride + 1];
    }
}

// ==========================================
// ISING KERNELS
// ==========================================

/**
 * VERSION 1: GLOBAL MEMORY (NAIVE)
 * Each thread reads its neighbors directly from Global Memory (VRAM).
 * This version is used as a baseline for performance benchmarking.
 */
__global__ void ising_step_global(int* lattice, int L, int row_stride, float* lookup_probs, int color, curandState* states) {
    // Calculate global coordinates in the lattice (accounting for +1 padding)
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Boundary check: ensure the thread is within the LxL lattice
    if (thread_x <= L && thread_y <= L) {
        
        // Checkerboard (Red-Black) update logic
        // Only threads matching the current 'color' update during this pass
        if ((thread_x + thread_y) % 2 == color) {
            int global_index = thread_y * row_stride + thread_x;
            int current_spin = lattice[global_index];

            /**
             * Direct Global Memory Access:
             * Every neighbor read travels through the L2 cache/VRAM bus.
             */
            int sum_neighbors = lattice[(thread_y - 1) * row_stride + thread_x] + // Top neighbor
                                lattice[(thread_y + 1) * row_stride + thread_x] + // Bottom neighbor
                                lattice[thread_y * row_stride + (thread_x - 1)] + // Left neighbor
                                lattice[thread_y * row_stride + (thread_x + 1)]; // Right neighbor

            // Map current state and neighbor sum to the 1D lookup table index
            int row = (current_spin == -1) ? 0 : 1;
            int col = (sum_neighbors + 4) / 2;

            // Generate a random float using the pre-initialized CURAND state
            float rand_val = curand_uniform(&states[global_index]);
            
            // Metropolis Acceptance Criterion
            if (rand_val < lookup_probs[row * 5 + col]) {
                lattice[global_index] = -current_spin;
            }
        }
    }
}


/**
 * VERSION 2: SHARED MEMORY + L2 CACHE
 * * This implementation uses a hybrid memory strategy:
 * 1. Shared Memory (L1): Used for high-speed access to spins within the block.
 * 2. L2 Cache (__ldg): Used to fetch 'padding' (boundary) spins from neighboring 
 * blocks via the Read-Only Data Cache, reducing Global Memory latency.
 * * Templates are used to define BLOCK_SIZE at compile-time, allowing the compiler 
 * to optimize register allocation and static shared memory partitioning.
 */

template <int BLOCK_SIZE>
__global__ void ising_step_shared(int* lattice, int L, int row_stride, float* lookup_probs, int color, curandState* states) {
    
    // local coordinates
    int thread_x = threadIdx.x; 
    int thread_y = threadIdx.y;

    // global coordinates
    int x = (blockIdx.x * blockDim.x) + thread_x + 1;
    int y = (blockIdx.y * blockDim.y) + thread_y + 1;

    int global_index = y * row_stride + x;

    // shared memory
    __shared__ int my_cache[BLOCK_SIZE][BLOCK_SIZE];

    // load from Global to Shared
    if (x <= L && y <= L) {
        my_cache[thread_y][thread_x] = lattice[global_index];
    }
    
    // wait for all threads to load their spin
    __syncthreads();

    // UPDATE LOGIC
    if (x <= L && y <= L) {
        if ((x + y) % 2 == color) {
            
            int current_spin = my_cache[thread_y][thread_x]; 
            int sum_n = 0;

            /* NEIGHBOR ACCESS
             * -) INNER (Shared Memory): 
             * Serves the reads from L1. Zero latency, no VRAM traffic.
             *
             * -) PADDING (__ldg): 
             * Serves boundaries via the Read-Only Cache (Texture Unit).
             * - Bypasses L1 coherency protocols for static data.
             * - Hits the L2 cache efficiently (high spatial locality).
             * - Prevents "warp divergence" penalties of complex loading schemes. */

            // left Neighbor
            if (thread_x > 0) {
                sum_n += my_cache[thread_y][thread_x - 1];
            }
            else{
                sum_n += __ldg(&lattice[global_index - 1]);
            }

            // right Neighbor
            if (thread_x < BLOCK_SIZE - 1){
                sum_n += my_cache[thread_y][thread_x + 1];
            } 
            else {                   
                sum_n += __ldg(&lattice[global_index + 1]);
            }

            // up Neighbor
            if (thread_y > 0){
                sum_n += my_cache[thread_y - 1][thread_x];
            } 
            else{
                sum_n += __ldg(&lattice[global_index - row_stride]);
            }        
            
            // down Neighbor
            if (thread_y < BLOCK_SIZE - 1){
                sum_n += my_cache[thread_y + 1][thread_x];
            } 
            else {
                sum_n += __ldg(&lattice[global_index + row_stride]);
            }                   
            
            // Metropolis update
            int lookup_index_0 = (current_spin == -1) ? 0 : 1;
            int lookup_index_1 = (sum_n + 4) / 2;

            float rand_val = curand_uniform(&states[global_index]);

            if (rand_val < lookup_probs[lookup_index_0 * 5 + lookup_index_1]) {
                lattice[global_index] = -current_spin;
            }
        }
    }
}


// ==========================================
// WRAPPERS (KERNEL LAUNCH)
// ==========================================

void launch_setup_rng(void* d_states, unsigned int seed, int L, int row_stride) {
    curandState* states = (curandState*)d_states;

    dim3 block(16, 16);
    dim3 grid((L + block.x - 1) / block.x, (L + block.y - 1) / block.y);

    setup_rand_kernel<<<grid, block>>>(states, seed, L, row_stride);

    cudaDeviceSynchronize();
}

void launch_sync_padding_gpu(int* d_lattice, int L, int row_stride) {
    // use fixed number of threads for padding.
    int threads = 256; 
    int grid = (L + threads - 1) / threads;
    
    sync_padding_kernel<<<grid, threads>>>(d_lattice, L, row_stride);
}



void launch_ising_shared(int* d_lattice, int L, int row_stride, float* d_lookup_probs, int color, void* d_states, int block_size) {
    // cast to the correct type
    curandState* states = (curandState*)d_states;

    dim3 block(block_size, block_size);
    dim3 grid((L + block.x - 1) / block.x, (L + block.y - 1) / block.y);
    
    // Choose the correct template based on runtime block_size
    switch(block_size) {
        case 8:
            ising_step_shared<8><<<grid, block>>>(d_lattice, L, row_stride, d_lookup_probs, color, states);
            break;
        case 16:
            ising_step_shared<16><<<grid, block>>>(d_lattice, L, row_stride, d_lookup_probs, color, states);
            break;
        case 32:
            ising_step_shared<32><<<grid, block>>>(d_lattice, L, row_stride, d_lookup_probs, color, states);
            break;
        default:
            std::cerr << "ERROR: Block size " << block_size << " not supported! (Use 8, 16, or 32)" << std::endl;
            
    }
}

void launch_ising_global(int* d_lattice, int L, int row_stride, float* d_lookup_probs, int color, void* d_states, int block_size){
    curandState* states = (curandState*) d_states;
    if (block_size > 32){
        std::cout << "It is not possible to have a block size of "<< block_size <<" --> casted to 32" << std::endl;
        block_size = 32;
    }

    dim3 block(block_size,block_size);
    dim3 grid((L + block.x - 1)/block.x, (L + block.y - 1)/block.y);

    ising_step_global<<<grid,block>>>(d_lattice, L, row_stride, d_lookup_probs, color, states);
}

void launch_cuda_sync(){
    cudaDeviceSynchronize();
}