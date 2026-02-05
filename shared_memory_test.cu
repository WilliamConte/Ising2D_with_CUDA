#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>

/* * TEMPLATE PARAMETER <int BLOCK_SIZE>:
 * This converts the block size from a runtime variable to a compile-time constant.
 * This is necessary to allocate static Shared Memory (__shared__ int my_cache[N][N])
 * and allows the compiler to unroll loops and optimize register usage.
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

    // SHARED MEMORY
    // Now valid because BLOCK_SIZE is a constant known at compile time
    __shared__ int my_cache[BLOCK_SIZE][BLOCK_SIZE];

    // LOADING
    if (x <= L && y <= L) {
        my_cache[thread_y][thread_x] = lattice[global_index];
    }
    
    __syncthreads();

    // UPDATE LOGIC
    if (x <= L && y <= L) {
        if ((x + y) % 2 == color) {
            
            int current_spin = my_cache[thread_y][thread_x]; 
            int sum_n = 0;

            /* HYBRID MEMORY STRATEGY
             * -) INNER: read from shared memory.
             * It accesses local data instantly (L1 Cache).
             * -) BORDERS: Read from Global Memory using ''__ldg()''.
             * It uses a special Read-Only Cache (Texture Unit) which is 
             * much faster than standard global reads for boundary data. */

            // LEFT
            if (thread_x > 0) {
                sum_n += my_cache[thread_y][thread_x - 1];
            }
            else{
                sum_n += __ldg(&lattice[global_index - 1]);
            }

            // RIGHT
            if (thread_x < BLOCK_SIZE - 1){
                sum_n += my_cache[thread_y][thread_x + 1];
            } 
            else {                   
                sum_n += __ldg(&lattice[global_index + 1]);
            }

            // UP
            if (thread_y > 0){
                sum_n += my_cache[thread_y - 1][thread_x];
            } 
            else{
                sum_n += __ldg(&lattice[global_index - row_stride]);
            }        
            
            // DOWN
            if (thread_y < BLOCK_SIZE - 1){
                sum_n += my_cache[thread_y + 1][thread_x];
            } 
            else {
                sum_n += __ldg(&lattice[global_index + row_stride]);
            }                   
            
            // METROPOLIS
            int lookup_index_0 = (current_spin == -1) ? 0 : 1;
            int lookup_index_1 = (sum_n + 4) / 2;

            float rand_val = curand_uniform(&states[global_index]);

            if (rand_val < lookup_probs[lookup_index_0 * 5 + lookup_index_1]) {
                lattice[global_index] = -current_spin;
            }
        }
    }
}

// --- HELPER KERNEL TO INIT RNG ---
__global__ void setup_rng(curandState* states, unsigned long seed, int L, int row_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int idx = y * row_stride + x;
    if (x <= L && y <= L) curand_init(seed, idx, 0, &states[idx]);
}

/* This function solves the "Runtime vs Compile-time" problem.
 * It takes the runtime variable 'bs' and uses a switch statement 
 * to launch the specific template version of the kernel.
 * This allows to benchmark different block sizes easily 
 * without recompiling the code every time.*/
void run_ising_step(int bs, dim3 grid, dim3 block, int* d_lat, int L, int stride, float* d_probs, int color, curandState* d_states) {
    switch (bs) {
        case 8:
            ising_step_shared<8><<<grid, block>>>(d_lat, L, stride, d_probs, color, d_states);
            break;
        case 16:
            ising_step_shared<16><<<grid, block>>>(d_lat, L, stride, d_probs, color, d_states);
            break;
        case 32:
            ising_step_shared<32><<<grid, block>>>(d_lat, L, stride, d_probs, color, d_states);
            break;
        default:
            std::cerr << "ERROR: Block Size " << bs << " not supported in switch statement." << std::endl;
            exit(1);
    }
}

// --- MAIN TO TEST IT ---
int main() {
    int L = 32; // Small size for testing
    int row_stride = L + 2; // Padding
    int size = row_stride * row_stride;
    float T = 2.27; // Critical Temp

    // PARAMETER TO BENCHMARK (Variable)
    int block_size = 16; 

    std::vector<int> h_lattice(size, 1); // Start all +1 (Cold start)
    // Add padding zeros
    for(int i=0; i<size; i++) {
        int r = i / row_stride;
        int c = i % row_stride;
        if (r == 0 || r == L+1 || c == 0 || c == L+1) h_lattice[i] = 0;
    }

    // 2. Lookup Table Calculation (CPU)
    float h_lookup[10];
    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        int spin = (s_idx == 0) ? -1 : 1;
        for (int n_idx = 0; n_idx < 5; ++n_idx) {
            int neighbor_sum = n_idx * 2 - 4;
            double dE = 2.0 * spin * neighbor_sum;
            h_lookup[s_idx * 5 + n_idx] = (dE <= 0) ? 1.0f : exp(-dE / T);
        }
    }

    // 3. Device Allocation
    int* d_lattice;
    float* d_lookup;
    curandState* d_states;

    cudaMalloc(&d_lattice, size * sizeof(int));
    cudaMalloc(&d_lookup, 10 * sizeof(float));
    cudaMalloc(&d_states, size * sizeof(curandState));

    // 4. Copy to Device
    cudaMemcpy(d_lattice, h_lattice.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lookup, h_lookup, 10 * sizeof(float), cudaMemcpyHostToDevice);

    // 5. Setup Grid
    dim3 block(block_size, block_size);
    dim3 grid((L + block.x - 1) / block.x, (L + block.y - 1) / block.y);

    std::cout << "Setting up RNG..." << std::endl;
    setup_rng<<<grid, block>>>(d_states, 1234, L, row_stride);
    cudaDeviceSynchronize();

    // 6. RUN THE KERNEL (Using the Wrapper)
    std::cout << "Running Kernel with Block Size: " << block_size << std::endl;
    
    // Call the wrapper instead of the kernel directly
    run_ising_step(block_size, grid, block, d_lattice, L, row_stride, d_lookup, 0, d_states); // Red
    run_ising_step(block_size, grid, block, d_lattice, L, row_stride, d_lookup, 1, d_states); // Black
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();

    // 7. Copy Back and Print Sample
    cudaMemcpy(h_lattice.data(), d_lattice, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Success! Top-left corner spins:" << std::endl;
    for(int i=1; i<=8; i++) {
        for(int j=1; j<=8; j++) {
            std::cout << (h_lattice[i*row_stride + j] > 0 ? "+" : "-") << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_lattice);
    cudaFree(d_lookup);
    cudaFree(d_states);

    return 0;
}