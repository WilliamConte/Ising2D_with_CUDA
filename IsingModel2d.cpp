#include "IsingModel2d.h"
#include "Ising_gpu_interface.h"

/* =====================================================================
 * OBJECTS LIFECYCLE MANAGEMENT
 * Handles initialization (Constructor) and cleanup (Destructor).
 * Sets up the lattice, RNGs, lookup tables, and allocates device memory.
 * ===================================================================== */

/****************** CONSTRUCTOR *********************/
IsingModel2d::IsingModel2d(int L, double T, double J, double h, unsigned int seed) : L(L), row_stride(L + 2), T(T), J(J), h(h), beta(1.0/T), serial_rng(seed) {
    // initialize lattice with total size including padding
    lattice.resize(row_stride * row_stride); 
    
    // random number generator for spins
    std::uniform_int_distribution<int> uni_dist(0, 1);
    
    // initialize core spins to +1 or -1
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            lattice[i * row_stride + j] = 2 * uni_dist(serial_rng) - 1;
        }
    }

    // initialize RNG engines for parallel threads in OPENMP
    int max_threads = omp_get_max_threads();
    for (int i = 0; i < max_threads; i++) {
        omp_rngs.emplace_back(seed + i + 1);
    }

    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        // assign to index 0 the values of spin -1 and
        // to index 1 the values of spin +1
        double s = (s_idx == 0) ? -1.0 : 1.0;
        
        // for a given spin value, store the probability of spin flip given the neighbors
        for (int i = 0; i < 5; ++i) {
            // Index mapping: index 0 is relative to value of neighbors -4 (i.e., all negative)
            int physical_sum = (i * 2) - 4; // possible sum of neighbors is -4,-2,0,2,4
            // compute delta E
            double delta_E = 2.0 * s * (J * physical_sum + h);
            lookup_probs[s_idx][i] = (delta_E > 0) ? std::exp(-delta_E * beta) : 1.0;
            }
    }

    // Cuda allocation:
    allocate_cuda();
    copy_to_device();
    upload_lookup_probs(); // Copy lookup tables
    unsigned int cuda_seed = seed + 123; // different cuda seed 
    // Initialize CUDA RNG states: computationally expensive, executed only once.
    launch_setup_rng(d_states, cuda_seed, L, row_stride);
}


/****************** DESTRUCTOR *********************/
IsingModel2d::~IsingModel2d() {
    // deallocate the memory used
    deallocate_cuda();
}

/* =====================================================================
 * BOUNDARY CONDITIONS
 * Manages ghost cells (padding) to enforce periodic boundary conditions.
 * ===================================================================== */

/***************** SYNC PADDING (ON HOST) **********/
void IsingModel2d::sync_padding() {
    for (int k = 1; k <= L; k++) {
        lattice[0 * row_stride + k] = lattice[L * row_stride + k];           // Top pad Row
        lattice[(L + 1) * row_stride + k] = lattice[1 * row_stride + k];     // Bottom pad Row
        lattice[k * row_stride + 0] = lattice[k * row_stride + L];           // Left pad Col
        lattice[k * row_stride + (L + 1)] = lattice[k * row_stride + 1];     // Right pad Col
    }
}

/* =====================================================================
 * PHYSICS STEP (HOST)
 * The single-spin update logic used by CPU solvers.
 * ===================================================================== */

/****************** METROPOLIS RULE *********************/

void IsingModel2d::Metropolis_update(int i, int j, std::mt19937& rng) {
    int array_index = i * row_stride + j;
    int neighbors = lattice[(i - 1) * row_stride + j] + 
                    lattice[(i + 1) * row_stride + j] + 
                    lattice[i * row_stride + (j - 1)] + 
                    lattice[i * row_stride + (j + 1)];

    int spin_idx = (lattice[array_index] == -1) ? 0 : 1;
    int sum_idx = (neighbors + 4) / 2;

    // We need a distribution instance locally or as a member
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (dist(rng) < lookup_probs[spin_idx][sum_idx]) {
        lattice[array_index] *= -1;
    }
}

/* =====================================================================
 * SOLVER IMPLEMENTATIONS
 * specific algorithms for evolving the system (Serial, OpenMP, CUDA).
 * ===================================================================== */

/****************** UPDATE STEP SERIAL *********************/

void IsingModel2d::step_serial() {
    //synchronize the padding 
    sync_padding();

    std::uniform_int_distribution<int> index_dist(1, L);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // perform L*L flip attempts (this is one step)
    for (int n = 0; n < L * L; n++) {
        int i = index_dist(serial_rng);
        int j = index_dist(serial_rng);
        
        Metropolis_update(i,j,serial_rng);
    }
}

/****************** UPDATE STEP OPENMP *********************/

void IsingModel2d::step_openmp(){
    sync_padding();
    
    // start parallel region
    #pragma omp parallel 
    {
        // assign threads indices
        int thread_index = omp_get_thread_num();
        std::mt19937& thread_rng = omp_rngs[thread_index]; 
        
        // update even indices
        #pragma omp for collapse(2)
        for(int i = 1; i <= L; i++){
            for(int j = 1; j <= L; j++){
                if ((i+j) % 2 == 0){
                    Metropolis_update(i, j, thread_rng);
                }
            }
        }
        
        // update the odd indices
        #pragma omp for collapse(2)
        for(int i = 1; i <= L; i++){
            for(int j = 1; j <= L; j++){
                if ((i+j) % 2 != 0){
                    Metropolis_update(i, j, thread_rng);
                }
            }
        }
    } // End of parallel region
}

/****************** UPDATE STEP CUDA *********************/

void IsingModel2d::step_cuda(){
    // synchronize the padding
    launch_sync_padding_gpu(d_lattice, L, row_stride);
    // update even indices
    launch_metropolis_step(d_lattice, L, row_stride, d_lookup_probs, 0, d_states, cuda_block_size);
    // update odd indices
    launch_metropolis_step(d_lattice, L, row_stride, d_lookup_probs, 1, d_states, cuda_block_size);
}

/* =====================================================================
 * MEMORY & DEVICE MANAGEMENT
 * Handles data transfer between Host (RAM) and Device (VRAM).
 * ===================================================================== */

/******************* CUDA MEMORY HANDLING ***************/

// allocate memory on device 
void IsingModel2d::allocate_cuda() {
    size_t lattice_bytes = row_stride * row_stride * sizeof(int);
    size_t states_bytes = row_stride * row_stride * 70; // excess CuRand state dimension estimation

    d_lattice = (int*)gpu_alloc(lattice_bytes);
    d_states = gpu_alloc(states_bytes);
    d_lookup_probs = (float*)gpu_alloc(10 * sizeof(float));
}

void IsingModel2d::copy_to_device() {
    gpu_memcpy_to_device(d_lattice, lattice.data(), lattice.size() * sizeof(int));
}

void IsingModel2d::copy_to_host() {
    gpu_memcpy_to_host(lattice.data(), d_lattice, lattice.size() * sizeof(int));
}

void IsingModel2d::deallocate_cuda() {
    gpu_free(d_lattice);
    gpu_free(d_states);
    gpu_free(d_lookup_probs);
}

void IsingModel2d::upload_lookup_probs(){
    float host_lookup[10];
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 5; j++){
            host_lookup[i*5 + j] = lookup_probs[i][j];
        }
    }

    gpu_memcpy_to_device(d_lookup_probs,&host_lookup,sizeof(float) * 10);
}

void IsingModel2d::device_synchronize(){
    launch_cuda_sync();
}

/* =====================================================================
 * OBSERVABLES & ANALYTICS
 * Functions to calculate physical properties (Magnetization, Energy).
 * Note: These usually run on Host after fetching data from Device.
 * ===================================================================== */

/*************** MAGNETIZATION ************************/
double IsingModel2d::magnetization(Mode mode) {
    double m = 0.0;
    
    // Serial version
    if (mode == Mode::serial) {
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                m += lattice[i * row_stride + j];
            }
        }
    }
    
    // Parallel OpenMP version
    else if (mode == Mode::parallel_cpu) {
        #pragma omp parallel for collapse(2) reduction(+:m)
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                m += lattice[i * row_stride + j];
            }
        }
    }

    // Parallel CUDA version

    else if (mode == Mode::parallel_CUDA){
        // note: we perform the reduction on the CPU to highlight the PCIe bottleneck.
        // transfer data from GPU (VRAM) to Host (RAM)
        // this copy_to_host() is the dominant cost for large lattices.
        copy_to_host();
        // compute magnetization on CPU
        return magnetization(Mode::serial);
    }
    
    return m / (double)(L * L);
}

/*************** ENERGY ************************/

double IsingModel2d::energy(Mode mode) {
    double E = 0.0;
    
    // Serial version
    if (mode == Mode::serial){
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                int array_index = i * row_stride + j;
                int neighbors = lattice[(i - 1) * row_stride + j] + //down
                                lattice[(i + 1) * row_stride + j] + //up
                                lattice[i * row_stride + (j - 1)] + //left
                                lattice[i * row_stride + (j + 1)]; //right
                
                // Interaction energy (halved for double counting) + field energy
                E += -0.5 * J * lattice[array_index] * neighbors - h * lattice[array_index];
            }
        }
    }
    
    // Parallel OpenMP version
    else if (mode == Mode::parallel_cpu){
        #pragma omp parallel for collapse(2) reduction(+:E)
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                int array_index = i * row_stride + j;
                int neighbors = lattice[(i - 1) * row_stride + j] + 
                                lattice[(i + 1) * row_stride + j] + 
                                lattice[i * row_stride + (j - 1)] + 
                                lattice[i * row_stride + (j + 1)];
                
                // Interaction energy (halved for double counting) + field energy
                E += -0.5 * J * lattice[array_index] * neighbors - h * lattice[array_index];
            }
        }
    }

    // Parallel CUDA version
    else if (mode == Mode::parallel_CUDA){
        // note: we perform the reduction on the CPU to highlight the PCIe bottleneck.
        // transfer data from GPU (VRAM) to Host (RAM)
        // this copy_to_host() is the dominant cost for large lattices.
        copy_to_host(); 
        
        // delegate the calculation to the CPU implementation
        return energy(Mode::serial); 
    }

    return E;
}

/* =====================================================================
 * SIMULATION DRIVER
 * The high-level interface to execute steps using the chosen backend.
 * ===================================================================== */

/******* UPDATE STEP ********/

void IsingModel2d::update(Mode mode, int steps) {
    switch(mode) {
        
        /****** SERIAL CPU EXECUTION ******/
        case Mode::serial:
            for (int i = 0; i < steps; i++) {
                // The serial step already includes sync_padding() internally
                step_serial();
            }
            break;

        /****** PARALLEL OPENMP EXECUTION *****/
        
        case Mode::parallel_cpu:
            for (int i = 0; i < steps; i++) {
                // The OpenMP step already includes sync_padding() internally
                step_openmp(); 
            }
            break;
        
        /****** PARALLEL CUDA EXECUTION *****/
        case Mode::parallel_CUDA:
            
            // d_lattice must be already allocated and populated on the GPU.
            // if this is the very first run, ensure copy_to_device() was called before.

            for (int i = 0; i < steps; i++) {
                // perform one full Monte Carlo step (Padding + Red + Black)
                step_cuda(); 
            }
            
            // wrapper for the threads synchronization
            launch_cuda_sync();
            
            // note: We do NOT copy data back to Host here. 
            
            break;
    }
}