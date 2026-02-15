#include "IsingModel2d.h"
#include "Ising_gpu_interface.h"

/* =====================================================================
 * OBJECTS LIFECYCLE MANAGEMENT
 * Handles initialization (Constructor) and cleanup (Destructor).
 * Sets up the lattice, RNGs, lookup tables, and allocates device memory.
 * ===================================================================== */

/****************** CONSTRUCTOR *********************/
IsingModel2d::IsingModel2d(int L, double T, double J, double h, unsigned int seed) : L(L), row_stride(L + 2), T(T), beta(1.0/T), J(J), h(h), serial_rng(seed),
m_seed(seed) {
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

    // Compute lookup probabilities to determine spin flips during Metropolis updates.
    // In this way we compute only once and we do only "read" operation in the following,
    // which are more efficient than computing an exponential function at each step
    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        // assign to index 0 the values of spin -1 and
        // to index 1 the values of spin +1
        double s = (s_idx == 0) ? -1.0 : 1.0;
        
        // for a given spin value, store the probability of spin flip given the neighbors
        for (int i = 0; i < 5; ++i) {
            // Index mapping: index 0 is relative to value of neighbors -4 (i.e., all negative)
            int physical_sum = (i * 2) - 4; // possible sum of neighbors is -4,-2,0,2,4
            // Globally we have this index mapping: (sum_neighbors --> index_lookup_table)
            // -4 --> 0 | -2 --> 1 | 0 --> 2 | 2 --> 3 | 4 --> 3
            
            // compute delta E
            double delta_E = 2.0 * s * (J * physical_sum + h);

            // assign to each position in the lookup prob the correspondent acceptance probability
            lookup_probs[s_idx][i] = (delta_E > 0) ? std::exp(-delta_E * beta) : 1.0;
            }
    }

    // Cuda allocation:
    allocate_cuda();
    copy_to_device();
    upload_lookup_probs(); // Copy lookup tables
    unsigned int cuda_seed = seed + 123; 
    // Initialize cud rng states: computationally expensive, executed only once.
    launch_setup_rng(d_states, cuda_seed, L, row_stride);
}


/****************** DESTRUCTOR *********************/
IsingModel2d::~IsingModel2d() {
    // deallocate the memory used
    deallocate_cuda();
}

/******** CHANGE TEMPERATURE OF THE OBJECT *******/
void IsingModel2d::set_T(double T_new){
    if (this->T == T_new) return;
    else {
        
        // update the new attributes
        this->T = T_new;
        this->beta = 1.0/T_new;

        // recompute the lookup probabilities
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

    // Allocate also on device:
    upload_lookup_probs();
    }
}

/* =====================================================================
 * BOUNDARY CONDITIONS
 * Originally we decided to use a padding to ensure boundary conditions both on CPU
 * and GPU. In particular, on CPU we used the function ''sync_padding''. This functions
 * manages ghost cells (padding) to enforce periodic boundary conditions, in particular:
 * - Top row is copied to the Bottom padding layer
 * - Bottom row is copied to the Top padding layer
 * - Left column is copied to the Right padding layer
 * - Right column is copied to the Left padding layer
 * But in the end this strategy was not used on CPU, we used a simpler ternary operator,
 * which is more efficient in this context. We maintain this function just for completeness.
 * On GPU, instead, we used that strategy (The detailed implementations are in "ising_CUDA_kernels.cu")
 * ===================================================================== */

/***************** SYNC PADDING (ON HOST) **********/
/*Note: in the final implementation this function is not used in the CPU, since read/copy 
from the RAM is less efficient than if-else operations.*/

void IsingModel2d::sync_padding() {
    for (int k = 1; k <= L; k++) {
        lattice[0 * row_stride + k] = lattice[L * row_stride + k];           // Top pad Row
        lattice[(L + 1) * row_stride + k] = lattice[1 * row_stride + k];     // Bottom pad Row
        lattice[k * row_stride + 0] = lattice[k * row_stride + L];           // Left pad Col
        lattice[k * row_stride + (L + 1)] = lattice[k * row_stride + 1];     // Right pad Col
    }
}


/* =====================================================================
 * METROPOLIS UPDATE (Optimized for CPU: Serial & OpenMP)
 * * This function attempts to flip a single spin at coordinates (i, j).
 * * Difference from GPU version:
 * it does not rely on "Ghost Cells" (padding) to find neighbors.
 * Instead, it uses branching to handle 
 * Periodic Boundary Conditions (PBC) on the fly.
 * This allows us not to use the expensive ''sync_padding()'' calls 
 * from the Serial and OpenMP loops, as we read directly from the 
 * real lattice data.
 * ===================================================================== */

void IsingModel2d::Metropolis_update(int i, int j, std::mt19937& rng) {
    
    // Identify the 1D index of the spin we are trying to flip
    int current_idx = i * row_stride + j;

    // Identify Neighbor Indices using Periodic Boundary Conditions (PBC).
    // check if we are on a boundary (1 or L) and "wrap around" explicitly.
    
    // if i is 1 (top row), the neighbor above is L (bottom row). Else i-1.
    int i_up    = (i == 1) ? L : i - 1;   
    // if i is L (bottom row), the neighbor below is 1 (top row). Else i+1.
    int i_down  = (i == L) ? 1 : i + 1;   
    // if j is 1 (left col), the neighbor left is L (right col). Else j-1.
    int j_left  = (j == 1) ? L : j - 1;   
    // if j is L (right col), the neighbor right is 1 (left col). Else j+1.
    int j_right = (j == L) ? 1 : j + 1;   
    // Sum the neighbors by reading directly from the real lattice locations.
    int neighbors = lattice[i_up   * row_stride + j] + 
                    lattice[i_down * row_stride + j] + 
                    lattice[i * row_stride + j_left] + 
                    lattice[i * row_stride + j_right];

    // Calculate Energy Change (Delta E) via Lookup Table
    // map the current spin (-1 or 1) to a table index (0 or 1)
    int spin_val = lattice[current_idx];
    int spin_idx = (spin_val < 0) ? 0 : 1; 
    
    // map the neighbor sum (-4, -2, 0, 2, 4) to a table index (0, 1, 2, 3, 4)
    int sum_idx  = (neighbors + 4) / 2;

    // use a static distribution to avoid the overhead of reconstructing 
    // the object at every single function call
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    // check the probability against the pre-computed lookup table.
    if (dist(rng) < lookup_probs[spin_idx][sum_idx]) {
        lattice[current_idx] *= -1; // Flip the spin
    }
}

/* =====================================================================
 * SOLVER IMPLEMENTATIONS
 * specific algorithms for evolving the system (Serial, OpenMP, CUDA).
 * ===================================================================== */

/****************** UPDATE STEP SERIAL *********************/

void IsingModel2d::step_serial() {
    //synchronize the padding 
    //sync_padding(); not useful on CPU

    static std::uniform_int_distribution<int> index_dist(1, L);
    static std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // perform L*L flip attempts (this is one step)
    for (int n = 0; n < L * L; n++) {
        int i = index_dist(serial_rng);
        int j = index_dist(serial_rng);
        
        Metropolis_update(i,j,serial_rng);
    }
}

/****************** UPDATE STEP OPENMP *********************/

void IsingModel2d::step_openmp(int steps) {
    
    //sync_padding();
    
    // Threads are created here and stay alive for all simulation
    #pragma omp parallel 
    {
        // each thread gets its own unique ID and private RNG state
        int thread_id = omp_get_thread_num();
        std::mt19937& thread_rng = omp_rngs[thread_id]; 

        // time loop
        for (int s = 0; s < steps; s++) {
            
            // even spins
            #pragma omp for collapse(2) 
            for(int i = 1; i <= L; i++){
                for(int j = 1; j <= L; j++){
                    // Checkerboard condition for even sites
                    if ((i + j) % 2 == 0) {
                        Metropolis_update(i, j, thread_rng);
                    }
                }
            }
            
            // odd spins
            #pragma omp for collapse(2)
            for(int i = 1; i <= L; i++){
                for(int j = 1; j <= L; j++){
                    // Checkerboard condition for odd sites
                    if ((i + j) % 2 != 0) {
                        Metropolis_update(i, j, thread_rng);
                    }
                }
            }
                    
        } // end of time loop
    } // end of parallel region (threads are destroyed here)
}

/****************** UPDATE STEP CUDA *********************/

void IsingModel2d::step_cuda_global(){
    // synchronize the padding
    launch_sync_padding_gpu(d_lattice, L, row_stride);
    // update even indices
    launch_ising_global(d_lattice, L, row_stride, d_lookup_probs, 0, d_states, cuda_block_size);

    launch_sync_padding_gpu(d_lattice, L, row_stride);
    // update odd indices
    launch_ising_global(d_lattice, L, row_stride, d_lookup_probs, 1, d_states, cuda_block_size);
}

void IsingModel2d::step_cuda_shared(){
    // synchronize the padding
    launch_sync_padding_gpu(d_lattice, L, row_stride);
    // update even indices
    launch_ising_shared(d_lattice, L, row_stride, d_lookup_probs, 0, d_states, cuda_block_size);

    launch_sync_padding_gpu(d_lattice, L, row_stride);
    // update odd indices
    launch_ising_shared(d_lattice, L, row_stride, d_lookup_probs, 1, d_states, cuda_block_size);
}

/* =====================================================================
 * MEMORY & DEVICE MANAGEMENT
 * Handles data transfer between Host (RAM) and Device (VRAM).
 * ===================================================================== */

/******************* CUDA MEMORY HANDLING ***************/

// allocate memory on device 
void IsingModel2d::allocate_cuda() {
    size_t lattice_bytes = row_stride * row_stride * sizeof(int);
    size_t states_bytes = row_stride * row_stride * 70; // conservative CuRand state dimension estimation

    // allocate lattice, curandstates and lookup table
    d_lattice = (int*)gpu_alloc(lattice_bytes);
    d_states = gpu_alloc(states_bytes);
    d_lookup_probs = (float*)gpu_alloc(10 * sizeof(float));
}

void IsingModel2d::copy_to_device() {
    // it copies the lattice from host to device
    gpu_memcpy_to_device(d_lattice, lattice.data(), lattice.size() * sizeof(int));
}

void IsingModel2d::copy_to_host() {
    // it copies the lattice from device to host
    gpu_memcpy_to_host(lattice.data(), d_lattice, lattice.size() * sizeof(int));
}

void IsingModel2d::deallocate_cuda() {
    // free the memory 
    gpu_free(d_lattice);
    gpu_free(d_states);
    gpu_free(d_lookup_probs);
}

void IsingModel2d::upload_lookup_probs(){
    // upload to device the lookup probability table
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

/*OpenMP function to get the number of threads*/
int IsingModel2d::get_openmp_threads(){
    return omp_get_max_threads();
}

void IsingModel2d::set_num_threads(int n) { 
    // change number of threads used in openMP
    // (useful for benchmarking)
    omp_set_num_threads(n); 
    
    if (n > (int)omp_rngs.size()) {
        int current_size = omp_rngs.size();
        for (int i = current_size; i < n; i++) {
            omp_rngs.emplace_back(m_seed + i + 1);
        }
    }
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
    else if (mode == Mode::openMP) {
        #pragma omp parallel for collapse(2) reduction(+:m)
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                m += lattice[i * row_stride + j];
            }
        }
    }

    // Parallel CUDA version

    else if (mode == Mode::cuda_global || mode == Mode::cuda_shared){
        // this ''copy_to_host()'' is the dominant cost for large lattices
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
                
                // boundary conditions with ternary operator
                int i_up    = (i == 1) ? L : i - 1;
                int i_down  = (i == L) ? 1 : i + 1;
                int j_left  = (j == 1) ? L : j - 1;
                int j_right = (j == L) ? 1 : j + 1;
                
                int neighbors = lattice[i_up   * row_stride + j] + // up
                        lattice[i_down * row_stride + j] + //down
                        lattice[i * row_stride + j_left] + // left
                        lattice[i * row_stride + j_right]; // right
                
                // Interaction energy (halved for double counting) + field energy
                E += -0.5 * J * lattice[array_index] * neighbors - h * lattice[array_index];
            }
        }
    }
    
    // Parallel OpenMP version
    else if (mode == Mode::openMP){
        #pragma omp parallel for collapse(2) reduction(+:E)
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                int array_index = i * row_stride + j;
                
                // boundary conditions with ternary operator
                int i_up    = (i == 1) ? L : i - 1;
                int i_down  = (i == L) ? 1 : i + 1;
                int j_left  = (j == 1) ? L : j - 1;
                int j_right = (j == L) ? 1 : j + 1;
                
                int neighbors = lattice[i_up   * row_stride + j] + // up
                        lattice[i_down * row_stride + j] + //down
                        lattice[i * row_stride + j_left] + // left
                        lattice[i * row_stride + j_right]; // right
                        
                // Interaction energy (halved for double counting) + field energy
                E += -0.5 * J * lattice[array_index] * neighbors - h * lattice[array_index];
            }
        }
    }

    // Parallel CUDA version
    else if (mode == Mode::cuda_global || mode == Mode::cuda_shared){
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
                step_serial();
            }
            break;

        /****** PARALLEL OPENMP EXECUTION *****/
        
        case Mode::openMP:
            step_openmp(steps); 
            break;
        
        /****** PARALLEL CUDA EXECUTION *****/
        case Mode::cuda_global:
            
            // d_lattice must be already allocated and populated on the GPU.
            // if this is the very first run, ensure copy_to_device() was called before.

            for (int i = 0; i < steps; i++) {
                // perform one full Monte Carlo step (Padding + Red + Black)
                step_cuda_global(); 
            }
            
            // wrapper for the blocks synchronization
            launch_cuda_sync();
            
            // note: We do NOT copy data back to Host here.

            break;
        
        case Mode::cuda_shared:
            
            // d_lattice must be already allocated and populated on the GPU.
            // if this is the very first run, ensure copy_to_device() was called before.

            for (int i = 0; i < steps; i++) {
                // perform one full Monte Carlo step (Padding + Red + Black)
                step_cuda_shared(); 
            }
            
            // wrapper for the blocks synchronization
            launch_cuda_sync();
            
            // note: We do NOT copy data back to Host here. 
            
            break;
    }
}