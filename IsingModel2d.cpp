#include "IsingModel2d.h"

// constructor definition
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

    // initialize RNG engines for parallel threads
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
        double dE = 2.0 * s * (J * physical_sum + h);
        lookup_probs[s_idx][i] = (dE > 0) ? std::exp(-dE * beta) : 1.0;
        }
    }
}

void IsingModel2d::sync_padding() {
    // 
    for (int k = 1; k <= L; k++) {
        lattice[0 * row_stride + k] = lattice[L * row_stride + k];           // Top pad Row
        lattice[(L + 1) * row_stride + k] = lattice[1 * row_stride + k];     // Bottom pad Row
        lattice[k * row_stride + 0] = lattice[k * row_stride + L];           // Left pad Col
        lattice[k * row_stride + (L + 1)] = lattice[k * row_stride + 1];     // Right pad Col
    }
}

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

void IsingModel2d::step_openmp(){
    sync_padding();
    // chessboard update: update only "alternating" spins to avoid race conditions
    
    
    // update even indices
    for(int i = 1; i <= L; i++){
        for(int j = 1; j <= L; j++){
            if ((i+j) % 2 == 0){
                Metropolis_update(i,j/*random number generator*/);
            }
        }
    }

    //update odd indices
    for(int i = 1; i <= L; i++){
        for(int j = 1; j <= L; j++){
            if ((i+j) % 2 != 0){
                Metropolis_update(i,j/*random number generator*/);
            }
        }
    }
}



double IsingModel2d::magnetization(Mode mode) {
    double m = 0.0;
    
    // 1. Serial Version
    if (mode == Mode::serial) {
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                m += lattice[i * row_stride + j];
            }
        }
    }
    
    // 2. Parallel OpenMP Version
    else if (mode == Mode::parallel_cpu) {
        #pragma omp parallel for collapse(2) reduction(+:m)
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                m += lattice[i * row_stride + j];
            }
        }
    }

    
    // Return the absolute average magnetization
    return m / (double)(L * L);
}


double IsingModel2d::energy(Mode mode) {
    double E = 0.0;
    if (mode == Mode::serial){
        for (int i = 1; i <= L; i++) {
            for (int j = 1; j <= L; j++) {
                int array_index = i * row_stride + j;
                int neighbors = lattice[(i - 1) * row_stride + j] + lattice[(i + 1) * row_stride + j] + lattice[i * row_stride + (j - 1)] + lattice[i * row_stride + (j + 1)];
                // Interaction energy (halved for double counting) + field energy
                E += -0.5 * J * lattice[array_index] * neighbors - h * lattice[array_index];
            }
        }
    }
    //parallel openMP
    /************/

    //parallel CUDA
    /************/

    
    return E;
}

void IsingModel2d::update(Mode mode, int steps) {
    switch(mode) {
        case Mode::serial:
            for (int i = 0; i < steps; i++) {
                sync_padding();
                step_serial();
            }
            break;
        /*case Mode::parallel_cpu:
            for (int i = 0; i < steps; i++) {
                step_openmp(); 
            }
            break;
            */
    }
}