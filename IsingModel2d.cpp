#include "IsingModel2d.h"

// constructor definition
IsingModel2d::IsingModel2d(int L, double T, unsigned int seed): L(L),T(T), beta(1.0/T), serial_rng(seed) {

    // initialize the lattice by pre-allocating the memory in the heap
    lattice.resize(L*L);
    // set the seed for all the 
    std::uniform_int_distribution<int> uni_dist(0,1);
    for (int i = 0; i<L*L; i++){
        // pass the engine in the uniform number generation object
        lattice[i] = 2 * uni_dist(serial_rng) - 1;
    }

    // initialize the seeds for each treas
    int max_threads = omp_get_max_threads();
    for (int i = 0; i < max_threads; i++){
        omp_rngs.emplace_back(seed + i + 1); // constructs the RNG engine directly inside the memory
    }
    
}