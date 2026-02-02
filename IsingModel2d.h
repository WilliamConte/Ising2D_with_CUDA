// preprocessor instructions to avoid multiple definitions
#ifndef ISINGMODEL2D_HPP
#define ISINGMODEL2D_HPP

#include <vector>
#include <random>
#include <omp.h>

// enum class to identify the type of algorithm implemented (done for readability)
enum class Mode {serial, parallel_cpu, parallel_CUDA};

class IsingModel2d{
    public:
        
        // constructor  
        IsingModel2d(int L, double T, unsigned int seed);

        void update(Mode mode, int steps);

        double energy();

        double magnetization();

        // setter function to change the block size once the object is already created
        void cuda_grid_size(int size){ cuda_block_size = size; };

        /// Pybind functions ///
        // helper functions for Pybind11: in this way Python will read directly from the RAM without copying 
        // the lattice's data (expensive for large lattices)
        int* get_data_ptr() { return lattice.data(); } // returns a pointer to where the lattice is stored in the RAM
        std::vector<size_t> get_shape() { return {(size_t)L, (size_t)L}; } // treat it as a 2d object
        std::vector<size_t> get_strides() { return { (size_t)L * sizeof(int), sizeof(int) }; } // each element is an int
        // functions that allow to change thread counts from Python for plots
        void set_num_threads(int n) { omp_set_num_threads(n); }
        void set_cuda_block_size(int s) { cuda_block_size = s; }
    
    private:
        
        int L;
        double T;
        double beta;
        int cuda_block_size = 16;

        // lattice is already flattened in 1D vector for simplicity on next phases 
        std::vector<int> lattice;

        // random Number Engines
        std::mt19937 serial_rng; 
        std::vector<std::mt19937> omp_rngs; // one number per thread to avoid race conditions

        // internal logic
        void step_serial();
        void step_openmp();
        void step_cuda();
};

#endif