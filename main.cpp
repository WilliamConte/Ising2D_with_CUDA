#include <vector>
#include <random>
#include <omp.h>
#include <iostream>

int main(){

    int L = 5;
    // lattice is already flattened in 1D vector for simplicity on next phases 
    std::vector<int> lattice;
    lattice.resize(L*L);
    // random Number Engines
    std::random_device rd;
    std::mt19937 serial_rng(rd());
    
    std::uniform_int_distribution<int> dist(0,1);
    for (int i = 0; i<L*L; i++){
        // pass the engine in the uniform number generation object
        lattice[i] = 2 * dist(serial_rng) - 1;
    }
    
    for (auto a : lattice){std::cout << a;}

    return 0;
}


