#include <iostream>
#include <iomanip>
#include "IsingModel2d.h"

int main() {
    // Parameters
    int L = 10;           // 100x100 lattice
    double T_high = 5.0;   // High temperature (Disordered)
    double T_low = 1.0;    // Low temperature (Ordered)
    double J = 1.0;
    double h = 0.0;
    unsigned int seed = 123;
    int steps = 1000;      // Number of Monte Carlo steps

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Testing Ising Model Serial Version ---" << std::endl;

    // 1. Test High Temperature (Should stay disordered)
    IsingModel2d model_high(L, T_high, J, h, seed);
    std::cout << "\n[T = 5.0] Initial Magnetization: " << model_high.magnetization() << std::endl;
    
    model_high.update(Mode::serial, steps);
    
    std::cout << "[T = 5.0] After " << steps << " steps: " << model_high.magnetization() << std::endl;
    std::cout << "   (Expected: Should stay close to 0)" << std::endl;

    // 2. Test Low Temperature (Should become ordered)
    IsingModel2d model_low(L, T_low, J, h, seed);
    std::cout << "\n[T = 1.0] Initial Magnetization: " << model_low.magnetization() << std::endl;
    
    model_low.update(Mode::serial, steps);
    
    std::cout << "[T = 1.0] After " << steps << " steps: " << model_low.magnetization() << std::endl;
    std::cout << "   (Expected: Should increase toward 1.0)" << std::endl;

    // 3. Test Energy Consistency
    std::cout << "\nFinal Energy (T=1.0): " << model_low.energy() << std::endl;

    return 0;
}