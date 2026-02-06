#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include "IsingModel2d.h" 

int main(int argc, char* argv[]) {
    // Ci aspettiamo: ./ising_sim <L> <Steps> <Temp> <Mode>
    if (argc < 5) {
        std::cerr << "Usage: ./ising_sim <L> <Steps> <Temp> <serial|omp|cuda>" << std::endl;
        return 1;
    }

    int L = std::stoi(argv[1]);
    int steps = std::stoi(argv[2]);
    double T = std::stod(argv[3]);
    std::string mode_str = argv[4];

    // --- CORREZIONE QUI SOTTO ---
    // Uso "Mode" direttamente, senza IsingModel2d:: davanti
    Mode mode;
    if (mode_str == "serial") mode = Mode::serial;
    else if (mode_str == "omp") mode = Mode::parallel_cpu;
    else if (mode_str == "cuda") mode = Mode::parallel_CUDA;
    else {
        std::cerr << "Invalid mode!" << std::endl;
        return 1;
    }

    // Inizializzazione
    IsingModel2d model(L, T, 1.0, 0.0, 1234);

    // Esecuzione e Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    model.update(mode, steps);
    
    // Sync per CUDA
    if (mode == Mode::parallel_CUDA) model.device_synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calcolo osservabili
    double mag = model.magnetization(mode);
    double energy = model.energy(mode);

    // Output per Python: TIME, MAGNETIZATION, ENERGY
    std::cout << elapsed.count() << "," << mag << "," << energy << std::endl;

    return 0;
}