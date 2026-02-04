#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include"Ising_CUDA_kernels.cu"

// Includi qui i tuoi kernel (o assicurati che siano sopra il main nello stesso file)

// --- HELPER PER IL TEST ---
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int main() {
    // 1. Configurazione
    int L = 128;               // Dimensione reticolo
    int row_stride = L + 2;    // Con padding
    int size = row_stride * row_stride;
    int B = 16;                // Dimensione blocco (16x16 = 256 thread)
    unsigned int seed = 1234;

    // 2. Allocazione Host (CPU)
    std::vector<int> h_lattice(size, 1); // Inizializziamo tutti gli spin a +1
    float h_lookup_probs[10] = {1.0, 1.0, 0.5, 0.1, 0.01,   // Probs per spin -1
                                0.01, 0.1, 0.5, 1.0, 1.0};  // Probs per spin +1 (valori a caso per test)

    // 3. Allocazione Device (GPU)
    int *d_lattice;
    float *d_lookup_probs;
    curandState *d_states;

    checkCuda(cudaMalloc(&d_lattice, size * sizeof(int)));
    checkCuda(cudaMalloc(&d_lookup_probs, 10 * sizeof(float)));
    checkCuda(cudaMalloc(&d_states, size * sizeof(curandState)));

    // 4. Trasferimento Host -> Device
    checkCuda(cudaMemcpy(d_lattice, h_lattice.data(), size * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_lookup_probs, h_lookup_probs, 10 * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Configurazione Lancio
    dim3 blockSize(B, B);
    dim3 gridSize((L + B - 1) / B, (L + B - 1) / B);

    std::cout << "Inizializzazione RNG..." << std::endl;
    setup_rand_kernel<<<gridSize, blockSize>>>(d_states, seed, L, row_stride);
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Esecuzione 100 passi Metropolis..." << std::endl;
    for (int i = 0; i < 100; i++) {
        // Fase Rossa
        Ising_step_kernel<<<gridSize, blockSize>>>(d_lattice, L, row_stride, d_lookup_probs, 0, d_states, B);
        // Fase Nera
        Ising_step_kernel<<<gridSize, blockSize>>>(d_lattice, L, row_stride, d_lookup_probs, 1, d_states, B);
    }
    checkCuda(cudaDeviceSynchronize());

    // 6. Trasferimento Device -> Host per verifica
    checkCuda(cudaMemcpy(h_lattice.data(), d_lattice, size * sizeof(int), cudaMemcpyDeviceToHost));

    // 7. Semplice Verifica: Calcoliamo la magnetizzazione finale
    double mag = 0;
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            mag += h_lattice[i * row_stride + j];
        }
    }
    std::cout << "Magnetizzazione finale media: " << mag / (L * L) << std::endl;
    std::cout << "Test completato con successo!" << std::endl;

    // 8. Pulizia
    cudaFree(d_lattice);
    cudaFree(d_lookup_probs);
    cudaFree(d_states);

    return 0;
}