# Parallelizing 2D Ising Model
**Different Implementations and Benchmarks**

## 📌 Project Overview
This project focuses on the simulation and parallelization of the **2D Ising Model**, a statistical mechanics framework used to describe ferromagnetic systems. The primary goal is to evaluate and compare the performance of different programming paradigms—**Serial, OpenMP, and CUDA**—in simulating thermal fluctuations and phase transitions.

The simulation utilizes the **Metropolis-Hastings Algorithm** (Markov Chain Monte Carlo) to sample configurations according to the Boltzmann distribution.

![Phase Transition Plot](./figures/phase_transition_ht.png)

## 🛠️ Implementation & Architectures

### Execution Modes
The project supports four distinct evolution modes:
1. **Serial**: Single-core CPU execution.
2. **OpenMP**: Multi-core CPU parallelization using a checkerboard update scheme.
3. **CUDA Global**: Baseline GPU implementation using global memory.
4. **CUDA Shared**: Optimized GPU implementation using **Shared Memory** and **Read-only cache** (`__ldg()`) to minimize access latency.

### Technical Optimizations
* **Lookup Probability Table**: Pre-computes Boltzmann weights for all possible energy changes to avoid expensive exponential calls during the simulation.
* **Checkerboard Algorithm**: Updates "Red" and "Black" cells in separate phases to prevent race conditions during parallel updates.
* **Periodic Boundary Conditions (PBC)**: 
    * **CPU**: Implemented via ternary operators for branch prediction efficiency.
    * **GPU**: Employs **Halo Cells (Padding)** to ensure memory coalescing and eliminate branch divergence.

## 📊 Benchmarks and Performance

### Test Environment
* **CPU**: Intel® Core™ i5-13420H (4 P-Cores / 4 E-Cores / 12 Threads).
* **GPU**: NVIDIA GeForce RTX 4050 Laptop (Ada Lovelace Architecture, 2560 CUDA Cores, 24 MB L2 Cache).

### Speedup Comparison ($L=512$, 5k steps)
| Mode | Time (s) | Speedup (vs Serial) |
| :--- | :--- | :--- |
| **Serial** | 32.7749 | 1.00x |
| **OpenMP** | 3.5483 | 9.24x |
| **CUDA (Global)** | 0.4315 | 75.96x |
| **CUDA (Shared)** | 0.3793 | **86.42x** |

### Scaling Insights
* **Latency-bound Regime ($L \le 512$)**: Shared Memory is superior as it minimizes access latency.
* **Bandwidth-bound Regime ($L > 512$)**: Performance is limited by VRAM bandwidth as data size exceeds L2 capacity.

## ⚙️ Compilation and Usage

The project is compiled into a high-performance Python library using **PyBind11**. The `Makefile` is located in the `src/` directory.

### Prerequisites
* CUDA Toolkit (including `nvcc` and `curand`)
* OpenMP
* Python 3 and `pybind11`

### Building the Library
Navigate to the source directory and run:
```bash
cd src
make
