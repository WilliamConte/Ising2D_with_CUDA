# --- CONFIGURAZIONE COMPILATORE ---
NVCC = nvcc

# --- ARCHITETTURA GPU ---
# sm_89 è specifico per RTX 4050/4060 (Ada Lovelace).
# Se ti da errore "architecture not supported", prova sm_86 o sm_75.
ARCH = -arch=sm_89

# --- FLAG DI COMPILAZIONE ---
# -O3: Massima ottimizzazione
# -Xcompiler -fopenmp: Passa il flag OpenMP al compilatore C++ sottostante (g++)
# -std=c++17: Usa standard C++ moderno
CFLAGS = -O3 $(ARCH) -Xcompiler -fopenmp -std=c++17

# --- FILE SORGENTI ---
# Elenco di tutti i file .cpp e .cu
SRCS = main.cpp IsingModel2d.cpp Ising_CUDA_kernels.cu

# --- NOME ESEGUIBILE ---
TARGET = ising_sim

# --- REGOLE DI COMPILAZIONE ---

# Regola di default (basta scrivere 'make')
all: $(TARGET)

# Compila tutto in un colpo solo
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Pulisce i file generati (scrivi 'make clean')
clean:
	rm -f $(TARGET)

# Stampa info di debug (scrivi 'make info')
info:
	@echo "Compilatore: $(NVCC)"
	@echo "Architettura: $(ARCH)"
	@echo "Sorgenti: $(SRCS)"