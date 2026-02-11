NVCC = nvcc
LIB_NAME = ising2d

# Paths
CUDA_PATH = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
PY_INCLUDES = $(shell python3 -m pybind11 --includes)
PY_SUFFIX = $(shell python3-config --extension-suffix)
PY_LDFLAGS = $(shell python3-config --ldflags)

# Compiler Flags
# -Xcompiler passes flags to the host compiler (g++).
# -fopenmp is essential to enable CPU parallelism via OpenMP.
HOST_COMPILER_FLAGS = -Xcompiler "-fPIC -fopenmp -O3 -march=native"

NVCCFLAGS = -O3 -shared -arch=sm_89 \
            $(HOST_COMPILER_FLAGS) \
            $(PY_INCLUDES) $(PY_LDFLAGS) \
            -L$(CUDA_LIB_PATH) -Xlinker -rpath,$(CUDA_LIB_PATH) \
            -lcudart -lcurand -lgomp

# SOURCE FILES
# Here we include ALL files required to build the Python library.
SRCS = main_wrapper.cu

all: $(LIB_NAME)$(PY_SUFFIX)

$(LIB_NAME)$(PY_SUFFIX): $(SRCS) IsingModel2d.h Ising_gpu_interface.h
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@
	# Move the library to the parent directory if necessary; otherwise, comment out the line below
	mv $@ ../

clean:
	rm -f ../*$(PY_SUFFIX) *$(PY_SUFFIX)

.PHONY: all clean