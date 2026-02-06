NVCC = nvcc
LIB_NAME = ising2d

# Paths
CUDA_PATH = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
PY_INCLUDES = $(shell python3 -m pybind11 --includes)
PY_SUFFIX = $(shell python3-config --extension-suffix)
PY_LDFLAGS = $(shell python3-config --ldflags)

# Compiler Flags
# -arch=sm_89 for RTX 4050
# -lgomp is needed for OpenMP
NVCCFLAGS = -O3 -shared -Xcompiler -fPIC -arch=sm_89 \
            $(PY_INCLUDES) $(PY_LDFLAGS) \
            -L$(CUDA_LIB_PATH) -Xlinker -rpath,$(CUDA_LIB_PATH) \
            -lcudart -lcurand -lgomp

all: $(LIB_NAME)$(PY_SUFFIX)

$(LIB_NAME)$(PY_SUFFIX): main_wrapper.cu IsingModel2d.h Ising_gpu_interface.h
	$(NVCC) $(NVCCFLAGS) main_wrapper.cu -o $@
	mv $@ ../

clean:
	rm -f *$(PY_SUFFIX)

.PHONY: all clean