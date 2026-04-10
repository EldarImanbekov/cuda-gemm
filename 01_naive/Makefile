# Makefile for CUDA GEMM project
# Usage: make        → builds the binary
#        make run    → builds and runs
#        make clean  → removes build artifacts

NVCC      = nvcc
# sm_80 = A100, sm_86 = RTX 3090, sm_89 = RTX 4090
# Change this to match your GPU
ARCH      = sm_80
CFLAGS    = -O3 -arch=$(ARCH) -lineinfo
TARGET    = naive_gemm

all: $(TARGET)

$(TARGET): naive_gemm.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) naive_gemm.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
