NVCC   = nvcc
ARCH = sm_75
CFLAGS = -O3 -arch=$(ARCH) -lineinfo
TARGET = tiled_gemm

all: $(TARGET)

$(TARGET): tiled_gemm.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) tiled_gemm.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
