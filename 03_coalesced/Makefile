NVCC   = nvcc
ARCH = sm_75
CFLAGS = -O3 -arch=$(ARCH) -lineinfo
TARGET = coalesced_gemm

all: $(TARGET)

$(TARGET): coalesced_gemm.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) coalesced_gemm.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
