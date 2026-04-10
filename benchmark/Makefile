NVCC   = nvcc
ARCH   = sm_80
CFLAGS = -O3 -arch=$(ARCH) -lineinfo -lm
TARGET = benchmark

all: $(TARGET)

$(TARGET): benchmark.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) benchmark.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
