# CUDA GEMM Optimization

Progressive optimization of matrix multiplication (GEMM) kernels in CUDA.
Benchmarked on an NVIDIA A100 80GB.

## Kernels

| Version | Description | Performance |
|---------|-------------|-------------|
| 01_naive | Baseline, pure global memory | TBD |
| 02_tiled | Shared memory tiling | TBD |
| 03_coalesced | Coalesced memory access | TBD |

## Build
```bash
cd 01_naive
make && make run
```