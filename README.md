## Results (Tesla T4, 1024×1024)

| Kernel | Time (ms) | GFLOP/s | Speedup |
|--------|-----------|---------|---------|
| Naive | 9.070 | 236.8 | 1.00x |
| Tiled (shared mem) | 4.810 | 446.4 | 1.89x |
| Coalesced | 5.612 | 382.7 | 1.62x |