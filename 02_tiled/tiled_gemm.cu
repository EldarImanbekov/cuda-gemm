/*
 * tiled_gemm.cu
 *
 * Optimization 1: Shared Memory Tiling
 *
 * Key idea: instead of every thread reading from slow global memory
 * for every multiply, the entire thread block cooperates to load
 * a tile of A and a tile of B into fast shared memory first.
 * Then everyone computes from shared memory.
 *
 * Global memory latency:  ~600 cycles
 * Shared memory latency:  ~5 cycles  (~100x faster)
 *
 * C = A * B
 * A: (M x K)
 * B: (K x N)
 * C: (M x N)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Tile size — each block handles a TILE_SIZE x TILE_SIZE chunk
// 16x16 = 256 threads per block, fits well in shared memory
#define TILE_SIZE 16

// ─────────────────────────────────────────────
// Tiled GEMM Kernel
// ─────────────────────────────────────────────

/*
 * How tiling works, step by step:
 *
 * The output matrix C is divided into tiles of size TILE_SIZE x TILE_SIZE.
 * Each thread block is responsible for computing one tile of C.
 *
 * To compute its tile of C, a block needs:
 *   - An entire row-strip of A  (row tiles across all of K)
 *   - An entire col-strip of B  (col tiles across all of K)
 *
 * We iterate over these strips one tile at a time:
 *   1. Load tile of A into shared memory  (all threads cooperate)
 *   2. Load tile of B into shared memory  (all threads cooperate)
 *   3. Sync — make sure everyone finished loading  (__syncthreads)
 *   4. Each thread computes its partial dot product from shared memory
 *   5. Sync again before loading the next tile
 *   6. Repeat until all K elements are covered
 *   7. Write final result to C in global memory
 */
__global__ void tiled_gemm_kernel(
    const float* A,
    const float* B,
    float*       C,
    int M, int K, int N
) {
    // Shared memory tiles — declared statically
    // These live in on-chip SRAM, shared by all threads in the block
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Which tile of C does this block own?
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Which element within the tile does this thread own?
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Global row and col this thread is responsible for in C
    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;

    // Accumulate partial results here (stays in a register — very fast)
    float sum = 0.0f;

    // Loop over tiles along the K dimension
    // numTiles = ceil(K / TILE_SIZE)
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // ── Step 1: Load tile of A into shared memory ──
        // Each thread loads one element of the tile
        int aRow = row;
        int aCol = t * TILE_SIZE + threadCol;

        // Bounds check — matrix may not be divisible by TILE_SIZE
        if (aRow < M && aCol < K)
            As[threadRow][threadCol] = A[aRow * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;

        // ── Step 2: Load tile of B into shared memory ──
        int bRow = t * TILE_SIZE + threadRow;
        int bCol = col;

        if (bRow < K && bCol < N)
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        else
            Bs[threadRow][threadCol] = 0.0f;

        // ── Step 3: Sync — CRITICAL ──
        // We MUST wait for ALL threads to finish loading
        // before any thread starts computing.
        // Without this, some threads might read unloaded data.
        __syncthreads();

        // ── Step 4: Compute partial dot product from shared memory ──
        // This is now reading from fast on-chip memory, not global VRAM
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        // ── Step 5: Sync before next tile ──
        // We MUST wait for all threads to finish computing
        // before any thread overwrites shared memory with the next tile.
        __syncthreads();
    }

    // ── Step 6: Write result to global memory ──
    // Only one write per thread total — same as naive
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ─────────────────────────────────────────────
// CPU Reference
// ─────────────────────────────────────────────

void cpu_gemm(
    const float* A,
    const float* B,
    float*       C,
    int M, int K, int N
) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    printf("Matrix dimensions: A(%d x %d) * B(%d x %d) = C(%d x %d)\n",
           M, K, K, N, M, N);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Total FLOPs: %.2f GFLOP\n", 2.0 * M * K * N / 1e9);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A    = (float*)malloc(bytes_A);
    float* h_B    = (float*)malloc(bytes_B);
    float* h_C    = (float*)malloc(bytes_C);
    float* h_Cref = (float*)malloc(bytes_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("\nLaunch config: grid(%d, %d), block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int NUM_RUNS = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++)
        tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_RUNS;

    double gflops = (2.0 * M * K * N) / (ms / 1000.0) / 1e9;
    printf("\nTiled kernel (TILE_SIZE=%d):\n", TILE_SIZE);
    printf("  Average time : %.3f ms\n", ms);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    printf("\nRunning CPU reference...\n");
    cpu_gemm(h_A, h_B, h_Cref, M, K, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C[i] - h_Cref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %.6f %s\n",
           max_err, max_err < 1e-2f ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
