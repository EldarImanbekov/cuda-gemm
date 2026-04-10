/*
 * naive_gemm.cu
 * 
 * Baseline matrix multiplication kernel.
 * Every thread computes one element of C by reading
 * directly from global memory — no caching, no tricks.
 * This is the starting point we will optimize.
 *
 * C = A * B
 * A: (M x K)
 * B: (K x N)
 * C: (M x N)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────
// Naive GEMM Kernel
// ─────────────────────────────────────────────

/*
 * Each thread is responsible for exactly ONE element of C.
 *
 * Thread (row, col) computes C[row][col] by:
 *   - Walking across row `row` of A        (K reads from global memory)
 *   - Walking down  col `col` of B         (K reads from global memory)
 *   - Multiplying each pair and accumulating
 *
 * Problem: every read hits global memory (VRAM).
 * Global memory latency ~600-800 cycles.
 * With large K, threads spend most of their time waiting.
 */
__global__ void naive_gemm_kernel(
    const float* A,   // [M x K]
    const float* B,   // [K x N]
    float*       C,   // [M x N]
    int M, int K, int N
) {
    // Which element of C does this thread own?
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check — matrix size may not be divisible by block size
    if (row >= M || col >= N) return;

    // Accumulate the dot product into a register
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        // A[row][k] in row-major layout = A[row * K + k]
        // B[k][col] in row-major layout = B[k * N + col]
        sum += A[row * K + k] * B[k * N + col];
    }

    // Write result once — only one global memory write per thread
    C[row * N + col] = sum;
}

// ─────────────────────────────────────────────
// CPU Reference (for correctness check)
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
    // Matrix dimensions
    // Using powers of 2 — GPU loves aligned sizes
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    printf("Matrix dimensions: A(%d x %d) * B(%d x %d) = C(%d x %d)\n",
           M, K, K, N, M, N);
    printf("Total FLOPs: %.2f GFLOP\n", 2.0 * M * K * N / 1e9);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // ── Allocate host memory ──
    float* h_A    = (float*)malloc(bytes_A);
    float* h_B    = (float*)malloc(bytes_B);
    float* h_C    = (float*)malloc(bytes_C);      // GPU result
    float* h_Cref = (float*)malloc(bytes_C);      // CPU reference

    // Fill A and B with random values
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // ── Allocate device memory ──
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // ── Copy input data to GPU ──
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // ── Kernel launch configuration ──
    // 16x16 = 256 threads per block — a common starting point
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    printf("\nLaunch config: grid(%d, %d), block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // ── Warmup run (GPU needs to warm up for accurate timing) ──
    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed run ──
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int NUM_RUNS = 10;
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < NUM_RUNS; i++)
        naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_RUNS;  // average over runs

    // ── Performance ──
    double gflops = (2.0 * M * K * N) / (ms / 1000.0) / 1e9;
    printf("\nNaive kernel:\n");
    printf("  Average time : %.3f ms\n", ms);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);

    // ── Copy result back and verify ──
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    printf("\nRunning CPU reference (this will take a moment)...\n");
    cpu_gemm(h_A, h_B, h_Cref, M, K, N);

    // Compare GPU vs CPU results
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C[i] - h_Cref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %.6f %s\n",
           max_err, max_err < 1e-2f ? "(PASS)" : "(FAIL)");

    // ── Cleanup ──
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
