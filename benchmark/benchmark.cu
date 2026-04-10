/*
 * benchmark.cu
 *
 * Runs all three GEMM kernels back to back and prints
 * a clean comparison table with speedups.
 *
 * This is the file you run on the A100 to get your results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define TILE_SIZE 16

// ─────────────────────────────────────────────
// All three kernels inline
// ─────────────────────────────────────────────

__global__ void naive_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

__global__ void tiled_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void transpose_kernel(
    const float* B, float* Bt, int K, int N
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    if (x < N && y < K) tile[threadIdx.y][threadIdx.x] = B[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (x < K && y < N) Bt[y * K + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void coalesced_gemm_kernel(
    const float* A, const float* Bt, float* C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bts[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int btRow = blockIdx.x * TILE_SIZE + threadIdx.y;
        int btCol = t * TILE_SIZE + threadIdx.x;
        Bts[threadIdx.y][threadIdx.x] = (btRow < N && btCol < K) ? Bt[btRow * K + btCol] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bts[threadIdx.x][k];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ─────────────────────────────────────────────
// Timing helper
// ─────────────────────────────────────────────

float time_kernel(void (*launch)(float*, float*, float*, int, int, int, dim3, dim3),
                  float* d_A, float* d_B, float* d_C,
                  int M, int K, int N, dim3 grid, dim3 block,
                  int num_runs) {
    // warmup
    launch(d_A, d_B, d_C, M, K, N, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++)
        launch(d_A, d_B, d_C, M, K, N, grid, block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / num_runs;
}

// Launch wrappers so we can pass them as function pointers
void launch_naive(float* A, float* B, float* C, int M, int K, int N, dim3 grid, dim3 block) {
    naive_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
void launch_tiled(float* A, float* B, float* C, int M, int K, int N, dim3 grid, dim3 block) {
    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
void launch_coalesced(float* A, float* Bt, float* C, int M, int K, int N, dim3 grid, dim3 block) {
    coalesced_gemm_kernel<<<grid, block>>>(A, Bt, C, M, K, N);
}

// ─────────────────────────────────────────────
// Correctness check
// ─────────────────────────────────────────────

bool verify(const float* gpu, const float* ref, int N) {
    for (int i = 0; i < N; i++)
        if (fabsf(gpu[i] - ref[i]) > 1e-2f) return false;
    return true;
}

void cpu_gemm(const float* A, const float* B, float* C, int M, int K, int N) {
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
    const int NUM_RUNS = 20;

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n");
    printf("=================================================\n");
    printf("  CUDA GEMM Benchmark\n");
    printf("=================================================\n");
    printf("GPU       : %s\n", prop.name);
    printf("Matrix    : %d x %d\n", M, N);
    printf("FLOPs     : %.2f GFLOP\n", 2.0 * M * K * N / 1e9);
    printf("Runs/kernel: %d\n", NUM_RUNS);
    printf("=================================================\n\n");

    size_t bytes = M * K * sizeof(float);

    // Host memory
    float* h_A    = (float*)malloc(bytes);
    float* h_B    = (float*)malloc(bytes);
    float* h_C    = (float*)malloc(bytes);
    float* h_Cref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Device memory
    float *d_A, *d_B, *d_Bt, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,  bytes));
    CUDA_CHECK(cudaMalloc(&d_B,  bytes));
    CUDA_CHECK(cudaMalloc(&d_Bt, bytes));
    CUDA_CHECK(cudaMalloc(&d_C,  bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Precompute transposed B for coalesced kernel
    dim3 transBlock(TILE_SIZE, TILE_SIZE);
    dim3 transGrid((N + TILE_SIZE-1)/TILE_SIZE, (K + TILE_SIZE-1)/TILE_SIZE);
    transpose_kernel<<<transGrid, transBlock>>>(d_B, d_Bt, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch configs
    dim3 block16(TILE_SIZE, TILE_SIZE);
    dim3 grid16((N + TILE_SIZE-1)/TILE_SIZE, (M + TILE_SIZE-1)/TILE_SIZE);

    // CPU reference (small matrix for speed)
    printf("Computing CPU reference...\n");
    cpu_gemm(h_A, h_B, h_Cref, M, K, N);
    printf("Done.\n\n");

    // Results storage
    float times[3];
    double gflops[3];
    const char* names[3] = {"Naive", "Tiled (shared mem)", "Coalesced"};

    // ── Run naive ──
    times[0] = time_kernel(launch_naive, d_A, d_B, d_C, M, K, N, grid16, block16, NUM_RUNS);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("Naive      : %s\n", verify(h_C, h_Cref, M*N) ? "PASS" : "FAIL");

    // ── Run tiled ──
    times[1] = time_kernel(launch_tiled, d_A, d_B, d_C, M, K, N, grid16, block16, NUM_RUNS);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("Tiled      : %s\n", verify(h_C, h_Cref, M*N) ? "PASS" : "FAIL");

    // ── Run coalesced ──
    times[2] = time_kernel(launch_coalesced, d_A, d_Bt, d_C, M, K, N, grid16, block16, NUM_RUNS);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("Coalesced  : %s\n", verify(h_C, h_Cref, M*N) ? "PASS" : "FAIL");

    // Compute GFLOP/s
    for (int i = 0; i < 3; i++)
        gflops[i] = (2.0 * M * K * N) / (times[i] / 1000.0) / 1e9;

    // ── Print results table ──
    printf("\n");
    printf("=================================================\n");
    printf("  Results\n");
    printf("=================================================\n");
    printf("%-22s %10s %12s %10s\n", "Kernel", "Time (ms)", "GFLOP/s", "Speedup");
    printf("%-22s %10s %12s %10s\n", "------", "---------", "-------", "-------");
    for (int i = 0; i < 3; i++) {
        printf("%-22s %10.3f %12.1f %9.2fx\n",
               names[i], times[i], gflops[i], times[0] / times[i]);
    }
    printf("=================================================\n\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Bt));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_Cref);

    return 0;
}
