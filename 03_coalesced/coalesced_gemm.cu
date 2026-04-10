/*
 * coalesced_gemm.cu
 *
 * Optimization 2: Coalesced Memory Access
 *
 * Problem with tiled kernel:
 *   When loading a tile of B into shared memory, consecutive threads
 *   read B[row][col] where col is fixed and row varies — these memory
 *   addresses are N floats apart. The GPU fetches 128-byte cache lines,
 *   so strided access wastes most of each fetch.
 *
 * Fix:
 *   Transpose B into B^T before the kernel runs.
 *   Now accessing B^T[col][row] means consecutive threads read
 *   consecutive memory addresses — perfectly coalesced.
 *   One cache line serves as many threads as possible.
 *
 * Memory coalescing rule:
 *   Consecutive threads (threadIdx.x = 0,1,2,...) should access
 *   consecutive memory addresses. Always.
 *
 * C = A * B  (we transpose B internally)
 * A:  (M x K)
 * B:  (K x N)  →  Bt: (N x K)
 * C:  (M x N)
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

#define TILE_SIZE 16

// ─────────────────────────────────────────────
// Transpose Kernel
// ─────────────────────────────────────────────

/*
 * Transposes matrix B (K x N) into Bt (N x K).
 * Each thread copies one element.
 * Also uses shared memory to make the transpose itself coalesced.
 */
__global__ void transpose_kernel(
    const float* B,   // [K x N]
    float*       Bt,  // [N x K]
    int K, int N
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Read from B in a coalesced pattern into shared memory
    if (x < N && y < K)
        tile[threadIdx.y][threadIdx.x] = B[y * N + x];

    __syncthreads();

    // Write to Bt in a coalesced pattern from shared memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < K && y < N)
        Bt[y * K + x] = tile[threadIdx.x][threadIdx.y];
}

// ─────────────────────────────────────────────
// Coalesced GEMM Kernel
// ─────────────────────────────────────────────

/*
 * Same tiling structure as before, but now uses Bt (transposed B).
 * Loading a tile of Bt into shared memory is now fully coalesced
 * because consecutive threads read consecutive memory addresses.
 *
 * C[row][col] = dot(A[row], B[:,col])
 *             = dot(A[row], Bt[col])   ← same math, better memory access
 */
__global__ void coalesced_gemm_kernel(
    const float* A,   // [M x K]
    const float* Bt,  // [N x K]  — transposed B
    float*       C,   // [M x N]
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bts[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;

    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // Load tile of A — coalesced (consecutive threads read consecutive cols)
        int aCol = t * TILE_SIZE + threadCol;
        if (row < M && aCol < K)
            As[threadRow][threadCol] = A[row * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;

        // Load tile of Bt — NOW COALESCED
        // Bt[col][k] — consecutive threads vary col, read consecutive memory
        int btRow = blockCol * TILE_SIZE + threadRow;  // col dimension
        int btCol = t * TILE_SIZE + threadCol;          // k dimension
        if (btRow < N && btCol < K)
            Bts[threadRow][threadCol] = Bt[btRow * K + btCol];
        else
            Bts[threadRow][threadCol] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadRow][k] * Bts[threadCol][k];
            //                        ↑ note: threadCol indexes into transposed tile
        }

        __syncthreads();
    }

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

    size_t bytes_A  = M * K * sizeof(float);
    size_t bytes_B  = K * N * sizeof(float);
    size_t bytes_C  = M * N * sizeof(float);

    float* h_A    = (float*)malloc(bytes_A);
    float* h_B    = (float*)malloc(bytes_B);
    float* h_C    = (float*)malloc(bytes_C);
    float* h_Cref = (float*)malloc(bytes_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_Bt, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,  bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,  bytes_B));
    CUDA_CHECK(cudaMalloc(&d_Bt, bytes_B));  // same size as B
    CUDA_CHECK(cudaMalloc(&d_C,  bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Transpose B on the GPU before the main kernel
    dim3 transBlock(TILE_SIZE, TILE_SIZE);
    dim3 transGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                   (K + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<transGrid, transBlock>>>(d_B, d_Bt, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("\nLaunch config: grid(%d, %d), block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    coalesced_gemm_kernel<<<grid, block>>>(d_A, d_Bt, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int NUM_RUNS = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++)
        coalesced_gemm_kernel<<<grid, block>>>(d_A, d_Bt, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_RUNS;

    double gflops = (2.0 * M * K * N) / (ms / 1000.0) / 1e9;
    printf("\nCoalesced kernel (TILE_SIZE=%d):\n", TILE_SIZE);
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
    CUDA_CHECK(cudaFree(d_Bt));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
