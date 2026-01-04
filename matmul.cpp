#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

/**
 * NOTE: my machine is row-major order memory
 */
void matmul_naive(int n,
                  const double* __restrict A,
                  const double* __restrict B,
                  double* __restrict C) {
    for (int r = 0; r < n; ++r) {          // row of A
        for (int c = 0; c < n; ++c) {      // col of B
            for (int k = 0; k < n; ++k) {  // col of A, row of B
                // Equivalent to: C[r][c] += A[r][k] * B[k][c];
                C[r * n + c] += A[r * n + k] * B[k * n + c];
            }
        }
    }
}

void matmul_cache_friendly(int n,
                           const double* __restrict A,
                           const double* __restrict B,
                           double* __restrict C) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                // C[i*n + j] → good, sequential in memory.
                // B[k*n + j] → good, sequential in memory.
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

void matmul_cache_tiled(int n,
                        const double* __restrict A,
                        const double* __restrict B,
                        double* __restrict C) {
    const int TILE = 128;  // tile size (tune for your cache)

    // Initialize C to zero
    for (int i = 0; i < n * n; ++i)
        C[i] = 0.0;

    for (int ii = 0; ii < n; ii += TILE) {
        for (int jj = 0; jj < n; jj += TILE) {
            for (int kk = 0; kk < n; kk += TILE) {
                int i_max = (ii + TILE < n) ? ii + TILE : n;
                int j_max = (jj + TILE < n) ? jj + TILE : n;
                int k_max = (kk + TILE < n) ? kk + TILE : n;

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a = A[i * n + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * n + j] += a * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    // Initialize A and B with some values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i + j * N] = i + j;
            B[i + j * N] = i - j;
            C[i + j * N] = 0.0;
        }
    }

    // basic version
    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(N, A, B, C);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Matrix multiplication time: " << duration.count() / 1000.0 << " ms" << std::endl;

    // cache friendly version
    start = std::chrono::high_resolution_clock::now();
    matmul_cache_friendly(N, A, B, C);
    duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "Cache-friendly matrix multiplication time: " << duration.count() / 1000.0
              << " ms" << std::endl;

    // memory blocked version
    start = std::chrono::high_resolution_clock::now();
    matmul_cache_tiled(N, A, B, C);
    duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "matmul_cache_tiled time: " << duration.count() / 1000.0 << " ms" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
