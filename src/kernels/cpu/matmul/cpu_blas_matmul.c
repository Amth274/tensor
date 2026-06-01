#include <stdint.h>
#include "cpu_ops.h"

#define MATMUL_BLOCK 32

void cpu_blas_matmul_fp32(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k)
{
    float* c = (float*)out;
    const float* x = (const float*)a;
    const float* y = (const float*)b;

    for (uint32_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }

    for (uint32_t ii = 0; ii < m; ii += MATMUL_BLOCK) {
        uint32_t i_end = ii + MATMUL_BLOCK < m ? ii + MATMUL_BLOCK : m;
        for (uint32_t kk = 0; kk < k; kk += MATMUL_BLOCK) {
            uint32_t k_end = kk + MATMUL_BLOCK < k ? kk + MATMUL_BLOCK : k;
            for (uint32_t jj = 0; jj < n; jj += MATMUL_BLOCK) {
                uint32_t j_end = jj + MATMUL_BLOCK < n ? jj + MATMUL_BLOCK : n;

                for (uint32_t i = ii; i < i_end; i++) {
                    for (uint32_t p = kk; p < k_end; p++) {
                        float a_ip = x[i * k + p];
                        for (uint32_t j = jj; j < j_end; j++) {
                            c[i * n + j] += a_ip * y[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

void cpu_blas_matmul_fp64(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k)
{
    double* c = (double*)out;
    const double* x = (const double*)a;
    const double* y = (const double*)b;

    for (uint32_t i = 0; i < m * n; i++) {
        c[i] = 0.0;
    }

    for (uint32_t ii = 0; ii < m; ii += MATMUL_BLOCK) {
        uint32_t i_end = ii + MATMUL_BLOCK < m ? ii + MATMUL_BLOCK : m;
        for (uint32_t kk = 0; kk < k; kk += MATMUL_BLOCK) {
            uint32_t k_end = kk + MATMUL_BLOCK < k ? kk + MATMUL_BLOCK : k;
            for (uint32_t jj = 0; jj < n; jj += MATMUL_BLOCK) {
                uint32_t j_end = jj + MATMUL_BLOCK < n ? jj + MATMUL_BLOCK : n;

                for (uint32_t i = ii; i < i_end; i++) {
                    for (uint32_t p = kk; p < k_end; p++) {
                        double a_ip = x[i * k + p];
                        for (uint32_t j = jj; j < j_end; j++) {
                            c[i * n + j] += a_ip * y[p * n + j];
                        }
                    }
                }
            }
        }
    }
}
