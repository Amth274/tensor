#include <stdint.h>
#include "cpu_ops.h"

#define MATMUL_BLOCK 32

int cpu_blas_matmul_fp32(KernelCall* call)
{
    TensorMeta* a_meta = call->input_metas[0];
    TensorMeta* b_meta = call->input_metas[1];
    uint32_t m = (uint32_t)a_meta->shape[0];
    uint32_t n = (uint32_t)b_meta->shape[1];
    uint32_t k = (uint32_t)a_meta->shape[1];
    float* c = (float*)call->outputs[0];
    const float* x = (const float*)call->inputs[0];
    const float* y = (const float*)call->inputs[1];

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
    return 0;
}

int cpu_blas_matmul_fp64(KernelCall* call)
{
    TensorMeta* a_meta = call->input_metas[0];
    TensorMeta* b_meta = call->input_metas[1];
    uint32_t m = (uint32_t)a_meta->shape[0];
    uint32_t n = (uint32_t)b_meta->shape[1];
    uint32_t k = (uint32_t)a_meta->shape[1];
    double* c = (double*)call->outputs[0];
    const double* x = (const double*)call->inputs[0];
    const double* y = (const double*)call->inputs[1];

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
    return 0;
}
