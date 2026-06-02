#include <immintrin.h>
#include "cpu_ops.h"

#define DEFINE_AVX2_BINARY_KERNEL(NAME, AVX_OP, SCALAR_OP)                  \
int NAME(KernelCall* call)                                                   \
{                                                                            \
    float* out = (float*)call->outputs[0];                                    \
    const float* a = (const float*)call->inputs[0];                           \
    const float* b = (const float*)call->inputs[1];                           \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel;                 \
    uint32_t i = 0;                                                           \
    uint32_t simd_end = numel & ~7U;                                          \
                                                                             \
    for (; i < simd_end; i += 8) {                                            \
        __m256 va = _mm256_loadu_ps(a + i);                                   \
        __m256 vb = _mm256_loadu_ps(b + i);                                   \
        __m256 vr = AVX_OP(va, vb);                                           \
        _mm256_storeu_ps(out + i, vr);                                        \
    }                                                                        \
                                                                             \
    for (; i < numel; i++) {                                                  \
        out[i] = a[i] SCALAR_OP b[i];                                         \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}

DEFINE_AVX2_BINARY_KERNEL(avx2_add_fp32, _mm256_add_ps, +)
DEFINE_AVX2_BINARY_KERNEL(avx2_sub_fp32, _mm256_sub_ps, -)
DEFINE_AVX2_BINARY_KERNEL(avx2_mul_fp32, _mm256_mul_ps, *)
DEFINE_AVX2_BINARY_KERNEL(avx2_div_fp32, _mm256_div_ps, /)
