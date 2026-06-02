#pragma once
#ifndef AVX_MACROS_H
#define AVX_MACROS_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include "kernel.h"

#define SIMD_BINARY_KERNEL(NAME, OP, SCALAR_OP)                     \
int NAME(KernelCall* call) {                                        \
    float* o = (float*)call->outputs[0];                            \
    const float* x = (const float*)call->inputs[0];                 \
    const float* y = (const float*)call->inputs[1];                 \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel;        \
                                                                    \
    uint32_t i = 0;                                                 \
                                                                    \
    uint32_t simd_end = numel & ~7U;                                \
                                                                    \
    for (; i < simd_end; i += 8) {                                  \
        __m256 va = _mm256_loadu_ps(x + i);                         \
        __m256 vb = _mm256_loadu_ps(y + i);                         \
                                                                    \
        __m256 vr = OP(va, vb);                                     \
                                                                    \
        _mm256_storeu_ps(o + i, vr);                                \
    }                                                               \
                                                                    \
    for (; i < numel; ++i) {                                        \
        o[i] = x[i] SCALAR_OP y[i];                                 \
    }                                                               \
    return 0;                                                       \
}

#endif
