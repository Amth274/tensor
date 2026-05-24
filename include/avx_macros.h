#pragma once
#ifndef AVX_MACROS_H
#define AVX_MACROS_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

#define SIMD_BINARY_KERNEL(NAME, OP, SCALAR_OP)                     \
void NAME(                                                          \
    void* out,                                                      \
    const void* a,                                                  \
    const void* b,                                                  \
    uint32_t numel                                                  \
) {                                                                 \
    float* o = (float*)out;                                         \
    const float* x = (const float*)a;                               \
    const float* y = (const float*)b;                               \
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
}

#endif