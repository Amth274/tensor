#include "avx_macros.h"

SIMD_BINARY_KERNEL(avx2_add_fp32, _mm256_add_ps, +)
SIMD_BINARY_KERNEL(avx2_sub_fp32, _mm256_sub_ps, -)
SIMD_BINARY_KERNEL(avx2_mul_fp32, _mm256_mul_ps, *)
SIMD_BINARY_KERNEL(avx2_div_fp32, _mm256_div_ps, /)