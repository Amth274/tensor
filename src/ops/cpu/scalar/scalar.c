#include "scalar.h"

void scalar_add_fp32(
    void* out,
    float* a,
    float* b,
    uint32_t numel
)
{
    float* o = (float*)out;

    for(uint32_t i = 0; i < numel; i++) {
        o[i] = a[i] + b[i];
    }
}

void scalar_mul_fp32(
    void* out,
    float* a,
    float* b,
    uint32_t numel
)
{
    float* o = (float*)out;

    for(uint32_t i = 0; i < numel; i++) {
        o[i] = a[i] * b[i];
    }
}