#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdlib.h>
#include <stdint.h>

typedef enum {
    ISA_SCALAR=0,
    ISA_AVX,
    ISA_AVX2,
    ISA_AVX512,
    ISA_NEON,
    ISA_MAX
} isa_t;

typedef struct {
    uint8_t avx2;
    uint8_t avx512;
    uint8_t fma;
    uint8_t sse42;
} CpuCaps;

extern CpuCaps g_cpu_caps;

void detect_cpu_caps(void);
void* gpu_managed_alloc(size_t size);
void gpu_managed_free(void* ptr);

#endif 
