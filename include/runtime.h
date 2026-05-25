#ifndef RUNTIME_H
#define RUNTIME_H

typedef enum {
    ISA_SCALAR=0,
    ISA_AVX,
    ISA_AVX2,
    ISA_AVX512,
    ISA_NEON,
    ISA_MAX
} isa_t;


#endif 
