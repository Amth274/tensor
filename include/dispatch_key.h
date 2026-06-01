#ifndef DISPATCH_KEY_H
#define DISPATCH_KEY_H

#include <stdint.h>
#include "runtime.h"
#include "tensor.h"



typedef struct {
    Device device;
    Op op;
    DType dtype;
    uint8_t contiguous;
    isa_t isa;
    
} DispatchKey;

/*
========================================
KERNEL ABI
========================================
*/

typedef void (*BinaryKernelFn)(
    void* out,
    const void* a,
    const void* b,
    uint32_t numel
);


typedef void (*UnaryKernelFn)(
    void* out,
    const void* a,
    double scalar,
    uint32_t numel
);

typedef void (*MatMulKernelFn)(
    void* out,
    const void* a,
    const void* b,
    uint32_t m,
    uint32_t n,
    uint32_t k
);

#endif
