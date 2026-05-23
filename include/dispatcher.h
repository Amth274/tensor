#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "tensor.h"

/*
========================================
OPS
========================================
*/

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_COUNT
} OpType;

/*
========================================
KERNEL ABI
========================================
*/

typedef void (*BinaryKernelFn)(
    void* out,
    void* a,
    void* b,
    uint32_t numel
    // uint32_t stride,
);

/*
========================================
DISPATCH KEY
========================================
*/

typedef struct {
    Device device;
    OpType op;
    DType dtype;
    uint8_t contiguous;
} DispatchKey;

/*
========================================
DISPATCH API
========================================
*/

void dispatcher_init();

BinaryKernelFn dispatch_binary_kernel(
    DispatchKey key
);

#endif