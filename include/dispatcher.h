#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "tensor.h"
// #include "graph.h"

typedef enum {
    OP_ADD,
    OP_MUL,
    OP_SUB,
    OP_DIV,
    OP_COUNT
} OpType;

// typedef enum {
//     DTYPE_FP32,
//     DTYPE_FP16,
//     DTYPE_BF16,
//     DTYPE_COUNT
// } DType;

// typedef enum {
//     DEVICE_CPU,
//     DEVICE_CUDA,
//     DEVICE_ROCM,
//     DEVICE_COUNT
// } Device;


/*
========================================
KERNEL FUNCTION TYPES
========================================
*/

typedef void (*BinaryKernelFn)(
    void* out,
    void* a,
    void* b,
    uint32_t numel
);

#endif