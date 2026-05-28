#ifndef DISPATCH_KEY_H
#define DISPATCH_KEY_H

#include <stdint.h>
#include "runtime.h"
#include "tensor.h"

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_NEG,
    OP_LOG,
    OP_COUNT
} Op;


typedef enum {
    OP_TYPE_CONTIGUOUS,
    OP_TYPE_STRIDED,
    OP_TYPE_BROADCASTED,
    OP_TYPE_COUNT
} OpType;


typedef enum {
    OP_KIND_UNARY,
    OP_KIND_BINARY,
    OP_KIND_REDUCTION,
    OP_KIND_COUNT
} OpKind;


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
    uint32_t numel
);

#endif
