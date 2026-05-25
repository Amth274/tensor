#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "tensor.h"
#include "runtime.h"


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
    const void* a,
    const void* b,
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
    isa_t isa;
    
} DispatchKey;

/*
========================================
DISPATCH API
========================================
*/

void dispatcher_init();

BinaryKernelFn dispatch_binary_kernel(DispatchKey key);
void register_binary_kernel(Device device,OpType op,DType dtype,uint8_t contiguous,BinaryKernelFn fn);

#endif