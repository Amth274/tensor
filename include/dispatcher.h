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
    OP_NEG,
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
);


typedef void (*UnaryKernelFn)(
    void* out,
    const void* a,
    uint32_t numel
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
    isa_t isa;
    
} DispatchKey;

/*
========================================
DISPATCH API
========================================
*/

void dispatcher_init();

BinaryKernelFn dispatch_binary_kernel(DispatchKey key);
UnaryKernelFn dispatch_unary_kernel(DispatchKey key);
void register_binary_kernel(Device device,OpType op,DType dtype,uint8_t contiguous,isa_t isa,BinaryKernelFn fn);
void register_unary_kernel(Device device,OpType op,DType dtype,uint8_t contiguous,isa_t isa,UnaryKernelFn fn);
void register_cpu_unary_kernels(void);
void register_cpu_binary_kernels(void);
#endif