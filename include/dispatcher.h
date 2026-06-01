#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "dispatch_key.h"
#include "types.h"


extern BinaryKernelFn binary_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

extern UnaryKernelFn unary_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

extern MatMulKernelFn matmul_registry
    [DEVICE_COUNT][DTYPE_COUNT][2][ISA_MAX];

void dispatcher_init();

BinaryKernelFn dispatch_binary_kernel(DispatchKey key);
UnaryKernelFn dispatch_unary_kernel(DispatchKey key);
MatMulKernelFn dispatch_matmul_kernel(DispatchKey key);


#endif
