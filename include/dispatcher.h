#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "tensor.h"
#include "runtime.h"
#include "dispatch_key.h"




/*
========================================
DISPATCH KEY
========================================
*/

void dispatcher_init();

BinaryKernelFn dispatch_binary_kernel(DispatchKey key);
UnaryKernelFn dispatch_unary_kernel(DispatchKey key);

void register_binary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,BinaryKernelFn fn);
void register_unary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,UnaryKernelFn fn);
void register_cpu_unary_kernels(void);
void register_cpu_binary_kernels(void);
#endif
