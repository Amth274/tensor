#ifndef REGISTERY_H
#define REGISTERY_H 

#include "dispatcher.h"

void register_cpu_unary_kernels(void);
void register_cpu_binary_kernels(void);
void register_cpu_matmul_kernels(void);
void register_gpu_matmul_kernels(void);

void register_binary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,BinaryKernelFn fn);
void register_unary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,UnaryKernelFn fn);
void register_matmul_kernel(Device device,DType dtype,uint8_t contiguous,isa_t isa,MatMulKernelFn fn);




#endif 
