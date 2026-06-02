#ifndef REGISTERY_H
#define REGISTERY_H 

#include "dispatcher.h"

void register_cpu_unary_kernels(void);
void register_cpu_binary_kernels(void);
void register_cpu_matmul_kernels(void);
void register_gpu_matmul_kernels(void);

void register_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,KernelFn fn);




#endif 
