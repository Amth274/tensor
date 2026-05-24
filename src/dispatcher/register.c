#include "dispatcher.h"
#include "scalar.h"

void register_cpu_kernels(void)
{
    register_binary_kernel(DEVICE_CPU,OP_ADD,DTYPE_FP32,1,scalar_add_fp32);
    register_binary_kernel(DEVICE_CPU,OP_MUL,DTYPE_FP32,1,scalar_mul_fp32);
    register_binary_kernel(DEVICE_CPU,OP_DIV,DTYPE_FP32,1,scalar_div_fp32);
    register_binary_kernel(DEVICE_CPU,OP_SUB,DTYPE_FP32,1,scalar_sub_fp32);

    register_binary_kernel(DEVICE_CPU,OP_ADD,DTYPE_FP64,1,scalar_add_fp64);
    register_binary_kernel(DEVICE_CPU,OP_MUL,DTYPE_FP64,1,scalar_mul_fp64);
    register_binary_kernel(DEVICE_CPU,OP_DIV,DTYPE_FP64,1,scalar_div_fp64);
    register_binary_kernel(DEVICE_CPU,OP_SUB,DTYPE_FP64,1,scalar_sub_fp64);
    
}