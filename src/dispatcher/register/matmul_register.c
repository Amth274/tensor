#include "registery.h"
#include "cpu_ops.h"

void register_cpu_matmul_kernels(void)
{
    register_kernel(DEVICE_CPU, OP_MATMUL, DTYPE_FP32, 1, ISA_SCALAR, cpu_blas_matmul_fp32);
    register_kernel(DEVICE_CPU, OP_MATMUL, DTYPE_FP64, 1, ISA_SCALAR, cpu_blas_matmul_fp64);
}

void register_gpu_matmul_kernels(void)
{
    register_kernel(DEVICE_CUDA, OP_MATMUL, DTYPE_FP32, 1, ISA_SCALAR, gpu_blas_matmul_fp32);
    register_kernel(DEVICE_CUDA, OP_MATMUL, DTYPE_FP64, 1, ISA_SCALAR, gpu_blas_matmul_fp64);
}
