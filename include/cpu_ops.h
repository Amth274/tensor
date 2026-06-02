#ifndef CPU_OPS_H
#define CPU_OPS_H

#include "kernel.h"

//elementary binary kernels 
int scalar_add_fp32(KernelCall* call);
int scalar_mul_fp32(KernelCall* call);
int scalar_sub_fp32(KernelCall* call);
int scalar_div_fp32(KernelCall* call);

int scalar_add_fp64(KernelCall* call);
int scalar_mul_fp64(KernelCall* call);
int scalar_sub_fp64(KernelCall* call);
int scalar_div_fp64(KernelCall* call);


//elementary unary kernels 
int scalar_neg_fp32(KernelCall* call);
int scalar_log_fp32(KernelCall* call);

//matmul kernels
int cpu_blas_matmul_fp32(KernelCall* call);
int cpu_blas_matmul_fp64(KernelCall* call);
int gpu_blas_matmul_fp32(KernelCall* call);
int gpu_blas_matmul_fp64(KernelCall* call);


//elementary binary simd kernels 
int avx2_add_fp32(KernelCall* call);
int avx2_mul_fp32(KernelCall* call);
int avx2_div_fp32(KernelCall* call);
int avx2_sub_fp32(KernelCall* call);



#endif
