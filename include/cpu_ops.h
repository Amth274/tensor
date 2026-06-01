#ifndef CPU_OPS_H
#define CPU_OPS_H

#include<stdint.h>

//elementary binary kernels 
void scalar_add_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_mul_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_sub_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_div_fp32(void* out,const void* a,const void* b,uint32_t numel);

void scalar_add_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_mul_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_sub_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_div_fp64(void* out,const void* a,const void* b,uint32_t numel);


//elementary unary kernels 
void scalar_neg_fp32(void* out,const void* a,double sc,uint32_t numel);
void scalar_log_fp32(void* out,const void* a,double sc,uint32_t numel);

//matmul kernels
void cpu_blas_matmul_fp32(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k);
void cpu_blas_matmul_fp64(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k);
void gpu_blas_matmul_fp32(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k);
void gpu_blas_matmul_fp64(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k);


//elementary binary simd kernels 
void avx2_add_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_mul_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_div_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_sub_fp32(void* out,const void* a,const void* b,uint32_t numel);



#endif
