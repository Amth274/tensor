#ifndef AVX2_H
#define AVX2_H

void avx2_add_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_mul_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_div_fp32(void* out,const void* a,const void* b,uint32_t numel);
void avx2_sub_fp32(void* out,const void* a,const void* b,uint32_t numel);


#endif