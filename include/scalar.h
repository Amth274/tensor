#ifndef SCALAR_H
#define SCALAR_H

#include "dispatcher.h"
                               

//elementary binary kernels 
void scalar_add_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_mul_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_sub_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_div_fp32(void* out,const void* a,const void* b,uint32_t numel);

void scalar_add_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_mul_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_sub_fp64(void* out,const void* a,const void* b,uint32_t numel);
void scalar_div_fp64(void* out,const void* a,const void* b,uint32_t numel);

void scalar_neg_fp32(void* out,const void* a,uint32_t numel);

#endif