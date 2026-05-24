#ifndef SCALAR_H
#define SCALAR_H

#include "dispatcher.h"

//elementary binary kernels 
void scalar_add_fp32(void* out,float* a,float* b,uint32_t numel);
void scalar_mul_fp32(void* out,float* a,float* b,uint32_t numel);
void scalar_sub_fp32(void* out,float* a,float* b,uint32_t numel);

#endif