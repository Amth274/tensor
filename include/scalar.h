#ifndef SCALAR_H
#define SCALAR_H

#include "dispatcher.h"

/*
==================================================
Binary kernel generator macro
=================================================
*/


#define DEFINE_BINARY_KERNEL(NAME,TYPE,OP) \
void NAME(void* out,const void* a,const void* b, uint32_t numel) \
{                               \
    TYPE* o = (TYPE*)out ;      \
    TYPE* x = (const TYPE*)a; \
    TYPE* y = (const TYPE*)b; \
                                    \
    for(uint32_t i=0;i<numel;i++){  \
        o[i] = x[i] OP y[i];        \
    }                               \
                                    \
}                                   

//elementary binary kernels 
void scalar_add_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_mul_fp32(void* out,const void* a,const void* b,uint32_t numel);
void scalar_sub_fp32(void* out,const void* a,const void* b,uint32_t numel);

#endif