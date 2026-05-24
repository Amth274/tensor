#ifndef MACROS_H
#define MACROS_H

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


#endif MACROS_H