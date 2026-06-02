#ifndef MACROS_H
#define MACROS_H
#include<math.h>

/*
==================================================
Binary kernel generator macro
=================================================
*/


#define DEFINE_BINARY_KERNEL(NAME,TYPE,OP) \
int NAME(KernelCall* call) \
{                               \
    TYPE* o = (TYPE*)call->outputs[0] ;      \
    const TYPE* x = (const TYPE*)call->inputs[0]; \
    const TYPE* y = (const TYPE*)call->inputs[1]; \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel; \
                                    \
    for(uint32_t i=0;i<numel;i++){  \
        o[i] = x[i] OP y[i];        \
    }                               \
    return 0;                       \
}    


/*=====================================
UNARY KERNELS                           
=======================================
*/

#define DEFINE_UNARY_KERNEL(NAME,TYPE,OP)           \
int NAME(KernelCall* call){                         \
                                                    \
    TYPE* o = (TYPE*)call->outputs[0];              \
    const TYPE* x = (const TYPE*)call->inputs[0];   \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel; \
                                                    \
    for(uint32_t i=0;i<numel;i++){                  \
        o[i] = OP x[i];                             \
    }                                               \
    return 0;                                       \
}

#define DEFINE_LOG_KERNEL(NAME,TYPE) \
int NAME(KernelCall* call){ \
    TYPE* o = (TYPE*)call->outputs[0]; \
    const TYPE* x = (const TYPE*)call->inputs[0]; \
    double sc = call->node->scalar; \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel; \
    for(uint32_t i=0;i<numel;i++){ \
        o[i] = (TYPE)(log((double)x[i]) / log(sc)); \
    } \
    return 0; \
}

#endif //MACROS_H
