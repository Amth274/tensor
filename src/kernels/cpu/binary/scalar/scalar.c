#include "cpu_ops.h"

#define DEFINE_SCALAR_BINARY_KERNEL(NAME, TYPE, OP)                         \
int NAME(KernelCall* call)                                                   \
{                                                                            \
    TYPE* out = (TYPE*)call->outputs[0];                                      \
    const TYPE* a = (const TYPE*)call->inputs[0];                             \
    const TYPE* b = (const TYPE*)call->inputs[1];                             \
    uint32_t numel = (uint32_t)call->output_metas[0]->numel;                 \
                                                                             \
    for (uint32_t i = 0; i < numel; i++) {                                    \
        out[i] = a[i] OP b[i];                                                \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}

DEFINE_SCALAR_BINARY_KERNEL(scalar_add_fp32, float, +)
DEFINE_SCALAR_BINARY_KERNEL(scalar_mul_fp32, float, *)
DEFINE_SCALAR_BINARY_KERNEL(scalar_sub_fp32, float, -)
DEFINE_SCALAR_BINARY_KERNEL(scalar_div_fp32, float, /)

DEFINE_SCALAR_BINARY_KERNEL(scalar_add_fp64, double, +)
DEFINE_SCALAR_BINARY_KERNEL(scalar_mul_fp64, double, *)
DEFINE_SCALAR_BINARY_KERNEL(scalar_sub_fp64, double, -)
DEFINE_SCALAR_BINARY_KERNEL(scalar_div_fp64, double, /)
