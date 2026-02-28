#ifndef BINARY_KERNEL_TEMPLATES_H
#define BINARY_KERNEL_TEMPLATES_H

#include "backend/common/tensor_iterator.h"

/* Helper macro */
#define DEFINE_BINARY_KERNEL_FOR_TYPE(OPNAME, SCALAR_EXPR, ctype, suffix) \
void cpu_##OPNAME##_##suffix(                                             \
    Tensor* t1, Tensor* t2, Tensor* out                                   \
) {                                                                       \
    TensorIterator it = tensor_iterator(t1, t2, out);                      \
                                                                           \
    ctype* a    = (ctype*)it.ptr_a;                                        \
    ctype* b    = (ctype*)it.ptr_b;                                        \
    ctype* outp = (ctype*)it.ptr_out;                                      \
                                                                           \
    for (int i = 0; i < it.numel; i++) {                                   \
        outp[i] = (SCALAR_EXPR);                                           \
    }                                                                      \
}

#endif