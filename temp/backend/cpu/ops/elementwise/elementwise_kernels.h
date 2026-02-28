#ifndef ELEMENTWISE_KERNELS_H
#define ELEMENTWISE_KERNELS_H

#include "tensor.h"
#include "backend/cpu/cpu_dtype_list.h"

/* ===== ADD ===== */
#define X(ctype, suffix, dtype_enum) \
    void cpu_add_##suffix(Tensor*, Tensor*, Tensor*);
CPU_DTYPE_LIST
#undef X

/* ===== SUB (if implemented later) */

#define X(ctype, suffix, dtype_enum) \
    void cpu_sub_##suffix(Tensor*, Tensor*, Tensor*);
CPU_DTYPE_LIST
#undef X


#endif