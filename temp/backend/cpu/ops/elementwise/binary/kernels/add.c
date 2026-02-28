#include "backend/cpu/ops/elementwise/binary/binary_kernel_templates.h"
#include "backend/cpu/cpu_dtype_list.h"

/* Define X for this op */
#define X(ctype, suffix, dtype_enum) \
    DEFINE_BINARY_KERNEL_FOR_TYPE(add, a[i] + b[i], ctype, suffix)

/* Expand for all dtypes */
CPU_DTYPE_LIST

/* Cleanup */
#undef X