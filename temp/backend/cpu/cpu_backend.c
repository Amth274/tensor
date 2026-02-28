#include "cpu_backend.h"
#include "dispatcher/dispatcher.h"
#include "cpu_dtype_list.h"
#include "backend/cpu/ops/elementwise/elementwise_kernels.h"

CPUFeatures g_cpu_features;

/* Forward declaration */
static void detect_cpu_features(void);

static void detect_cpu_features(void)
{
    g_cpu_features.has_avx2 = 0;
    g_cpu_features.has_avx  = 0;
}

void cpu_backend_init(void)
{
    detect_cpu_features();

    /* Register ADD for all CPU dtypes */
#define X(ctype, suffix, dtype_enum) \
    dispatch_register_binary( \
        OP_ADD, \
        DEVICE_CPU, \
        dtype_enum, \
        cpu_add_##suffix \
    );

    CPU_DTYPE_LIST
#undef X

    /* Register SUB for all CPU dtypes */
#define X(ctype, suffix, dtype_enum) \
    dispatch_register_binary( \
        OP_SUB, \
        DEVICE_CPU, \
        dtype_enum, \
        cpu_sub_##suffix \
    );

    CPU_DTYPE_LIST
#undef X

}