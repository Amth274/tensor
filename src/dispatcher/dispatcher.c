#include "dispatcher.h"
#include "scalar.h"

#include <stdio.h>

static BinaryKernelFn
binary_registry
[DEVICE_COUNT]
[OP_COUNT]
[DTYPE_COUNT]
[2]; // contiguous/noncontiguous

static void register_cpu_kernels()
{
    binary_registry[DEVICE_CPU][OP_ADD][DTYPE_FP32][1]= scalar_add_fp32;
    // binary_registry[DEVICE_CPU][OP_ADD][DTYPE_FP16] = scalar_add_fp16;
    binary_registry[DEVICE_CPU][OP_MUL][DTYPE_FP32][1] = scalar_mul_fp32;
    binary_registry[DEVICE_CPU][OP_DIV][DTYPE_FP32][1] = scalar_div_fp32;
    binary_registry[DEVICE_CPU][OP_SUB][DTYPE_FP32][1] = scalar_sub_fp32;
}




void dispatcher_init()
{
    register_cpu_kernels();
}

BinaryKernelFn dispatch_binary_kernel(
    DispatchKey key
)
{
    BinaryKernelFn fn =
        binary_registry
        [key.device]
        [key.op]
        [key.dtype]
        [key.contiguous];

    if (!fn) {
        fprintf(stderr,
            "ERROR: kernel not found "
            "(device=%d op=%d dtype=%d contig=%d)\n",
            key.device,
            key.op,
            key.dtype,
            key.contiguous
        );

        return NULL;
    }

    return fn;
}