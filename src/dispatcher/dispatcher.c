#include "graph.h"
#include <stdio.h>

#include "dispatcher.h"
#include "scalar.h"

static BinaryKernelFn
registry[DEVICE_COUNT][OP_COUNT][DTYPE_COUNT];

void register_kernels()
{
    registry[DEVICE_CPU][OP_ADD][DTYPE_FP32]
        = scalar_add_fp32;
}


BinaryKernelFn dispatch_kernel(
    Device device,
    OpType op,
    DType dtype
)
{
    return registry[device][op][dtype];
}