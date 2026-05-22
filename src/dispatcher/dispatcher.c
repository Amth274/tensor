#include "graph.h"
#include <stdio.h>
#include "dispatcher.h"

static KernelFn registry[DEVICE_COUNT][OP_COUNT];

void register_kernels()
{
    registry[DEVICE_CPU][OP_ADD] = cpu_add_kernel;
}

