#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdint.h>
#include "dispatch_key.h"
#include "kernel.h"
#include "types.h"


extern KernelFn kernel_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

void dispatcher_init();

KernelFn dispatch_kernel(DispatchKey key);


#endif
