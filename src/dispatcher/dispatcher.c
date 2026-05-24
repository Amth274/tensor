#include "dispatcher.h"
#include "register.h"
#include <stdio.h>

static BinaryKernelFn
binary_registry
[DEVICE_COUNT]
[OP_COUNT]
[DTYPE_COUNT]
[2];

void register_binary_kernel(Device device,OpType op,DType dtype,uint8_t contiguous,BinaryKernelFn fn)
{
    binary_registry[device][op][dtype][contiguous]= fn;
}

void dispatcher_init()
{
    register_cpu_kernels();
}

BinaryKernelFn dispatch_binary_kernel(
    DispatchKey key
)
{
    BinaryKernelFn fn =binary_registry[key.device][key.op][key.dtype][key.contiguous];

    if (!fn) {

        fprintf(stderr,"ERROR: kernel not found ""(device=%d op=%d dtype=%d contig=%d)\n",key.device,key.op,key.dtype,key.contiguous);
        return NULL;
    }
    return fn;
}