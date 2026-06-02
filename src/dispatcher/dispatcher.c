#include "dispatcher.h"
#include <stdio.h>
#include "registery.h"


KernelFn kernel_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

void dispatcher_init(){
    register_cpu_binary_kernels();
    register_cpu_unary_kernels();
    register_cpu_matmul_kernels();
    register_gpu_matmul_kernels();
}


KernelFn dispatch_kernel(DispatchKey key){
    KernelFn fn = kernel_registry[key.device][key.op][key.dtype][key.contiguous][key.isa];

    if (!fn && key.isa != ISA_SCALAR) {
        fn = kernel_registry[key.device][key.op][key.dtype][key.contiguous][ISA_SCALAR];
    }

    if (!fn) {
        fprintf(stderr,"ERROR: kernel not found ""(device=%d op=%d dtype=%d contig=%d isa=%d)\n",key.device,key.op,key.dtype,key.contiguous,key.isa);
        return NULL;
    }
    return fn;
}
