#include "dispatcher.h"
#include <stdio.h>
#include "registery.h"


// kernelfn incte

BinaryKernelFn binary_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

UnaryKernelFn unary_registry
    [DEVICE_COUNT][OP_COUNT][DTYPE_COUNT][2][ISA_MAX];

MatMulKernelFn matmul_registry
    [DEVICE_COUNT][DTYPE_COUNT][2][ISA_MAX];

void dispatcher_init(){
    register_cpu_binary_kernels();
    register_cpu_unary_kernels();
    register_cpu_matmul_kernels();
    register_gpu_matmul_kernels();
}


// binary dispatcher
BinaryKernelFn dispatch_binary_kernel(DispatchKey key){
    BinaryKernelFn fn =binary_registry[key.device][key.op][key.dtype][key.contiguous][key.isa];

    if (!fn) {
        fprintf(stderr,"ERROR: kernel not found ""(device=%d op=%d dtype=%d contig=%d)\n",key.device,key.op,key.dtype,key.contiguous);
        return NULL;
    }
    return fn;
}

MatMulKernelFn dispatch_matmul_kernel(DispatchKey key){
    MatMulKernelFn fn = matmul_registry[key.device][key.dtype][key.contiguous][key.isa];

    if (!fn && key.isa != ISA_SCALAR) {
        fn = matmul_registry[key.device][key.dtype][key.contiguous][ISA_SCALAR];
    }

    if (!fn) {
        fprintf(stderr,"ERROR: matmul kernel not found ""(device=%d dtype=%d contig=%d)\n",key.device,key.dtype,key.contiguous);
        return NULL;
    }
    return fn;
}

//unary dispatcher
UnaryKernelFn dispatch_unary_kernel(DispatchKey key){
    UnaryKernelFn fn =unary_registry[key.device][key.op][key.dtype][key.contiguous][key.isa];

    if (!fn) {
        fprintf(stderr,"ERROR: kernel not found ""(device=%d op=%d dtype=%d contig=%d)\n",key.device,key.op,key.dtype,key.contiguous);
        return NULL;
    }
    return fn;
}
