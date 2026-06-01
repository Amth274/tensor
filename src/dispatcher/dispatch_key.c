#include "dispatch_key_builder.h"
#include "runtime.h"

DispatchKey get_binary_dispatch_key(Op op,TensorMeta* a,TensorMeta* b,TensorMeta* out,Graph* g){
    
    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = op;
    dpk.dtype = out->dtype;
    dpk.contiguous = 1;

    if (a->dtype != b->dtype || a->dtype != out->dtype){
        dpk.isa = ISA_SCALAR;
        return dpk;
    }

    if (g->device == DEVICE_CPU && dpk.contiguous && g_cpu_caps.avx2 && dpk.dtype == DTYPE_FP32){
        dpk.isa = ISA_AVX2;
    } else {
        dpk.isa = ISA_SCALAR;
    }

    return dpk;
}


DispatchKey get_unary_dispatch_key(Op op,TensorMeta* a,TensorMeta* out,Graph* g){
    
    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = op;
    dpk.dtype = out->dtype;
    dpk.contiguous = 1;

    if (a->dtype != out->dtype) {
        dpk.isa = ISA_SCALAR;
        return dpk;
    }

    if (g->device == DEVICE_CPU && dpk.contiguous && g_cpu_caps.avx2 && dpk.dtype == DTYPE_FP32){
        dpk.isa = ISA_AVX2;
    } else {
        dpk.isa = ISA_SCALAR;
    }

    return dpk;
}

DispatchKey get_matmul_dispatch_key(TensorMeta* a,TensorMeta* b,TensorMeta* out,Graph* g){
    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = OP_MATMUL;
    dpk.dtype = out->dtype;
    dpk.contiguous = 1;

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        dpk.isa = ISA_SCALAR;
        return dpk;
    }

    dpk.isa = ISA_SCALAR;
    return dpk;
}
