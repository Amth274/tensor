#include "dispatcher.h"
#include "scalar.h"
#include "avx2.h"

void register_cpu_unary_kernels(void){

    register_unary_kernel(DEVICE_CPU,OP_NEG,DTYPE_FP32,1,ISA_SCALAR,scalar_neg_fp32);
    register_unary_kernel(DEVICE_CPU,OP_LOG,DTYPE_FP32,1,ISA_SCALAR,scalar_log_fp32);
    
}