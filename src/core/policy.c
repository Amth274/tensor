#include "tensor.h"
#include "dispatcher.h"
#include "graph.h"
#include<stdio.h>
#include "policy.h"

DispatchKey execution_policy(Node* n,Graph* g){

    // OpType op = n->op;
    int avx2_flag = 0;
    

    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = n->op;
    dpk.dtype = DTYPE_FP32;
    dpk.contiguous = 1;
    for(uint32_t i=0;i<n->num_inputs;i++){
        TensorMeta* tm = n->inputs[i];
        if(tm->numel > 1024){
            dpk.isa = ISA_AVX2;
        }
        else{
            dpk.isa = ISA_SCALAR;
        }
    }
    

    

    return dpk;
}