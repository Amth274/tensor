#include "tensor.h"
#include "dispatcher.h"
#include "graph.h"
#include<stdio.h>
#include<stdlib.h>
#include "policy.h"

DispatchKey execution_policy(Node* n,Graph* g){

    // OpType op = n->op;

    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = n->op;
    dpk.dtype = DTYPE_FP32;
    dpk.contiguous = 1;
    dpk.isa = ISA_SCALAR;
    dpk.isa = ISA_AVX2;
      
    return dpk;
}