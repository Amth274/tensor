#include "tensor.h"
#include "dispatcher.h"
#include "graph.h"
#include<stdio.h>
#include "policy.h"

DispatchKey execution_policy(Node* n,Graph* g){

    OpType op = n->op;
    for(uint32_t i=0;i<n->num_inputs;i++){
        i++;
    }

    DispatchKey dpk;
    dpk.device = g->device;
    dpk.op = n->op;
    dpk.dtype = DTYPE_FP32;
    dpk.contiguous = 1;

    return dpk;
}