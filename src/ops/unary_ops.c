#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "ops.h"
#include "tensor.h"


Tensor* tensor_neg(Tensor* t)
{
    if (!t) {
        fprintf(stderr, "ERROR: null tensor recieved\n");
        return NULL;
    }

    Graph* g = t->graph;
    TensorMeta* m = &g->tensors[t->id];

    Tensor* out = tensor_create(g,m->ndims,m->shape,m->dtype,m->requires_grad);

    if (!out) {
        fprintf(stderr, "ERROR: output tensor alloc failed\n");
        return NULL;
    }

    uint32_t inputs[1] = { t->id };
    uint32_t outpus[1] = {out->id};

    if (graph_add_node(g,OP_NEG,inputs,1,outpus,1) == INVALID_ID){
        free(out);
        return NULL;
    }

    return out;
}



Tensor* tensor_log(Tensor* t,float a)
{
    if (!t) {
        fprintf(stderr, "ERROR: null tensor recieved\n");
        return NULL;
    }

    Graph* g = t->graph;
    TensorMeta* m = &g->tensors[t->id];

    Tensor* out = tensor_create(g,m->ndims,m->shape,m->dtype,m->requires_grad);

    if (!out) {
        fprintf(stderr, "ERROR: output tensor alloc failed\n");
        return NULL;
    }

    uint32_t inputs[1] = { t->id };
    uint32_t outpus[1] = {out->id};

    if (graph_add_node(g,OP_LOG,inputs,1,outpus,1) == INVALID_ID){
        free(out);
        return NULL;
    }

    g->nodes[g->node_count-1].scalar = a;

    return out;
}
