#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "ops.h"
#include "tensor.h"

static int same_shape(const TensorMeta* a, const TensorMeta* b)
{
    if (a->ndims != b->ndims)
        return 0;

    for (int32_t i = 0; i < a->ndims; i++) {
        if (a->shape[i] != b->shape[i])
            return 0;
    }

    return 1;
}

Tensor* tensor_add(Tensor* t1, Tensor* t2)
{
    if (!t1 || !t2) {
        fprintf(stderr, "ERROR: null tensor received\n");
        return NULL;
    }

    if (!t1->graph || t1->graph != t2->graph) {
        fprintf(stderr, "ERROR: tensors must belong to the same graph\n");
        return NULL;
    }

    Graph* g = t1->graph;

    if (t1->id >= g->tensor_count || t2->id >= g->tensor_count) {
        fprintf(stderr, "ERROR: tensor id does not exist in graph\n");
        return NULL;
    }

    TensorMeta* m1 = &g->tensors[t1->id];
    TensorMeta* m2 = &g->tensors[t2->id];

    if (m1->dtype != m2->dtype) {
        fprintf(stderr, "ERROR: tensor_add requires matching dtypes for now\n");
        return NULL;
    }

    if (!same_shape(m1, m2)) {
        fprintf(stderr, "ERROR: tensor_add requires matching shapes for now\n");
        return NULL;
    }

    uint8_t requires_grad = m1->requires_grad || m2->requires_grad;
    Tensor* out = tensor_create(g, m1->ndims, m1->shape, m1->dtype, requires_grad);
    if (!out)
        return NULL;

    uint32_t inputs[] = {t1->id, t2->id};
    uint32_t outputs[] = {out->id};

    if (graph_add_node(g, OP_ADD, inputs, 2, outputs, 1) == INVALID_ID) {
        free(out);
        return NULL;
    }

    return out;
}


