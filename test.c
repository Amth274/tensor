#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "ops.h"
#include "tensor.h"

int main(void)
{
    Graph* g = graph_create(DEVICE_CPU, 0);

    int64_t shape[] = {4, 5};

    Tensor* a = tensor_create(g, 2, shape, DTYPE_FP32, 0);
    Tensor* b = tensor_create(g, 2, shape, DTYPE_FP32, 0);

    Tensor* c = tensor_add(a, b);

    graph_compile(g);
    graph_execute(g);

    printf("Created: c = a + b\n");
    printf("a tensor id: %u\n", a->id);
    printf("b tensor id: %u\n", b->id);
    printf("c tensor id: %u\n", c->id);
    printf("nodes: %u\n", g->node_count);
    printf("tensors: %u\n", g->tensor_count);
    printf("compiled: %u\n", g->compiled);
    printf("executed: %u\n", g->executed);
    printf("arena size: %zu bytes\n", g->arena_size);

    free(a);
    free(b);
    free(c);
    graph_destroy(g);

    return 0;
}
