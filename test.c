#include <stdio.h>
#include <stdint.h>
#include "graph.h"

int main()
{
    // Create graph
    Graph* g = graph_create(DEVICE_CPU, 0);
    if (!g) {
        printf("Failed to create graph\n");
        return 1;
    }

    int64_t shape[] = {1};

    // Create tensors
    uint32_t t0 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t1 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t2 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);

    // Node0: produces t0
    graph_add_node(g, OP_ADD, NULL, 0, &t0, 1);

    // Node1: consumes t0 → produces t1
    uint32_t in1[] = {t0};
    uint32_t out1[] = {t1};
    graph_add_node(g, OP_ADD, in1, 1, out1, 1);

    // Node2: consumes t1 → produces t2
    uint32_t in2[] = {t1};
    uint32_t out2[] = {t2};
    graph_add_node(g, OP_ADD, in2, 1, out2, 1);

    // Run topo sort
    if (graph_topo_sort(g) != 0) {
        printf("Topo sort failed\n");
        graph_destroy(g);
        return 1;
    }

    printf("Execution Order:\n");
    for (uint32_t i = 0; i < g->execution_count; i++) {
        printf("%u ", g->execution_order[i]);
    }
    printf("\n\n");

    // Run lifetime analysis
    if (graph_compute_lifetimes(g) != 0) {
        printf("Lifetime analysis failed\n");
        graph_destroy(g);
        return 1;
    }

    printf("Tensor Lifetimes:\n");
    for (uint32_t i = 0; i < g->tensor_count; i++) {
        TensorMeta* t = &g->tensors[i];
        printf("Tensor %u: first_use = %u, last_use = %u\n",
               i, t->first_use, t->last_use);
    }

    graph_destroy(g);
    return 0;
}