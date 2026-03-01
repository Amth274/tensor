#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "graph.h"
#include "tensor.h"
// #include "backend/cpu/cpu_backend.h"
// #include "elementwise_ops.h"
#include <assert.h>

int main() {
    // Create graph
Graph* g = graph_create(DEVICE_CPU, 0);

// Create tensors
uint32_t t0 = graph_add_tensor_meta(g, 1, (int64_t[]){1}, DTYPE_FP32, 0);
uint32_t t1 = graph_add_tensor_meta(g, 1, (int64_t[]){1}, DTYPE_FP32, 0);
uint32_t t2 = graph_add_tensor_meta(g, 1, (int64_t[]){1}, DTYPE_FP32, 0);

// Node0: produces t0 (input-like)
graph_add_node(g, OP_ADD, NULL, 0, &t0, 1);

    // Node1: consumes t0, produces t1
    uint32_t in1[] = {t0};
    uint32_t out1[] = {t1};
    graph_add_node(g, OP_ADD, in1, 1, out1, 1);

    // Node2: consumes t1, produces t2
    uint32_t in2[] = {t1};
    uint32_t out2[] = {t2};
    graph_add_node(g, OP_ADD, in2, 1, out2, 1);
    graph_topo_sort(g);
    for (int i = 0; i < g->execution_count; i++) {
        printf("%u ", g->execution_order[i]);
    }
    printf("\n");
    graph_destroy(g);
}