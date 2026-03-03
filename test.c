#include <stdio.h>
#include <stdint.h>
#include "graph.h"

int main()
{
    Graph* g = graph_create(DEVICE_CPU, 0);
    if (!g) return 1;

    // Make tensors larger so difference is visible
    int64_t shape[] = {1024};  // 1024 * 4 bytes = 4096 bytes

    // -------- Chain A --------
    uint32_t t0 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t1 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t2 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);

    graph_add_node(g, OP_ADD, NULL, 0, &t0, 1);

    uint32_t in1[] = {t0};
    uint32_t out1[] = {t1};
    graph_add_node(g, OP_ADD, in1, 1, out1, 1);

    uint32_t in2[] = {t1};
    uint32_t out2[] = {t2};
    graph_add_node(g, OP_ADD, in2, 1, out2, 1);

    // -------- Chain B --------
    uint32_t t3 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t4 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);
    uint32_t t5 = graph_add_tensor_meta(g, 1, shape, DTYPE_FP32, 0);

    // graph_add_node(g, OP_ADD, NULL, 0, &t3, 1);
    uint32_t in3[] = {t2};
graph_add_node(g, OP_ADD, in3, 1, &t3, 1);

    uint32_t in4[] = {t3};
    uint32_t out4[] = {t4};
    graph_add_node(g, OP_ADD, in4, 1, out4, 1);

    uint32_t in5[] = {t4};
    uint32_t out5[] = {t5};
    graph_add_node(g, OP_ADD, in5, 1, out5, 1);

    // Run passes
    graph_topo_sort(g);
    graph_compute_lifetimes(g);

    size_t baseline = graph_simulate_peak_memory(g);
    graph_plan_memory(g);

    printf("Baseline Peak: %zu bytes\n", baseline);
    printf("Planned Arena Size: %zu bytes\n", g->arena_size);

    printf("\nTensor Lifetimes:\n");
    for (uint32_t i = 0; i < g->tensor_count; i++) {
        TensorMeta* t = &g->tensors[i];
        printf("Tensor %u: [%u, %u], offset=%zu\n",
               i, t->first_use, t->last_use, t->offset);
    }

    graph_destroy(g);
    return 0;
}