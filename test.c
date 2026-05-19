#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "tensor.h"

static int require_tensor(Tensor* t, const char* name)
{
    if (!t || !t->graph || t->id == INVALID_ID) {
        fprintf(stderr, "failed to create %s\n", name);
        return -1;
    }

    return 0;
}

static void print_execution_order(Graph* g)
{
    printf("Execution order:");
    for (uint32_t i = 0; i < g->execution_count; i++) {
        printf(" %u", g->execution_order[i]);
    }
    printf("\n");
}

static void print_tensor_table(Graph* g)
{
    printf("\nTensor table:\n");
    printf("id | producer | consumers | lifetime | offset | bytes\n");
    printf("---+----------+-----------+----------+--------+------\n");

    for (uint32_t i = 0; i < g->tensor_count; i++) {
        TensorMeta* t = &g->tensors[i];

        if (t->producer == INVALID_ID) {
            printf("%2u | input    ", i);
        } else {
            printf("%2u | %8u", i, t->producer);
        }

        printf(" | %9u | [%u, %u]   | %6zu | %zu\n",
               t->num_consumers,
               t->first_use,
               t->last_use,
               t->offset,
               t->size_bytes);
    }
}

int main(void)
{
    Graph* g = graph_create(DEVICE_CPU, 0);
    if (!g) {
        fprintf(stderr, "failed to create graph\n");
        return 1;
    }

    int64_t shape[] = {1024};

    Tensor* a = tensor_create(g, 1, shape, DTYPE_FP32, 0);
    Tensor* b = tensor_create(g, 1, shape, DTYPE_FP32, 0);
    Tensor* c = tensor_create(g, 1, shape, DTYPE_FP32, 0);
    Tensor* d = tensor_create(g, 1, shape, DTYPE_FP32, 0);
    Tensor* e = tensor_create(g, 1, shape, DTYPE_FP32, 0);
    Tensor* f = tensor_create(g, 1, shape, DTYPE_FP32, 0);

    if (require_tensor(a, "tensor a") != 0 ||
        require_tensor(b, "tensor b") != 0 ||
        require_tensor(c, "tensor c") != 0 ||
        require_tensor(d, "tensor d") != 0 ||
        require_tensor(e, "tensor e") != 0 ||
        require_tensor(f, "tensor f") != 0) {
        free(a);
        free(b);
        free(c);
        free(d);
        free(e);
        free(f);
        graph_destroy(g);
        return 1;
    }

    uint32_t add0_inputs[] = {a->id, b->id};
    uint32_t add0_outputs[] = {c->id};
    uint32_t n0 = graph_add_node(g, OP_ADD, add0_inputs, 2, add0_outputs, 1);

    uint32_t add1_inputs[] = {c->id, b->id};
    uint32_t add1_outputs[] = {d->id};
    uint32_t n1 = graph_add_node(g, OP_ADD, add1_inputs, 2, add1_outputs, 1);

    uint32_t mul0_inputs[] = {c->id, d->id};
    uint32_t mul0_outputs[] = {e->id};
    uint32_t n2 = graph_add_node(g, OP_MUL, mul0_inputs, 2, mul0_outputs, 1);

    uint32_t sub0_inputs[] = {e->id, a->id};
    uint32_t sub0_outputs[] = {f->id};
    uint32_t n3 = graph_add_node(g, OP_SUB, sub0_inputs, 2, sub0_outputs, 1);

    if (n0 == INVALID_ID || n1 == INVALID_ID ||
        n2 == INVALID_ID || n3 == INVALID_ID) {
        fprintf(stderr, "failed to add one or more nodes\n");
        free(a);
        free(b);
        free(c);
        free(d);
        free(e);
        free(f);
        graph_destroy(g);
        return 1;
    }

    if (graph_compile(g) != 0) {
        fprintf(stderr, "graph_compile failed\n");
        free(a);
        free(b);
        free(c);
        free(d);
        free(e);
        free(f);
        graph_destroy(g);
        return 1;
    }

    size_t peak = graph_simulate_peak_memory(g);

    printf("Graph compiled: %u\n", g->compiled);
    printf("Nodes: %u\n", g->node_count);
    printf("Tensors: %u\n", g->tensor_count);
    printf("Simulated peak live memory: %zu bytes\n", peak);
    printf("Planned arena size: %zu bytes\n", g->arena_size);
    printf("Arena allocated: %s\n", g->arena ? "yes" : "no");

    print_execution_order(g);
    print_tensor_table(g);

    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
    graph_destroy(g);
    return 0;
}
