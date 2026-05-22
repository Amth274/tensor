#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "ops/ops.h"
#include "tensor.h"

static int check(int condition, const char* message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return -1;
    }

    printf("PASS: %s\n", message);
    return 0;
}

int main(void)
{
    Graph* g = graph_create(DEVICE_CPU, 0);
    if (!g) {
        fprintf(stderr, "FAIL: graph_create returned NULL\n");
        return 1;
    }

    int status = 0;
    int64_t shape[] = {4,5};

    Tensor* a = tensor_create(g, 2, shape, DTYPE_FP32, 0);
    Tensor* b = tensor_create(g, 2, shape, DTYPE_FP32, 1);
    Tensor* c = tensor_add(a, b);

    status |= check(a != NULL, "tensor_create creates input a");
    status |= check(b != NULL, "tensor_create creates input b");
    status |= check(c != NULL, "tensor_add returns output tensor");

    if (status == 0) {
        status |= check(g->tensor_count == 3, "tensor_add creates one output tensor");
        status |= check(g->node_count == 1, "tensor_add creates one graph node");

        Node* node = &g->nodes[0];
        TensorMeta* a_meta = &g->tensors[a->id];
        TensorMeta* b_meta = &g->tensors[b->id];
        TensorMeta* c_meta = &g->tensors[c->id];

        status |= check(node->op == OP_ADD, "created node op is OP_ADD");
        status |= check(node->num_inputs == 2, "created node has two inputs");
        status |= check(node->num_outputs == 1, "created node has one output");
        status |= check(node->inputs[0] == a->id, "node input 0 is tensor a");
        status |= check(node->inputs[1] == b->id, "node input 1 is tensor b");
        status |= check(node->outputs[0] == c->id, "node output is tensor c");

        status |= check(a_meta->producer == INVALID_ID, "input a has no producer");
        status |= check(b_meta->producer == INVALID_ID, "input b has no producer");
        status |= check(c_meta->producer == node->id, "output c producer is add node");
        status |= check(a_meta->num_consumers == 1, "input a has one consumer");
        status |= check(b_meta->num_consumers == 1, "input b has one consumer");
        status |= check(a_meta->consumers[0] == node->id, "input a consumer is add node");
        status |= check(b_meta->consumers[0] == node->id, "input b consumer is add node");

        status |= check(c_meta->requires_grad == 1, "output requires_grad propagates");
        status |= check(c_meta->dtype == DTYPE_FP32, "output dtype matches inputs");
        status |= check(c_meta->ndims == 2, "output ndims matches inputs");
        status |= check(c_meta->shape[0] == 4, "output shape dim 0 matches inputs");
        status |= check(c_meta->shape[1] == 5, "output shape dim 1 matches inputs");
    }

    if (status == 0) {
        status |= check(graph_compile(g) == 0, "graph_compile succeeds after tensor_add");
        status |= check(g->compiled == 1, "graph compiled flag is set");
        status |= check(g->execution_count == 1, "execution order has one node");
        status |= check(g->arena != NULL, "arena is allocated");
    }

    free(a);
    free(b);
    free(c);
    graph_destroy(g);

    if (status != 0) {
        fprintf(stderr, "tensor_add test failed\n");
        return 1;
    }

    printf("tensor_add test passed\n");
    return 0;
}
