#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "graph.h"
#include "tensor.h"
// #include "backend/cpu/cpu_backend.h"
// #include "elementwise_ops.h"
#include <assert.h>

int main() {
    Graph* g = graph_create(DEVICE_CPU, 0);

    int64_t shape[2] = {2,3};

    Tensor* a = tensor_create(g, 2, shape, DTYPE_FP32, 0);

    printf("tensor created\n");

    free(a);
    graph_destroy(g);

    return 0;
}