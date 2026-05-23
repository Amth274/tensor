#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "graph.h"
#include "tensor.h"
#include "ops.h"
#include "dispatcher.h"

/*
=========================================================
Temporary helper
=========================================================
*/

static void* get_tensor_ptr(Graph* g, uint32_t tensor_id)
{
    TensorMeta* t = &g->tensors[tensor_id];
    return (uint8_t*)g->arena + t->offset;
}

/*
=========================================================
Main
=========================================================
*/

int main()
{
    /*
    -----------------------------------------------------
    Init dispatcher
    -----------------------------------------------------
    */

    dispatcher_init();

    /*
    -----------------------------------------------------
    Create graph
    -----------------------------------------------------
    */

    Graph* g = graph_create(DEVICE_CPU, 0);

    if (!g) {
        printf("FAILED: graph_create\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Create tensors
    -----------------------------------------------------
    */

    int64_t shape[1] = {4};

    Tensor* a = tensor_create(
        g,
        1,
        shape,
        DTYPE_FP32,
        0
    );

    Tensor* b = tensor_create(
        g,
        1,
        shape,
        DTYPE_FP32,
        0
    );

    if (!a || !b) {
        printf("FAILED: tensor_create\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Add operation
    -----------------------------------------------------
    */

    Tensor* out = tensor_add(a, b);

    if (!out) {
        printf("FAILED: tensor_add\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Compile graph
    -----------------------------------------------------
    */

    if (graph_compile(g) != 0) {
        printf("FAILED: graph_compile\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Get tensor memory
    -----------------------------------------------------
    */

    float* a_ptr = (float*)get_tensor_ptr(g, a->id);
    float* b_ptr = (float*)get_tensor_ptr(g, b->id);
    float* out_ptr = (float*)get_tensor_ptr(g, out->id);

    /*
    -----------------------------------------------------
    Fill inputs
    -----------------------------------------------------
    */

    a_ptr[0] = 1.0f;
    a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f;

    b_ptr[0] = 10.0f;
    b_ptr[1] = 20.0f;
    b_ptr[2] = 30.0f;
    b_ptr[3] = 40.0f;

    
    // -----------------------------------------------------
    // Validate output
    // -----------------------------------------------------
    // */

    graph_execute(g);

    printf("Output:\n");

    for (int i = 0; i < 4; i++) {
        printf("%f\n", out_ptr[i]);
    }

    /*
    Expected:
    11
    22
    33
    44
    */


    /*
    -----------------------------------------------------
    Cleanup
    -----------------------------------------------------
    */

    graph_destroy(g);

    free(a);
    free(b);
    free(out);

    return 0;
}