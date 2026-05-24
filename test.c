#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "graph.h"
#include "tensor.h"
#include "ops.h"
#include "dispatcher.h"

/*
=========================================================
Helper
=========================================================
*/

static void* get_tensor_ptr(
    Graph* g,
    uint32_t tensor_id
)
{
    TensorMeta* t = &g->tensors[tensor_id];

    return (uint8_t*)g->arena + t->offset;
}

/*
=========================================================
Print helper
=========================================================
*/

static void print_tensor(
    const char* name,
    float* data,
    uint32_t numel
)
{
    printf("%s:\n", name);

    for(uint32_t i = 0; i < numel; i++) {
        printf("%f\n", data[i]);
    }

    printf("\n");
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

    Graph* g = graph_create(
        DEVICE_CPU,
        0
    );

    if (!g) {
        printf("FAILED: graph_create\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Shape
    -----------------------------------------------------
    */

    int64_t shape[1] = {4};

    /*
    -----------------------------------------------------
    Create input tensors
    -----------------------------------------------------
    */

    Tensor* a = tensor_create(
        g,
        1,
        shape,
        DTYPE_FP64,
        0
    );

    Tensor* b = tensor_create(
        g,
        1,
        shape,
        DTYPE_FP64,
        0
    );

    if (!a || !b) {
        printf("FAILED: tensor_create\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Create ops
    -----------------------------------------------------
    */

    Tensor* add_out = tensor_add(a, b);
    Tensor* mul_out = tensor_mul(a, b);
    Tensor* sub_out = tensor_sub(a, b);
    Tensor* div_out = tensor_div(a, b);

    if (!add_out ||
        !mul_out ||
        !sub_out ||
        !div_out)
    {
        printf("FAILED: op creation\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Compile graph
    -----------------------------------------------------
    */

    g->tensors[add_out->id].is_output = 1;
    g->tensors[mul_out->id].is_output = 1;
    g->tensors[sub_out->id].is_output = 1;
    g->tensors[div_out->id].is_output = 1;
    
    if (graph_compile(g) != 0) {
        printf("FAILED: graph_compile\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Get tensor memory
    -----------------------------------------------------
    */

    float* a_ptr =
        (float*)get_tensor_ptr(g, a->id);

    float* b_ptr =
        (float*)get_tensor_ptr(g, b->id);

    float* add_ptr =
        (float*)get_tensor_ptr(g, add_out->id);

    float* mul_ptr =
        (float*)get_tensor_ptr(g, mul_out->id);

    float* sub_ptr =
        (float*)get_tensor_ptr(g, sub_out->id);

    float* div_ptr =
        (float*)get_tensor_ptr(g, div_out->id);

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

    /*
    -----------------------------------------------------
    Execute graph
    -----------------------------------------------------
    */

    if (graph_execute(g) != 0) {
        printf("FAILED: graph_execute\n");
        return -1;
    }

    /*
    -----------------------------------------------------
    Print outputs
    -----------------------------------------------------
    */

    print_tensor("ADD", add_ptr, 4);
    print_tensor("MUL", mul_ptr, 4);
    print_tensor("SUB", sub_ptr, 4);
    print_tensor("DIV", div_ptr, 4);

    /*
    -----------------------------------------------------
    Validation
    -----------------------------------------------------
    */

    int success = 1;

    /*
    ADD
    */

    if (add_ptr[0] != 11.0f) success = 0;
    if (add_ptr[1] != 22.0f) success = 0;
    if (add_ptr[2] != 33.0f) success = 0;
    if (add_ptr[3] != 44.0f) success = 0;

    /*
    MUL
    */

    if (mul_ptr[0] != 10.0f) success = 0;
    if (mul_ptr[1] != 40.0f) success = 0;
    if (mul_ptr[2] != 90.0f) success = 0;
    if (mul_ptr[3] != 160.0f) success = 0;

    /*
    SUB
    */

    if (sub_ptr[0] != -9.0f) success = 0;
    if (sub_ptr[1] != -18.0f) success = 0;
    if (sub_ptr[2] != -27.0f) success = 0;
    if (sub_ptr[3] != -36.0f) success = 0;

    /*
    DIV
    */

    if (div_ptr[0] != 0.1f) success = 0;
    if (div_ptr[1] != 0.1f) success = 0;
    if (div_ptr[2] != 0.1f) success = 0;
    if (div_ptr[3] != 0.1f) success = 0;

    /*
    -----------------------------------------------------
    Final result
    -----------------------------------------------------
    */

    if (success) {
        printf("ALL TESTS PASSED\n");
    }
    else {
        printf("TEST FAILED\n");
    }

    /*
    -----------------------------------------------------
    Cleanup
    -----------------------------------------------------
    */

    graph_destroy(g);

    free(a);
    free(b);

    free(add_out);
    free(mul_out);
    free(sub_out);
    free(div_out);

    return 0;
}