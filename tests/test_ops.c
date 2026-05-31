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

    dispatcher_init();

    Graph* g = graph_create(DEVICE_CPU,0);

    if (!g) {
        printf("FAILED: graph_create\n");
        return -1;
    }

    int64_t shape[1] = {4};

    Tensor* a = tensor_create(g,1,shape,DTYPE_FP32,0);

    Tensor* b = tensor_create(g,1,shape,DTYPE_FP32,0);


    double temp = 2.0;
    Tensor* t_out = tensor_log(a,temp);


    if (graph_compile(g) != 0) {
        printf("FAILED: graph_compile\n");
        return -1;
    }

    float* a_ptr =
        (float*)get_tensor_ptr(g, a->id);



    a_ptr[0] = 1.0f;
    a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f;


    if (graph_execute(g) != 0) {
        printf("FAILED: graph_execute\n");
        return -1;
    }

    float* out_ptr = (float*)get_tensor_ptr(g,t_out->id);

    print_tensor("LOG", out_ptr, 4);


 
    graph_destroy(g);

    return 0;
}