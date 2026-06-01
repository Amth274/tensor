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
    int64_t mat_a_shape[2] = {2, 3};
    int64_t mat_b_shape[2] = {3, 2};

    Tensor* a = tensor_create(g,1,shape,DTYPE_FP32,0);

    Tensor* mat_a = tensor_create(g,2,mat_a_shape,DTYPE_FP32,0);
    Tensor* mat_b = tensor_create(g,2,mat_b_shape,DTYPE_FP32,0);


    double temp = 2.0;
    Tensor* t_out = tensor_log(a,temp);
    Tensor* mat_out = tensor_mat_mul(mat_a, mat_b);

    g->tensors[t_out->id].is_output = 1;
    g->tensors[mat_out->id].is_output = 1;


    if (graph_compile(g) != 0) {
        printf("FAILED: graph_compile\n");
        return -1;
    }

    float* a_ptr =
        (float*)get_tensor_ptr(g, a->id);
    float* mat_a_ptr =
        (float*)get_tensor_ptr(g, mat_a->id);
    float* mat_b_ptr =
        (float*)get_tensor_ptr(g, mat_b->id);



    a_ptr[0] = 1.0f;
    a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f;

    mat_a_ptr[0] = 1.0f;
    mat_a_ptr[1] = 2.0f;
    mat_a_ptr[2] = 3.0f;
    mat_a_ptr[3] = 4.0f;
    mat_a_ptr[4] = 5.0f;
    mat_a_ptr[5] = 6.0f;

    mat_b_ptr[0] = 7.0f;
    mat_b_ptr[1] = 8.0f;
    mat_b_ptr[2] = 9.0f;
    mat_b_ptr[3] = 10.0f;
    mat_b_ptr[4] = 11.0f;
    mat_b_ptr[5] = 12.0f;


    if (graph_execute(g) != 0) {
        printf("FAILED: graph_execute\n");
        return -1;
    }

    float* out_ptr = (float*)get_tensor_ptr(g,t_out->id);
    float* mat_out_ptr = (float*)get_tensor_ptr(g,mat_out->id);

    print_tensor("LOG", out_ptr, 4);
    print_tensor("MATMUL", mat_out_ptr, 4);

    if (mat_out_ptr[0] != 58.0f ||
        mat_out_ptr[1] != 64.0f ||
        mat_out_ptr[2] != 139.0f ||
        mat_out_ptr[3] != 154.0f) {
        printf("FAILED: tensor_mat_mul\n");
        return -1;
    }


 
    graph_destroy(g);
    free(a);
    free(mat_a);
    free(mat_b);
    free(t_out);
    free(mat_out);

    return 0;
}
