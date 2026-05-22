#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "graph.h"
#include "tensor.h"
#include "memory.h"
// #include "dispatcher.h"

//helper function 
static void compute_contiguous_strides(
    int32_t ndims,
    const int64_t* shape,
    int64_t* strides)
{
    if (ndims == 0) return;

    strides[ndims - 1] = 1;
    for (int32_t i = ndims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static size_t compute_numel(int32_t ndims, const int64_t* shape) {
    size_t numel = 1;
    for (int32_t i = 0; i < ndims; i++) {
        numel *= shape[i];
    }
    return numel;
}

static size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FP32: return 4;
        case DTYPE_FP16: return 2;
        case DTYPE_BF16: return 2;
        default: return 0;
    }
}

/*
===================================================================================================
*/

// create tensor class
Tensor* tensor_create(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad)
{
    if (!g || ndims <= 0 || !shape)
        return NULL;

    // Create symbolic tensor inside graph
    uint32_t id = graph_add_tensor_meta(
        g,
        ndims,
        shape,
        dtype,
        requires_grad
    );

    if (id == UINT32_MAX)
        return NULL;

        

    // Allocate tensor handle
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->graph = g;
    t->id = id;

    return t;
}
