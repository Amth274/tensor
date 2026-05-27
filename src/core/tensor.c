#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "graph.h"
#include "tensor.h"
#include "memory.h"
// #include "dispatcher.h"


/*
===================================================================================================
*/

// create tensor class
Tensor* tensor_create(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad)
{
    if (!g || ndims <= 0 || !shape)
        return NULL;

    // Create symbolic tensor inside graph
    uint32_t id = graph_add_tensor_meta(g,ndims,shape,dtype,requires_grad);

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
