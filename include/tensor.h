#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include "types.h"
// =======================
// Device & DType
// =======================




// Forward declaration
typedef struct Graph Graph;


// =======================
// Public Tensor Handle
// =======================

typedef struct Tensor {
    Graph* graph;     // Owning graph
    uint32_t id;      // Index into graph->tensors
} Tensor;


// =======================
// Tensor Creation
// =======================

Tensor* tensor_create(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad);

#endif // TENSOR_H