#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

// =======================
// Device & DType
// =======================

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_ROCM, 
    DEVICE_COUNT
} Device;

typedef enum {
    DTYPE_FP32,
    DTYPE_FP16,
    DTYPE_BF16,
    DTYPE_COUNT
} DType;


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