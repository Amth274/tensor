#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H

#include "backend/cpu/cpu_backend.h"
#include "tensor.h"



typedef struct {

    int64_t ndim;
    int64_t numel;
    const int64_t* shape;
    const int64_t* stride_a;
    const int64_t* stride_b;
    const int64_t* stride_out;
    void* ptr_a;
    void* ptr_b;
    void* ptr_out;
    size_t elem_size;
    int contiguous;
    int parallelizable;
    int vectorizable;

} TensorIterator;

TensorIterator tensor_iterator(
    Tensor* a,
    Tensor* b,
    Tensor* out
);

#endif