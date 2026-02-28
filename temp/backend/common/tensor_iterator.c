#include "tensor.h"
#include "tensor_iterator.h"

TensorIterator tensor_iterator(
    Tensor* a,
    Tensor* b,
    Tensor* out
)
{
    TensorIterator it;

    // it.ndim = a->ndims;
    // it.numel = a->numel;
    // it.shape = a->shape;

    // it.stride_a   = a->strides;
    // it.stride_b   = b->strides;
    // it.stride_out = out->strides;

    // it.ptr_a   = a->storage->data;
    // it.ptr_b   = b->storage->data;
    // it.ptr_out = out->storage->data;

    // it.elem_size = sizeof(a->dtype);

    it.contiguous = 0;
        // a->is_contiguous &&
        // b->is_contiguous &&
        // out->is_contiguous;

    return ;
}