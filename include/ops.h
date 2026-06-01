#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// elementwise ops 
Tensor* tensor_add(Tensor* t1, Tensor* t2);
Tensor* tensor_mul(Tensor* t1, Tensor* t2);
Tensor* tensor_div(Tensor* t1, Tensor* t2);
Tensor* tensor_sub(Tensor* t1, Tensor* t2);
Tensor* tensor_neg(Tensor* t);
Tensor* tensor_log(Tensor* t,float a);
Tensor* tensor_mat_mul(Tensor* t1, Tensor* t2);

#endif // OPS_H
