#ifndef DISPATCH_H
#define DISPATCH_H

#include "tensor.h"
#include <stdio.h>

void add_dispatch(Tensor* t1,Tensor* t2,Tensor* result);
void fill_dispatch(Tensor* t,float avg,float var);

#endif //DISPATCH_H