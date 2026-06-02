#ifndef DISPATCH_KEY_H
#define DISPATCH_KEY_H

#include <stdint.h>
#include "runtime.h"
#include "tensor.h"



typedef struct {
    Device device;
    Op op;
    DType dtype;
    uint8_t contiguous;
    isa_t isa;
    
} DispatchKey;

#endif
