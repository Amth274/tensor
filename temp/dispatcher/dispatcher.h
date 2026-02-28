#include "tensor.h"
#include "dispatch.h"
#include <stdint.h>

/*
===============================================================================
    Operation Types
===============================================================================
*/

typedef enum {
    OP_ADD = 0,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_COUNT
} OpType;

typedef enum {
    OP_FILL_NORMAL = 0,
    OP_FILL_UNIFORM,
    OP_FILL_CUSTOM_VALS,
    OP_FILL_COUNT
} OPFill;


/*
===============================================================================
    Kernel Function Types
===============================================================================
*/

// Binary op: out = f(a, b)
typedef void (*BinaryKernelFn)(Tensor* a, Tensor* b, Tensor* out);

// Unary op: out = f(a)
typedef void (*UnaryKernelFn)(Tensor* a, Tensor* out);

// Fill op
typedef void (*FillKernelFn)(Tensor* t, float arg1, float arg2);


/*
===============================================================================
    Dispatcher API
===============================================================================
*/

// Registration
void dispatch_register_binary(
    OpType op,
    Device device,
    DType dtype,
    BinaryKernelFn fn
);

// void dispatch_register_fill(
//     OPFill fill,
//     Device device,
//     DType dtype,
//     UnaryKernelFn fn
// );

void dispatch_register_unary(
    OpType op,
    Device device,
    DType dtype,
    UnaryKernelFn fn
);

void dispatch_register_fill(
    OpType op,
    Device device,
    DType dtype,
    FillKernelFn fn
);


// Lookup
BinaryKernelFn dispatch_lookup_binary(
    OpType op,
    Device device,
    DType dtype
);

UnaryKernelFn dispatch_lookup_unary(
    OpType op,
    Device device,
    DType dtype
);

FillKernelFn dispatch_lookup_fill(
    OpType OPFill,
    Device device,
    DType dtype
);

#ifdef __cplusplus
}
#endif

// #endif 