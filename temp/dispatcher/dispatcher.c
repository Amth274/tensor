#include <stdio.h>
#include <stdlib.h>
#include "dispatcher.h"

/*
=============================================================================== 
    Static Dispatch Tables
===============================================================================
*/

static BinaryKernelFn
binary_table[OP_COUNT][DEVICE_COUNT][DTYPE_COUNT] = {0};

static UnaryKernelFn
unary_table[OP_COUNT][DEVICE_COUNT][DTYPE_COUNT] = {0};

static FillKernelFn
fill_table[OP_FILL_COUNT][DEVICE_COUNT][DTYPE_COUNT] = {0};


/*
===============================================================================
    Registration
===============================================================================
*/

void dispatch_register_binary(
    OpType op,
    Device device,
    DType dtype,
    BinaryKernelFn fn
)
{
    binary_table[op][device][dtype] = fn;
}

void dispatch_register_unary(
    OpType op,
    Device device,
    DType dtype,
    UnaryKernelFn fn
)
{
    unary_table[op][device][dtype] = fn;
}

// void dispatch_register_fill(
//     OPFill fill,
//     Device device,
//     DType dtype,
//     FillKernelFn fn
// )
// {
//     fill_table[fill][device][dtype] = fn;
// }


/*
===============================================================================
    Lookup
===============================================================================
*/

BinaryKernelFn dispatch_lookup_binary(
    OpType op,
    Device device,
    DType dtype
)
{
    return binary_table[op][device][dtype];
}

UnaryKernelFn dispatch_lookup_unary(
    OpType op,
    Device device,
    DType dtype
)
{
    return unary_table[op][device][dtype];
}

// FillKernelFn dispatch_lookup_fill(
//     OPFill fill,
//     Device device,
//     DType dtype
// )
// {
//     return fill_table[fill][device][dtype];
// }