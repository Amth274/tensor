#ifndef DISPATCHER_H
#define DISPATCHER_H

typedef enum {
    //ELEMENTWISE KERNELS
    //BINARY KERNELS
    OP_ADD,
    OP_MUL,
    OP_SUB,
    OP_DIV,
    OP_COUNT
}OpType;

typedef enum {
    DTYPE_FP32,
    DTYPE_FP16,
    DTYPE_BF16,
    DTYPE_COUNT
} DType;

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_ROCM, 
    DEVICE_COUNT
} Device;

