#ifndef TYPES_H
#define TYPES_H



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
    DTYPE_FP64,
    DTYPE_COUNT
} DType;

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_NEG,
    OP_LOG,
    OP_COUNT
} Op;


typedef enum {
    OP_TYPE_CONTIGUOUS,
    OP_TYPE_STRIDED,
    OP_TYPE_BROADCASTED,
    OP_TYPE_COUNT
} OpType;


typedef enum {
    OP_KIND_UNARY,
    OP_KIND_BINARY,
    OP_KIND_REDUCTION,
    OP_KIND_COUNT
} OpKind;

#endif