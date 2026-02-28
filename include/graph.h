#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"

#define MAX_INPUTS   8
#define MAX_OUTPUTS  4
#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32



// =======================
// OpType
// =======================

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_COUNT
} OpType;


// =======================
// TensorMeta
// =======================

typedef struct {
    uint32_t id;

    int32_t  ndims;
    int64_t* shape;
    int64_t* strides;

    DType dtype;

    int32_t producer;

    size_t numel;
    size_t size_bytes;

    size_t offset;
    uint32_t first_use;
    uint32_t last_use;

    uint8_t is_input;
    uint8_t is_output;
    uint8_t requires_grad;

} TensorMeta;


// =======================
// Node
// =======================

typedef struct {
    uint32_t id;

    OpType op;

    uint32_t num_inputs;
    uint32_t inputs[MAX_INPUTS];

    uint32_t num_outputs;
    uint32_t outputs[MAX_OUTPUTS];

    void* kernel_cache;
    uint8_t is_fused;

} Node;


// =======================
// Graph
// =======================

typedef struct Graph {

    Node* nodes;
    uint32_t node_count;
    uint32_t node_capacity;

    TensorMeta* tensors;
    uint32_t tensor_count;
    uint32_t tensor_capacity;

    Device device;
    int device_id;

    void* arena;
    size_t arena_size;

    uint8_t compiled;
    uint8_t executed;

} Graph;


// Graph lifecycle
Graph* graph_create(Device device, int device_id);
void graph_destroy(Graph* g);

// IR building
uint32_t graph_add_tensor_meta(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad);

uint32_t graph_add_node(Graph* g,OpType op,uint32_t* inputs,uint32_t num_inputs,uint32_t* outputs,uint32_t num_outputs);

void graph_destroy(Graph* g);




#endif // GRAPH_H