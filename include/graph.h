#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"
#include "dispatcher.h"

/*
MACROS 
*/
#define INVALID_ID 0xFFFFFFFFu
#define MAX_INPUTS   8
#define MAX_OUTPUTS  4
#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32


// typedef int (*KernelFn)(Graph* g, Node* node);


// =======================
// OpType`
// ===========`============

// typedef enum {
//     OP_ADD,
//     OP_SUB,
//     OP_MUL,`
//     OP_DIV,
//     OP_COUNT
// // } OpType;


// // ============================================================================================================
// TensorMeta
// ============================================================================================================

typedef struct {
    uint32_t id;

    int32_t  ndims;
    int64_t* shape;
    int64_t* strides;

    DType dtype;

    // Classification
    uint8_t requires_grad;
    uint8_t is_parameter;
    uint8_t is_gradient;
    uint8_t is_activation;
    uint8_t is_optimizer_state;
    uint8_t is_input;
    uint8_t is_output;

    // Graph connectivity
    uint32_t producer; 
    uint32_t* consumers;
    uint32_t  num_consumers;
    uint32_t  consumer_capacity;

    // Size info
    size_t numel;
    size_t size_bytes;
    size_t aligned_size;

    // Memory planning
    size_t offset;
    uint32_t first_use;
    uint32_t last_use;

} TensorMeta;


// ===========================================================================================================
// Node
// ===========================================================================================================

typedef struct {
    uint32_t id;

    OpType op;

    uint32_t num_inputs;
    uint32_t inputs[MAX_INPUTS];

    uint32_t num_outputs;
    uint32_t outputs[MAX_OUTPUTS];

    uint32_t exec_index;
    uint8_t memory_class; // for params,acts,grads

    size_t aligned_size;

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

    uint32_t* grad_of;

    void* arena;
    size_t arena_size;

    uint32_t* node_refcount; // for backward accumulation tracking

    uint32_t* execution_order;
    uint32_t  execution_count;

    uint8_t compiled;
    uint8_t executed;

} Graph;


// Graph lifecycle
Graph* graph_create(Device device, int device_id);
void graph_destroy(Graph* g);

// IR building
uint32_t graph_add_tensor_meta(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad);
uint32_t graph_add_node(Graph* g,OpType op,uint32_t* inputs,uint32_t num_inputs,uint32_t* outputs,uint32_t num_outputs);
int graph_topo_sort(Graph* g);
int graph_lifetime_analysis(Graph* g);
size_t graph_simulate_peak_memory(Graph* g);
int graph_plan_memory(Graph* g);
// void graph_destroy(Graph* g);
// backward_graph 
int graph_validate(Graph* g);
int graph_allocate_arena(Graph* g);
int graph_compile(Graph* g);

int graph_execute(Graph* g);





#endif // GRAPH_H