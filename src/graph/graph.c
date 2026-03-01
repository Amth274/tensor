#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include <stdio.h>

#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32

Graph* graph_create(Device device, int device_id)
{
    Graph* g = (Graph*)malloc(sizeof(Graph));
    if (!g) return NULL;


    // Zero entire struct for safety
    memset(g, 0, sizeof(Graph));


    // Initialize device
    g->device = device;
    g->device_id = device_id;

    // Allocate node array
    g->node_capacity = INITIAL_NODE_CAPACITY;
    g->nodes = (Node*)malloc(sizeof(Node) * g->node_capacity);
    if (!g->nodes) {
        free(g);
        return NULL;
    }

    // Allocate tensor array
    g->tensor_capacity = INITIAL_TENSOR_CAPACITY;
    g->tensors = (TensorMeta*)malloc(sizeof(TensorMeta) * g->tensor_capacity);
    if (!g->tensors) {
        free(g->nodes);
        free(g);
        return NULL;
    }

    // Execution state
    g->arena = NULL;
    g->arena_size = 0;

    g->compiled = 0;
    g->executed = 0;
    

    return g;
}


void graph_destroy(Graph* g)
{
    if (!g) return;

    // Free tensor shapes
    for (uint32_t i = 0; i < g->tensor_count; i++) {
        free(g->tensors[i].shape);
        free(g->tensors[i].strides);
    }

    free(g->nodes);
    free(g->tensors);

    if (g->arena) {
        // replace with device_free later
        free(g->arena);
    }

    free(g);
}

static size_t dtype_size(DType dtype)
{
    switch (dtype) {
        case DTYPE_FP32: return 4;
        case DTYPE_FP16: return 2;
        case DTYPE_BF16: return 2;
        default: return 0;
    }
}


uint32_t graph_add_tensor_meta(Graph* g,int32_t ndims,const int64_t* shape,DType dtype,uint8_t requires_grad)
{
    if (!g || ndims <= 0 || !shape){
        return UINT32_MAX;
    }

    // Grow tensor array if needed
    if (g->tensor_count == g->tensor_capacity) {
        uint32_t new_capacity = g->tensor_capacity * 2;
        TensorMeta* new_tensors = (TensorMeta*)realloc(
            g->tensors,
            sizeof(TensorMeta) * new_capacity
        );

        if (!new_tensors)
            return UINT32_MAX;

        g->tensors = new_tensors;
        g->tensor_capacity = new_capacity;
    }



    uint32_t id = g->tensor_count;
    TensorMeta* t = &g->tensors[id];

    // Initialize
    t->id = id;
    t->ndims = ndims;
    t->dtype = dtype;
    t->requires_grad = requires_grad;

    t->is_input = 0;
    t->is_output = 0;

    t->offset = 0;
    t->first_use = 0;
    t->last_use = 0;

    t->producer = INVALID_ID;
    t->consumers = NULL;

    // Allocate shape
    t->shape = (int64_t*)malloc(sizeof(int64_t) * ndims);
    t->strides = (int64_t*)malloc(sizeof(int64_t) * ndims);

    if (!t->shape || !t->strides){
        return UINT32_MAX;
    }

    printf("shape allocated\n");

    // Copy shape
    size_t numel = 1;
    for (int32_t i = 0; i < ndims; i++) {
        t->shape[i] = shape[i];
        numel *= shape[i];
    }

    t->numel = numel;
    t->size_bytes = numel * dtype_size(dtype);

    // Compute contiguous strides
    t->strides[ndims - 1] = 1;
    for (int32_t i = ndims - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }

    g->tensor_count++;

    return id;
}


uint32_t graph_add_node(Graph* g,OpType op,uint32_t* inputs,uint32_t num_inputs,uint32_t* outputs,uint32_t num_outputs)
{
    if (!g || num_inputs > MAX_INPUTS || num_outputs > MAX_OUTPUTS)
        return UINT32_MAX;

    // Grow node array if needed
    if (g->node_count == g->node_capacity) {
        uint32_t new_capacity = g->node_capacity * 2;
        Node* new_nodes = (Node*)realloc(
            g->nodes,
            sizeof(Node) * new_capacity
        );

        if (!new_nodes)
            return UINT32_MAX;

        g->nodes = new_nodes;
        g->node_capacity = new_capacity;
    }

    uint32_t id = g->node_count;
    Node* n = &g->nodes[id];

    n->id = id;
    n->op = op;

    n->num_inputs = num_inputs;
    n->num_outputs = num_outputs;

    // Copy inputs
    for (uint32_t i = 0; i < num_inputs; i++) {
        n->inputs[i] = inputs[i];

        // Update last_use placeholder
        // g->tensors[inputs[i]].last_use = id;
    }

    // Copy outputs
    for (uint32_t i = 0; i < num_outputs; i++) {
        n->outputs[i] = outputs[i];

        // Set producer
        g->tensors[outputs[i]].producer = id;
        // g->tensors[outputs[i]].first_use = id;
    }

    n->kernel_cache = NULL;
    n->is_fused = 0;

    g->node_count++;

    return id;
}

int graph_topo_sort(Graph* g){
    
}