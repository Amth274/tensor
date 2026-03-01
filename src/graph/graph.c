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
    free(g->tensors[i].consumers);   // <-- THIS is the missing one
}

    free(g->nodes);
    free(g->execution_order);
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
    t->num_consumers = 0;
    t->consumer_capacity = 0;
    

    // Allocate shape
    t->shape = (int64_t*)malloc(sizeof(int64_t) * ndims);
    t->strides = (int64_t*)malloc(sizeof(int64_t) * ndims);

    if (!t->shape || !t->strides){
        return UINT32_MAX;
    }

    // printf("shape allocated\n");

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


uint32_t graph_add_node(Graph* g,
                        OpType op,
                        uint32_t* inputs,
                        uint32_t num_inputs,
                        uint32_t* outputs,
                        uint32_t num_outputs)
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

    // ---- Copy inputs + register consumers ----
    for (uint32_t i = 0; i < num_inputs; i++) {
        uint32_t tensor_id = inputs[i];
        n->inputs[i] = tensor_id;

        TensorMeta* t = &g->tensors[tensor_id];

        // Grow consumer list if needed
        if (t->num_consumers == t->consumer_capacity) {
            uint32_t new_cap = (t->consumer_capacity == 0)
                ? 4
                : t->consumer_capacity * 2;

            uint32_t* new_arr = (uint32_t*)realloc(
                t->consumers,
                new_cap * sizeof(uint32_t)
            );

            if (!new_arr)
                return UINT32_MAX;

            t->consumers = new_arr;
            t->consumer_capacity = new_cap;
        }

        t->consumers[t->num_consumers++] = id;
    }

    // ---- Copy outputs + set producer ----
    for (uint32_t i = 0; i < num_outputs; i++) {
        uint32_t tensor_id = outputs[i];
        n->outputs[i] = tensor_id;

        g->tensors[tensor_id].producer = id;
    }

    n->kernel_cache = NULL;
    n->is_fused = 0;

    g->node_count++;

    return id;
}

int graph_topo_sort(Graph* g)
{
    if (!g || g->node_count == 0)
        return -1;

    uint32_t node_count = g->node_count;

    // Allocate execution order array
    if (g->execution_order)
        free(g->execution_order);

    g->execution_order = (uint32_t*)malloc(sizeof(uint32_t) * node_count);
    if (!g->execution_order)
        return -1;

    g->execution_count = 0;

    // Allocate indegree array
    uint32_t* indegree = (uint32_t*)calloc(node_count, sizeof(uint32_t));
    if (!indegree) {
        free(g->execution_order);
        g->execution_order = NULL;
        return -1;
    }

    // Compute indegree for each node
    for (uint32_t n = 0; n < node_count; n++) {
        Node* node = &g->nodes[n];

        for (uint32_t i = 0; i < node->num_inputs; i++) {
            uint32_t tensor_id = node->inputs[i];
            TensorMeta* t = &g->tensors[tensor_id];

            if (t->producer != INVALID_ID) {
                indegree[n]++;
            }
        }
    }

    // Allocate queue (max size = node_count)
    uint32_t* queue = (uint32_t*)malloc(sizeof(uint32_t) * node_count);
    if (!queue) {
        free(indegree);
        free(g->execution_order);
        g->execution_order = NULL;
        return -1;
    }

    uint32_t head = 0;
    uint32_t tail = 0;

    // Push nodes with indegree 0
    for (uint32_t n = 0; n < node_count; n++) {
        if (indegree[n] == 0) {
            queue[tail++] = n;
        }
    }

    // Kahn's algorithm
    while (head < tail) {
        uint32_t node_id = queue[head++];
        g->execution_order[g->execution_count++] = node_id;

        Node* node = &g->nodes[node_id];

        // For each output tensor
        for (uint32_t i = 0; i < node->num_outputs; i++) {
            uint32_t tensor_id = node->outputs[i];
            TensorMeta* t = &g->tensors[tensor_id];

            // For each consumer of that tensor
            for (uint32_t c = 0; c < t->num_consumers; c++) {
                uint32_t consumer_id = t->consumers[c];

                indegree[consumer_id]--;

                if (indegree[consumer_id] == 0) {
                    queue[tail++] = consumer_id;
                }
            }
        }
    }

    free(queue);
    free(indegree);

    // Detect cycle
    if (g->execution_count != node_count) {
        // Cycle detected
        free(g->execution_order);
        g->execution_order = NULL;
        g->execution_count = 0;
        return -1;
    }

    return 0;
}


// lifetime analysis

int graph_compute_lifetimes(Graph* g)
{
    if (!g || !g->execution_order || g->execution_count == 0)
        return -1;

    uint32_t node_count = g->node_count;
    uint32_t tensor_count = g->tensor_count;

    // Allocate node_id -> execution index map
    uint32_t* position = (uint32_t*)malloc(sizeof(uint32_t) * node_count);
    if (!position)
        return -1;

    for (uint32_t i = 0; i < g->execution_count; i++) {
        uint32_t node_id = g->execution_order[i];
        position[node_id] = i;
    }

    // Compute lifetimes for each tensor
    for (uint32_t t_id = 0; t_id < tensor_count; t_id++) {

        TensorMeta* t = &g->tensors[t_id];

        // ---- Case 1: Tensor has consumers ----
        if (t->num_consumers > 0) {

            uint32_t first = UINT32_MAX;
            uint32_t last  = 0;

            for (uint32_t c = 0; c < t->num_consumers; c++) {
                uint32_t consumer_id = t->consumers[c];
                uint32_t exec_pos = position[consumer_id];

                if (exec_pos < first)
                    first = exec_pos;

                if (exec_pos > last)
                    last = exec_pos;
            }

            // Determine first_use
            if (t->producer != INVALID_ID) {
                t->first_use = position[t->producer];
            } else {
                // No producer: lifetime starts at first actual use
                t->first_use = first;
            }

            t->last_use = last;
        }

        // ---- Case 2: Tensor has NO consumers ----
        else {

            if (t->producer != INVALID_ID) {
                uint32_t exec_pos = position[t->producer];
                t->first_use = exec_pos;
                t->last_use  = exec_pos;
            } else {
                // Completely unused tensor
                t->first_use = 0;
                t->last_use  = 0;
            }
        }
    }

    free(position);
    return 0;
}