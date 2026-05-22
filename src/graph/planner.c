#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include <stdio.h>

#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32


int graph_topo_sort(Graph* g)
{
    if (!g || g->node_count == 0)
        return -1;

    uint32_t node_count = g->node_count;

    // Allocate execution order array
    // if (g->execution_order)
    //     free(g->execution_order);

    g->execution_order = (uint32_t*)malloc(sizeof(uint32_t) * node_count);
    if (!g->execution_order)
        return -1;

    g->execution_count = 0;

    // Allocate indegree arry
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

    // Allocate queue (max size = node_count) (unitl only when max_size<node_count)
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

    // Kahns algorithm
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

size_t graph_simulate_peak_memory(Graph* g)
{
    if (!g || !g->execution_order || g->execution_count == 0)
        return 0;

    size_t peak = 0;

    // Iterate over each execution step
    for (uint32_t step = 0; step < g->execution_count; step++) {

        size_t current_live_memory = 0;

        // Sum sizes of all live tensors at this step
        for (uint32_t t_id = 0; t_id < g->tensor_count; t_id++) {

            TensorMeta* t = &g->tensors[t_id];

            if (t->first_use <= step && step <= t->last_use) {
                current_live_memory += t->size_bytes;
            }
        }

        if (current_live_memory > peak)
            peak = current_live_memory;
    }

    return peak;
}


typedef struct {
    uint32_t tensor_id;
} TensorRef;

typedef struct {
    size_t offset;
    size_t size;
} FreeBlock;

static int compare_by_first_use(const void* a, const void* b)
{
    const TensorMeta* ta = *(const TensorMeta**)a;
    const TensorMeta* tb = *(const TensorMeta**)b;

    if (ta->first_use < tb->first_use) return -1;
    if (ta->first_use > tb->first_use) return 1;
    return 0;
}

int graph_plan_memory(Graph* g)
{
    if (!g)
        return -1;

    uint32_t tensor_count = g->tensor_count;

    if (tensor_count == 0)
        return 0;

    // Collect tensor pointers
    TensorMeta** tensors = malloc(sizeof(TensorMeta*) * tensor_count);
    if (!tensors)
        return -1;

    for (uint32_t i = 0; i < tensor_count; i++)
        tensors[i] = &g->tensors[i];

    // Sort by first_use
    qsort(tensors, tensor_count, sizeof(TensorMeta*), compare_by_first_use);

    // Active tensors (worst-case size = tensor_count)
    TensorMeta** active = malloc(sizeof(TensorMeta*) * tensor_count);
    if (!active) {
        free(tensors);
        return -1;
    }
    uint32_t active_count = 0;

    // Free blocks (worst-case size = tensor_count)
    FreeBlock* free_blocks = malloc(sizeof(FreeBlock) * tensor_count);
    if (!free_blocks) {
        free(tensors);
        free(active);
        return -1;
    }
    uint32_t free_count = 0;

    size_t arena_size = 0;

    for (uint32_t i = 0; i < tensor_count; i++) {

        TensorMeta* current = tensors[i];

        // Skip zero-sized tensors
        if (current->size_bytes == 0)
            continue;

        // 1. Expire old tensors
        for (uint32_t j = 0; j < active_count;) {

            if (active[j]->last_use < current->first_use) {

                // Move its memory into free_blocks
                free_blocks[free_count].offset = active[j]->offset;
                free_blocks[free_count].size   = active[j]->size_bytes;
                free_count++;

                // Remove from active (swap-remove)
                active[j] = active[active_count - 1];
                active_count--;
            }
            else {
                j++;
            }
        }

        // 2. Try to reuse a free block (first-fit)
        size_t assigned_offset = SIZE_MAX;
        uint32_t reused_index = UINT32_MAX;

        for (uint32_t j = 0; j < free_count; j++) {
            if (free_blocks[j].size >= current->size_bytes) {
                assigned_offset = free_blocks[j].offset;
                reused_index = j;
                break;
            }
        }

        if (assigned_offset != SIZE_MAX) {
            // Reuse block
            current->offset = assigned_offset;

            // Remove used free block (swap-remove)
            free_blocks[reused_index] = free_blocks[free_count - 1];
            free_count--;
        }
        else {
            // Allocate new memory at end
            current->offset = arena_size;
            arena_size += current->size_bytes;
        }

        // Add to active list
        active[active_count++] = current;
    }

    g->arena_size = arena_size;

    free(tensors);
    free(active);
    free(free_blocks);

    return 0;
}


// graph validation 
int graph_validate(Graph* g){
    if(!g){
        fprintf(stderr,"ERROR: graph does not exist. validation failed\n");
        return -1;
    }
    if(!g->nodes){
        fprintf(stderr,"ERROR: graph contains null nodes. validation failed\n");
        return -1;
    }

    if(!g->tensors){
        fprintf(stderr,"ERROR: graph contains nulltensors. validation failed\n");
        return -1;
    }
    if(g->node_count>g->node_capacity){
        fprintf(stderr,"ERROR: graph contains nodes>node capacity. validation failed\n");
        return -1;
    }
    if(g->tensor_count>g->tensor_capacity){
        fprintf(stderr,"ERROR: graph has tensor > tensor capacity. validation failed\n");
        return -1;
    }

    for(uint_fast32_t i=0;i<g->node_count;i++){
        if(g->nodes[i].id!=i){
            fprintf(stderr,"ERROR: node id mismatch. validation failed\n");
            return -1;
            break;
        }
        if(g->nodes[i].num_inputs>MAX_INPUTS || g->nodes[i].num_outputs>MAX_OUTPUTS){
            fprintf(stderr,"ERROR: num inputs/outputs exceed MAX_INPUT/MAX_OUTPUTS capacity. validatin failed\n");
            return -1;
            break;
        }
    }
    for(uint_fast32_t i=0;i<g->tensor_count;i++){
        if(g->tensors[i].id!=i){
            fprintf(stderr,"ERROR: tensor desnt exist(id mismatch). validation failed\n");
            return -1;
            break;
        }
        //tensor shape stride mismatch will be validated in tensor_add_meta
    }

    return 0;
}


// arena allocator 
int graph_allocate_arena(Graph* g)
{
    if (!g)
        return -1;

    if (g->arena_size == 0)
        return 0;

    if (g->arena) {
        free(g->arena);
        g->arena = NULL;
    }

    g->arena = malloc(g->arena_size);
    if (!g->arena)
        return -1;

    return 0;
}


// graph_compile 

int graph_compile(Graph* g)
{
    if (graph_validate(g) != 0)
        return -1;

    if (graph_topo_sort(g) != 0)
        return -1;

    if (graph_compute_lifetimes(g) != 0)
        return -1;

    if (graph_plan_memory(g) != 0)
        return -1;

    if (graph_allocate_arena(g) != 0)
        return -1;

    g->compiled = 1;
    return 0;
}
