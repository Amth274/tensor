#ifndef KERNEL_H
#define KERNEL_H

#include "graph.h"

typedef struct {
    Graph* graph;
    Node* node;

    void* outputs[MAX_OUTPUTS];
    const void* inputs[MAX_INPUTS];

    TensorMeta* output_metas[MAX_OUTPUTS];
    TensorMeta* input_metas[MAX_INPUTS];

    uint32_t num_inputs;
    uint32_t num_outputs;
} KernelCall;

typedef int (*KernelFn)(KernelCall* call);

#endif
