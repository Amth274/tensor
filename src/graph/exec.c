#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include <stdio.h>

#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32

void* get_tensor_data(Graph* g,uint32_t tensor_id){
    TensorMeta* t = &g->tensors[tensor_id];
    return (uint8_t*)g->arena + t->offset;
}


int graph_execute(Graph* g)
{
    if(!g){
        fprintf(stderr,
            "ERROR: graph does not exist\n");
        return -1;
    }

    if(g->compiled != 1){
        fprintf(stderr,
            "ERROR: graph not compiled\n");
        return -1;
    }

    for(uint32_t i = 0;
        i < g->execution_count;
        i++)
    {
        uint32_t node_id =
            g->execution_order[i];

        Node* node =
            &g->nodes[node_id];

        if (node->num_inputs != 2 ||
            node->num_outputs != 1)
        {
            fprintf(stderr,
                "ERROR: unsupported node\n");

            return -1;
        }

        TensorMeta* out_meta =
            &g->tensors[node->outputs[0]];

        void* out =
            get_tensor_data(
                g,
                node->outputs[0]
            );

        void* in0 =
            get_tensor_data(
                g,
                node->inputs[0]
            );

        void* in1 =
            get_tensor_data(
                g,
                node->inputs[1]
            );

        DispatchKey key = {
            .device = g->device,
            .op = node->op,
            .dtype = out_meta->dtype,
            .contiguous = 1
        };

        BinaryKernelFn fn =
            dispatch_binary_kernel(key);

        if (!fn) {
            fprintf(stderr,
                "ERROR: kernel dispatch failed\n");

            return -1;
        }

        fn(
            out,
            in0,
            in1,
            out_meta->numel
        );
    }

    g->executed = 1;

    return 0;
}