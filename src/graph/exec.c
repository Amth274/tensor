#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include "dispatch_key_builder.h"
#include "dispatcher.h"
#include <stdio.h>

#define INITIAL_NODE_CAPACITY   16
#define INITIAL_TENSOR_CAPACITY 32

void* get_tensor_data(Graph* g,uint32_t tensor_id){
    TensorMeta* t = &g->tensors[tensor_id];
    return (uint8_t*)g->arena + t->offset;
}

static void build_kernel_call(Graph* g, Node* node, KernelCall* call)
{
    memset(call, 0, sizeof(*call));
    call->graph = g;
    call->node = node;
    call->num_inputs = node->num_inputs;
    call->num_outputs = node->num_outputs;

    for (uint32_t i = 0; i < node->num_inputs; i++) {
        uint32_t tensor_id = node->inputs[i];
        call->inputs[i] = get_tensor_data(g, tensor_id);
        call->input_metas[i] = &g->tensors[tensor_id];
    }

    for (uint32_t i = 0; i < node->num_outputs; i++) {
        uint32_t tensor_id = node->outputs[i];
        call->outputs[i] = get_tensor_data(g, tensor_id);
        call->output_metas[i] = &g->tensors[tensor_id];
    }
}

OpKind classify_op(Op op){
    
    if( op==OP_ADD || op==OP_DIV || op==OP_MUL || op==OP_SUB){
        return OP_KIND_BINARY;
    }
    if(op==OP_MATMUL){
        return OP_KIND_MATMUL;
    }
    if(op==OP_NEG || op==OP_LOG){
        return OP_KIND_UNARY;
    }

    return OP_KIND_COUNT;
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

    for(uint32_t i = 0;i < g->execution_count;i++){
        uint32_t node_id = g->execution_order[i];
        Node* node = &g->nodes[node_id];

        OpKind op_kind = classify_op(node->op);
        KernelCall call;
        build_kernel_call(g, node, &call);
        DispatchKey key;

        switch (op_kind)
        {
        case OP_KIND_BINARY:{
        key = get_binary_dispatch_key(node->op,call.input_metas[0],call.input_metas[1],call.output_metas[0],g);
            break;
        }
        
        case OP_KIND_UNARY:{
        key = get_unary_dispatch_key(node->op,call.input_metas[0],call.output_metas[0],g);
        break;
        }
        
        case OP_KIND_MATMUL:{
        key = get_matmul_dispatch_key(call.input_metas[0],call.input_metas[1],call.output_metas[0],g);
        break;
        }
    
        default:
            fprintf(stderr,"ERROR: unsupported op kind\n");
            return -1;
        }       

        KernelFn fn = dispatch_kernel(key);
        if (!fn) {
            fprintf(stderr,"ERROR: kernel dispatch failed\n");
            return -1;
        }

        if (fn(&call) != 0) {
            fprintf(stderr,"ERROR: kernel execution failed\n");
            return -1;
        }
        
    }

    g->executed = 1;

    return 0;
}
