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

OpKind classify_op(Op op){
    
    if( op==OP_ADD || op==OP_DIV || op==OP_MUL || op==OP_SUB){
        return OP_KIND_BINARY;
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
        switch (op_kind)
        {
        case OP_KIND_BINARY:{
        TensorMeta* m1 = &g->tensors[node->inputs[0]];
        TensorMeta* m2 = &g->tensors[node->inputs[1]];
        TensorMeta* out = &g->tensors[node->outputs[0]];
        DispatchKey key = get_binary_dispatch_key(node->op,m1,m2,out,g);
        BinaryKernelFn fn = dispatch_binary_kernel(key);

        void* out_ptr = get_tensor_data(g,node->outputs[0]);
        void* a_ptr = get_tensor_data(g,node->inputs[0]);
        void* b_ptr = get_tensor_data(g,node->inputs[1]);

        fn(out_ptr,a_ptr,b_ptr,out->numel);
            break;
        }
        
        case OP_KIND_UNARY:{
        TensorMeta* m = &g->tensors[node->inputs[0]];
        TensorMeta* out = &g->tensors[node->outputs[0]];
        DispatchKey key = get_unary_dispatch_key(node->op,m,out,g);
        UnaryKernelFn fn = dispatch_unary_kernel(key);

        void* out_ptr = get_tensor_data(g,node->outputs[0]);
        void* in_ptr = get_tensor_data(g,node->inputs[0]);

        fn(out_ptr,in_ptr,node->scalar,out->numel);
        break;
        }
    
        default:
            break;
        }       
        
    }

    g->executed = 1;

    return 0;
}
