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


int graph_execute(Graph* g){
    if(!g){
        fprintf(stderr,"ERROR: graph does not exist. execution failed \n");
        return -1;
    }
    if(g->compiled!=1){
        fprintf(stderr,"ERROR: graph not compiled. execution failed\n");
        return -1;
    }
    for(uint32_t i=0;i<g->execution_count;i++){
        uint32_t node_id =  g->execution_order[i];

        Node* node = &g->nodes[node_id];

        // dispatch function invoked 
        //testcase for tensor_add 
        //some add kernel 
        
    }
    g->executed = 1;

    return 0;
}
