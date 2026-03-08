#include <stdlib.h>
#include <stdio.h>
#include "graph.h"

int graph_build_backward(Graph* g){

    // init g->grad_of to invalid
    g->grad_of = malloc(sizeof(uint32_t)*g->tensor_capacity);
    for(int i=0;i<g->tensor_capacity;i++){
        g->grad_of[i] = INVALID_ID;
    }

    

    

}