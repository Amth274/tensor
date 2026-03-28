// #include <stdlib.h>
// #include <stdio.h>
// #include "graph.h"

// int graph_build_backward(Graph* g){

//     // init g->grad_of to invalid
//     g->grad_of = malloc(sizeof(uint32_t)*g->tensor_capacity);
//     for(int i=0;i<g->tensor_capacity;i++){
//         g->grad_of[i] = INVALID_ID;
//     }

//     for(int i=0;i<g->tensor_capacity;i++){
//         if(g->tensors[i]->is_output==1){
//             // TensorMeta* seed = create_seed_tensor() // create_seed_tensor implementation to be done
//         }
//         g->grad_of[g->tensors->id] = seed->id;
//     }

//     for(i=0;i<)

    

// }