#ifndef DISPATCH_KEY_BUILDER_H
#define DISPATCH_KEY_BUILDER_H

#include "dispatch_key.h"
#include "graph.h"

DispatchKey get_binary_dispatch_key(
    Op op,
    TensorMeta* a,
    TensorMeta* b,
    TensorMeta* out,
    Graph* g
);

DispatchKey get_unary_dispatch_key(
    Op op,
    TensorMeta* a,
    TensorMeta* out,
    Graph* g
);

#endif
