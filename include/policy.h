#ifndef POLICY_H
#define POLICY_H

#include "dispatcher.h"
#include "graph.h"

DispatchKey execution_policy(
    Node* n,
    Graph* g
);

#endif