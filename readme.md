# Tensor Runtime

A lightweight tensor runtime written in C featuring graph-based execution, arena memory planning, dispatch-driven kernel execution, and modular CPU backends.

---

# Overview

This project is an experimental tensor runtime focused on understanding and implementing the core systems behind modern deep learning frameworks and compiler runtimes.

The goal is not to build another high-level ML framework, but to explore low-level runtime infrastructure such as:

- graph IR execution
- tensor lifetime analysis
- arena memory allocation
- dispatch systems
- execution policies
- modular backend kernels
- dtype-aware execution
- future SIMD/threaded execution

The runtime currently supports contiguous CPU tensor execution with extensible operator dispatch.

---

# Current Features

## Graph Runtime

- Tensor graph construction
- Topological execution ordering
- Node-based IR
- Generic graph executor

## Memory System

- Arena allocator
- Tensor lifetime analysis
- Memory reuse for temporaries
- Output tensor lifetime pinning

## Dispatcher

- Dispatch-key-based kernel lookup
- Modular backend registration
- DType-aware execution
- Contiguous/non-contiguous dispatch preparation

## CPU Backend

Current scalar CPU kernels:

- Add
- Multiply
- Subtract
- Divide

Supported dtypes:

- FP32
- FP64

## Runtime Infrastructure

- Execution policy layer
- Kernel registration system
- Backend abstraction
- Extensible operator architecture

---

# Architecture

```text
Tensor API
    ↓
Graph IR
    ↓
Topological Scheduler
    ↓
Lifetime Analysis
    ↓
Arena Memory Planner
    ↓
Execution Policy
    ↓
Dispatcher
    ↓
CPU Kernels
```

---

# Project Structure

```text
src/
├── autograd/
├── core/
│   ├── ops.c
│   ├── policy.c
│   └── tensor.c
├── dispatcher/
│   ├── dispatcher.c
│   └── register.c
├── graph/
│   ├── exec.c
│   ├── graph.c
│   └── planner.c
├── ops/
│   ├── cpu/
│   │   ├── scalar/
│   │   └── simd/
│   └── gpu/

include/
├── dispatcher.h
├── graph.h
├── ops.h
├── policy.h
├── register.h
├── scalar.h
├── tensor.h
```

---

# Build

```bash
make
```

Run:

```bash
make run
```

---

# Example

```c
Tensor* a = tensor_create(...);
Tensor* b = tensor_create(...);

Tensor* out = tensor_add(a, b);

graph_compile(g);
graph_execute(g);
```

---

# Example Output

```text
ADD:
11
22
33
44

MUL:
10
40
90
160

SUB:
-9
-18
-27
-36

DIV:
0.1
0.1
0.1
0.1
```

---

# Design Goals

This project prioritizes:

- clean runtime architecture
- explicit memory management
- modular backend design
- low-level systems understanding
- extensibility over framework features

---

# Planned Features

## Execution

- AVX2 kernels
- threaded execution
- kernel caching
- backend specialization

## Tensor Semantics

- broadcasting
- strided tensors
- tensor views
- reductions

## Linear Algebra

- matrix multiplication
- cache blocking
- SIMD GEMM kernels

## Runtime

- benchmark infrastructure
- profiling utilities
- threadpool runtime
- scheduler improvements

## Autograd

- backward graph generation
- gradient accumulation
- trainable graphs

---

# Motivation

This runtime is primarily a systems-learning project exploring the implementation details behind modern tensor runtimes and ML execution engines such as:

- TensorRT
- XLA
- tinygrad
- TVM
- PyTorch internals

---

# Status

The project is currently in active development and architecture exploration.

The runtime core, dispatcher, planner, and execution engine are functional, while SIMD, threading, broadcasting, and autograd are still under development.

---

# License

MIT License.