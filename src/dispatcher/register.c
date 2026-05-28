#include "registery.h"
#include "dispatcher.h"


void register_binary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,BinaryKernelFn fn){
    binary_registry[device][op][dtype][contiguous][isa]= fn;
}

void register_unary_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,UnaryKernelFn fn){
    unary_registry[device][op][dtype][contiguous][isa]= fn;
}

