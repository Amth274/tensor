#include "registery.h"
#include "dispatcher.h"


void register_kernel(Device device,Op op,DType dtype,uint8_t contiguous,isa_t isa,KernelFn fn){
    kernel_registry[device][op][dtype][contiguous][isa]= fn;
}
