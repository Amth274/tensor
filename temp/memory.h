#ifndef MEMORY_H
#define MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

void* device_alloc(Device device, int device_id, size_t size);
void* device_free(Device device,int device_id, size_t size);


#endif //MEMORY_H