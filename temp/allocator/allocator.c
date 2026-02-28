#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

// device_allocator

void* device_alloc(Device device, int device_id, size_t size) {
    switch (device) {

        case DEVICE_CPU: {
            void* ptr = aligned_alloc(64, size);
            if (!ptr) {
                fprintf(stderr, "CPU allocation failed\n");
                return NULL;
            }
            return ptr;
        }

        case DEVICE_CUDA:
            // Placeholder for now
            fprintf(stderr, "CUDA allocator not implemented\n");
            return NULL;

        case DEVICE_ROCM:
            fprintf(stderr, "ROCM allocator not implemented\n");
            return NULL;

        default:
            return NULL;
    }
}

void device_free(Device device, int device_id, void* ptr) {
    if (!ptr) return;

    switch (device) {

        case DEVICE_CPU:
            free(ptr);
            break;

        case DEVICE_CUDA:
            // cudaSetDevice(device_id);
            // cudaFree(ptr);
            break;

        case DEVICE_ROCM:
            // hipSetDevice(device_id);
            // hipFree(ptr);
            break;
    }
}
