// #include "elementwise_ops.h"
// #include "tensor.h"
// #include "dispatcher/dispatcher.h"
// #include <stdlib.h>
// #include <stdio.h>


// Tensor* tensor_add(Tensor* t1, Tensor* t2)
// {
//     if (!t1 || !t2) {
//         fprintf(stderr, "ERROR: null tensor passed to tensor_add\n");
//         return NULL;
//     }

//     //  Device check
//     if (t1->storage->device != t2->storage->device) {
//         fprintf(stderr, "ERROR: tensors must be on same device\n");
//         return NULL;
//     }

//     //broadcasting and dtype promotion logic to be implemented


//     //  Create output tensor
//     Tensor* out = tensor_create(
//         t1->ndims,
//         t1->shape,
//         t1->dtype,
//         0,
//         t1->storage->device,
//         t1->storage->device_id
//     );

//     if (!out) {
//         fprintf(stderr, "ERROR: failed to create output tensor\n");
//         return NULL;
//     }

//     // Lookup kernel
//     BinaryKernelFn fn = dispatch_lookup_binary(
//         OP_ADD,
//         t1->storage->device,
//         t1->dtype
//     );

//     if (!fn) {
//         fprintf(stderr, "ERROR: kernel not registered\n");
//         abort();
//     }

//     fn(t1, t2, out);

//     return out;
// }


// Tensor* tensor_sub(Tensor* t1, Tensor* t2)
// {
//     if (!t1 || !t2) {
//         fprintf(stderr, "ERROR: null tensor passed to tensor_add\n");
//         return NULL;
//     }

//     //  Device check
//     if (t1->storage->device != t2->storage->device) {
//         fprintf(stderr, "ERROR: tensors must be on same device\n");
//         return NULL;
//     }

//     //broadcasting and dtype promotion logic to be implemented


//     //  Create output tensor
//     Tensor* out = tensor_create(
//         t1->ndims,
//         t1->shape,
//         t1->dtype,
//         0,
//         t1->storage->device,
//         t1->storage->device_id
//     );

//     if (!out) {
//         fprintf(stderr, "ERROR: failed to create output tensor\n");
//         return NULL;
//     }

//     // Lookup kernel
//     BinaryKernelFn fn = dispatch_lookup_binary(
//         OP_SUB,
//         t1->storage->device,
//         t1->dtype
//     );

//     if (!fn) {
//         fprintf(stderr, "ERROR: kernel not registered\n");
//         abort();
//     }

//     fn(t1, t2, out);

//     return out;
// }