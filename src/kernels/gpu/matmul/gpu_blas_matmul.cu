#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "kernel.h"

extern "C" void* gpu_managed_alloc(size_t size)
{
    void* ptr = NULL;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess)
        return NULL;
    return ptr;
}

extern "C" void gpu_managed_free(void* ptr)
{
    if (ptr)
        cudaFree(ptr);
}

__global__ static void matmul_fp32_kernel(float* c,const float* a,const float* b,uint32_t m,uint32_t n,uint32_t k)
{
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
        return;

    float sum = 0.0f;
    for (uint32_t p = 0; p < k; p++) {
        sum += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = sum;
}

__global__ static void matmul_fp64_kernel(double* c,const double* a,const double* b,uint32_t m,uint32_t n,uint32_t k)
{
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
        return;

    double sum = 0.0;
    for (uint32_t p = 0; p < k; p++) {
        sum += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = sum;
}

extern "C" int gpu_blas_matmul_fp32(KernelCall* call)
{
    TensorMeta* a_meta = call->input_metas[0];
    TensorMeta* b_meta = call->input_metas[1];
    uint32_t m = (uint32_t)a_meta->shape[0];
    uint32_t n = (uint32_t)b_meta->shape[1];
    uint32_t k = (uint32_t)a_meta->shape[1];
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_fp32_kernel<<<grid, block>>>((float*)call->outputs[0], (const float*)call->inputs[0], (const float*)call->inputs[1], m, n, k);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_blas_matmul_fp64(KernelCall* call)
{
    TensorMeta* a_meta = call->input_metas[0];
    TensorMeta* b_meta = call->input_metas[1];
    uint32_t m = (uint32_t)a_meta->shape[0];
    uint32_t n = (uint32_t)b_meta->shape[1];
    uint32_t k = (uint32_t)a_meta->shape[1];
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_fp64_kernel<<<grid, block>>>((double*)call->outputs[0], (const double*)call->inputs[0], (const double*)call->inputs[1], m, n, k);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}
