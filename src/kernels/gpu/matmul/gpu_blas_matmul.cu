#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

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

extern "C" void gpu_blas_matmul_fp32(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k)
{
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_fp32_kernel<<<grid, block>>>((float*)out, (const float*)a, (const float*)b, m, n, k);
    cudaDeviceSynchronize();
}

extern "C" void gpu_blas_matmul_fp64(void* out,const void* a,const void* b,uint32_t m,uint32_t n,uint32_t k)
{
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_fp64_kernel<<<grid, block>>>((double*)out, (const double*)a, (const double*)b, m, n, k);
    cudaDeviceSynchronize();
}
