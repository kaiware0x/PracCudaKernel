#include <iostream>

__global__ void minimal(int *d_a)
{
    *d_a = 13;
}

__global__ void assign(int *d_a, int value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    d_a[idx] = value;
}

__global__ void assign2D(int *d_a, int w, int h, int value)
{
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = iy * w + ix;
    d_a[idx] = value;
}

void inc_cpu(int *a, int N)
{
    for (size_t idx = 0; idx < N; idx++)
    {
        a[idx]++;
    }
}

__global__ void inc_gpu(int *d_a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_a[idx]++;
    }
}
__device__ void device_strcpy(char *dst, const char *src)
{
    while (*dst++ = *src++)
        ;
}

__global__ void kernel(char *A)
{
    device_strcpy(A, "Hello, World!");
}

int main()
{
    char *d_hello;

    cudaMalloc((void **)&d_hello, 32);
    
    kernel<<<1, 1>>>(d_hello);

    char hello[32];
    cudaMemcpy(hello, d_hello, 32, cudaMemcpyDeviceToHost);

    cudaFree(d_hello);

    puts(hello);
}