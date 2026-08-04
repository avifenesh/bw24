// FMA spin-kernel board-health probe (prior-session protocol): saturate FMA pipes
// on every SM for ~35s while nvidia-smi logs 1Hz power/clocks/temp.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void spin(float* out, long iters) {
    float a = threadIdx.x * 0.001f + 1.0f, b = blockIdx.x * 0.002f + 1.5f, c = 0.0f;
    for (long i = 0; i < iters; i++) { c = fmaf(a, b, c); a = fmaf(c, 0.9999f, b); b = fmaf(a, 1.0001f, c); }
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = c;
}
int main() {
    int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("device=%s SMs=%d\n", p.name, p.multiProcessorCount);
    float* d; cudaMalloc(&d, 4);
    // 4 blocks/SM x 512 threads, iterate until ~35s
    dim3 grid(p.multiProcessorCount * 4), blk(512);
    spin<<<grid, blk>>>(d, 3500000000L);
    cudaError_t e = cudaDeviceSynchronize();
    printf("spin done rc=%d (%s)\n", (int)e, cudaGetErrorString(e));
    return e != cudaSuccess;
}
