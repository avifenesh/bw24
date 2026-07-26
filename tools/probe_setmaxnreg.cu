// setmaxnreg contract probe: can a 384-thread kernel compiled WITHOUT launch_bounds
// (ptxas free to use ~250 regs) launch successfully when warpgroups rebalance via
// setmaxnreg (consumers .inc 240, producer .dec 24)? Checks: ptxas static regs,
// launch rc, correctness of a register-heavy consumer computation.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1;} } while (0)

__device__ __forceinline__ void reg_inc(int n) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(240));
}
__device__ __forceinline__ void reg_dec(int n) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(24));
}

extern "C" __global__ void __launch_bounds__(384, 1) probe(float* out, const float* in) {
    int wg = threadIdx.x / 128;
    if (wg == 2) {
        reg_dec(24);
        return;   // producer idles
    }
    reg_inc(240);
    // register-heavy consumer: 192 live accumulators
    float acc[192];
    #pragma unroll
    for (int i = 0; i < 192; i++) acc[i] = in[i];
    for (int it = 0; it < 100; it++)
        #pragma unroll
        for (int i = 0; i < 192; i++) acc[i] = fmaf(acc[i], 1.0001f, (float)i);
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 192; i++) s += acc[i];
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;
}

int main() {
    float *d_out, *d_in;
    CK(cudaMalloc(&d_out, 384 * 4 * 4));
    CK(cudaMalloc(&d_in, 192 * 4));
    CK(cudaMemset(d_in, 0, 192 * 4));
    probe<<<4, 384>>>(d_out, d_in);
    cudaError_t e = cudaGetLastError();
    printf("launch: %s\n", cudaGetErrorString(e));
    CK(cudaDeviceSynchronize());
    float h[4];
    CK(cudaMemcpy(h, d_out, 16, cudaMemcpyDeviceToHost));
    printf("result[0]=%f (expect finite, identical across threads reading same in)\n", h[0]);
    return 0;
}
