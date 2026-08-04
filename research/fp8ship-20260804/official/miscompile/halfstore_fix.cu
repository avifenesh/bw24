// Variant B: same kernel but the d half is stored as ONE aligned uint16_t (dst offsets are
// all even at 34-byte stride). If this is correct where the byte-split store is wrong, the
// miscompile is specifically the two-byte-store pattern.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
__global__ __launch_bounds__(128) void store_pattern(const float *__restrict__ dvals,
                                                     uint8_t *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qb = blockIdx.x * 4 + warp;
    float x = dvals[qb] * (lane + 1);
    float amax = fabsf(x);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    const int qi = (int)rintf(x * id);
    uint8_t *dst = out + (size_t)qb * 34;
    if (lane == 0) {
        *(uint16_t *)dst = __half_as_ushort(__float2half_rn(d));
    }
    dst[2 + lane] = (uint8_t)(int8_t)qi;
}
int main() {
    const int nblk = 8;
    float h_d[nblk];
    for (int i = 0; i < nblk; ++i) h_d[i] = 0.001234f * (i + 1) + 0.0771f;
    float *d_d; uint8_t *d_o;
    cudaMalloc(&d_d, sizeof(h_d)); cudaMalloc(&d_o, nblk * 34);
    cudaMemset(d_o, 0xAA, nblk * 34);
    cudaMemcpy(d_d, h_d, sizeof(h_d), cudaMemcpyHostToDevice);
    store_pattern<<<nblk / 4, 128>>>(d_d, d_o);
    cudaDeviceSynchronize();
    uint8_t h_o[nblk * 34];
    cudaMemcpy(h_o, d_o, sizeof(h_o), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int b = 0; b < nblk; ++b) {
        float amax = 0.f;
        for (int l = 0; l < 32; ++l) { float x = h_d[b] * (l + 1); amax = fmaxf(amax, fabsf(x)); }
        float d = amax / 127.0f;
        uint16_t ref = __half_as_ushort(__float2half_rn(d));
        uint16_t got = (uint16_t)h_o[b * 34] | ((uint16_t)h_o[b * 34 + 1] << 8);
        if (ref != got) { ++bad; printf("blk %d: ref 0x%04x got 0x%04x\n", b, ref, got); }
    }
    printf(bad ? "STILL BAD: %d/%d\n" : "u16-store variant CORRECT (%d/%d bad)\n", bad, nblk);
    return 0;
}
