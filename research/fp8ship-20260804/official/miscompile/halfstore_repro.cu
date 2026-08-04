// Minimal repro for the fp8_blk_dequant d-half low-byte loss seen on nvcc 13.0.88/sm_120a
// (vast 2x5090 box, 2026-08-04). Reproduces the EXACT store pattern of
// fp8_blk_dequant_q8_0_kernel: lane-0 splits __half_as_ushort(__float2half_rn(d)) into two
// byte stores at dst[0]/dst[1] while every lane stores dst[2+lane]. If dst[0] lands as 0
// (out = in & 0xff00) the miscompile is in this isolated pattern; if correct here, the
// trigger is elsewhere in the full kernel.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

__global__ __launch_bounds__(128) void store_pattern(const float *__restrict__ dvals,
                                                     uint8_t *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qb = blockIdx.x * 4 + warp;
    float x = dvals[qb] * (lane + 1); // touch all lanes so the shuffle below is live
    float amax = fabsf(x);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
    }
    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    const int qi = (int)rintf(x * id);
    uint8_t *dst = out + (size_t)qb * 34;
    if (lane == 0) {
        const uint16_t bits = __half_as_ushort(__float2half_rn(d));
        dst[0] = (uint8_t)(bits & 0xffu);
        dst[1] = (uint8_t)(bits >> 8);
    }
    dst[2 + lane] = (uint8_t)(int8_t)qi;
}

int main() {
    const int nblk = 8;
    float h_d[nblk];
    for (int i = 0; i < nblk; ++i) h_d[i] = 0.001234f * (i + 1) + 0.0771f;
    float *d_d; uint8_t *d_o;
    cudaMalloc(&d_d, sizeof(h_d));
    cudaMalloc(&d_o, nblk * 34);
    cudaMemset(d_o, 0xAA, nblk * 34); // poison so a LOST write reads 0xAA, not 0x00
    cudaMemcpy(d_d, h_d, sizeof(h_d), cudaMemcpyHostToDevice);
    store_pattern<<<nblk / 4, 128>>>(d_d, d_o);
    cudaDeviceSynchronize();
    uint8_t h_o[nblk * 34];
    cudaMemcpy(h_o, d_o, sizeof(h_o), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int b = 0; b < nblk; ++b) {
        // host reference of the same math
        float amax = 0.f;
        for (int l = 0; l < 32; ++l) { float x = h_d[b] * (l + 1); amax = fmaxf(amax, fabsf(x)); }
        float d = amax / 127.0f;
        uint16_t ref = __half_as_ushort(__float2half_rn(d));
        uint16_t got = (uint16_t)h_o[b * 34] | ((uint16_t)h_o[b * 34 + 1] << 8);
        if (ref != got) {
            ++bad;
            printf("blk %d: ref 0x%04x got 0x%04x (got==ref&0xff00: %s; low byte 0x%02x)\n",
                   b, ref, got, (got == (ref & 0xff00)) ? "YES" : "no", h_o[b * 34]);
        }
    }
    printf(bad ? "REPRO: %d/%d d-half stores wrong in the ISOLATED pattern\n"
               : "isolated pattern OK (%d/%d bad) — trigger is elsewhere in the full kernel\n",
           bad, nblk);
    return 0;
}
