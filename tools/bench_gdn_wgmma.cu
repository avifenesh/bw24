// GDN K4 wgmma harness v1 (round 32+, ARCHITECTURE-H100.md): the chunk-stack rewrite
// on the FA3 toolkit. K4 per (head, chunk): step A: Y = U - W.M (W 32x128, M 128x128);
// step B: M = diag-scale(M)*bC + ys^T.k (ys 32x128 scaled rows, k 32x128).
// Design (ledger 5fb1aca4): M as wgmma accumulator for step B; per-chunk fragment->smem
// bf16 round-trip feeds step A's B operand; m64 rows carry 32-real + 32-pad.
//
// v1 = ONE (head, chunk) step-A tile proof vs CPU ref: Y32x128 = U - W.M via
// wgmma m64n64k16 x (2 n-blocks x 8 k-steps), canonical descriptors (proven).
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/gw tools/bench_gdn_wgmma.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ unsigned long long make_desc(const void* smem_ptr,
                                                        unsigned lead_bytes,
                                                        unsigned stride_bytes) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
template<int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
}
__device__ __forceinline__ void wgmma_m64n64k16_bf16(float acc[32], unsigned long long da,
                                                     unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// canonical core-matrix staging (proven): element (r, kk) of k-step st at
// st*2048 + (r/8)*256 + (kk/8)*128 + (r%8)*16 + (kk%8)*2  (64-row tiles)
__device__ __forceinline__ size_t canon_off(int st, int r, int kk) {
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
}

// ---- v1: step-A tile — Y(32x128) = U - W(32x128).M(128x128), one warpgroup ----
extern "C" __global__ void gdn_k4_stepA_probe(const __nv_bfloat16* __restrict__ W,
                                              const __nv_bfloat16* __restrict__ M,
                                              const float* __restrict__ U,
                                              float* __restrict__ Y) {
    // A = W rows (m64: 32 real + 32 pad), K-major canonical; B = M K-major canonical
    // (M[k][n]: rows of M are the k dim — M stored row-major [128][128], B tile per
    // (k-step st, n-block nb): element (n, kk) staged by n like the QK^T probe's K).
    __shared__ __align__(128) __nv_bfloat16 sA[64 * 128];
    __shared__ __align__(128) __nv_bfloat16 sB[2][64 * 128];   // 2 n-blocks x (64n x 128k)
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int r = idx / 128, c = idx % 128;
        int st = c / 16, kk = c % 16;
        float v = (r < 32) ? __bfloat162float(W[r * 128 + c]) : 0.0f;
        *(__nv_bfloat16*)((char*)sA + canon_off(st, r, kk)) = __float2bfloat16(v);
        // B: n-block nb, n-row = M column n; element (n, kk of k-step st) = M[st*16+kk][n]
        for (int nb = 0; nb < 2; nb++) {
            int n = nb * 64 + r;   // reuse r as the n index (0..63)
            *(__nv_bfloat16*)((char*)sB[nb] + canon_off(st, r, kk)) =
                M[(st * 16 + kk) * 128 + n];
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    float acc[2][32];
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[nb][i] = 0.0f;
    wgmma_fence();
    for (int st = 0; st < 8; st++) {
        unsigned long long da = make_desc((char*)sA + st * 2048, 128, 256);
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            unsigned long long db = make_desc((char*)sB[nb] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc[nb], da, db, st == 0 ? 0 : 1);
        }
    }
    wgmma_commit();
    wgmma_wait<0>();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            int n8 = i / 4;
            int cc = nb * 64 + c0 + n8 * 8;
            if (r0 < 32) {
                Y[r0 * 128 + cc + 0] = U[r0 * 128 + cc + 0] - acc[nb][i + 0];
                Y[r0 * 128 + cc + 1] = U[r0 * 128 + cc + 1] - acc[nb][i + 1];
            }
            if (r0 + 8 < 32) {
                Y[(r0 + 8) * 128 + cc + 0] = U[(r0 + 8) * 128 + cc + 0] - acc[nb][i + 2];
                Y[(r0 + 8) * 128 + cc + 1] = U[(r0 + 8) * 128 + cc + 1] - acc[nb][i + 3];
            }
        }
}

int main() {
    __nv_bfloat16 *hW = (__nv_bfloat16*)malloc(32 * 128 * 2);
    __nv_bfloat16 *hM = (__nv_bfloat16*)malloc(128 * 128 * 2);
    float *hU = (float*)malloc(32 * 128 * 4), *ref = (float*)malloc(32 * 128 * 4);
    srand(17);
    for (int i = 0; i < 32 * 128; i++) hW[i] = __float2bfloat16((rand() % 255 - 127) * 0.012f);
    for (int i = 0; i < 128 * 128; i++) hM[i] = __float2bfloat16((rand() % 255 - 127) * 0.009f);
    for (int i = 0; i < 32 * 128; i++) hU[i] = (rand() % 255 - 127) * 0.02f;
    for (int r = 0; r < 32; r++)
        for (int n = 0; n < 128; n++) {
            float s = 0;
            for (int k = 0; k < 128; k++)
                s += __bfloat162float(hW[r * 128 + k]) * __bfloat162float(hM[k * 128 + n]);
            ref[r * 128 + n] = hU[r * 128 + n] - s;
        }
    __nv_bfloat16 *dW, *dM; float *dU, *dY;
    CK(cudaMalloc(&dW, 32 * 128 * 2)); CK(cudaMalloc(&dM, 128 * 128 * 2));
    CK(cudaMalloc(&dU, 32 * 128 * 4)); CK(cudaMalloc(&dY, 32 * 128 * 4));
    CK(cudaMemcpy(dW, hW, 32 * 128 * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM, hM, 128 * 128 * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dU, hU, 32 * 128 * 4, cudaMemcpyHostToDevice));
    gdn_k4_stepA_probe<<<1, 128>>>(dW, dM, dU, dY);
    CK(cudaDeviceSynchronize());
    float* hY = (float*)malloc(32 * 128 * 4);
    CK(cudaMemcpy(hY, dY, 32 * 128 * 4, cudaMemcpyDeviceToHost));
    float mr = 0; int bad = 0;
    for (int i = 0; i < 32 * 128; i++) {
        float rl = fabsf(hY[i] - ref[i]) / fmaxf(fabsf(ref[i]), 1e-3f);
        if (rl > mr) mr = rl;
        if (rl > 2e-2f) bad++;
    }
    printf("K4 step-A tile: max_rel %.3e bad %d/4096 %s\n", mr, bad, bad == 0 ? "MATCH" : "MISMATCH");
    return bad != 0;
}
