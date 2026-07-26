// FA3 harness v1 (round 27+, ARCHITECTURE-H100.md): can a wgmma warpgroup FA beat the
// mma fa_prefill_bf16kv_pp (993us/layer at T=2048 = ~3.5% of bf16 TC peak, ncu: 11.6%
// occupancy, 255 regs — register-bound; sm_90a wgmma reads operands from SMEM, removing
// the Q/K register residency that caps the current kernel)?
//
// Playbook (the K4/K5 harness discipline): v1 proves ONE warpgroup QK^T tile against the
// CPU reference (descriptor layout iterated via -D knobs), then softmax+PV, then the
// pipelined kernel, then engine integration behind BW24_FA3.
//
// Shapes: qwen35 attn — T=2048, H=16, HKV=4 (GQA 4:1), D=256, causal.
// Layouts (engine): Q/O [T, H, D] token-major; K/V [T, HKV, D] token-major (bf16 mirrors).
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/fa3 tools/bench_fa3.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#ifndef TRANS_A
#define TRANS_A 0
#endif
#ifndef TRANS_B
#define TRANS_B 0     // canonical K-major core layout for both operands (s8-probe proven)
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

// ---- wgmma helpers (lifted from tools/bench_q8_gemm_wgmma.cu — the proven set) ----
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

// m64n64k16 bf16 -> f32: 32 accumulator regs per thread across the warpgroup.
// A, B from smem descriptors. scale_d=1 accumulates onto acc.
__device__ __forceinline__ void wgmma_m64n64k16_bf16(float acc[32], unsigned long long da,
                                                     unsigned long long db, int scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, " STR(TRANS_A) ", " STR(TRANS_B) ";\n"
        "}\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// ---- v1 probe: ONE warpgroup computes S = Q_tile(64xD) . K_tile(64xD)^T, k16-stepped ----
// Descriptor layout knobs (iterate on box until MATCH):
//   A tile in smem: [64][16] bf16 per k-step, contiguous k-steps -> sQ [64][D]
//   B (K^T) tile:   [64][16] bf16 per k-step from K rows       -> sK [64][D]
// wgmma B for n=64: B is k16 x n64 — K rows ARE the n dim, so sK holds K rows and the
// descriptor must present them k-major. Canonical no-swizzle core matrix = 8x(16B);
// knobs below express lead/stride in bytes for both operands.
#ifndef A_LEAD
#define A_LEAD 32      // bytes between core-matrix rows groups of A
#endif
#ifndef A_STRIDE
#define A_STRIDE 256   // bytes between 8-row core groups of A
#endif
#ifndef B_LEAD
#define B_LEAD 32
#endif
#ifndef B_STRIDE
#define B_STRIDE 256
#endif

// sQ/sK: per k-step 2KB tiles in the CANONICAL core-matrix layout the s8 probe proved:
// core matrix = 8 rows x 16B; element (r, kk) of step s lives at byte
// s*2048 + (r/8)*256 + (kk/8)*128 + (r%8)*16 + (kk%8)*2  — desc (lead=128, stride=256).
extern "C" __global__ void fa3_qk_probe(const __nv_bfloat16* __restrict__ Q,
                                        const __nv_bfloat16* __restrict__ K,
                                        float* __restrict__ S,
                                        int D) {
    __shared__ __align__(128) __nv_bfloat16 sQ[64 * 256];
    __shared__ __align__(128) __nv_bfloat16 sK[64 * 256];
    const int tid = threadIdx.x;   // 128 threads = 1 warpgroup
    char* bQ = (char*)sQ;
    char* bK = (char*)sK;
    for (int idx = tid; idx < 64 * D; idx += 128) {
        int r = idx / D, c = idx % D;
        int st = c / 16, kk = c % 16;
        size_t off = (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
        *(__nv_bfloat16*)(bQ + off) = Q[r * D + c];
        *(__nv_bfloat16*)(bK + off) = K[r * D + c];
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    wgmma_fence();
    for (int s = 0; s < D / 16; s++) {
        unsigned long long da = make_desc(sQ + s * 64 * 16, A_LEAD, A_STRIDE);
        unsigned long long db = make_desc(sK + s * 64 * 16, B_LEAD, B_STRIDE);
        wgmma_m64n64k16_bf16(acc, da, db, s == 0 ? 0 : 1);
    }
    wgmma_commit();
    wgmma_wait<0>();

    // scatter acc to S[64][64]: wgmma m64nN f32 fragment layout — thread t of warp w owns
    // rows (w*16 + t/4, +8) and cols (t%4)*2 + n8*8 (+1), n8 = reg pair index /2.
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        int n8 = i / 4;
        S[(r0 + 0) * 64 + c0 + n8 * 8 + 0] = acc[i + 0];
        S[(r0 + 0) * 64 + c0 + n8 * 8 + 1] = acc[i + 1];
        S[(r0 + 8) * 64 + c0 + n8 * 8 + 0] = acc[i + 2];
        S[(r0 + 8) * 64 + c0 + n8 * 8 + 1] = acc[i + 3];
    }
}

// ---- v2 probe: ONE full FA tile — S = QK^T -> causal+scale -> softmax -> P(bf16) ->
// O = P.V, all wgmma (PV = 4x m64n64k16 over V restaged n-major canonical).
extern "C" __global__ void fa3_tile_probe(const __nv_bfloat16* __restrict__ Q,
                                          const __nv_bfloat16* __restrict__ K,
                                          const __nv_bfloat16* __restrict__ V,
                                          float* __restrict__ O,
                                          int D, float scale) {
    __shared__ __align__(128) __nv_bfloat16 sQ[64 * 256];
    __shared__ __align__(128) __nv_bfloat16 sK[64 * 256];
    __shared__ __align__(128) __nv_bfloat16 sV[64 * 256];   // V^T n-major canonical (4 k-steps x n256)
    __shared__ __align__(128) __nv_bfloat16 sP[64 * 64];    // P canonical (4 k-steps x 64x16)
    const int tid = threadIdx.x;
    char *bQ = (char*)sQ, *bK = (char*)sK, *bP = (char*)sP, *bV = (char*)sV;
    for (int idx = tid; idx < 64 * D; idx += 128) {
        int r = idx / D, c = idx % D;
        int st = c / 16, kk = c % 16;
        size_t off = (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
        *(__nv_bfloat16*)(bQ + off) = Q[r * D + c];
        *(__nv_bfloat16*)(bK + off) = K[r * D + c];
        // V^T: B element (n=d, k=kv_row): steps over kv (4 x 16), 256 n-rows of 8KB tiles
        int kv = r, d = c;
        int stv = kv / 16, kkv = kv % 16;
        size_t offv = (size_t)stv * 8192 + (d / 8) * 256 + (kkv / 8) * 128 + (d % 8) * 16 + (kkv % 8) * 2;
        *(__nv_bfloat16*)(bV + offv) = V[kv * D + d];
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    wgmma_fence();
    for (int st = 0; st < D / 16; st++) {
        unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
        unsigned long long db = make_desc(bK + st * 2048, 128, 256);
        wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
    }
    wgmma_commit();
    wgmma_wait<0>();

    // fragment coords: thread owns rows r0/r0+8, cols c0 + n8*8 (+1)
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    // causal+scale, rowmax/rowsum over the 4-thread row group (shfl over lane%4 dim = xor 1,2)
    float m[2] = {-1e30f, -1e30f};
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int rr = r0 + ((i % 4) / 2) * 8;
        int cc = c0 + (i / 4) * 8 + (i % 2);
        acc[i] = (cc <= rr) ? acc[i] * scale : -1e30f;
        int half = (i % 4) / 2;
        if (acc[i] > m[half]) m[half] = acc[i];
    }
    #pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
        m[0] = fmaxf(m[0], __shfl_xor_sync(0xffffffffu, m[0], o));
        m[1] = fmaxf(m[1], __shfl_xor_sync(0xffffffffu, m[1], o));
    }
    float l[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int half = (i % 4) / 2;
        float pv = expf(acc[i] - m[half]);
        acc[i] = pv;
        l[half] += pv;
    }
    #pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
        l[0] += __shfl_xor_sync(0xffffffffu, l[0], o);
        l[1] += __shfl_xor_sync(0xffffffffu, l[1], o);
    }
    // P -> bf16 canonical smem (steps over the kv dim: st = col/16)
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int rr = r0 + ((i % 4) / 2) * 8;
        int cc = c0 + (i / 4) * 8 + (i % 2);
        int st = cc / 16, kk = cc % 16;
        size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
        *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    // O = P.V : 4 n64-blocks x 4 k16-steps of m64n64k16
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;
    wgmma_fence();
    for (int st = 0; st < 4; st++) {
        unsigned long long da = make_desc(bP + st * 2048, 128, 256);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            unsigned long long db = make_desc(bV + st * 8192 + nb * 64 * 32, 128, 256);
            wgmma_m64n64k16_bf16(oacc[nb], da, db, st == 0 ? 0 : 1);
        }
    }
    wgmma_commit();
    wgmma_wait<0>();

    // O scatter with 1/l normalization
    float il[2] = {1.0f / l[0], 1.0f / l[1]};
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            int n8 = i / 4;
            int cc = nb * 64 + c0 + n8 * 8;
            O[(r0 + 0) * D + cc + 0] = oacc[nb][i + 0] * il[0];
            O[(r0 + 0) * D + cc + 1] = oacc[nb][i + 1] * il[0];
            O[(r0 + 8) * D + cc + 0] = oacc[nb][i + 2] * il[1];
            O[(r0 + 8) * D + cc + 1] = oacc[nb][i + 3] * il[1];
        }
}

int main() {
    const int D = 256;
    // hostile-ish operands
    __nv_bfloat16 *hQ = (__nv_bfloat16*)malloc(64 * D * 2), *hK = (__nv_bfloat16*)malloc(64 * D * 2);
    float* hS = (float*)malloc(64 * 64 * 4);
    float* refS = (float*)malloc(64 * 64 * 4);
    srand(11);
    for (int i = 0; i < 64 * D; i++) {
        hQ[i] = __float2bfloat16((rand() % 255 - 127) * 0.013f);
        hK[i] = __float2bfloat16((rand() % 255 - 127) * 0.009f);
    }
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++) {
            float s = 0.0f;
            for (int d = 0; d < D; d++)
                s += __bfloat162float(hQ[r * D + d]) * __bfloat162float(hK[c * D + d]);
            refS[r * 64 + c] = s;
        }
    __nv_bfloat16 *dQ, *dK; float* dS;
    CK(cudaMalloc(&dQ, 64 * D * 2)); CK(cudaMalloc(&dK, 64 * D * 2)); CK(cudaMalloc(&dS, 64 * 64 * 4));
    CK(cudaMemcpy(dQ, hQ, 64 * D * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, 64 * D * 2, cudaMemcpyHostToDevice));
    fa3_qk_probe<<<1, 128>>>(dQ, dK, dS, D);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hS, dS, 64 * 64 * 4, cudaMemcpyDeviceToHost));
    float max_rel = 0.0f; int bad = 0;
    for (int i = 0; i < 64 * 64; i++) {
        float r = fabsf(hS[i] - refS[i]) / fmaxf(fabsf(refS[i]), 1e-3f);
        if (r > max_rel) max_rel = r;
        if (r > 2e-2f) bad++;
    }
    printf("QK^T tile: max_rel %.3e bad %d/4096 %s (AL=%d AS=%d BL=%d BS=%d TA=%d TB=%d)\n",
           max_rel, bad, bad == 0 ? "MATCH" : "MISMATCH", A_LEAD, A_STRIDE, B_LEAD, B_STRIDE, TRANS_A, TRANS_B);
    if (bad) return 1;

    // ---- v2: full FA tile ----
    float scale = 1.0f / sqrtf((float)D);
    __nv_bfloat16* hV = (__nv_bfloat16*)malloc(64 * D * 2);
    for (int i = 0; i < 64 * D; i++) hV[i] = __float2bfloat16((rand() % 255 - 127) * 0.011f);
    float* refO = (float*)malloc(64 * D * 4);
    for (int r = 0; r < 64; r++) {
        float mx = -1e30f, sm = 0.0f, p[64];
        for (int c = 0; c <= r; c++) {
            float sv = refS[r * 64 + c] * scale;
            if (sv > mx) mx = sv;
        }
        for (int c = 0; c <= r; c++) {
            p[c] = expf(refS[r * 64 + c] * scale - mx);
            sm += p[c];
        }
        for (int d = 0; d < D; d++) {
            float o = 0.0f;
            // kernel semantics: P rounds to bf16 BEFORE PV (same as the shipped mma
            // kernel's sP); the l denominator is the f32 sum of unrounded p.
            for (int c = 0; c <= r; c++)
                o += __bfloat162float(__float2bfloat16(p[c])) * __bfloat162float(hV[c * D + d]);
            refO[r * D + d] = o / sm;
        }
    }
    __nv_bfloat16* dV; float* dO;
    CK(cudaMalloc(&dV, 64 * D * 2)); CK(cudaMalloc(&dO, 64 * D * 4));
    CK(cudaMemcpy(dV, hV, 64 * D * 2, cudaMemcpyHostToDevice));
    fa3_tile_probe<<<1, 128>>>(dQ, dK, dV, dO, D, scale);
    CK(cudaDeviceSynchronize());
    float* hO = (float*)malloc(64 * D * 4);
    CK(cudaMemcpy(hO, dO, 64 * D * 4, cudaMemcpyDeviceToHost));
    max_rel = 0.0f; bad = 0;
    for (int i = 0; i < 64 * D; i++) {
        float r = fabsf(hO[i] - refO[i]) / fmaxf(fabsf(refO[i]), 1e-3f);
        if (r > max_rel) max_rel = r;
        if (r > 3e-2f) bad++;
    }
    printf("FA tile:   max_rel %.3e bad %d/%d %s\n", max_rel, bad, 64 * D,
           bad == 0 ? "MATCH" : "MISMATCH");
    return bad == 0 ? 0 : 1;
}
