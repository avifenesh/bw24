// FA3 harness v1 (round 27+, ARCHITECTURE-H100.md): can a wgmma warpgroup FA beat the
// mma fa_prefill_bf16kv_pp (993us/layer at T=2048 = ~3.5% of bf16 TC peak, ncu: 11.6%
// occupancy, 255 regs — register-bound; sm_90a wgmma reads operands from SMEM, removing
// the Q/K register residency that caps the current kernel)?
//
// Playbook (the K4/K5 harness discipline): v1 proves ONE warpgroup QK^T tile against the
// CPU reference (descriptor layout iterated via -D knobs), then softmax+PV, then the
// pipelined kernel, then engine integration behind MEMRA_FA3.
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
#ifndef V9_REGS
#define V9_REGS 240
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

// ---- v3: the full kernel — grid (T/64, H); KV tile loop with online rescale; GQA. ----
// Layouts (engine): Q/O [T, H, D] rows; K/V [T, HKV, D] rows (bf16 mirrors).
// Q tile staged once (canonical); per KV tile: stage K,V^T -> S wgmma -> causal/scale ->
// online max/sum rescale (O *= alpha) -> P restage -> PV wgmma accumulate.
extern "C" __global__ void __launch_bounds__(128, 1)
fa3_v3(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    char* bQ = smem;                    // 32KB (16 steps x 2KB)
    char* bK = bQ + 64 * 256 * 2;       // 32KB
    char* bV = bK + 64 * 256 * 2;       // 32KB (4 kv-steps x 8KB, n-major)
    char* bP = bV + 64 * 256 * 2;       // 8KB
    const int tid = threadIdx.x;
    const int q0 = blockIdx.x * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);
    if (q0 >= T) return;

    for (int idx = tid; idx < 64 * D; idx += 128) {
        int r = idx / D, c = idx % D;
        int st = c / 16, kk = c % 16;
        size_t off = (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
        float qv = (q0 + r < T) ? __bfloat162float(Q[((size_t)(q0 + r) * H + head) * D + c]) : 0.0f;
        *(__nv_bfloat16*)(bQ + off) = __float2bfloat16(qv);
    }

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int kv_end = q0 + 64 <= T ? q0 + 64 : T;   // causal: tiles up to the diagonal
    for (int k0 = 0; k0 < kv_end; k0 += 64) {
        // stage K rows + V^T for this tile
        for (int idx = tid; idx < 64 * D; idx += 128) {
            int r = idx / D, c = idx % D;
            int st = c / 16, kk = c % 16;
            size_t off = (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
            float kvv = (k0 + r < T) ? __bfloat162float(K[((size_t)(k0 + r) * HKV + kvh) * D + c]) : 0.0f;
            *(__nv_bfloat16*)(bK + off) = __float2bfloat16(kvv);
            int stv = r / 16, kkv = r % 16;
            size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2;
            float vv = (k0 + r < T) ? __bfloat162float(V[((size_t)(k0 + r) * HKV + kvh) * D + c]) : 0.0f;
            *(__nv_bfloat16*)(bV + offv) = __float2bfloat16(vv);
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        float acc[32];
        wgmma_fence();
        for (int st = 0; st < D / 16; st++) {
            unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
            unsigned long long db = make_desc(bK + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait<0>();

        // causal mask (global coords) + scale; new running max
        float mn[2] = {m[0], m[1]};
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int rr = q0 + r0 + ((i % 4) / 2) * 8;
            int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
            acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
            int half = (i % 4) / 2;
            if (acc[i] > mn[half]) mn[half] = acc[i];
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
        if (m[0] == -1e30f) alpha[0] = 0.0f;
        if (m[1] == -1e30f) alpha[1] = 0.0f;
        m[0] = mn[0]; m[1] = mn[1];
        float ladd[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int half = (i % 4) / 2;
            float pv = expf(acc[i] - m[half]);
            acc[i] = pv;
            ladd[half] += pv;
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
            ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
        }
        l[0] = l[0] * alpha[0] + ladd[0];
        l[1] = l[1] * alpha[1] + ladd[1];
        // O rescale
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
        // P restage
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
        wgmma_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long da = make_desc(bP + st * 2048, 128, 256);
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                unsigned long long db = make_desc(bV + st * 8192 + nb * 64 * 32, 128, 256);
                wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
            }
        }
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();   // K/V smem reused next tile
    }
    // write O rows (skip pads)
    float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            int n8 = i / 4;
            int cc = nb * 64 + c0 + n8 * 8;
            int ra = q0 + r0, rb = q0 + r0 + 8;
            if (ra < T) {
                O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
            }
            if (rb < T) {
                O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
            }
        }
}

// ---- v4: v3 + int4 cp.async K staging (canonical 16B segments are contiguous in
// global K rows) + 2-stage K/V ring (prefetch next tile behind current compute).
// V^T staging stays scalar (transposed scatter) — measured next if it dominates.
__device__ __forceinline__ void fa3_cp16(void* dst, const void* src, int pred_bytes) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(pred_bytes));
}
__device__ __forceinline__ void fa3_cp_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void fa3_cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

extern "C" __global__ void __launch_bounds__(128, 1)
fa3_v4(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    char* bQ = smem;                          // 32KB
    char* bK[2] = { bQ + 65536, bQ + 65536 + 32768 };
    char* bV[2] = { bQ + 65536 + 65536, bQ + 65536 + 65536 + 32768 };
    char* bP = bQ + 65536 + 131072;           // 8KB  (total 168KB)
    const int tid = threadIdx.x;
    const int q0 = blockIdx.x * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);
    if (q0 >= T) return;

    // Q staged once: int4 segments (r, st, h16): src Q row contiguous 16B
    for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
        int r = seg / (D / 8), s8v = seg % (D / 8);
        int st = s8v / 2, h16 = s8v % 2;
        char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
        int gr = q0 + r;
        fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                 gr < T ? 16 : 0);
    }
    fa3_cp_commit();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int kv_end = q0 + 64 <= T ? q0 + 64 : T;
    const int n_tiles = (kv_end + 63) / 64;

    // stage(kv-tile, buf): K via int4 cp.async; V^T scalar (after the cp group commits)
    #define V4_STAGE_K(t_, b_) do {                                                     \
        int k0_ = (t_) * 64;                                                            \
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {                           \
            int r = seg / (D / 8), s8v = seg % (D / 8);                                 \
            int st = s8v / 2, h16 = s8v % 2;                                            \
            char* dst = bK[b_] + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;  \
            int gr = k0_ + r;                                                           \
            fa3_cp16(dst, K + ((size_t)(gr < T ? gr : T - 1) * HKV + kvh) * D + st * 16 + h16 * 8, \
                     gr < T ? 16 : 0);                                                  \
        }                                                                               \
        fa3_cp_commit();                                                                \
    } while (0)
    #define V4_STAGE_V(t_, b_) do {                                                    \
        int k0_ = (t_) * 64;                                                            \
        for (int idx = tid; idx < 64 * D; idx += 128) {                                 \
            int r = idx / D, c = idx % D;                                               \
            int stv = r / 16, kkv = r % 16;                                             \
            size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2; \
            int gr = k0_ + r;                                                           \
            *(__nv_bfloat16*)(bV[b_] + offv) = (gr < T)                                 \
                ? V[((size_t)gr * HKV + kvh) * D + c] : __float2bfloat16(0.0f);         \
        }                                                                               \
    } while (0)

    V4_STAGE_K(0, 0);
    V4_STAGE_V(0, 0);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        fa3_cp_wait<0>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // prefetch next tile behind this tile's compute
        if (t + 1 < n_tiles) V4_STAGE_K(t + 1, cur ^ 1);

        float acc[32];
        wgmma_fence();
        for (int st = 0; st < D / 16; st++) {
            unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
            unsigned long long db = make_desc(bK[cur] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        // stage next V while S cooks
        if (t + 1 < n_tiles) V4_STAGE_V(t + 1, cur ^ 1);
        wgmma_wait<0>();

        float mn[2] = {m[0], m[1]};
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int rr = q0 + r0 + ((i % 4) / 2) * 8;
            int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
            acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
            int half = (i % 4) / 2;
            if (acc[i] > mn[half]) mn[half] = acc[i];
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
        if (m[0] == -1e30f) alpha[0] = 0.0f;
        if (m[1] == -1e30f) alpha[1] = 0.0f;
        m[0] = mn[0]; m[1] = mn[1];
        float ladd[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int half = (i % 4) / 2;
            float pv = expf(acc[i] - m[half]);
            acc[i] = pv;
            ladd[half] += pv;
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
            ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
        }
        l[0] = l[0] * alpha[0] + ladd[0];
        l[1] = l[1] * alpha[1] + ladd[1];
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
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
        wgmma_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long da = make_desc(bP + st * 2048, 128, 256);
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
            }
        }
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();
    }
    float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            int n8 = i / 4;
            int cc = nb * 64 + c0 + n8 * 8;
            int ra = q0 + r0, rb = q0 + r0 + 8;
            if (ra < T) {
                O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
            }
            if (rb < T) {
                O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
            }
        }
}

// ---- v5: 2 warpgroups x 64 q-rows per CTA sharing the K/V ring (halves K/V staging
// traffic; the warpgroups' softmax/wgmma phases interleave on the SM). smem 208KB.
extern "C" __global__ void __launch_bounds__(256, 1)
fa3_v5(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;         // warpgroup 0/1
    const int tid = threadIdx.x % 128;
    char* bQ = smem + wg * 32768;             // 2 x 32KB
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + wg * 8192;     // 2 x 8KB (total 212992 = 208KB)
    const int q0 = (blockIdx.x * 2 + wg) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    // Q staged per warpgroup
    if (q0 < T) {
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8v = seg % (D / 8);
            int st = s8v / 2, h16 = s8v % 2;
            char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
            int gr = q0 + r;
            fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                     gr < T ? 16 : 0);
        }
    }
    fa3_cp_commit();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    // BOTH warpgroups loop over the union of needed kv tiles (wg1's diagonal is 64
    // further); wg0 skips tiles past its own diagonal (no work, but stays in the
    // __syncthreads() cadence — staging is cooperative across all 256 threads).
    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;   // wg1's q0
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;

    #define V5_STAGE_K(t_, b_) do {                                                     \
        int k0_ = (t_) * 64;                                                            \
        for (int seg = threadIdx.x; seg < 64 * (D / 8); seg += 256) {                   \
            int r = seg / (D / 8), s8v = seg % (D / 8);                                 \
            int st = s8v / 2, h16 = s8v % 2;                                            \
            char* dst = bK[b_] + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;  \
            int gr = k0_ + r;                                                           \
            fa3_cp16(dst, K + ((size_t)(gr < T ? gr : T - 1) * HKV + kvh) * D + st * 16 + h16 * 8, \
                     gr < T ? 16 : 0);                                                  \
        }                                                                               \
        fa3_cp_commit();                                                                \
    } while (0)
    #define V5_STAGE_V(t_, b_) do {                                                    \
        int k0_ = (t_) * 64;                                                            \
        for (int idx = threadIdx.x; idx < 64 * D; idx += 256) {                         \
            int r = idx / D, c = idx % D;                                               \
            int stv = r / 16, kkv = r % 16;                                             \
            size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2; \
            int gr = k0_ + r;                                                           \
            *(__nv_bfloat16*)(bV[b_] + offv) = (gr < T)                                 \
                ? V[((size_t)gr * HKV + kvh) * D + c] : __float2bfloat16(0.0f);         \
        }                                                                               \
    } while (0)

    V5_STAGE_K(0, 0);
    V5_STAGE_V(0, 0);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        fa3_cp_wait<0>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        if (t + 1 < n_tiles) V5_STAGE_K(t + 1, cur ^ 1);

        const bool active = (q0 < T) && (k0 < kv_end_own);
        float acc[32];
        if (active) {
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
                unsigned long long db = make_desc(bK[cur] + st * 2048, 128, 256);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
        }
        if (t + 1 < n_tiles) V5_STAGE_V(t + 1, cur ^ 1);
        if (active) {
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            // bP is per-warpgroup: warpgroup-local barrier suffices
            asm volatile("bar.sync %0, 128;" :: "r"(wg + 1));
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

// ---- v6 (corrected cadence): softmax/PV overlap. PV(t) stays PENDING across the
// iteration boundary and retires just before the oacc rescale of softmax(t+1) — the
// scalar mask/max/exp work runs under the PV wgmmas. V(t+1) restage happens only
// after PV(t-1) (same buffer parity) retires; K(t+2) restage after S(t) consumed it.
extern "C" __global__ void __launch_bounds__(256, 1)
fa3_v6(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    char* bQ = smem + wg * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP[2] = { smem + 196608 + wg * 16384, smem + 196608 + wg * 16384 + 8192 };
    const int q0 = (blockIdx.x * 2 + wg) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    if (q0 < T) {
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8v = seg % (D / 8);
            int st = s8v / 2, h16 = s8v % 2;
            char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
            int gr = q0 + r;
            fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                     gr < T ? 16 : 0);
        }
    }
    fa3_cp_commit();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    // prologue: K0+V0; K1 behind; S(0)
    V5_STAGE_K(0, 0);
    V5_STAGE_V(0, 0);
    fa3_cp_wait<0>();
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    if (1 < n_tiles) V5_STAGE_K(1, 1);
    float acc[2][32];
    if (n_own > 0) {
        wgmma_fence();
        for (int st = 0; st < D / 16; st++) {
            unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
            unsigned long long db = make_desc(bK[0] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc[0], da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
    }
    int have_s = 0;
    int pv_pending = 0;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        const bool has_next_own = (t + 1) < n_own;
        if (active) {
            float* a = acc[have_s];
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                a[i] = (cc <= rr && cc < T) ? a[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (a[i] > mn[half]) mn[half] = a[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(a[i] - m[half]);
                a[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            // retire PV(t-1) ONLY NOW (it ran under the scalar work above)
            if (pv_pending) { wgmma_wait<0>(); pv_pending = 0; }
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP[cur] + off) = __float2bfloat16(a[i]);
            }
            asm volatile("bar.sync %0, 128;" :: "r"(wg + 1));
            asm volatile("fence.proxy.async.shared::cta;");
        } else if (pv_pending) {
            wgmma_wait<0>(); pv_pending = 0;
        }
        // V(t+1) into bV[(t+1)&1]: its previous reader PV(t-1) (same parity) is retired.
        if (t + 1 < n_tiles) V5_STAGE_V(t + 1, cur ^ 1);
        // K(t+1) cp group must be retired before S(t+1) reads it (K(t+2) group may pend)
        __syncthreads();
        if (t + 2 < n_tiles) { V5_STAGE_K(t + 2, cur); fa3_cp_wait<1>(); }
        else fa3_cp_wait<0>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        if (active) {
            wgmma_fence();
            if (has_next_own) {
                for (int st = 0; st < D / 16; st++) {
                    unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
                    unsigned long long db = make_desc(bK[cur ^ 1] + st * 2048, 128, 256);
                    wgmma_m64n64k16_bf16(acc[have_s ^ 1], da, db, st == 0 ? 0 : 1);
                }
                wgmma_commit();
            }
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP[cur] + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            if (has_next_own) {
                wgmma_wait<1>();     // S(t+1) retired; PV(t) pends into softmax(t+1)
                pv_pending = 1;
            } else {
                wgmma_wait<0>();
                pv_pending = 0;
            }
            have_s ^= 1;
        }
        __syncthreads();
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

// ---- v7: split-D (canonical FA3 hd256 shape). ONE 64-row q-tile per CTA; WG0 owns
// S/softmax/P + PV cols 0-127; WG1 waits on bP and PVs cols 128-255. O per warpgroup
// = 64x128 f32 / 128 thr = 64 regs (v5 halved). l broadcast via smem for WG1's write.
extern "C" __global__ void __launch_bounds__(256, 1)
fa3_v7(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    char* bQ = smem;                       // 32KB (WG0 only reads)
    char* bK[2] = { smem + 32768, smem + 65536 };
    char* bV[2] = { smem + 98304, smem + 131072 };
    char* bP = smem + 163840;              // 8KB
    float* sL = (float*)(smem + 172032);   // 64 f32 running-l + 64 alpha
    float* sAl = sL + 64;
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    const int q0 = blockIdx.x * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);
    if (q0 >= T) return;

    if (wg == 0) {
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8v = seg % (D / 8);
            int st = s8v / 2, h16 = s8v % 2;
            char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
            int gr = q0 + r;
            fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                     gr < T ? 16 : 0);
        }
    }
    fa3_cp_commit();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[2][32];                     // this WG's 128-col half (2 n64 blocks)
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int kv_end = q0 + 64 <= T ? q0 + 64 : T;
    const int n_tiles = (kv_end + 63) / 64;

    V5_STAGE_K(0, 0);
    V5_STAGE_V(0, 0);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        fa3_cp_wait<0>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        if (t + 1 < n_tiles) V5_STAGE_K(t + 1, cur ^ 1);

        if (wg == 0) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
                unsigned long long db = make_desc(bK[cur] + st * 2048, 128, 256);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            // publish alpha (and final l after the loop) for WG1
            if (lane % 4 == 0) {
                sAl[r0] = alpha[0];
                sAl[r0 + 8] = alpha[1];
            }
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
        }
        if (t + 1 < n_tiles) V5_STAGE_V(t + 1, cur ^ 1);
        __syncthreads();                    // bP + sAl visible to WG1
        asm volatile("fence.proxy.async.shared::cta;");
        {
            float al[2] = {sAl[r0], sAl[r0 + 8]};
            #pragma unroll
            for (int nb = 0; nb < 2; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= al[(i % 4) / 2];
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 2; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + (wg * 2 + nb) * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();
    }
    if (wg == 0 && (lane % 4) == 0) {
        sL[r0] = l[0];
        sL[r0 + 8] = l[1];
    }
    __syncthreads();
    {
        float ll[2] = {sL[r0], sL[r0 + 8]};
        float il[2] = {ll[0] > 0.0f ? 1.0f / ll[0] : 0.0f, ll[1] > 0.0f ? 1.0f / ll[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 2; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = (wg * 2 + nb) * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

// ---- v8a: producer warpgroup (no TMA yet). CTA = 384 thr: WG0/WG1 = v5's consumers
// (staging removed), WG2 stages the K/V ring and signals via named barriers:
//   FULL[s] (id 2+s, count 384): producer arrives after staging stage s; consumers sync.
//   EMPTY[s] (id 4+s, count 384): consumers arrive when done with stage s; producer syncs.
// Consumer-only sync: id 1 count 256. Per-WG P barrier: ids 6/7 count 128.
__device__ __forceinline__ void bar_sync(int id, int cnt) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt));
}
__device__ __forceinline__ void bar_arrive(int id, int cnt) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(cnt));
}

extern "C" __global__ void __launch_bounds__(384, 1)
fa3_v8(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;          // 0,1 consumers; 2 producer
    const int tid = threadIdx.x % 128;

    char* bQ = smem + (wg < 2 ? wg : 0) * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + (wg < 2 ? wg : 0) * 8192;
    const int q0 = (blockIdx.x * 2 + (wg < 2 ? wg : 0)) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;

    if (wg == 2) {
        // ---- producer (setmaxnreg contract: release to 24; probe-verified) ----
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        for (int t = 0; t < n_tiles; t++) {
            const int b = t & 1;
            if (t >= 2) bar_sync(4 + b, 384);
            int k0 = t * 64;
            for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
                int r = seg / (D / 8), s8v = seg % (D / 8);
                int st = s8v / 2, h16 = s8v % 2;
                char* dst = bK[b] + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
                int gr = k0 + r;
                fa3_cp16(dst, K + ((size_t)(gr < T ? gr : T - 1) * HKV + kvh) * D + st * 16 + h16 * 8,
                         gr < T ? 16 : 0);
            }
            fa3_cp_commit();
            for (int idx = tid; idx < 64 * D; idx += 128) {
                int r = idx / D, c = idx % D;
                int stv = r / 16, kkv = r % 16;
                size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2;
                int gr = k0 + r;
                *(__nv_bfloat16*)(bV[b] + offv) = (gr < T)
                    ? V[((size_t)gr * HKV + kvh) * D + c] : __float2bfloat16(0.0f);
            }
            fa3_cp_wait<0>();
            asm volatile("fence.proxy.async.shared::cta;");
            bar_arrive(2 + b, 384);
        }
        return;
    }

    // ---- consumers (v5 shape; acquire 240 — post-inc region, probe-verified) ----
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    if (q0 < T) {
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8v = seg % (D / 8);
            int st = s8v / 2, h16 = s8v % 2;
            char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
            int gr = q0 + r;
            fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                     gr < T ? 16 : 0);
        }
    }
    fa3_cp_commit();
    fa3_cp_wait<0>();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        bar_sync(2 + cur, 384);                 // stage `cur` FULL
        asm volatile("fence.proxy.async.shared::cta;");
        if (active) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
                unsigned long long db = make_desc(bK[cur] + st * 2048, 128, 256);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            bar_sync(6 + wg, 128);              // per-WG P visibility
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        bar_arrive(4 + cur, 384);               // stage `cur` EMPTY
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

// ---- v9: async producer via mbarriers (no TMA needed). Producer issues each stage's
// cp.asyncs + V stores, then cp.async.mbarrier.arrive (fires when ITS copies drain) and
// moves on WITHOUT waiting. Consumers mbarrier.try_wait FULL; arrive EMPTY when done.
// setmaxnreg split per the cracked contract.
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(void* mbar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(a));
}
__device__ __forceinline__ void mbar_cp_arrive(void* mbar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(a));
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(a), "r"(phase));
    }
}

extern "C" __global__ void __launch_bounds__(384, 1)
fa3_v9(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
       const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
       int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    char* bQ = smem + (wg < 2 ? wg : 0) * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + (wg < 2 ? wg : 0) * 8192;
    // mbarriers: FULL[2] (producer 128 arrivals), EMPTY[2] (consumer 256 arrivals)
    unsigned long long* mFull = (unsigned long long*)(smem + 212992);
    unsigned long long* mEmpty = mFull + 2;
    const int q0 = (blockIdx.x * 2 + (wg < 2 ? wg : 0)) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    if (threadIdx.x == 0) {
        mbar_init(&mFull[0], 128); mbar_init(&mFull[1], 128);
        mbar_init(&mEmpty[0], 256); mbar_init(&mEmpty[1], 256);
    }
    __syncthreads();

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;

    if (wg == 2) {
        // ---- producer: never self-waits on copies ----
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        for (int t = 0; t < n_tiles; t++) {
            const int b = t & 1;
            if (t >= 2) mbar_wait(&mEmpty[b], ((t - 2) >> 1) & 1);
            int k0 = t * 64;
            for (int idx = tid; idx < 64 * D; idx += 128) {
                int r = idx / D, c = idx % D;
                int stv = r / 16, kkv = r % 16;
                size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2;
                int gr = k0 + r;
                *(__nv_bfloat16*)(bV[b] + offv) = (gr < T)
                    ? V[((size_t)gr * HKV + kvh) * D + c] : __float2bfloat16(0.0f);
            }
            for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
                int r = seg / (D / 8), s8v = seg % (D / 8);
                int st = s8v / 2, h16 = s8v % 2;
                char* dst = bK[b] + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
                int gr = k0 + r;
                fa3_cp16(dst, K + ((size_t)(gr < T ? gr : T - 1) * HKV + kvh) * D + st * 16 + h16 * 8,
                         gr < T ? 16 : 0);
            }
            mbar_cp_arrive(&mFull[b]);   // fires when THIS thread's cp.asyncs drain
        }
        return;
    }

    // ---- consumers ----
    asm volatile("setmaxnreg.inc.sync.aligned.u32 " STR(V9_REGS) ";");
    if (q0 < T) {
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8v = seg % (D / 8);
            int st = s8v / 2, h16 = s8v % 2;
            char* dst = bQ + st * 2048 + (r / 8) * 256 + h16 * 128 + (r % 8) * 16;
            int gr = q0 + r;
            fa3_cp16(dst, Q + ((size_t)(gr < T ? gr : T - 1) * H + head) * D + st * 16 + h16 * 8,
                     gr < T ? 16 : 0);
        }
    }
    fa3_cp_commit();
    fa3_cp_wait<0>();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        mbar_wait(&mFull[cur], (t >> 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;");
        if (active) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = make_desc(bQ + st * 2048, 128, 256);
                unsigned long long db = make_desc(bK[cur] + st * 2048, 128, 256);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            bar_sync(6 + wg, 128);
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        mbar_arrive(&mEmpty[cur]);
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

__device__ __forceinline__ void wgmma_m64n64k16_bf16_tb(float acc[32], unsigned long long da,
                                                        unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 1;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// ---- v10: TMA swizzled Q/K inside the v5 shape. Single thread issues TMA per ring
// stage (expect-tx mbarrier); V^T stays cooperative-scalar (transpose); PV path keeps
// canonical descriptors. Swizzle pairing per probe_tma: mode=1, SBO=1024, LBO ignored,
// k16 slice = atom base + j*32.
#include <cuda.h>
__device__ __forceinline__ void v10_mbar_init(void* mbar, unsigned count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void v10_expect_tx(void* mbar, unsigned bytes) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void v10_arrive(void* mbar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(a));
}
__device__ __forceinline__ void v10_wait(void* mbar, unsigned phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(a), "r"(phase));
    }
}
__device__ __forceinline__ void v10_tma3d(const CUtensorMap* map, void* dst,
                                          int c0, int c1, int c2, void* mbar) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    unsigned b = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
}
__device__ __forceinline__ unsigned long long v10_desc_swz(const void* p) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((1024u >> 4) & 0x3FFF) << 32;   // SBO=1024, LBO=0
    d |= (unsigned long long)1 << 62;                          // 128B swizzle
    return d;
}

extern "C" __global__ void __launch_bounds__(256, 1)
fa3_v10(const __grid_constant__ CUtensorMap tQ, const __grid_constant__ CUtensorMap tK,
        const __nv_bfloat16* __restrict__ V, float* __restrict__ O,
        int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    char* bQ = smem + wg * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + wg * 8192;
    unsigned long long* mFull = (unsigned long long*)(smem + 212992);
    unsigned long long* mEmpty = mFull + 2;
    const int q0 = (blockIdx.x * 2 + wg) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    if (threadIdx.x == 0) {
        v10_mbar_init(&mFull[0], 1); v10_mbar_init(&mFull[1], 1);
        v10_mbar_init(&mEmpty[0], 256); v10_mbar_init(&mEmpty[1], 256);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;
    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    // prologue: thread 0 issues Q (both WGs) + K(0) on FULL[0]
    if (threadIdx.x == 0) {
        v10_expect_tx(&mFull[0], 2 * 32768 + 32768);
        for (int a = 0; a < 4; a++) {
            v10_tma3d(&tQ, smem + a * 8192, a * 64, head, (blockIdx.x * 2) * 64, &mFull[0]);
            v10_tma3d(&tQ, smem + 32768 + a * 8192, a * 64, head, (blockIdx.x * 2 + 1) * 64, &mFull[0]);
            v10_tma3d(&tK, bK[0] + a * 8192, a * 64, kvh, 0, &mFull[0]);
        }
    }
    // V(0) cooperative
    {
        for (int idx = threadIdx.x; idx < 64 * D; idx += 256) {
            int r = idx / D, c = idx % D;
            int stv = r / 16, kkv = r % 16;
            size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2;
            *(__nv_bfloat16*)(bV[0] + offv) = (r < T)
                ? V[((size_t)r * HKV + kvh) * D + c] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        v10_wait(&mFull[cur], (t >> 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;");
        // issue K(t+1) (stage freed two tiles ago; EMPTY guards reuse)
        if (threadIdx.x == 0 && t + 1 < n_tiles) {
            if (t + 1 >= 2) v10_wait(&mEmpty[cur ^ 1], ((t - 1) >> 1) & 1);
            v10_expect_tx(&mFull[cur ^ 1], 32768);
            for (int a = 0; a < 4; a++)
                v10_tma3d(&tK, bK[cur ^ 1] + a * 8192, a * 64, kvh, (t + 1) * 64, &mFull[cur ^ 1]);
        }
        if (active) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = v10_desc_swz(bQ + (st / 4) * 8192 + (st % 4) * 32);
                unsigned long long db = v10_desc_swz(bK[cur] + (st / 4) * 8192 + (st % 4) * 32);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            bar_sync(6 + wg, 128);
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    unsigned long long db = make_desc(bV[cur] + st * 8192 + nb * 64 * 32, 128, 256);
                    wgmma_m64n64k16_bf16(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        // V(t+1) cooperative into the other stage, then flag EMPTY for K reuse
        if (t + 1 < n_tiles) {
            int k1 = (t + 1) * 64;
            for (int idx = threadIdx.x; idx < 64 * D; idx += 256) {
                int r = idx / D, c = idx % D;
                int stv = r / 16, kkv = r % 16;
                size_t offv = (size_t)stv * 8192 + (c / 8) * 256 + (kkv / 8) * 128 + (c % 8) * 16 + (kkv % 8) * 2;
                int gr = k1 + r;
                *(__nv_bfloat16*)(bV[cur ^ 1] + offv) = (gr < T)
                    ? V[((size_t)gr * HKV + kvh) * D + c] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        v10_arrive(&mEmpty[cur]);
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
fa3_v11(const __grid_constant__ CUtensorMap tQ, const __grid_constant__ CUtensorMap tK,
        const __grid_constant__ CUtensorMap tV, float* __restrict__ O,
        int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    char* bQ = smem + wg * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + wg * 8192;
    unsigned long long* mFull = (unsigned long long*)(smem + 212992);
    unsigned long long* mEmpty = mFull + 2;
    const int q0 = (blockIdx.x * 2 + wg) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    if (threadIdx.x == 0) {
        v10_mbar_init(&mFull[0], 1); v10_mbar_init(&mFull[1], 1);
        v10_mbar_init(&mEmpty[0], 256); v10_mbar_init(&mEmpty[1], 256);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;
    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    // prologue: thread 0 issues Q (both WGs) + K(0) + V(0) on FULL[0] — all TMA
    if (threadIdx.x == 0) {
        v10_expect_tx(&mFull[0], 2 * 32768 + 2 * 32768);
        for (int a = 0; a < 4; a++) {
            v10_tma3d(&tQ, smem + a * 8192, a * 64, head, (blockIdx.x * 2) * 64, &mFull[0]);
            v10_tma3d(&tQ, smem + 32768 + a * 8192, a * 64, head, (blockIdx.x * 2 + 1) * 64, &mFull[0]);
            v10_tma3d(&tK, bK[0] + a * 8192, a * 64, kvh, 0, &mFull[0]);
            v10_tma3d(&tV, bV[0] + a * 8192, a * 64, kvh, 0, &mFull[0]);
        }
    }
    __syncthreads();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        v10_wait(&mFull[cur], (t >> 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;");
        // issue K(t+1) (stage freed two tiles ago; EMPTY guards reuse)
        if (threadIdx.x == 0 && t + 1 < n_tiles) {
            if (t + 1 >= 2) v10_wait(&mEmpty[cur ^ 1], ((t - 1) >> 1) & 1);
            v10_expect_tx(&mFull[cur ^ 1], 2 * 32768);
            for (int a = 0; a < 4; a++) {
                v10_tma3d(&tK, bK[cur ^ 1] + a * 8192, a * 64, kvh, (t + 1) * 64, &mFull[cur ^ 1]);
                v10_tma3d(&tV, bV[cur ^ 1] + a * 8192, a * 64, kvh, (t + 1) * 64, &mFull[cur ^ 1]);
            }
        }
        if (active) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = v10_desc_swz(bQ + (st / 4) * 8192 + (st % 4) * 32);
                unsigned long long db = v10_desc_swz(bK[cur] + (st / 4) * 8192 + (st % 4) * 32);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            bar_sync(6 + wg, 128);
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    // V rows TMA-swizzled; trans_b MN-major B: atom = nb*8192, k-slice = st*2048
                    unsigned long long db = v10_desc_swz(bV[cur] + nb * 8192 + st * 2048);
                    wgmma_m64n64k16_bf16_tb(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();
        v10_arrive(&mEmpty[cur]);
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
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
    if (bad) return 1;

    // ---- v3: correctness at T=192 (3 tiles, exercises rescale + GQA) + rate at T=2048 ----
    {
        const int H = 16, HKV = 4;
        for (int T : {64, 128, 192, 2048}) {
            size_t nq = (size_t)T * H * D, nkv = (size_t)T * HKV * D;
            __nv_bfloat16 *q = (__nv_bfloat16*)malloc(nq * 2), *k = (__nv_bfloat16*)malloc(nkv * 2), *vv = (__nv_bfloat16*)malloc(nkv * 2);
            for (size_t i = 0; i < nq; i++) q[i] = __float2bfloat16((rand() % 255 - 127) * 0.011f);
            for (size_t i = 0; i < nkv; i++) { k[i] = __float2bfloat16((rand() % 255 - 127) * 0.009f); vv[i] = __float2bfloat16((rand() % 255 - 127) * 0.012f); }
            __nv_bfloat16 *dq, *dk, *dv; float* dout;
            CK(cudaMalloc(&dq, nq * 2)); CK(cudaMalloc(&dk, nkv * 2)); CK(cudaMalloc(&dv, nkv * 2)); CK(cudaMalloc(&dout, nq * 4));
            CK(cudaMemcpy(dq, q, nq * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dk, k, nkv * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, vv, nkv * 2, cudaMemcpyHostToDevice));
            int shmem = 64 * 256 * 2 * 3 + 64 * 64 * 2;
            int shmem4 = 65536 + 131072 + 8192;
            int shmem5 = 212992;
            int shmem6 = 196608 + 32768;
            int shmem7 = 172032 + 512;
            int shmem8 = 212992;
            int shmem9 = 212992 + 64;
            int shmem10 = 212992 + 64;
            CUtensorMap tQm, tKm;
            {
                cuuint64_t gdq[3] = { (cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T };
                cuuint64_t gsq[2] = { (cuuint64_t)D * 2, (cuuint64_t)H * D * 2 };
                cuuint32_t bxq[3] = { 64, 1, 64 };
                cuuint32_t esq[3] = { 1, 1, 1 };
                if (cuTensorMapEncodeTiled(&tQm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dq,
                        gdq, gsq, bxq, esq, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) { printf("tQ encode fail\n"); exit(1); }
                cuuint64_t gdk[3] = { (cuuint64_t)D, (cuuint64_t)HKV, (cuuint64_t)T };
                cuuint64_t gsk[2] = { (cuuint64_t)D * 2, (cuuint64_t)HKV * D * 2 };
                if (cuTensorMapEncodeTiled(&tKm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dk,
                        gdk, gsk, bxq, esq, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) { printf("tK encode fail\n"); exit(1); }
            }
            CK(cudaFuncSetAttribute(fa3_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            CK(cudaFuncSetAttribute(fa3_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem4));
            CK(cudaFuncSetAttribute(fa3_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem5));
            CK(cudaFuncSetAttribute(fa3_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem6));
            CK(cudaFuncSetAttribute(fa3_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem7));
            CK(cudaFuncSetAttribute(fa3_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem8));
            CK(cudaFuncSetAttribute(fa3_v9, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem9));
            CK(cudaFuncSetAttribute(fa3_v10, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem10));
            dim3 grid((T + 63) / 64, H);
            dim3 grid5((T + 127) / 128, H);
            CUtensorMap tVm;
            {
                cuuint64_t gdk[3] = { (cuuint64_t)D, (cuuint64_t)HKV, (cuuint64_t)T };
                cuuint64_t gsk[2] = { (cuuint64_t)D * 2, (cuuint64_t)HKV * D * 2 };
                cuuint32_t bxq[3] = { 64, 1, 64 };
                cuuint32_t esq[3] = { 1, 1, 1 };
                if (cuTensorMapEncodeTiled(&tVm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dv,
                        gdk, gsk, bxq, esq, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) { printf("tV encode fail\n"); exit(1); }
            }
            CK(cudaFuncSetAttribute(fa3_v11, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem10));
            fa3_v11<<<grid5, 256, shmem10>>>(tQm, tKm, tVm, dout, T, H, HKV, D, scale);
            CK(cudaDeviceSynchronize());
            if (T != 2048) {
                float* ho = (float*)malloc(nq * 4);
                CK(cudaMemcpy(ho, dout, nq * 4, cudaMemcpyDeviceToHost));
                double worst = 0; int nb2 = 0;
                // ONLINE reference — the kernel's exact arithmetic class: per-64-tile max,
                // P rounded bf16 at the tile max, O rescaled by alpha (the shipped mma
                // kernel makes the same tradeoff; engine arbitration = the argmax battery).
                for (int h = 0; h < H; h++) {
                    int kh = h / (H / HKV);
                    for (int r = 0; r < T; r++) {
                        static float sv_row[4096];
                        for (int c = 0; c <= r; c++) {
                            float sv = 0;
                            for (int d = 0; d < D; d++)
                                sv += __bfloat162float(q[((size_t)r * H + h) * D + d]) * __bfloat162float(k[((size_t)c * HKV + kh) * D + d]);
                            sv_row[c] = sv * scale;
                        }
                        for (int d = 0; d < D; d += 37) {   // spot columns
                            float m2 = -1e30f, l2 = 0.0f, o = 0.0f;
                            for (int k0 = 0; k0 <= r; k0 += 64) {
                                int ce = (k0 + 64 <= r + 1) ? k0 + 64 : r + 1;
                                float tm = m2;
                                for (int c = k0; c < ce; c++) if (sv_row[c] > tm) tm = sv_row[c];
                                float al = (m2 == -1e30f) ? 0.0f : expf(m2 - tm);
                                m2 = tm;
                                o *= al; l2 *= al;
                                for (int c = k0; c < ce; c++) {
                                    float pv = expf(sv_row[c] - m2);
                                    l2 += pv;
                                    o += __bfloat162float(__float2bfloat16(pv)) * __bfloat162float(vv[((size_t)c * HKV + kh) * D + d]);
                                }
                            }
                            o /= l2;
                            float got = ho[((size_t)r * H + h) * D + d];
                            float rel = fabsf(got - o) / fmaxf(fabsf(o), 1e-3f);
                            if (rel > worst) worst = rel;
                            if (rel > 5e-3f && fabsf(got - o) > 2e-3f) nb2++;
                        }
                    }
                }
                printf("v3 T=%d:  max_rel %.3e bad %d %s\n", T, worst, nb2, nb2 == 0 ? "MATCH" : "MISMATCH");
                if (nb2 && T == 128) {   // per-row diagnostics, head 0
                    for (int r = 0; r < T; r += 8) {
                        double w = 0;
                        for (int rr2 = r; rr2 < r + 8; rr2++) {
                            // recompute row worst for head 0 only
                            int h = 0, kh = 0;
                            float mx = -1e30f, sm = 0.0f; static float p2[4096];
                            for (int c = 0; c <= rr2; c++) {
                                float sv = 0;
                                for (int d = 0; d < D; d++) sv += __bfloat162float(q[((size_t)rr2 * H + h) * D + d]) * __bfloat162float(k[((size_t)c * HKV + kh) * D + d]);
                                p2[c] = sv * scale; if (p2[c] > mx) mx = p2[c];
                            }
                            for (int c = 0; c <= rr2; c++) { p2[c] = expf(p2[c] - mx); sm += p2[c]; }
                            float* ho2 = (float*)malloc(4);
                            for (int d = 0; d < D; d += 64) {
                                float o = 0;
                                for (int c = 0; c <= rr2; c++) o += __bfloat162float(__float2bfloat16(p2[c])) * __bfloat162float(vv[((size_t)c * HKV + kh) * D + d]);
                                o /= sm;
                                float got;
                                CK(cudaMemcpy(&got, dout + ((size_t)rr2 * H + h) * D + d, 4, cudaMemcpyDeviceToHost));
                                float rel = fabsf(got - o) / fmaxf(fabsf(o), 1e-3f);
                                if (rel > w) w = rel;
                            }
                            free(ho2);
                        }
                        printf("  rows %3d-%3d: worst %.3e\n", r, r + 7, w);
                    }
                }
                free(ho);
                if (nb2 && T > 64) return 1;
            } else {
                cudaEvent_t a, b2; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b2));
                for (int i = 0; i < 3; i++) fa3_v3<<<grid, 128, shmem>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v3<<<grid, 128, shmem>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                float ms; CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v3 T=2048: %.0fus/call (engine mma baseline 993us)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v4<<<grid, 128, shmem4>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v4<<<grid, 128, shmem4>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v4 T=2048: %.0fus/call (int4 cp.async K + 2-stage ring)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v5<<<grid5, 256, shmem5>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v5<<<grid5, 256, shmem5>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v5 T=2048: %.0fus/call (2 warpgroups, shared K/V ring)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v6<<<grid5, 256, shmem6>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v6<<<grid5, 256, shmem6>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v6 T=2048: %.0fus/call (S(t+1)-before-PV(t) overlap)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v7<<<grid, 256, shmem7>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v7<<<grid, 256, shmem7>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v7 T=2048: %.0fus/call (split-D, 2 WGs one q-tile)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v8<<<grid5, 384, shmem8>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v8<<<grid5, 384, shmem8>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v8 T=2048: %.0fus/call (producer warpgroup, named barriers)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v9<<<grid5, 384, shmem9>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v9<<<grid5, 384, shmem9>>>(dq, dk, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v9 T=2048: %.0fus/call (async producer, mbarriers)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v10<<<grid5, 256, shmem10>>>(tQm, tKm, dv, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v10<<<grid5, 256, shmem10>>>(tQm, tKm, dv, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v10 T=2048: %.0fus/call (TMA swizzled Q/K in the v5 shape)\n", ms * 50.0);
                for (int i = 0; i < 3; i++) fa3_v11<<<grid5, 256, shmem10>>>(tQm, tKm, tVm, dout, T, H, HKV, D, scale);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) fa3_v11<<<grid5, 256, shmem10>>>(tQm, tKm, tVm, dout, T, H, HKV, D, scale);
                CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
                CK(cudaEventElapsedTime(&ms, a, b2));
                printf("v11 T=2048: %.0fus/call (full-TMA staging, trans_b PV)\n", ms * 50.0);
            }
            cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
            free(q); free(k); free(vv);
        }
    }
    return 0;
}
