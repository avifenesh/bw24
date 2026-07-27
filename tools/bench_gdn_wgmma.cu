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
#include <vector>

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

// trans_b wgmma (PV-proven pairing: canonical row staging + trans bit)
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


// ---- v4: 2 warpgroups x i-halves — CTA (head, 32-col block), 256 threads.
// wg owns M(64col x 64i half) in Macc[32]; step A partials (k = own i-half, 4 k-steps)
// exchanged through an f32 smem overlay on sM; wg0 does the Y/ys^T epilogue; step B
// per wg on its own k^T atom (2 k-steps). Halves both wgmma chains and doubles
// warps/scheduler vs v3 (ncu v3: 1.0 active warps/scheduler, no-eligible 86.5%).
extern "C" __global__ void __launch_bounds__(256, 1)
gdn_k4_wgmma_v4(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                const float* __restrict__ beta,
                const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                float* __restrict__ Y, __half* __restrict__ Ssnap,
                const float* __restrict__ state_in, float* __restrict__ state_out,
                int H, int T, int C) {
    constexpr int D = 128;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int warp = wtid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;
    const int ih = wg * 64;                       // own i-half base

    __shared__ __align__(128) __nv_bfloat16 sM[2][64 * 64];      // per-wg B (n=col, k=own i): 4 k-steps
    __shared__ __align__(128) __nv_bfloat16 sW[1][64 * 128];     // shared A (m=j, k=i full): 8 k-steps
    __shared__ __align__(128) __nv_bfloat16 sK[1][2][64 * 32];   // [buf][i-half] k^T (n=i atom, k=j): 2 k-steps
    __shared__ __align__(128) __nv_bfloat16 sYs[64 * 32];     // shared A step B (m=col, k=j)
    __shared__ float gk[32];
    float* sS = (float*)&sM[0][0];                // step-A partial exchange overlay (64x64 f32 = sM[0..1])

    float Macc[32];
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        bool re0 = cll < 32, re1 = cll + 8 < 32;
        Macc[q + 0] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 0] : 0.0f;
        Macc[q + 1] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 1] : 0.0f;
        Macc[q + 2] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] : 0.0f;
        Macc[q + 3] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] : 0.0f;
    }

    const int NC = (T + C - 1) / C;
    // sW pad rows (j 32..63) never change: zero once
    for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
        int r = 32 + seg / (D / 8), s8 = seg % (D / 8);
        *(uint4*)((char*)sW[0] + canon_off(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        // Ssnap (column-block layout) + M restage into own sM half (real col rows only)
        {
            __half* snap = Ssnap + ((size_t)c * H + h) * D * D;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cll = warp * 16 + fr;
                int col = col0 + cll;
                int il = fc + n8 * 8;
                int i0 = ih + il;
                if (cll < 32) {
#ifndef SKIP_SNAP
                    snap[(size_t)(col >> 5) * (D * 32) + (size_t)(i0 + 0) * 32 + (col & 31)] = __float2half(Macc[q + 0]);
                    snap[(size_t)(col >> 5) * (D * 32) + (size_t)(i0 + 1) * 32 + (col & 31)] = __float2half(Macc[q + 1]);
                    snap[(size_t)((col + 8) >> 5) * (D * 32) + (size_t)(i0 + 0) * 32 + ((col + 8) & 31)] = __float2half(Macc[q + 2]);
                    snap[(size_t)((col + 8) >> 5) * (D * 32) + (size_t)(i0 + 1) * 32 + ((col + 8) & 31)] = __float2half(Macc[q + 3]);
#endif
                    *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll, il % 16)) =
                        __floats2bfloat162_rn(Macc[q + 0], Macc[q + 1]);
                    *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll + 8, il % 16)) =
                        __floats2bfloat162_rn(Macc[q + 2], Macc[q + 3]);
                }
            }
        }
        // stage W real rows via cp.async 16B (canonical block = direct copy; pad zeroed once),
        // then own k^T half (register transpose) + gk under the async shadow
#ifndef SKIP_STAGE
        for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
            int r = seg / (D / 8), s8 = seg % (D / 8);
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sW[0] + canon_off(st, r, kk8));
            const void* src = Wb16 + (((size_t)c * H + h) * C + r) * D + st * 16 + kk8;
            int sz = (r < Cc) ? 16 : 0;   // tail chunk: zero-fill
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(sz));
        }
        asm volatile("cp.async.commit_group;");
        for (int idx = wtid; idx < 32 * 8; idx += 128) {
            int j = idx / 8, i8l = (idx % 8) * 8;
            __nv_bfloat16 kv8[8];
            if (j < Cc) *(uint4*)kv8 = *(const uint4*)(kb16 + ((size_t)(t0 + j) * H + h) * D + ih + i8l);
            else        *(uint4*)kv8 = make_uint4(0u, 0u, 0u, 0u);
            #pragma unroll
            for (int e2 = 0; e2 < 8; e2++)
                *(__nv_bfloat16*)((char*)sK[0][wg] + canon_off(j >> 4, i8l + e2, j & 15)) = kv8[e2];
        }
        asm volatile("cp.async.wait_group 0;");
#endif
        if (tid < 32) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = (tid < Cc) ? expf(gC - gcum[(size_t)(t0 + tid) * H + h]) * beta[(size_t)(t0 + tid) * H + h] : 0.0f;
        }
        // U prefetch (wg0 epilogue operands): register loads under staging + step-A shadow
        float2 uPre[16];
        if (wg == 0) {
            const int j0p = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4, cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0p + pr * 8;
                    uPre[n8 * 2 + pr] = (j < Cc && cl < 32)
                        ? *(const float2*)(U + (((size_t)c * H + h) * C + j) * D + col0 + cl)
                        : make_float2(0.0f, 0.0f);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        // step A: partial S over own i-half (4 k-steps)
        float acc[32];
#ifdef SKIP_STEPA
        #pragma unroll
        for (int q = 0; q < 32; q++) acc[q] = 0.0f;
#endif
#ifndef SKIP_STEPA
        wgmma_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long da = make_desc((char*)sW[0] + (wg * 4 + st) * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sM[wg] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
#endif
        __syncthreads();                            // both wgs done reading sM (= sS region)
        // exchange: wg1 partials -> sS; wg0 sums
        if (wg == 1) {
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int r = warp * 16 + fr, cc = fc + n8 * 8;
                sS[(r + 0) * 64 + cc + 0] = acc[q + 0];
                sS[(r + 0) * 64 + cc + 1] = acc[q + 1];
                sS[(r + 8) * 64 + cc + 0] = acc[q + 2];
                sS[(r + 8) * 64 + cc + 1] = acc[q + 3];
            }
        }
        __syncthreads();
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
#ifndef SKIP_EPI
        if (wg == 0) {
            // full S = own + peer; Y epilogue + ys^T staging (j rows 0..31 live here)
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    float yv0 = 0.0f, yv1 = 0.0f;
                    if (j < Cc && cl < 32) {
                        float2 u2 = uPre[n8 * 2 + pr];
                        yv0 = u2.x - (acc[q + pr * 2 + 0] + sS[j * 64 + cl + 0]);
                        yv1 = u2.y - (acc[q + pr * 2 + 1] + sS[j * 64 + cl + 1]);
                        *(float2*)(Y + (((size_t)c * H + h) * C + j) * D + col0 + cl) = make_float2(yv0, yv1);
                        yv0 *= gk[j]; yv1 *= gk[j];
                    }
                    if (j0 < 32 && cl < 32) {
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 0, j % 16)) = __float2bfloat16(yv0);
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 1, j % 16)) = __float2bfloat16(yv1);
                    }
                }
            }
        }
#endif
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // step B: M(col, own i) = bC*M + ys^T . k^T-half (2 k-steps, 1 atom)
        #pragma unroll
        for (int q = 0; q < 32; q++) Macc[q] *= bC;
#ifndef SKIP_STEPB
        wgmma_fence();
        for (int st = 0; st < 2; st++) {
            unsigned long long da = make_desc((char*)sYs + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sK[0][wg] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Macc, da, db, 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
#endif
        __syncthreads();                            // next restage overwrites sM/sS
    }
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        if (cll < 32) {
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 0] = Macc[q + 0];
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 1] = Macc[q + 1];
        }
        if (cll + 8 < 32) {
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] = Macc[q + 2];
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] = Macc[q + 3];
        }
    }
}


// ---- v5: K4+K5 FUSION — v4 shape + K5's output pass absorbed. Per chunk:
// phase 1 (o += exp(gcum_j) * q_j . M_pre[col]) rides step A's commit group (same
// B = sM[wg]); phase 2 (o += P . Y) rides step B's (B = plain Y^T = sYs). gk folds
// into sK staging so sYs stays unfolded and serves step B's A and phase 2's B.
// Ssnap global round-trip DELETED (M is already on-SM). wg1 posts BOTH partial sets
// (step A -> sS on sM, phase 1 -> sQ overlay) in the one exchange window.
extern "C" __global__ void __launch_bounds__(256, 1)
gdn_k45_wgmma_v5(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                 const float* __restrict__ beta,
                 const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                 float* __restrict__ Y,
                 const __nv_bfloat16* __restrict__ qb16, const __nv_bfloat16* __restrict__ Pb16,
                 float* __restrict__ o, float scale,
                 const float* __restrict__ state_in, float* __restrict__ state_out,
                 int H, int T, int C) {
    constexpr int D = 128;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int warp = wtid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;
    const int ih = wg * 64;

    __shared__ __align__(128) __nv_bfloat16 sM[2][64 * 64];      // per-wg B (n=col, k=own i)
    __shared__ __align__(128) __nv_bfloat16 sW[64 * 128];        // A step A (m=j, k=i full)
    __shared__ __align__(128) __nv_bfloat16 sK[2][64 * 32];      // per-wg gk-folded k^T (n=i atom, k=j)
    __shared__ __align__(128) __nv_bfloat16 sQ[2][64 * 64];      // per-wg A phase 1 (m=j, k=own i)
    __shared__ __align__(128) __nv_bfloat16 sP2[64 * 32];        // A phase 2 (m=j, k=j2, tri-masked)
    __shared__ __align__(128) __nv_bfloat16 sYs[64 * 32];        // plain Y^T (m/n=col, k=j)
    float* sS = (float*)&sM[0][0];    // step-A partial exchange overlay
    float* sO = (float*)&sQ[0][0];    // phase-1 partial exchange overlay (sQ consumed by then)

    float Macc[32];
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        bool re0 = cll < 32, re1 = cll + 8 < 32;
        Macc[q + 0] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 0] : 0.0f;
        Macc[q + 1] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 1] : 0.0f;
        Macc[q + 2] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] : 0.0f;
        Macc[q + 3] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] : 0.0f;
    }

    const int NC = (T + C - 1) / C;
    // pad rows (m 32..63) of sW/sP2 never change: zero once
    for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
        int r = 32 + seg / (D / 8), s8 = seg % (D / 8);
        *(uint4*)((char*)sW + canon_off(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int seg = tid; seg < 32 * 4; seg += 256) {
        int r = 32 + seg / 4, s8 = seg % 4;
        *(uint4*)((char*)sP2 + canon_off(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        // M restage into own sM half (real col rows only) — no Ssnap in the fused form
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cll = warp * 16 + fr;
            int il = fc + n8 * 8;
            if (cll < 32) {
                *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 0], Macc[q + 1]);
                *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll + 8, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 2], Macc[q + 3]);
            }
        }
        // W rows via cp.async; k^T (gk-folded) + q + P staged under the async shadow
        for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
            int r = seg / (D / 8), s8 = seg % (D / 8);
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sW + canon_off(st, r, kk8));
            const void* src = Wb16 + (((size_t)c * H + h) * C + r) * D + st * 16 + kk8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((r < Cc) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        {
            const float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            for (int idx = wtid; idx < 32 * 8; idx += 128) {
                int j = idx / 8, i8l = (idx % 8) * 8;
                float gkj = 0.0f;
                __nv_bfloat16 kv8[8];
                if (j < Cc) {
                    gkj = expf(gC - gcum[(size_t)(t0 + j) * H + h]) * beta[(size_t)(t0 + j) * H + h];
                    *(uint4*)kv8 = *(const uint4*)(kb16 + ((size_t)(t0 + j) * H + h) * D + ih + i8l);
                } else *(uint4*)kv8 = make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int e2 = 0; e2 < 8; e2++)
                    *(__nv_bfloat16*)((char*)sK[wg] + canon_off(j >> 4, i8l + e2, j & 15)) =
                        __float2bfloat16(gkj * __bfloat162float(kv8[e2]));
            }
        }
        // q rows -> canonical A via cp.async (bf16 mirror; both i-halves)
#ifndef SKIP5_Q
        for (int seg = tid; seg < 2 * 32 * (D / 16); seg += 256) {
            int half = seg / 256, rem = seg % 256;
            int j = rem / 8, s8 = rem % 8;
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sQ[half] + canon_off(st, j, kk8));
            const void* src = qb16 + ((size_t)(t0 + j) * H + h) * D + half * 64 + st * 16 + kk8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((j < Cc) ? 16 : 0));
        }
#endif
        // P (pre-masked bf16 mirror) -> canonical A via cp.async
#ifndef SKIP5_P
        for (int seg = tid; seg < 32 * 4; seg += 256) {
            int j = seg / 4, s8 = seg % 4;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sP2 + canon_off(s8 / 2, j, (s8 % 2) * 8));
            const void* src = Pb16 + (((size_t)c * H + h) * C + j) * C + (s8 / 2) * 16 + (s8 % 2) * 8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((j < Cc) ? 16 : 0));
        }
#endif
        asm volatile("cp.async.commit_group;");   // q/P group (W group committed above)
        // U prefetch (wg0 epilogue operands)
        float2 uPre[16];
        if (wg == 0) {
            const int j0p = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4, cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0p + pr * 8;
                    uPre[n8 * 2 + pr] = (j < Cc && cl < 32)
                        ? *(const float2*)(U + (((size_t)c * H + h) * C + j) * D + col0 + cl)
                        : make_float2(0.0f, 0.0f);
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        // step A (acc) + phase 1 (Oacc) in one group — shared B slices
        float acc[32], Oacc[32];
        wgmma_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long da = make_desc((char*)sW + (wg * 4 + st) * 2048, 128, 256);
            unsigned long long dq = make_desc((char*)sQ[wg] + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sM[wg] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
#ifndef SKIP5_PH1
            wgmma_m64n64k16_bf16(Oacc, dq, db, st == 0 ? 0 : 1);
#endif
        }
#ifdef SKIP5_PH1
        #pragma unroll
        for (int q = 0; q < 32; q++) Oacc[q] = 0.0f;
#endif
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();                            // both wgs done reading sM/sQ (overlay regions)
        // exchange window: wg1 posts step-A AND phase-1 partials
        if (wg == 1) {
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int r = warp * 16 + fr, cc = fc + n8 * 8;
                sS[(r + 0) * 64 + cc + 0] = acc[q + 0];
                sS[(r + 0) * 64 + cc + 1] = acc[q + 1];
                sS[(r + 8) * 64 + cc + 0] = acc[q + 2];
                sS[(r + 8) * 64 + cc + 1] = acc[q + 3];
#ifndef SKIP5_EXO
                sO[(r + 0) * 64 + cc + 0] = Oacc[q + 0];
                sO[(r + 0) * 64 + cc + 1] = Oacc[q + 1];
                sO[(r + 8) * 64 + cc + 0] = Oacc[q + 2];
                sO[(r + 8) * 64 + cc + 1] = Oacc[q + 3];
#endif
            }
        }
        __syncthreads();
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        if (wg == 0) {
            // Y epilogue + plain Y^T staging; then phase-1 sum + exp(gcum_j) row scale
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    float yv0 = 0.0f, yv1 = 0.0f;
                    if (j < Cc && cl < 32) {
                        float2 u2 = uPre[n8 * 2 + pr];
                        yv0 = u2.x - (acc[q + pr * 2 + 0] + sS[j * 64 + cl + 0]);
                        yv1 = u2.y - (acc[q + pr * 2 + 1] + sS[j * 64 + cl + 1]);
                        *(float2*)(Y + (((size_t)c * H + h) * C + j) * D + col0 + cl) = make_float2(yv0, yv1);
                    }
                    if (j0 < 32 && cl < 32) {
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 0, j % 16)) = __float2bfloat16(yv0);
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 1, j % 16)) = __float2bfloat16(yv1);
                    }
                }
            }
#ifndef SKIP5_EXO
            const float b0 = (j0 < Cc) ? expf(gcum[(size_t)(t0 + j0) * H + h]) : 0.0f;
            const float b1 = (j0 + 8 < Cc) ? expf(gcum[(size_t)(t0 + j0 + 8) * H + h]) : 0.0f;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int r = j0, cc = fc + n8 * 8;
                Oacc[q + 0] = (Oacc[q + 0] + sO[(r + 0) * 64 + cc + 0]) * b0;
                Oacc[q + 1] = (Oacc[q + 1] + sO[(r + 0) * 64 + cc + 1]) * b0;
                Oacc[q + 2] = (Oacc[q + 2] + sO[(r + 8) * 64 + cc + 0]) * b1;
                Oacc[q + 3] = (Oacc[q + 3] + sO[(r + 8) * 64 + cc + 1]) * b1;
            }
#endif
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // step B (both wgs, Macc) + phase 2 (wg0, Oacc += P . Y) in one group
        #pragma unroll
        for (int q = 0; q < 32; q++) Macc[q] *= bC;
        wgmma_fence();
        for (int st = 0; st < 2; st++) {
            unsigned long long da = make_desc((char*)sYs + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sK[wg] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Macc, da, db, 1);
        }
        // phase 2 on BOTH wgs (C7519: wgmma in a divergent path serializes) — wg1's Oacc
        // result is discarded (never stored); only the O store below is wg-gated
#ifndef SKIP5_PH2
        for (int st = 0; st < 2; st++) {
            unsigned long long da = make_desc((char*)sP2 + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sYs + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Oacc, da, db, 1);
        }
#endif
        wgmma_commit();
        wgmma_wait<0>();
        // O out (wg0): o[j, col0+cl] = scale * Oacc
#ifndef SKIP5_OST
        if (wg == 0) {
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4, cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    if (j < Cc && cl < 32)
                        *(float2*)(o + ((size_t)(t0 + j) * H + h) * D + col0 + cl) =
                            make_float2(scale * Oacc[q + pr * 2 + 0], scale * Oacc[q + pr * 2 + 1]);
                }
            }
        }
#endif
        __syncthreads();                            // next restage overwrites sM/sQ overlays
    }
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        if (cll < 32) {
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 0] = Macc[q + 0];
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 1] = Macc[q + 1];
        }
        if (cll + 8 < 32) {
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] = Macc[q + 2];
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] = Macc[q + 3];
        }
    }
}


// ---- v6: exchange-free — BOTH wgs run FULL-k step A + phase 1 (tensor pipe ~16%,
// duplication is free) -> no partial exchange, no sS/sO overlays, 5 -> 3 barriers.
// Epilogue + O store split by column halves (wg0: n8 0..1 = cl 0..15, wg1: n8 2..3).
// Step B / Macc stay per-i-half (the register-resident state partition, untouched).
extern "C" __global__ void __launch_bounds__(256, 1)
gdn_k45_wgmma_v6(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                 const float* __restrict__ beta,
                 const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                 float* __restrict__ Y,
                 const __nv_bfloat16* __restrict__ qb16, const __nv_bfloat16* __restrict__ Pb16,
                 float* __restrict__ o, float scale,
                 const float* __restrict__ state_in, float* __restrict__ state_out,
                 int H, int T, int C) {
    constexpr int D = 128;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int warp = wtid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;
    const int ih = wg * 64;
    const int nlo = wg * 2;                        // this wg's n8 quads: {nlo, nlo+1}

    __shared__ __align__(128) __nv_bfloat16 sM[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 sW[64 * 128];
    __shared__ __align__(128) __nv_bfloat16 sK[2][64 * 32];
    __shared__ __align__(128) __nv_bfloat16 sQ[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 sP2[64 * 32];
    __shared__ __align__(128) __nv_bfloat16 sYs[64 * 32];
    __shared__ float gk[32];

    float Macc[32];
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        bool re0 = cll < 32, re1 = cll + 8 < 32;
        Macc[q + 0] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 0] : 0.0f;
        Macc[q + 1] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 1] : 0.0f;
        Macc[q + 2] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] : 0.0f;
        Macc[q + 3] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] : 0.0f;
    }

    const int NC = (T + C - 1) / C;
    for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
        int r = 32 + seg / (D / 8), s8 = seg % (D / 8);
        *(uint4*)((char*)sW + canon_off(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int seg = tid; seg < 32 * 4; seg += 256) {
        int r = 32 + seg / 4, s8 = seg % 4;
        *(uint4*)((char*)sP2 + canon_off(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cll = warp * 16 + fr;
            int il = fc + n8 * 8;
            if (cll < 32) {
                *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 0], Macc[q + 1]);
                *(__nv_bfloat162*)((char*)sM[wg] + canon_off(il / 16, cll + 8, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 2], Macc[q + 3]);
            }
        }
        for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
            int r = seg / (D / 8), s8 = seg % (D / 8);
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sW + canon_off(st, r, kk8));
            const void* src = Wb16 + (((size_t)c * H + h) * C + r) * D + st * 16 + kk8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((r < Cc) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        {
            const float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            for (int idx = wtid; idx < 32 * 8; idx += 128) {
                int j = idx / 8, i8l = (idx % 8) * 8;
                float gkj = 0.0f;
                __nv_bfloat16 kv8[8];
                if (j < Cc) {
                    gkj = expf(gC - gcum[(size_t)(t0 + j) * H + h]) * beta[(size_t)(t0 + j) * H + h];
                    *(uint4*)kv8 = *(const uint4*)(kb16 + ((size_t)(t0 + j) * H + h) * D + ih + i8l);
                } else *(uint4*)kv8 = make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int e2 = 0; e2 < 8; e2++)
                    *(__nv_bfloat16*)((char*)sK[wg] + canon_off(j >> 4, i8l + e2, j & 15)) =
                        __float2bfloat16(gkj * __bfloat162float(kv8[e2]));
            }
        }
        for (int seg = tid; seg < 2 * 32 * (D / 16); seg += 256) {
            int half = seg / 256, rem = seg % 256;
            int j = rem / 8, s8 = rem % 8;
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sQ[half] + canon_off(st, j, kk8));
            const void* src = qb16 + ((size_t)(t0 + j) * H + h) * D + half * 64 + st * 16 + kk8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((j < Cc) ? 16 : 0));
        }
        for (int seg = tid; seg < 32 * 4; seg += 256) {
            int j = seg / 4, s8 = seg % 4;
            unsigned dst = (unsigned)__cvta_generic_to_shared((char*)sP2 + canon_off(s8 / 2, j, (s8 % 2) * 8));
            const void* src = Pb16 + (((size_t)c * H + h) * C + j) * C + (s8 / 2) * 16 + (s8 % 2) * 8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"((j < Cc) ? 16 : 0));
        }
        if (tid < 32) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = (tid < Cc) ? expf(gC - gcum[(size_t)(t0 + tid) * H + h]) * beta[(size_t)(t0 + tid) * H + h] : 0.0f;
        }
        asm volatile("cp.async.commit_group;");
        // U prefetch: OWN column quads only (n8 = nlo..nlo+1)
        float2 uPre[8];
        {
            const int j0p = warp * 16 + fr;
            #pragma unroll
            for (int qq = 0; qq < 2; qq++) {
                int cl = fc + (nlo + qq) * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0p + pr * 8;
                    uPre[qq * 2 + pr] = (j < Cc && cl < 32)
                        ? *(const float2*)(U + (((size_t)c * H + h) * C + j) * D + col0 + cl)
                        : make_float2(0.0f, 0.0f);
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        // step A + phase 1: FULL k on BOTH wgs (8 k-steps over sM[0..1])
        float acc[32], Oacc[32];
        wgmma_fence();
        for (int st = 0; st < 8; st++) {
            unsigned long long da = make_desc((char*)sW + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sM[st >> 2] + (st & 3) * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        for (int st = 0; st < 8; st++) {
            unsigned long long dq = make_desc((char*)sQ[st >> 2] + (st & 3) * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sM[st >> 2] + (st & 3) * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Oacc, dq, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait<1>();                            // acc ready; phase-1 group still in flight
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        // epilogue on OWN column quads: Y + plain Y^T staging; then b_j scale (no sums)
        {
            const int j0 = warp * 16 + fr;
            const float b0 = (j0 < Cc) ? expf(gcum[(size_t)(t0 + j0) * H + h]) : 0.0f;
            const float b1 = (j0 + 8 < Cc) ? expf(gcum[(size_t)(t0 + j0 + 8) * H + h]) : 0.0f;
            #pragma unroll
            for (int qq = 0; qq < 2; qq++) {
                int n8 = nlo + qq;
                int q = n8 * 4;
                int cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    float yv0 = 0.0f, yv1 = 0.0f;
                    if (j < Cc && cl < 32) {
                        float2 u2 = uPre[qq * 2 + pr];
                        yv0 = u2.x - acc[q + pr * 2 + 0];
                        yv1 = u2.y - acc[q + pr * 2 + 1];
                        *(float2*)(Y + (((size_t)c * H + h) * C + j) * D + col0 + cl) = make_float2(yv0, yv1);
                    }
                    if (j0 < 32 && cl < 32) {
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 0, j % 16)) = __float2bfloat16(yv0);
                        *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 1, j % 16)) = __float2bfloat16(yv1);
                    }
                }
            }
            wgmma_wait<0>();                        // phase-1 done under the epilogue
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                Oacc[q + 0] *= b0; Oacc[q + 1] *= b0;
                Oacc[q + 2] *= b1; Oacc[q + 3] *= b1;
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // step B (per-wg i-half, Macc) + phase 2 (full, both wgs)
        #pragma unroll
        for (int q = 0; q < 32; q++) Macc[q] *= bC;
        wgmma_fence();
        for (int st = 0; st < 2; st++) {
            unsigned long long da = make_desc((char*)sYs + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sK[wg] + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Macc, da, db, 1);
        }
        for (int st = 0; st < 2; st++) {
            unsigned long long da = make_desc((char*)sP2 + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sYs + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(Oacc, da, db, 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
        // O store: OWN column quads
        {
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int qq = 0; qq < 2; qq++) {
                int n8 = nlo + qq;
                int q = n8 * 4;
                int cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    if (j < Cc && cl < 32)
                        *(float2*)(o + ((size_t)(t0 + j) * H + h) * D + col0 + cl) =
                            make_float2(scale * Oacc[q + pr * 2 + 0], scale * Oacc[q + pr * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        if (cll < 32) {
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 0] = Macc[q + 0];
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 1] = Macc[q + 1];
        }
        if (cll + 8 < 32) {
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] = Macc[q + 2];
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] = Macc[q + 3];
        }
    }
}

// ---- v2: FULL K4 chain — CTA = (head, 64-col block); M(64col x 128i) lives in
// acc[2 n-blocks... careful: step-B OUTPUT M'(col,i): m = col(64), n = i(128 = 2 n64)
// -> Macc[2][32]. Step A consumes M as B (n = col 64, k = i 128) via a per-chunk
// fragment->smem bf16 restage (on-SM). Engine semantics from bench_gdn_k4 cpu_ref.
extern "C" __global__ void __launch_bounds__(128, 1)
gdn_k4_wgmma_v2(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                const float* __restrict__ beta,
                const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                float* __restrict__ Y, __half* __restrict__ Ssnap,
                const float* __restrict__ state_in, float* __restrict__ state_out,
                int H, int T, int C) {
    constexpr int D = 128;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * 32;   // v3: 32-col slices -> grid (H,4) = 128 CTAs (machine fill)
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int fr = lane / 4, fc = (lane % 4) * 2;

    __shared__ __align__(128) __nv_bfloat16 sM[64 * 128];      // M-slice canonical (B for step A)
    __shared__ __align__(128) __nv_bfloat16 sW[2][64 * 128];   // W chunk rows (32 real + pad), double-buffered
    __shared__ __align__(128) __nv_bfloat16 sK[2][64 * 64];    // k^T canonical B (2 atoms x 2 k-steps), double-buffered
    __shared__ __align__(128) __nv_bfloat16 sYs[64 * 64];   // ys^T (A for step B): [col-m][j-k]
    __shared__ float gk[32];

    // Macc[m-block over col? m=col 64 -> ONE m64; n = i 128 -> 2 n64 blocks]
    float Macc[2][32];
    // load initial state: M[col][i] fragment mapping (m=col0+.., n=i)
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int col = warp * 16 + fr;
            int i0 = nb * 64 + fc + n8 * 8;
            bool re0 = col < 32, re1 = col + 8 < 32;
            Macc[nb][q + 0] = re0 ? state_in[((size_t)h * D + col0 + col) * D + i0 + 0] : 0.0f;
            Macc[nb][q + 1] = re0 ? state_in[((size_t)h * D + col0 + col) * D + i0 + 1] : 0.0f;
            Macc[nb][q + 2] = re1 ? state_in[((size_t)h * D + col0 + col + 8) * D + i0 + 0] : 0.0f;
            Macc[nb][q + 3] = re1 ? state_in[((size_t)h * D + col0 + col + 8) * D + i0 + 1] : 0.0f;
        }

    const int NC = (T + C - 1) / C;
    // W/k staging for chunk cc2 into buffer buf (16B copies; issued under wgmma shadows)
    auto stage_wk = [&](int cc2, int buf) {
        const int t0s = cc2 * C;
        const int Ccs = min(C, T - t0s);
        for (int seg = tid; seg < 64 * (D / 8); seg += 128) {
            int r = seg / (D / 8), s8 = seg % (D / 8);
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (r < Ccs) v = *(const uint4*)(Wb16 + (((size_t)cc2 * H + h) * C + r) * D + st * 16 + kk8);
            *(uint4*)((char*)sW[buf] + canon_off(st, r, kk8)) = v;
        }
        for (int idx = tid; idx < 32 * (D / 8); idx += 128) {
            int j = idx / (D / 8), i8 = (idx % (D / 8)) * 8;
            __nv_bfloat16 kv8[8];
            if (j < Ccs) *(uint4*)kv8 = *(const uint4*)(kb16 + ((size_t)(t0s + j) * H + h) * D + i8);
            else         *(uint4*)kv8 = make_uint4(0u, 0u, 0u, 0u);
            #pragma unroll
            for (int e2 = 0; e2 < 8; e2++) {
                int i = i8 + e2;
                *(__nv_bfloat16*)((char*)sK[buf] + (i >> 6) * 4096 + canon_off(j >> 4, i & 63, j & 15)) = kv8[e2];
            }
        }
    };
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        const int buf = c & 1;
        stage_wk(c, buf);
        // Ssnap (shipped column-block layout: [4 cb][128 i][32 col] per (c,h)) + M restage
        {
            __half* snap = Ssnap + ((size_t)c * H + h) * D * D;
            #pragma unroll
            for (int nb = 0; nb < 2; nb++)
                #pragma unroll
                for (int q = 0; q < 32; q += 4) {
                    int n8 = q / 4;
                    int cll = warp * 16 + fr;
                    int col = col0 + cll;
                    int i0 = nb * 64 + fc + n8 * 8;
                    // two col-rows x two i (pad rows 32..63 skip):
                    if (cll < 32) {
                        snap[(size_t)(col >> 5) * (D * 32) + (size_t)(i0 + 0) * 32 + (col & 31)] = __float2half(Macc[nb][q + 0]);
                        snap[(size_t)(col >> 5) * (D * 32) + (size_t)(i0 + 1) * 32 + (col & 31)] = __float2half(Macc[nb][q + 1]);
                    }
                    if (cll + 8 < 32) {
                        snap[(size_t)((col + 8) >> 5) * (D * 32) + (size_t)(i0 + 0) * 32 + ((col + 8) & 31)] = __float2half(Macc[nb][q + 2]);
                        snap[(size_t)((col + 8) >> 5) * (D * 32) + (size_t)(i0 + 1) * 32 + ((col + 8) & 31)] = __float2half(Macc[nb][q + 3]);
                    }
                    // M -> smem canonical for step A's B: B(n=col-local, k=i), 4B pair stores
                    *(__nv_bfloat162*)((char*)sM + canon_off(i0 / 16, cll, i0 % 16)) =
                        __floats2bfloat162_rn(Macc[nb][q + 0], Macc[nb][q + 1]);
                    *(__nv_bfloat162*)((char*)sM + canon_off(i0 / 16, cll + 8, i0 % 16)) =
                        __floats2bfloat162_rn(Macc[nb][q + 2], Macc[nb][q + 3]);
                }
        }
        if (tid < 32) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = (tid < Cc) ? expf(gC - gcum[(size_t)(t0 + tid) * H + h]) * beta[(size_t)(t0 + tid) * H + h] : 0.0f;
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        // step A: S(j, col-local) = W(64j x 128i) . M^T -> wgmma A=sW, B=sM (n=col)
        float acc[32];
        wgmma_fence();
        for (int st = 0; st < 8; st++) {
            unsigned long long da = make_desc((char*)sW[buf] + st * 2048, 128, 256);
            unsigned long long db = make_desc((char*)sM + st * 2048, 128, 256);
            wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();
        // Y epilogue + ys^T restage (A for step B: A(m=col-local 64, k=j 32))
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        {
            const int j0 = warp * 16 + fr;     // fragment rows = j
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cl = fc + n8 * 8;          // col-local 0..63
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    float yv0 = 0.0f, yv1 = 0.0f;
                    if (j < Cc && cl < 32) {
                        float2 u2 = *(const float2*)(U + (((size_t)c * H + h) * C + j) * D + col0 + cl);
                        yv0 = u2.x - acc[q + pr * 2 + 0];
                        yv1 = u2.y - acc[q + pr * 2 + 1];
                        *(float2*)(Y + (((size_t)c * H + h) * C + j) * D + col0 + cl) = make_float2(yv0, yv1);
                        yv0 *= gk[j]; yv1 *= gk[j];
                    }
                    // ys^T: A(m=cc, k=j) = gk[j]*yv (pad j/cols stage 0)
                    *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 0, j % 16)) = __float2bfloat16(yv0);
                    *(__nv_bfloat16*)((char*)sYs + canon_off(j / 16, cl + 1, j % 16)) = __float2bfloat16(yv1);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // step B: M(col,i) = bC*M + ys^T(64col x 32j) . k(32j x 128i)
        #pragma unroll
        for (int nb = 0; nb < 2; nb++)
            #pragma unroll
            for (int q = 0; q < 32; q++) Macc[nb][q] *= bC;
        wgmma_fence();
        for (int st = 0; st < 2; st++) {                    // k-steps over j (32 = 2 x 16)
            unsigned long long da = make_desc((char*)sYs + st * 2048, 128, 256);
            #pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                // B = k^T canonical K-major (n=i local, k=j): atom nb, k-slice st
                unsigned long long db = make_desc((char*)sK[buf] + nb * 4096 + st * 2048, 128, 256);
                wgmma_m64n64k16_bf16(Macc[nb], da, db, 1);
            }
        }
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();
    }
    // state out
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int col = warp * 16 + fr;
            int i0 = nb * 64 + fc + n8 * 8;
            if (col < 32) {
                state_out[((size_t)h * D + col0 + col) * D + i0 + 0] = Macc[nb][q + 0];
                state_out[((size_t)h * D + col0 + col) * D + i0 + 1] = Macc[nb][q + 1];
            }
            if (col + 8 < 32) {
                state_out[((size_t)h * D + col0 + col + 8) * D + i0 + 0] = Macc[nb][q + 2];
                state_out[((size_t)h * D + col0 + col + 8) * D + i0 + 1] = Macc[nb][q + 3];
            }
        }
}

int main(int argc, char** argv) {
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
    if (bad) return 1;

    // ---- v2: full chain at the calibrated dims (H=32, T=512, C=32) ----
    {
        const int H = 32, T = (argc > 1 ? atoi(argv[1]) : 512), C = 32, D = 128, NC = (T + C - 1) / C;
        size_t nk = (size_t)T * H * D, ng = (size_t)T * H, nu = (size_t)NC * H * C * D;
        float *k = (float*)malloc(nk * 4), *gc = (float*)malloc(ng * 4), *bt = (float*)malloc(ng * 4);
        float *U2 = (float*)malloc(nu * 4), *W2 = (float*)malloc(nu * 4);
        float *si = (float*)malloc((size_t)H * D * D * 4);
        // input law copied from bench_gdn_k4 main (the gate that validated the shipped mma):
        // k/U scale 1.0, W 0.3, state 0.5, beta 0.5..1.0, gcum continuous small-negative cumsum
        srand(7);
        auto rf = [](float sc) { return ((rand() % 2001) - 1000) * 1e-3f * sc; };
        for (size_t i = 0; i < nk; i++) k[i] = rf(1.0f);
        for (size_t i = 0; i < nu; i++) { U2[i] = rf(1.0f); W2[i] = rf(0.3f); }
        for (size_t i = 0; i < (size_t)H * D * D; i++) si[i] = rf(0.5f);
        for (size_t i = 0; i < ng; i++) bt[i] = 0.5f + ((rand() % 1000) * 5e-4f);
        for (int h2 = 0; h2 < H; h2++) {
            float acc2 = 0.0f;
            for (int t = 0; t < T; t++) {
                acc2 += -0.02f - (rand() % 100) * 2e-4f;
                gc[(size_t)t * H + h2] = acc2;
            }
        }
        // v5 fusion inputs: q rows + triangular intra-chunk P; K5 scale
        const float scale5 = 0.125f;
        float *hq5 = (float*)malloc(nk * 4);
        float *hP5 = (float*)malloc((size_t)NC * H * C * C * 4);
        for (size_t i = 0; i < nk; i++) hq5[i] = rf(1.0f);
        for (size_t i = 0; i < (size_t)NC * H * C * C; i++) hP5[i] = rf(0.5f);
        // CPU ref (bench_gdn_k4 semantics) with bf16-rounded W/k inputs (the mma class)
        for (size_t i = 0; i < nk; i++) k[i] = __bfloat162float(__float2bfloat16(k[i]));
        for (size_t i = 0; i < nu; i++) W2[i] = __bfloat162float(__float2bfloat16(W2[i]));
        float *refY = (float*)malloc(nu * 4), *refS = (float*)malloc((size_t)H * D * D * 4);
        float *refO = (float*)malloc(nk * 4);
        {
            std::vector<float> M((size_t)D * D), Yc((size_t)C * D), Mn((size_t)D * D);
            for (int h2 = 0; h2 < H; h2++) {
                for (int col = 0; col < D; col++)
                    for (int i = 0; i < D; i++)
                        M[(size_t)col * D + i] = si[((size_t)h2 * D + col) * D + i];
                for (int c2 = 0; c2 < NC; c2++) {
                    const int t0 = c2 * C, Cc = (T - t0 < C) ? (T - t0) : C;
                    const float gC = gc[(size_t)(t0 + Cc - 1) * H + h2];
                    for (int j = 0; j < Cc; j++)
                        for (int col = 0; col < D; col++) {
                            double a2 = 0;
                            for (int i = 0; i < D; i++)
                                a2 += (double)W2[(((size_t)c2 * H + h2) * C + j) * D + i] * M[(size_t)col * D + i];
                            float y = U2[(((size_t)c2 * H + h2) * C + j) * D + col] - (float)a2;
                            Yc[(size_t)j * D + col] = y;
                            refY[(((size_t)c2 * H + h2) * C + j) * D + col] = y;
                        }
                    // fused-K5 O reference: o[j,col] = scale*(exp(gc_j)*q_j.M_pre[col] + sum_{j2<=j} P[j,j2]*Y[j2,col])
                    for (int j = 0; j < Cc; j++) {
                        float bj = expf(gc[(size_t)(t0 + j) * H + h2]);
                        for (int col = 0; col < D; col++) {
                            double p1 = 0, p2 = 0;
                            for (int i = 0; i < D; i++)
                                p1 += (double)__bfloat162float(__float2bfloat16(hq5[((size_t)(t0 + j) * H + h2) * D + i])) * M[(size_t)col * D + i];
                            for (int j2 = 0; j2 <= j; j2++)
                                p2 += (double)hP5[(((size_t)c2 * H + h2) * C + j) * C + j2] * Yc[(size_t)j2 * D + col];
                            refO[((size_t)(t0 + j) * H + h2) * D + col] = scale5 * (float)(bj * p1 + p2);
                        }
                    }
                    const float bC = expf(gC);
                    for (int col = 0; col < D; col++)
                        for (int i = 0; i < D; i++) {
                            double a2 = bC * (double)M[(size_t)col * D + i];
                            for (int j = 0; j < Cc; j++) {
                                float g2 = expf(gC - gc[(size_t)(t0 + j) * H + h2]) * bt[(size_t)(t0 + j) * H + h2];
                                a2 += (double)g2 * k[((size_t)(t0 + j) * H + h2) * D + i] * Yc[(size_t)j * D + col];
                            }
                            Mn[(size_t)col * D + i] = (float)a2;
                        }
                    M.swap(Mn);
                }
                for (int col = 0; col < D; col++)
                    for (int i = 0; i < D; i++)
                        refS[((size_t)h2 * D + col) * D + i] = M[(size_t)col * D + i];
            }
        }
        // device: bf16 mirrors for k/W (engine convention)
        __nv_bfloat16 *kb = (__nv_bfloat16*)malloc(nk * 2), *wb = (__nv_bfloat16*)malloc(nu * 2);
        for (size_t i = 0; i < nk; i++) kb[i] = __float2bfloat16(k[i]);
        for (size_t i = 0; i < nu; i++) wb[i] = __float2bfloat16(W2[i]);
        __nv_bfloat16 *dkb, *dwb; float *dgc, *dbt, *dU, *dsi, *dso;
        float *dY; __half *dS;
        CK(cudaMalloc(&dkb, nk * 2)); CK(cudaMalloc(&dwb, nu * 2));
        CK(cudaMalloc(&dgc, ng * 4)); CK(cudaMalloc(&dbt, ng * 4)); CK(cudaMalloc(&dU, nu * 4));
        CK(cudaMalloc(&dsi, (size_t)H * D * D * 4)); CK(cudaMalloc(&dso, (size_t)H * D * D * 4));
        CK(cudaMalloc(&dY, nu * 4)); CK(cudaMalloc(&dS, (size_t)NC * H * D * D * 2));
        CK(cudaMemcpy(dkb, kb, nk * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dwb, wb, nu * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dgc, gc, ng * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbt, bt, ng * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dU, U2, nu * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsi, si, (size_t)H * D * D * 4, cudaMemcpyHostToDevice));
        dim3 grid(H, 4);
        gdn_k4_wgmma_v2<<<grid, 128>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        float* hY2 = (float*)malloc(nu * 4);
        float* hS2 = (float*)malloc((size_t)H * D * D * 4);
        CK(cudaMemcpy(hY2, dY, nu * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hS2, dso, (size_t)H * D * D * 4, cudaMemcpyDeviceToHost));
        // bench_gdn_k4 class metric: maxdiff / max|ref| against the bf16 band (3e-2);
        // Y out is f16 (engine class) so allow the same band
        double mY = 0, sY = 0, mS = 0, sS = 0;
        for (size_t i = 0; i < nu; i++) {
            mY = fmax(mY, fabs((double)hY2[i] - refY[i]));
            sY = fmax(sY, fabs((double)refY[i]));
        }
        for (size_t i = 0; i < (size_t)H * D * D; i++) {
            mS = fmax(mS, fabs((double)hS2[i] - refS[i]));
            sS = fmax(sS, fabs((double)refS[i]));
        }
        double relY = mY / fmax(sY, 1e-3), relS = mS / fmax(sS, 1e-3);
        int ok = relY < 3e-2 && relS < 3e-2;
        printf("v2 chain: Y rel %.3e (scale %.2f)  state rel %.3e (scale %.2f)  %s (band 3e-2)\n",
               relY, sY, relS, sS, ok ? "IN-BAND" : "OUT-OF-BAND");
        if (!ok) {
            for (int c2 = 0; c2 < NC; c2++) {
                double cm = 0, cs = 0;
                for (int h2 = 0; h2 < H; h2++)
                    for (int j = 0; j < C; j++)
                        for (int d2 = 0; d2 < D; d2++) {
                            size_t ix = (((size_t)c2 * H + h2) * C + j) * D + d2;
                            cm = fmax(cm, fabs((double)hY2[ix] - refY[ix]));
                            cs = fmax(cs, fabs((double)refY[ix]));
                        }
                printf("  chunk %2d: maxdiff %.3e scale %.2f rel %.3e\n", c2, cm, cs, cm / fmax(cs, 1e-3));
            }
        }
        // timing (shipped mma calibration point: 68.3us at these dims)
        cudaEvent_t a, b2; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b2));
        for (int i = 0; i < 5; i++) gdn_k4_wgmma_v2<<<grid, 128>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 50; i++) gdn_k4_wgmma_v2<<<grid, 128>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        float ms; CK(cudaEventElapsedTime(&ms, a, b2));
        printf("v2 timing: %.1fus/call (shipped mma = 68.3us at H=32 T=512 C=32)\n", ms * 20.0f);

        // ---- v4 ----
        CK(cudaMemset(dY, 0, nu * 4));
        CK(cudaMemset(dso, 0, (size_t)H * D * D * 4));
        gdn_k4_wgmma_v4<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hY2, dY, nu * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hS2, dso, (size_t)H * D * D * 4, cudaMemcpyDeviceToHost));
        double mY4 = 0, mS4 = 0;
        for (size_t i = 0; i < nu; i++) mY4 = fmax(mY4, fabs((double)hY2[i] - refY[i]));
        for (size_t i = 0; i < (size_t)H * D * D; i++) mS4 = fmax(mS4, fabs((double)hS2[i] - refS[i]));
        double relY4 = mY4 / fmax(sY, 1e-3), relS4 = mS4 / fmax(sS, 1e-3);
        int ok4 = relY4 < 3e-2 && relS4 < 3e-2;
        printf("v4 chain: Y rel %.3e  state rel %.3e  %s (band 3e-2)\n",
               relY4, relS4, ok4 ? "IN-BAND" : "OUT-OF-BAND");
        for (int i = 0; i < 5; i++) gdn_k4_wgmma_v4<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 50; i++) gdn_k4_wgmma_v4<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dS, dsi, dso, H, T, C);
        CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        CK(cudaEventElapsedTime(&ms, a, b2));
        printf("v4 timing: %.1fus/call\n", ms * 20.0f);

        // ---- v5: K4+K5 fused ----
        __nv_bfloat16 *hqb = (__nv_bfloat16*)malloc(nk * 2);
        __nv_bfloat16 *hPb = (__nv_bfloat16*)malloc((size_t)NC * H * C * C * 2);
        for (size_t i = 0; i < nk; i++) hqb[i] = __float2bfloat16(hq5[i]);
        for (int c2 = 0; c2 < NC; c2++)
            for (int h2 = 0; h2 < H; h2++)
                for (int j = 0; j < C; j++)
                    for (int j2 = 0; j2 < C; j2++) {
                        size_t ix = (((size_t)c2 * H + h2) * C + j) * C + j2;
                        hPb[ix] = __float2bfloat16(j2 <= j ? hP5[ix] : 0.0f);
                    }
        __nv_bfloat16 *dq5, *dP5; float *dO5;
        CK(cudaMalloc(&dq5, nk * 2)); CK(cudaMalloc(&dP5, (size_t)NC * H * C * C * 2));
        CK(cudaMalloc(&dO5, nk * 4));
        CK(cudaMemcpy(dq5, hqb, nk * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dP5, hPb, (size_t)NC * H * C * C * 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(dY, 0, nu * 4));
        CK(cudaMemset(dO5, 0, nk * 4));
        CK(cudaMemset(dso, 0, (size_t)H * D * D * 4));
        gdn_k45_wgmma_v5<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hY2, dY, nu * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hS2, dso, (size_t)H * D * D * 4, cudaMemcpyDeviceToHost));
        float* hO5 = (float*)malloc(nk * 4);
        CK(cudaMemcpy(hO5, dO5, nk * 4, cudaMemcpyDeviceToHost));
        double mY5 = 0, mS5 = 0, mO5 = 0, sO5 = 0;
        for (size_t i = 0; i < nu; i++) mY5 = fmax(mY5, fabs((double)hY2[i] - refY[i]));
        for (size_t i = 0; i < (size_t)H * D * D; i++) mS5 = fmax(mS5, fabs((double)hS2[i] - refS[i]));
        for (size_t i = 0; i < nk; i++) { mO5 = fmax(mO5, fabs((double)hO5[i] - refO[i])); sO5 = fmax(sO5, fabs((double)refO[i])); }
        double relY5 = mY5 / fmax(sY, 1e-3), relS5 = mS5 / fmax(sS, 1e-3), relO5 = mO5 / fmax(sO5, 1e-3);
        int ok5 = relY5 < 3e-2 && relS5 < 3e-2 && relO5 < 3e-2;
        printf("v5 fused: Y rel %.3e  state rel %.3e  O rel %.3e  %s (band 3e-2)\n",
               relY5, relS5, relO5, ok5 ? "IN-BAND" : "OUT-OF-BAND");
        for (int i = 0; i < 5; i++) gdn_k45_wgmma_v5<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 50; i++) gdn_k45_wgmma_v5<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        CK(cudaEventElapsedTime(&ms, a, b2));
        printf("v5 timing: %.1fus/call (v4 K4-only above; fused replaces K4 + K5 + Ssnap traffic)\n", ms * 20.0f);

        // ---- v6: exchange-free ----
        CK(cudaMemset(dY, 0, nu * 4));
        CK(cudaMemset(dO5, 0, nk * 4));
        CK(cudaMemset(dso, 0, (size_t)H * D * D * 4));
        gdn_k45_wgmma_v6<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hY2, dY, nu * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hS2, dso, (size_t)H * D * D * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hO5, dO5, nk * 4, cudaMemcpyDeviceToHost));
        double mY6 = 0, mS6 = 0, mO6 = 0;
        for (size_t i = 0; i < nu; i++) mY6 = fmax(mY6, fabs((double)hY2[i] - refY[i]));
        for (size_t i = 0; i < (size_t)H * D * D; i++) mS6 = fmax(mS6, fabs((double)hS2[i] - refS[i]));
        for (size_t i = 0; i < nk; i++) mO6 = fmax(mO6, fabs((double)hO5[i] - refO[i]));
        double relY6 = mY6 / fmax(sY, 1e-3), relS6 = mS6 / fmax(sS, 1e-3), relO6 = mO6 / fmax(sO5, 1e-3);
        int ok6 = relY6 < 3e-2 && relS6 < 3e-2 && relO6 < 3e-2;
        printf("v6 nox: Y rel %.3e  state rel %.3e  O rel %.3e  %s (band 3e-2)\n",
               relY6, relS6, relO6, ok6 ? "IN-BAND" : "OUT-OF-BAND");
        for (int i = 0; i < 5; i++) gdn_k45_wgmma_v6<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 50; i++) gdn_k45_wgmma_v6<<<grid, 256>>>(dkb, dgc, dbt, dU, dwb, dY, dq5, dP5, dO5, scale5, dsi, dso, H, T, C);
        CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        CK(cudaEventElapsedTime(&ms, a, b2));
        printf("v6 timing: %.1fus/call\n", ms * 20.0f);
        return (ok && ok4 && ok5 && ok6) ? 0 : 1;
    }
}
