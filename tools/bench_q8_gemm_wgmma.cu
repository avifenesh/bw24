// wgmma prefill-GEMM dev harness v1 (task 8, ARCHITECTURE-H100.md): pipelined warpgroup
// int8 GEMM vs the vendored MMQ per-shape numbers (nsys 2026-07-26, m=512 pp512 prime):
//   out 12288: MMQ 253us | out 8192: 168us | out 4096: 144us | out 1024: 82us
// v0 verdict: unpipelined 64x64 lost 3.1x at model shapes (541us avg vs MMQ 178us) — the
// harness's old "688us MMQ ref" was a pp2048-shape figure (baseline error, ledger'd).
//
// v1: 64(M) x 128(N) tile per warpgroup, NSTAGE=3 cp.async ring for the A/B int8 slices,
// ONE barrier per 32-K block, wgmma.m64n128k32 (single instr per block), scale folds read
// dplane/ad direct via __ldg (both are per-thread SEQUENTIAL in blk -> L1 line serves 64
// iters; smem-staging them measured as the stall in v0's profile shape).
// CORRECTNESS LAW: fence.proxy.async.shared::cta between generic/cp.async smem writes and
// wgmma's async-proxy reads (the v0 root bug; keep unconditionally).
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/wgmma tools/bench_q8_gemm_wgmma.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

// smem descriptor: start addr (bits 0-13, >>4), leading byte offset (16-29, >>4),
// stride byte offset (32-45, >>4), swizzle (62-63). Canonical no-swizzle K-major cores:
// core(i,j) = 8 strided rows x 16 contiguous bytes at i*SBO(256) + j*LBO(128), row r at +r*16.
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
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_size) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int src_size) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(d), "l"(src), "r"(src_size));
}
template<int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ void wgmma_m64n128k32_s8(int acc[64], unsigned long long da,
                                                    unsigned long long db, int scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, p;\n"
        "}\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]), "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        : "l"(da), "l"(db), "r"(scale_d));
}

// ---------------- v0 (kept as the reference/regression arm) ----------------
__device__ __forceinline__ void wgmma_m64n64k32_s8(int acc[32], unsigned long long da,
                                                   unsigned long long db, int scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p;\n"
        "}\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]),
          "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]),
          "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
          "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
          "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]),
          "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]),
          "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
          "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

extern "C" __global__ void __launch_bounds__(128, 1)
q8_gemm_wgmma_v0(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bscale,
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 64;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    __shared__ __align__(128) signed char sA[64 * 32];
    __shared__ __align__(128) signed char sB[64 * 32];
    float facc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) facc[i] = 0.0f;
    for (int blk = 0; blk < nblk; blk++) {
        {
            int r = tid / 2, seg = tid % 2;
            const signed char* src = A + (size_t)(row0 + r) * in_f + blk * 32 + seg * 16;
            signed char* dst = sA + (r / 8) * 256 + seg * 128 + (r % 8) * 16;
            *(int4*)dst = (row0 + r < out_f) ? *(const int4*)src : make_int4(0,0,0,0);
        }
        {
            int c = tid / 2, seg = tid % 2;
            const signed char* src = B + (size_t)(col0 + c) * in_f + blk * 32 + seg * 16;
            signed char* dst = sB + (c / 8) * 256 + seg * 128 + (c % 8) * 16;
            *(int4*)dst = (col0 + c < n_tok) ? *(const int4*)src : make_int4(0,0,0,0);
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        int acc[32];
        wgmma_fence();
        unsigned long long da = make_desc(sA, 128, 256);
        unsigned long long db = make_desc(sB, 128, 256);
        wgmma_m64n64k32_s8(acc, da, db, 0);
        wgmma_commit();
        wgmma_wait<0>();
        float wsc[2];
        {
            int warp = tid / 32;
            int r_base = warp * 16 + (tid % 32) / 4;
            wsc[0] = __half2float(Ascale[(size_t)(row0 + r_base) * nblk + blk]);
            wsc[1] = __half2float(Ascale[(size_t)(row0 + r_base + 8) * nblk + blk]);
        }
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4, reg = i % 4;
            int col = (tid % 4) * 2 + n8 * 8 + (reg % 2);
            float bs = Bscale[(size_t)(col0 + col) * nblk + blk];
            facc[i] += (float)acc[i] * wsc[reg / 2] * bs;
        }
        __syncthreads();
    }
    {
        int warp = tid / 32;
        int r_base = row0 + warp * 16 + (tid % 32) / 4;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4, reg = i % 4;
            int col = col0 + (tid % 4) * 2 + n8 * 8 + (reg % 2);
            int row = r_base + (reg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}

// ---------------- v1: pipelined 64x128 ----------------
#define V1_NSTAGE 3
// per stage: A 64x32 (2048B) + B 128x32 (4096B), both canonical-core layout.
extern "C" __global__ void __launch_bounds__(128, 1)
q8_gemm_wgmma_v1(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bscale,
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 128;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    __shared__ __align__(1024) signed char sA[V1_NSTAGE][64 * 32];
    __shared__ __align__(1024) signed char sB[V1_NSTAGE][128 * 32];

    float facc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) facc[i] = 0.0f;

    // per-thread cp.async targets (fixed across stages):
    //   A: 1 chunk  — r = tid/2, seg = tid%2
    //   B: 2 chunks — idx = tid + i*128, c = idx>>1, seg = idx&1
    int a_r = tid / 2, a_seg = tid % 2;
    int a_dst_off = (a_r / 8) * 256 + a_seg * 128 + (a_r % 8) * 16;
    const signed char* a_src_base = A + (size_t)(row0 + a_r) * in_f + a_seg * 16;

    #define V1_ISSUE(stage, blk) do {                                                     \
        cp_async16(sA[stage] + a_dst_off, a_src_base + (size_t)(blk) * 32, 16);            \
        _Pragma("unroll")                                                                  \
        for (int i_ = 0; i_ < 2; i_++) {                                                   \
            int idx_ = tid + i_ * 128;                                                     \
            int c_ = idx_ >> 1, seg_ = idx_ & 1;                                           \
            signed char* d_ = sB[stage] + (c_ / 8) * 256 + seg_ * 128 + (c_ % 8) * 16;     \
            const signed char* s_ = B + (size_t)(col0 + c_) * in_f + (size_t)(blk) * 32 + seg_ * 16; \
            cp_async16(d_, s_, (col0 + c_ < n_tok) ? 16 : 0);                              \
        }                                                                                  \
        cp_async_commit();                                                                 \
    } while (0)

    // prologue: stages 0..NSTAGE-2
    #pragma unroll
    for (int s = 0; s < V1_NSTAGE - 1; s++) if (s < nblk) V1_ISSUE(s, s);

    int warp = tid / 32;
    int r_base = warp * 16 + (tid % 32) / 4;
    const half* wsc_p0 = Ascale + (size_t)(row0 + r_base) * nblk;
    const half* wsc_p1 = Ascale + (size_t)(row0 + r_base + 8) * nblk;

    for (int blk = 0; blk < nblk; blk++) {
        int stage = blk % V1_NSTAGE;
        cp_async_wait<V1_NSTAGE - 2>();          // current stage landed
        __syncthreads();                          // all folds of blk-1 done (WAR for re-issue)
        int nxt = blk + V1_NSTAGE - 1;
        if (nxt < nblk) V1_ISSUE((nxt) % V1_NSTAGE, nxt);
        asm volatile("fence.proxy.async.shared::cta;");

        int acc[64];
        wgmma_fence();
        unsigned long long da = make_desc(sA[stage], 128, 256);
        unsigned long long db = make_desc(sB[stage], 128, 256);
        wgmma_m64n128k32_s8(acc, da, db, 0);      // fresh s32 accumulate per 32-K block
        wgmma_commit();
        wgmma_wait<0>();

        float wsc0 = __half2float(__ldg(wsc_p0 + blk));
        float wsc1 = __half2float(__ldg(wsc_p1 + blk));
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            int n8 = i / 4, reg = i % 4;
            int col = (tid % 4) * 2 + n8 * 8 + (reg % 2);
            float bs = (col0 + col < n_tok) ? __ldg(Bscale + (size_t)(col0 + col) * nblk + blk) : 0.0f;
            facc[i] += (float)acc[i] * (reg < 2 ? wsc0 : wsc1) * bs;
        }
    }
    {
        int rb = row0 + r_base;
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            int n8 = i / 4, reg = i % 4;
            int col = col0 + (tid % 4) * 2 + n8 * 8 + (reg % 2);
            int row = rb + (reg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}


// ---------------- v2: n64 dual-accumulator pipeline ----------------
// v1 verdict: 246 regs (acc64+facc64) -> 2 CTA/SM occupancy collapse; slower than v0.
// v2 keeps the 64x64 tile (32+32 dual acc sets + 32 facc ~= 140 regs), overlaps the
// fold of block k-1 with the wgmma of block k (wgmma.wait_group<1>), stages A/B AND
// the activation block-scales through a 4-deep cp.async ring with lookahead 2 (ring
// deeper than lookahead by 2 so the re-issue target is never the stage the in-flight
// fold reads). Weight scales stay direct __ldg (per-thread sequential in blk).
#define V2_NSTAGE 4
#define V2_LA 2
extern "C" __global__ void __launch_bounds__(128, 1)
q8_gemm_wgmma_v2(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bscale,
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 64;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    __shared__ __align__(1024) signed char sA[V2_NSTAGE][64 * 32];
    __shared__ __align__(1024) signed char sB[V2_NSTAGE][64 * 32];
    __shared__ __align__(128) float sBs[V2_NSTAGE][64];

    float facc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) facc[i] = 0.0f;

    int r = tid / 2, seg = tid % 2;
    int core_off = (r / 8) * 256 + seg * 128 + (r % 8) * 16;
    const signed char* a_src = A + (size_t)(row0 + r) * in_f + seg * 16;
    const signed char* b_src = B + (size_t)(col0 + r) * in_f + seg * 16;
    int b_ok = (col0 + r < n_tok) ? 16 : 0;

    #define V2_ISSUE(stage, blk) do {                                                   \
        cp_async16(sA[stage] + core_off, a_src + (size_t)(blk) * 32, 16);                \
        cp_async16(sB[stage] + core_off, b_src + (size_t)(blk) * 32, b_ok);              \
        if (tid < 64)                                                                    \
            cp_async4(&sBs[stage][tid], Bscale + (size_t)(col0 + tid) * nblk + (blk),    \
                      (col0 + tid < n_tok) ? 4 : 0);                                     \
        cp_async_commit();                                                               \
    } while (0)

    #pragma unroll
    for (int s = 0; s < V2_LA; s++) if (s < nblk) V2_ISSUE(s, s);

    int warp = tid / 32;
    int r_base = warp * 16 + (tid % 32) / 4;
    const half* wsc_p0 = Ascale + (size_t)(row0 + r_base) * nblk;
    const half* wsc_p1 = Ascale + (size_t)(row0 + r_base + 8) * nblk;

    // acc sets MUST be statically indexed (a runtime acc[blk&1] put them in local mem:
    // 256B stack frame, 1116us — the v2a verdict). Unroll the K loop in PAIRS: even
    // block -> acc0, odd -> acc1. nblk is even for every gated shape (in_f % 64 == 0).
    int acc0[32], acc1[32];
    #define V2_FOLD(pa, pblk) do {                                                       \
        int ps = (pblk) % V2_NSTAGE;                                                     \
        float w0 = __half2float(__ldg(wsc_p0 + (pblk)));                                 \
        float w1 = __half2float(__ldg(wsc_p1 + (pblk)));                                 \
        _Pragma("unroll")                                                                \
        for (int i = 0; i < 32; i++) {                                                   \
            int n8 = i / 4, rg = i % 4;                                                  \
            int col = (tid % 4) * 2 + n8 * 8 + (rg % 2);                                 \
            facc[i] += (float)(pa)[i] * (rg < 2 ? w0 : w1) * sBs[ps][col];               \
        }                                                                                \
    } while (0)
    #define V2_STEP(pa, blk) do {                                                        \
        int stage_ = (blk) % V2_NSTAGE;                                                  \
        cp_async_wait<V2_LA - 1>();                                                      \
        __syncthreads();                                                                 \
        asm volatile("fence.proxy.async.shared::cta;");                                  \
        wgmma_fence();                                                                   \
        unsigned long long da_ = make_desc(sA[stage_], 128, 256);                        \
        unsigned long long db_ = make_desc(sB[stage_], 128, 256);                        \
        wgmma_m64n64k32_s8(pa, da_, db_, 0);                                             \
        wgmma_commit();                                                                  \
        int nxt_ = (blk) + V2_LA;                                                        \
        if (nxt_ < nblk) V2_ISSUE(nxt_ % V2_NSTAGE, nxt_);                               \
    } while (0)

    for (int blk = 0; blk < nblk; blk += 2) {
        V2_STEP(acc0, blk);
        if (blk > 0) { wgmma_wait<1>(); V2_FOLD(acc1, blk - 1); }
        V2_STEP(acc1, blk + 1);
        wgmma_wait<1>(); V2_FOLD(acc0, blk);
    }
    wgmma_wait<0>();
    V2_FOLD(acc1, nblk - 1);

    {
        int rb = row0 + r_base;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4, rg = i % 4;
            int col = col0 + (tid % 4) * 2 + n8 * 8 + (rg % 2);
            int row = rb + (rg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}

// ---------------- v3: v2 + TRANSPOSED activation scales ----------------
// v2's Bscale staging is 64 uncoalesced 4B cp.asyncs per 32-K block (col-major layout
// ad[col*nblk+blk] scatters the per-block slice). The engine owns quantize_q8_1's store
// index, so a [blk][tok] twin is one extra fused store — here modeled as Bst.
// v1 verdict: 246 regs (acc64+facc64) -> 2 CTA/SM occupancy collapse; slower than v0.
// v2 keeps the 64x64 tile (32+32 dual acc sets + 32 facc ~= 140 regs), overlaps the
// fold of block k-1 with the wgmma of block k (wgmma.wait_group<1>), stages A/B AND
// the activation block-scales through a 4-deep cp.async ring with lookahead 2 (ring
// deeper than lookahead by 2 so the re-issue target is never the stage the in-flight
// fold reads). Weight scales stay direct __ldg (per-thread sequential in blk).
#define V3_NSTAGE 4
#define V3_LA 2
extern "C" __global__ void __launch_bounds__(128, 1)
q8_gemm_wgmma_v3(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bst,  /* TRANSPOSED [blk][tok] */
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 64;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    __shared__ __align__(1024) signed char sA[V3_NSTAGE][64 * 32];
    __shared__ __align__(1024) signed char sB[V3_NSTAGE][64 * 32];
    __shared__ __align__(128) float sBs[V3_NSTAGE][64];

    float facc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) facc[i] = 0.0f;

    int r = tid / 2, seg = tid % 2;
    int core_off = (r / 8) * 256 + seg * 128 + (r % 8) * 16;
    const signed char* a_src = A + (size_t)(row0 + r) * in_f + seg * 16;
    const signed char* b_src = B + (size_t)(col0 + r) * in_f + seg * 16;
    int b_ok = (col0 + r < n_tok) ? 16 : 0;

    #define V3_ISSUE(stage, blk) do {                                                   \
        cp_async16(sA[stage] + core_off, a_src + (size_t)(blk) * 32, 16);                \
        cp_async16(sB[stage] + core_off, b_src + (size_t)(blk) * 32, b_ok);              \
        if (tid < 16)                                                                    \
            cp_async16(&sBs[stage][tid * 4],                                             \
                       Bst + (size_t)(blk) * n_tok + col0 + tid * 4,                     \
                       (col0 + tid * 4 < n_tok) ? 16 : 0);                               \
        cp_async_commit();                                                               \
    } while (0)

    #pragma unroll
    for (int s = 0; s < V3_LA; s++) if (s < nblk) V3_ISSUE(s, s);

    int warp = tid / 32;
    int r_base = warp * 16 + (tid % 32) / 4;
    const half* wsc_p0 = Ascale + (size_t)(row0 + r_base) * nblk;
    const half* wsc_p1 = Ascale + (size_t)(row0 + r_base + 8) * nblk;

    // acc sets MUST be statically indexed (a runtime acc[blk&1] put them in local mem:
    // 256B stack frame, 1116us — the v2a verdict). Unroll the K loop in PAIRS: even
    // block -> acc0, odd -> acc1. nblk is even for every gated shape (in_f % 64 == 0).
    int acc0[32], acc1[32];
    #define V3_FOLD(pa, pblk) do {                                                       \
        int ps = (pblk) % V3_NSTAGE;                                                     \
        float w0 = __half2float(__ldg(wsc_p0 + (pblk)));                                 \
        float w1 = __half2float(__ldg(wsc_p1 + (pblk)));                                 \
        _Pragma("unroll")                                                                \
        for (int i = 0; i < 32; i++) {                                                   \
            int n8 = i / 4, rg = i % 4;                                                  \
            int col = (tid % 4) * 2 + n8 * 8 + (rg % 2);                                 \
            facc[i] += (float)(pa)[i] * (rg < 2 ? w0 : w1) * sBs[ps][col];               \
        }                                                                                \
    } while (0)
    #define V3_STEP(pa, blk) do {                                                        \
        int stage_ = (blk) % V3_NSTAGE;                                                  \
        cp_async_wait<V3_LA - 1>();                                                      \
        __syncthreads();                                                                 \
        asm volatile("fence.proxy.async.shared::cta;");                                  \
        wgmma_fence();                                                                   \
        unsigned long long da_ = make_desc(sA[stage_], 128, 256);                        \
        unsigned long long db_ = make_desc(sB[stage_], 128, 256);                        \
        wgmma_m64n64k32_s8(pa, da_, db_, 0);                                             \
        wgmma_commit();                                                                  \
        int nxt_ = (blk) + V3_LA;                                                        \
        if (nxt_ < nblk) V3_ISSUE(nxt_ % V3_NSTAGE, nxt_);                               \
    } while (0)

#ifdef V3_CEIL
    // PERF CEILING PROBE (numerically wrong on purpose): s32 accumulate across ALL K
    // (scale_d=1), zero mid-loop acc reads -> wgmma streams. Same loads/ring as v3.
    // Measures what the pipeline delivers when the per-32-block fold law is lifted.
    (void)acc1;
    for (int blk = 0; blk < nblk; blk++) {
        int stage_ = blk % V3_NSTAGE;
        cp_async_wait<V3_LA - 1>();
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        wgmma_fence();
        unsigned long long da_ = make_desc(sA[stage_], 128, 256);
        unsigned long long db_ = make_desc(sB[stage_], 128, 256);
        wgmma_m64n64k32_s8(acc0, da_, db_, blk == 0 ? 0 : 1);
        wgmma_commit();
        int nxt_ = blk + V3_LA;
        if (nxt_ < nblk) V3_ISSUE(nxt_ % V3_NSTAGE, nxt_);
    }
    wgmma_wait<0>();
    V3_FOLD(acc0, nblk - 1);
#else
    for (int blk = 0; blk < nblk; blk += 2) {
        V3_STEP(acc0, blk);
        if (blk > 0) { wgmma_wait<1>(); V3_FOLD(acc1, blk - 1); }
        V3_STEP(acc1, blk + 1);
        wgmma_wait<1>(); V3_FOLD(acc0, blk);
    }
    wgmma_wait<0>();
    V3_FOLD(acc1, nblk - 1);
#endif

    {
        int rb = row0 + r_base;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4, rg = i % 4;
            int col = col0 + (tid % 4) * 2 + n8 * 8 + (rg % 2);
            int row = rb + (rg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}



// ---------------- v4: 2-warpgroup ping-pong, M64 x N128 CTA tile ----------------
// Ceiling probe verdict (V3_CEIL): fold-free n64 still lands ~2.6x above the W-traffic
// floor — grid.y=8 re-reads W 8x (the dominant cost), and the per-32-block fold law
// serializes each warpgroup's wgmma pipe (~100us at wqkv). One design kills both:
// 256 threads = 2 warpgroups SHARING the A tile (N=128 per CTA -> grid.y=4, W traffic
// halved), each warpgroup owning 64 columns. While wg0 drains its accs to fold, wg1's
// wgmma occupies the tensor pipe (CUTLASS ping-pong, jitter-scheduled not barriered).
#ifndef V4_NSTAGE
#define V4_NSTAGE 4
#endif
#ifndef V4_LA
#define V4_LA 2
#endif
#ifndef V4_CTAS
#define V4_CTAS 1
#endif
extern "C" __global__ void __launch_bounds__(256, V4_CTAS)
q8_gemm_wgmma_v4(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bst,
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 128;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;          // 0..255
    int wg = tid / 128;             // warpgroup 0/1 -> columns col0 + wg*64 ..
    int ltid = tid % 128;
    int nblk = in_f / 32;
    __shared__ __align__(1024) signed char sA[V4_NSTAGE][64 * 32];
    __shared__ __align__(1024) signed char sB[V4_NSTAGE][128 * 32];
    __shared__ __align__(128) float sBs[V4_NSTAGE][128];

    float facc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) facc[i] = 0.0f;

    // cp.async split across 256 threads: A (128 chunks) by tid<128, B (256 chunks) by all,
    // transposed scales (8 x 16B) by tid<8.
    int a_r = tid / 2, a_seg = tid % 2;
    int a_off = (a_r / 8) * 256 + a_seg * 128 + (a_r % 8) * 16;
    const signed char* a_src = A + (size_t)(row0 + a_r) * in_f + a_seg * 16;
    int b_c = tid / 2, b_seg = tid % 2;
    int b_off = (b_c / 8) * 256 + b_seg * 128 + (b_c % 8) * 16;
    const signed char* b_src = B + (size_t)(col0 + b_c) * in_f + b_seg * 16;
    int b_ok = (col0 + b_c < n_tok) ? 16 : 0;

#ifdef V4_ADPLAIN
    // engine-native activation scales ad[col*nblk + blk]: 4 uncoalesced 4B cp.asyncs per
    // thread (tid<32 covers 128 cols) instead of 8x16B from a transposed twin.
    #define V4_SCALE_CP(stage, blk) do {                                                 \
        _Pragma("unroll")                                                                \
        for (int j_ = 0; j_ < 4; j_++) {                                                 \
            int c_ = tid * 4 + j_;                                                       \
            cp_async4(&sBs[stage][c_], Bst + (size_t)(col0 + c_) * nblk + (blk),         \
                      (col0 + c_ < n_tok) ? 4 : 0);                                      \
        }                                                                                \
    } while (0)
#else
    #define V4_SCALE_CP(stage, blk)                                                      \
        cp_async16(&sBs[stage][tid * 4],                                                 \
                   Bst + (size_t)(blk) * n_tok + col0 + tid * 4,                         \
                   (col0 + tid * 4 < n_tok) ? 16 : 0)
#endif

    #define V4_ISSUE(stage, blk) do {                                                   \
        if (tid < 128) cp_async16(sA[stage] + a_off, a_src + (size_t)(blk) * 32, 16);    \
        cp_async16(sB[stage] + b_off, b_src + (size_t)(blk) * 32, b_ok);                 \
        if (tid < 32)                                                                    \
            V4_SCALE_CP(stage, blk);                                                     \
        cp_async_commit();                                                               \
    } while (0)

    #pragma unroll
    for (int s = 0; s < V4_LA; s++) if (s < nblk) V4_ISSUE(s, s);

    int warp = ltid / 32;
    int r_base = warp * 16 + (ltid % 32) / 4;
    const half* wsc_p0 = Ascale + (size_t)(row0 + r_base) * nblk;
    const half* wsc_p1 = Ascale + (size_t)(row0 + r_base + 8) * nblk;

    int acc0[32], acc1[32];
    #define V4_FOLD(pa, pblk) do {                                                       \
        int ps = (pblk) % V4_NSTAGE;                                                     \
        float w0 = __half2float(__ldg(wsc_p0 + (pblk)));                                 \
        float w1 = __half2float(__ldg(wsc_p1 + (pblk)));                                 \
        _Pragma("unroll")                                                                \
        for (int i = 0; i < 32; i++) {                                                   \
            int n8 = i / 4, rg = i % 4;                                                  \
            int col = wg * 64 + (ltid % 4) * 2 + n8 * 8 + (rg % 2);                      \
            facc[i] += (float)(pa)[i] * (rg < 2 ? w0 : w1) * sBs[ps][col];               \
        }                                                                                \
    } while (0)
    #define V4_STEP(pa, blk) do {                                                        \
        int stage_ = (blk) % V4_NSTAGE;                                                  \
        cp_async_wait<V4_LA - 1>();                                                      \
        __syncthreads();                                                                 \
        asm volatile("fence.proxy.async.shared::cta;");                                  \
        wgmma_fence();                                                                   \
        unsigned long long da_ = make_desc(sA[stage_], 128, 256);                        \
        unsigned long long db_ = make_desc(sB[stage_] + wg * 2048, 128, 256);            \
        wgmma_m64n64k32_s8(pa, da_, db_, 0);                                             \
        wgmma_commit();                                                                  \
        int nxt_ = (blk) + V4_LA;                                                        \
        if (nxt_ < nblk) V4_ISSUE(nxt_ % V4_NSTAGE, nxt_);                               \
    } while (0)

    for (int blk = 0; blk < nblk; blk += 2) {
        V4_STEP(acc0, blk);
        if (blk > 0) { wgmma_wait<1>(); V4_FOLD(acc1, blk - 1); }
        V4_STEP(acc1, blk + 1);
        wgmma_wait<1>(); V4_FOLD(acc0, blk);
    }
    wgmma_wait<0>();
    V4_FOLD(acc1, nblk - 1);

    {
        int rb = row0 + r_base;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4, rg = i % 4;
            int col = col0 + wg * 64 + (ltid % 4) * 2 + n8 * 8 + (rg % 2);
            int row = rb + (rg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}

// ---- host: real model shapes, CPU ref probes, timing vs MMQ nsys medians ----
struct ShapeRef { int in_f, out_f, mmq_us; const char* tag; };

int main() {
    const int n_tok = 512;
    ShapeRef shapes[] = {
        {4096, 12288, 253, "wqkv (lin)"},
        {4096,  8192, 168, "mid"},
        {4096,  4096, 144, "square"},
        {11008, 4096, 144, "ffn_down"},
        {4096, 11008, 236, "ffn_gate/up"},   // grid 86x128: MMQ bucket ~ out12288 shape class
        {4096,  1024,  82, "small"},
    };
    srand(7);
    for (auto& sh : shapes) {
        int in_f = sh.in_f, out_f = sh.out_f, nblk = in_f / 32;
        signed char* hA = (signed char*)malloc((size_t)out_f * in_f);
        half* hAs = (half*)malloc((size_t)out_f * nblk * 2);
        signed char* hB = (signed char*)malloc((size_t)n_tok * in_f);
        float* hBs = (float*)malloc((size_t)n_tok * nblk * 4);
        float* hBst = (float*)malloc((size_t)n_tok * nblk * 4);
        for (size_t i = 0; i < (size_t)out_f * in_f; i++) hA[i] = (signed char)(rand() % 17 - 8);
        for (size_t i = 0; i < (size_t)out_f * nblk; i++) hAs[i] = __float2half(0.01f + (rand() % 100) * 1e-4f);
        for (size_t i = 0; i < (size_t)n_tok * in_f; i++) hB[i] = (signed char)(rand() % 17 - 8);
        for (size_t i = 0; i < (size_t)n_tok * nblk; i++) hBs[i] = 0.02f + (rand() % 100) * 1e-4f;
        for (int t = 0; t < n_tok; t++) for (int b = 0; b < nblk; b++)
            hBst[(size_t)b * n_tok + t] = hBs[(size_t)t * nblk + b];

        signed char *dA, *dB; half* dAs; float *dBs, *dBst, *dC;
        CK(cudaMalloc(&dA, (size_t)out_f * in_f));
        CK(cudaMalloc(&dAs, (size_t)out_f * nblk * 2));
        CK(cudaMalloc(&dB, (size_t)n_tok * in_f));
        CK(cudaMalloc(&dBs, (size_t)n_tok * nblk * 4));
        CK(cudaMalloc(&dBst, (size_t)n_tok * nblk * 4));
        CK(cudaMalloc(&dC, (size_t)n_tok * out_f * 4));
        CK(cudaMemcpy(dA, hA, (size_t)out_f * in_f, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dAs, hAs, (size_t)out_f * nblk * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, (size_t)n_tok * in_f, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBs, hBs, (size_t)n_tok * nblk * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBst, hBst, (size_t)n_tok * nblk * 4, cudaMemcpyHostToDevice));

        dim3 g0((out_f + 63) / 64, (n_tok + 63) / 64), g1((out_f + 63) / 64, (n_tok + 127) / 128), blk128(128);
        float* hC = (float*)malloc((size_t)n_tok * out_f * 4);
        double maxrel = 0;
        q8_gemm_wgmma_v2<<<g0, blk128>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC, dC, (size_t)n_tok * out_f * 4, cudaMemcpyDeviceToHost));
        for (int probe = 0; probe < 48; probe++) {
            int r = (probe * 131) % out_f, c = (probe * 197) % n_tok;
            double acc = 0;
            for (int b = 0; b < nblk; b++) {
                long s = 0;
                for (int k = 0; k < 32; k++)
                    s += (long)hA[(size_t)r * in_f + b * 32 + k] * hB[(size_t)c * in_f + b * 32 + k];
                acc += (double)s * __half2float(hAs[(size_t)r * nblk + b]) * hBs[(size_t)c * nblk + b];
            }
            double got = hC[(size_t)c * out_f + r];
            double rel = fabs(got - acc) / fmax(fabs(acc), 1e-3);
            if (rel > maxrel) maxrel = rel;
        }

        cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
        dim3 blk256(256);
        double us[5];
        for (int v = 0; v < 5; v++) {
            #define V_LAUNCH() do { \
                if (v == 0)      q8_gemm_wgmma_v0<<<g0, blk128>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok); \
                else if (v == 1) q8_gemm_wgmma_v1<<<g1, blk128>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok); \
                else if (v == 2) q8_gemm_wgmma_v2<<<g0, blk128>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok); \
                else if (v == 3) q8_gemm_wgmma_v3<<<g0, blk128>>>(dA, dAs, dB, dBst, dC, in_f, out_f, n_tok); \
                else             q8_gemm_wgmma_v4<<<g1, blk256>>>(dA, dAs, dB, dBst, dC, in_f, out_f, n_tok); \
            } while (0)
            for (int i = 0; i < 5; i++) V_LAUNCH();
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ea));
            for (int i = 0; i < 50; i++) V_LAUNCH();
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            us[v] = ms * 1000.0 / 50;
            #undef V_LAUNCH
        }
        // v3 correctness probe (transposed scales must not change one bit vs v2)
        double v3rel = 0;
        {
            q8_gemm_wgmma_v4<<<g1, blk256>>>(dA, dAs, dB, dBst, dC, in_f, out_f, n_tok);
            CK(cudaDeviceSynchronize());
            float* hC3 = (float*)malloc((size_t)n_tok * out_f * 4);
            CK(cudaMemcpy(hC3, dC, (size_t)n_tok * out_f * 4, cudaMemcpyDeviceToHost));
            for (int probe = 0; probe < 48; probe++) {
                int rr = (probe * 131) % out_f, cc = (probe * 197) % n_tok;
                double d = fabs((double)hC3[(size_t)cc * out_f + rr] - hC[(size_t)cc * out_f + rr]);
                if (d > v3rel) v3rel = d;
            }
            free(hC3);
        }
        printf("%-12s in=%5d out=%6d | v0 %5.1f v1 %5.1f v2 %5.1f v3 %5.1f v4 %5.1fus  MMQ %4dus  v4/MMQ %.2fx  rel %.1e %s v4d %.1e\n",
               sh.tag, in_f, out_f, us[0], us[1], us[2], us[3], us[4], sh.mmq_us, us[4] / sh.mmq_us,
               maxrel, maxrel < 1e-3 ? "OK" : "FAIL", v3rel);
        cudaFree(dA); cudaFree(dAs); cudaFree(dB); cudaFree(dBs); cudaFree(dC);
        cudaFree(dBst);
        free(hA); free(hAs); free(hB); free(hBs); free(hBst); free(hC);
    }
    return 0;
}
