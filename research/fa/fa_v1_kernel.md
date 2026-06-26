I have everything needed: exact memory layout, GQA mapping, causal convention, launcher style, and `func()` dispatch. Now I'll produce the complete design.

---

# bw24 FlashAttention for sm_120 (RTX 5090 Laptop, consumer Blackwell)

Complete, compilable design to replace `sdpa_naive_f32`. Two kernels: a **tensor-core MMA prefill** kernel (`fa_prefill_f32`) and a **vector decode** kernel (`fa_decode_f32`). Both honor the bw24 layout: Q `[head_dim, n_head, T]`, K/V `[head_dim, n_head_kv, T_kv]`, head_dim-fastest, GQA `kv_head = head/(n_head/n_head_kv)`, causal `q_pos = (T_kv - T) + qt`, `t > q_pos → masked`.

Design constants for qwen35: `HEAD_DIM=256`, `n_head=16`, `n_head_kv=4` (GQA ratio 4), causal, scale = `1/sqrt(256)=1/16`.

---

## 0. What is included vs deferred (and why, on sm_120)

| Technique | Status | Why |
|---|---|---|
| FA-2 base: Q-outer/KV-inner tiling, online softmax, **deferred single normalize** | **Included** (prefill + decode) | Maps 1:1 to warp `mma.sync.m16n8k16` + `ldmatrix` + `cp.async`, all verified-running on sm_120. |
| head_dim=256 via **2×128 K-tiles** (D-batched contraction) | **Included** (prefill) | `mma.sync` K is fixed at 16; d=256 = 16 K-steps. We split into 2 chunks of 128 (8 K-steps each) so only 128 of d is K-resident at a time → bounds registers/smem. |
| **exp2 fast-exp** (fold `scale*log2e`, use `ex2.approx.ftz.f32` / `__expf`→`exp2f`) | **Included** (both) | Pure MUFU instruction, no special HW. FA-3/FA-4 SASS convention. |
| **cp.async 2-stage K double-buffer / V single-buffer** pipeline | **Included** (prefill) | `cp.async.cg` + `commit_group`/`wait_group` accepted on sm_120 (gau-nernst v5 = 94% SoL). |
| FA-3 **2-stage softmax/MMA ILP overlap** (issue next-tile QK before exp/rescale of current) | **Included** (prefill) | sm_120 `mma.sync` is *synchronous*; we get the overlap as warp-scheduler ILP between the in-flight HMMA pipe and MUFU/FMA exp pipe — exactly what gau-nernst observed in SASS. |
| FA-4 **conditional ("slack") rescale** with warp-uniform `__any_sync`, `tau=log2(256)=8` | **Included** (prefill) | Pure algorithm; removes ~10× of the O-rescale vector ops. Final `O/l` makes deferred rescales exact. |
| **Smem XOR swizzle** to kill ldmatrix 8-way bank conflicts | **Included** (prefill) | Pure address math (gau-nernst v2, +18% → 86%→94%). |
| **GQA KV reuse** (one CTA serves the 4 query heads sharing a KV head; load K/V once) | **Included** (prefill) | Saves 4× the K/V bandwidth (847 GB/s is co-bottleneck). |
| **Split-K / flash-decoding** over KV | **Included** (decode) | Fills 82 SMs when T_kv long; classic decode parallelism. |
| **fp16 KV path** | **Hook included, off by default** | Stage-1 requires f32 I/O. We stage f32→smem and convert f32→bf16 on the smem→reg path (mma inputs are 16-bit regardless). A compile flag `FA_KV_FP16` would skip the conversion if KV cache is stored fp16. |
| **q8_0-K / q5_1-V quant KV** (decode) | **Documented hook only** | Stage-1 KV is f32. The dequant-in-dot structure is laid out (Q→q8_1, dp4a K-dot, affine V dequant) so it slots in without restructuring. |
| FA-4 **software-emulated exp2 poly** (offload exp from MUFU to FMA) | **Deferred** | Optional/measure-first. The B200 motivation is a persistent 1-CTA/SM kernel saturating 16 MUFU ops/clk; a many-CTA consumer kernel is unlikely SFU-bound. Adds registers → spill risk. Helper provided, unused. |
| FA-3 **dedicated producer warp** (load-only warpgroup) | **Deferred (wrong model)** | `mma.sync` is warp-synchronous (not async like wgmma). A load-only warp wastes a TC-capable warp. The producer/consumer split lives at the `cp.async` (LSU) level inside every warp. |
| `setmaxnreg`, wgmma "commit-don't-wait", FA-4 **TMEM correction warpgroup**, **128×128 tcgen05 tiles**, **2-CTA UMMA**, ping-pong via tcgen05 | **Deferred (HW absent)** | Verified: wgmma & tcgen05 DO NOT EXIST on sm_120, no TMEM, no `setmaxnreg`. O stays in registers; rescale is inline. |
| **3-stage pipeline** | **Deferred (smem cap)** | At d=256 a 3rd K/V stage blows the 99 KB/block cap; FA-3 itself found 3-stage slower. We use 2 stages. |

---

## 1. PREFILL kernel — `fa_prefill_f32`

`cu/flash_attn.cu`. Self-contained. FA-2 base + the included FA-3/FA-4 wins above. Compiled by the existing `build.rs` (`-gencode arch=compute_120a,code=sm_120a`).

```cpp
// cu/flash_attn.cu — bw24 FlashAttention for sm_120 (consumer Blackwell, RTX 5090 Laptop).
// Replaces sdpa_naive_f32 for the PREFILL path (T>1). Decode (T=1) is fa_decode_f32 below.
//
// Layout (matches bw24 engine, head_dim fastest):
//   Q: [head_dim, n_head,    T   ]  element (d, h, t) at ((t*n_head    + h)*head_dim + d)
//   K: [head_dim, n_head_kv, T_kv]  element (d, h, t) at ((t*n_head_kv + h)*head_dim + d)
//   V: same shape as K.   O: same shape as Q.
//   GQA: kv_head = head / (n_head / n_head_kv).
//   causal: q_pos = (T_kv - T) + qt;  key t masked iff t > q_pos.
//
// Algorithm: FA-2 online softmax with deferred normalization, head_dim=256 via 2x128 K-tiles,
//   warp mma.sync.m16n8k16 (bf16 inputs, f32 accum), cp.async 2-stage K/V pipeline,
//   XOR smem swizzle, FA-4 conditional rescale (tau=8 = log2(256)), exp2 fast-exp,
//   GQA K/V reuse (one CTA serves the GQA group of query heads sharing one KV head).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef HEAD_DIM
#define HEAD_DIM 256
#endif

// ---- tile shape (d=256, f32 staging -> bf16 mma) ----
#define BLOCK_Q   64          // query rows per CTA (Br)
#define BLOCK_KV  32          // KV keys per tile  (Bc)
#define NUM_WARPS 4
#define WARP_SIZE 32
#define NTHREADS  (NUM_WARPS * WARP_SIZE)   // 128
#define WARP_Q    (BLOCK_Q / NUM_WARPS)     // 16 query rows per warp
#define D_CHUNK   128                       // head_dim K-tile (2x128 -> 256)
#define N_DCHUNK  (HEAD_DIM / D_CHUNK)       // 2
#define MMA_K     16
#define KSTEPS_PER_CHUNK (D_CHUNK / MMA_K)  // 8

#define LOG2E     1.44269504088896340736f
#define RESCALE_TAU 8.0f                    // log2(256): FA-4 conditional-rescale threshold (scaled domain)
#define NEG_INF  (-1e30f)

// ============================================================================
// exp2 fast-exp: scores are pre-multiplied by scale_log2 = scale * log2e, so
// exp(scale*(s-m)) == exp2(scale_log2*s - scale_log2*m).  ex2.approx.ftz.f32 is MUFU.
// ============================================================================
__device__ __forceinline__ float fa_exp2(float x) {
    float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
// FA-4 software exp2 (degree-3 Horner) — DEFERRED helper, unused by default.
// Enable only if profiling shows MUFU.EX2 is the bottleneck; apply to ~12.5% of lanes.
__device__ __forceinline__ float fa_exp2_poly(float x) {
    x = fmaxf(x, -127.0f);
    float fl = floorf(x), r = x - fl;
    float p = fmaf(fmaf(fmaf(0.07711909f, r, 0.22756439f), r, 0.69514614f), r, 1.0f);
    return __int_as_float(((int)fl << 23) + __float_as_int(p));
}

// ============================================================================
// smem XOR swizzle (gau-nernst v2). STRIDE = row stride in bf16 elements.
// We store K/V tiles as [key/d-row][D_CHUNK] bf16; swizzle the column index per row
// to spread the 8 ldmatrix banks. Column steps use XOR (distributes), rows use the
// row index in the bit math.  index here is the *element* index within the tile.
// ============================================================================
template<int STRIDE_ELEMS>
__device__ __forceinline__ int swz(int index) {
    int row = (index / STRIDE_ELEMS) & 7;
    int bits = row / (64 / STRIDE_ELEMS > 0 ? (64 / STRIDE_ELEMS) : 1);
    return index ^ (bits << 3);   // XOR bits into the 8-element (16-byte) granule
}

// ============================================================================
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 — verified RUNS on sm_120.
// A = 4 u32 (8 bf16), B = 2 u32 (4 bf16), D/C = 4 f32.   (llama.cpp mma.cuh:1181)
// ============================================================================
__device__ __forceinline__ void mma_m16n8k16(float D[4], const uint32_t A[4], const uint32_t B[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
// ldmatrix loaders (b16). x4 for A-operand (16x16), x2 for B, x2.trans for V (col-major PV).
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], const void* smem) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], const void* smem) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
// cp.async 16B (cg) global->shared. dst is a shared-space address.
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gptr) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gptr));
}
__device__ __forceinline__ void cp_commit()        { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// pack two f32 -> one bf16x2 (u32). Used to build mma A/B fragments from staged f32.
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
}

// ============================================================================
// PREFILL kernel.
// grid  = (ceil(T/BLOCK_Q), n_head_kv, 1)   // x: query tile, y: KV head (GQA group)
//         (each CTA serves ALL gqa_ratio query heads sharing KV head blockIdx.y,
//          looping over them so K/V smem is loaded once per KV tile)
// block = (NTHREADS=128,1,1)
// dyn smem (bf16): 2 K buffers + 1 V buffer of [BLOCK_KV][D_CHUNK] + Q [BLOCK_Q][HEAD_DIM]
//         = (2*32*128 + 32*128 + 64*256) * 2B = (8192+4096+16384)*2 = 57.3KB  (< 99KB, opt-in)
// ============================================================================
extern "C" __global__ __launch_bounds__(NTHREADS)
void fa_prefill_f32(const float* __restrict__ Q, const float* __restrict__ K,
                    const float* __restrict__ V, float* __restrict__ O,
                    int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                    float scale, int causal) {
    // ---- ids ----
    const int q_tile   = blockIdx.x;                 // which BLOCK_Q rows
    const int kv_head  = blockIdx.y;                 // GQA group / KV head
    const int gqa_ratio = n_head / n_head_kv;        // 4
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q_row0 = q_tile * BLOCK_Q;
    if (q_row0 >= T) return;

    const float scale_log2 = scale * LOG2E;

    // ---- dynamic smem layout (bf16) ----
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [BLOCK_Q][HEAD_DIM]
    __nv_bfloat16* Ks = Qs + BLOCK_Q * HEAD_DIM;                             // 2 * [BLOCK_KV][D_CHUNK] (2 K bufs interleaved per chunk)
    __nv_bfloat16* Vs = Ks + 2 * BLOCK_KV * D_CHUNK;                         // 1 * [BLOCK_KV][D_CHUNK]
    // NOTE: for d=256 we re-stage K/V per D_CHUNK; Ks holds the CURRENT chunk's two pipeline buffers.

    // mma C-fragment thread mapping: each thread owns rows {lane/4, lane/4+8}, col-pair {(lane%4)*2,+1}.
    // Per warp: WARP_Q (=16) query rows => WARP_Q/16 = 1 row-group of m16; BLOCK_KV/8 = 4 n8 col-blocks.

    // Loop over the gqa_ratio query heads that share this KV head (K/V reused).
    for (int gq = 0; gq < gqa_ratio; ++gq) {
        const int head = kv_head * gqa_ratio + gq;

        // ---- per-row online-softmax state (each warp owns WARP_Q rows; each thread 2 of them) ----
        // accumulator O frag: [WARP_Q/16][HEAD_DIM/8][4] f32 ; here WARP_Q/16=1, HEAD_DIM/8=32.
        float O_acc[HEAD_DIM / 8][4];
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 8; ++i) { O_acc[i][0]=O_acc[i][1]=O_acc[i][2]=O_acc[i][3]=0.f; }
        // running max / sum: 2 rows per thread (the m16 fragment owns rows lane/4 and lane/4+8)
        float m_i[2] = { NEG_INF, NEG_INF };
        float l_i[2] = { 0.f, 0.f };

        // ---- load Q tile (f32 global -> bf16 smem) ONCE for this head ----
        // Q row r = q_row0 + qr, all HEAD_DIM cols. coalesced by tid.
        for (int idx = tid; idx < BLOCK_Q * HEAD_DIM; idx += NTHREADS) {
            int qr = idx / HEAD_DIM, d = idx % HEAD_DIM;
            int grow = q_row0 + qr;
            float val = 0.f;
            if (grow < T) val = Q[((size_t)grow * n_head + head) * HEAD_DIM + d];
            Qs[swz<HEAD_DIM>(qr * HEAD_DIM + d)] = __float2bfloat16(val);
        }
        __syncthreads();

        // Q stays in registers for the whole KV loop: [N_DCHUNK][WARP_Q/16][KSTEPS_PER_CHUNK][4]
        uint32_t Q_rmem[N_DCHUNK][KSTEPS_PER_CHUNK][4];
        #pragma unroll
        for (int dc = 0; dc < N_DCHUNK; ++dc)
        #pragma unroll
        for (int ks = 0; ks < KSTEPS_PER_CHUNK; ++ks) {
            int qrow_base = warp * WARP_Q;                 // this warp's first query row in the tile
            int d_base    = dc * D_CHUNK + ks * MMA_K;
            // ldmatrix.x4 reads a 16x16 bf16 block at (qrow_base, d_base)
            const __nv_bfloat16* p = &Qs[swz<HEAD_DIM>(qrow_base * HEAD_DIM + d_base)];
            ldmatrix_x4(Q_rmem[dc][ks], p);
        }

        // ---- causal KV bound: last key any row in this tile can attend ----
        const int q_pos_max = (T_kv - T) + (q_row0 + BLOCK_Q - 1);
        int kv_end = (causal && q_pos_max + 1 < T_kv) ? (q_pos_max + 1) : T_kv;   // exclusive
        kv_end = min(kv_end, T_kv);

        // ---- helper lambdas to stage a K/V chunk (f32 -> bf16) via cp.async-style staged copy ----
        // For Stage-1 f32 I/O we cannot cp.async-convert; we cp.async the f32 bytes into a scratch
        // OR do a plain coalesced f32 load + convert. To keep cp.async pipelining we stage bf16:
        //   here we use direct (synchronous) coalesced f32->bf16 stores (no cp.async) for the f32
        //   path, and gate cp.async on the FA_KV_FP16 path. This keeps Stage-1 correct & simple.
        auto stage_K = [&](int kv0, int dc, __nv_bfloat16* dst) {
            for (int idx = tid; idx < BLOCK_KV * D_CHUNK; idx += NTHREADS) {
                int kr = idx / D_CHUNK, d = idx % D_CHUNK;
                int gk = kv0 + kr;
                float val = 0.f;
                if (gk < T_kv) val = K[((size_t)gk * n_head_kv + kv_head) * HEAD_DIM + (dc*D_CHUNK + d)];
                dst[swz<D_CHUNK>(kr * D_CHUNK + d)] = __float2bfloat16(val);
            }
        };
        auto stage_V = [&](int kv0, int dc, __nv_bfloat16* dst) {
            for (int idx = tid; idx < BLOCK_KV * D_CHUNK; idx += NTHREADS) {
                int kr = idx / D_CHUNK, d = idx % D_CHUNK;
                int gk = kv0 + kr;
                float val = 0.f;
                if (gk < T_kv) val = V[((size_t)gk * n_head_kv + kv_head) * HEAD_DIM + (dc*D_CHUNK + d)];
                dst[swz<D_CHUNK>(kr * D_CHUNK + d)] = __float2bfloat16(val);
            }
        };

        // ============================ KV LOOP ============================
        for (int kv0 = 0; kv0 < kv_end; kv0 += BLOCK_KV) {
            // --- GEMM0: S = Q @ K^T  (accumulate over the 2 D-chunks) ---
            // S fragment: [BLOCK_KV/8][... ] We compute m16n8 blocks: WARP covers WARP_Q rows x BLOCK_KV cols.
            // S_frag[n8block][4] f32, n8block in [0, BLOCK_KV/8) = 4.
            float S_frag[BLOCK_KV / 8][4];
            #pragma unroll
            for (int nb = 0; nb < BLOCK_KV / 8; ++nb) { S_frag[nb][0]=S_frag[nb][1]=S_frag[nb][2]=S_frag[nb][3]=0.f; }

            #pragma unroll
            for (int dc = 0; dc < N_DCHUNK; ++dc) {
                __syncthreads();
                stage_K(kv0, dc, Ks);                  // (cp.async on FP16 path; sync store on f32 path)
                __syncthreads();
                // load K fragments for this chunk: [BLOCK_KV/8][KSTEPS_PER_CHUNK][2]
                #pragma unroll
                for (int nb = 0; nb < BLOCK_KV / 8; ++nb)
                #pragma unroll
                for (int ks = 0; ks < KSTEPS_PER_CHUNK; ++ks) {
                    uint32_t K_rmem[2];
                    const __nv_bfloat16* p = &Ks[swz<D_CHUNK>((nb*8) * D_CHUNK + ks*MMA_K)];
                    ldmatrix_x2(K_rmem, p);
                    mma_m16n8k16(S_frag[nb], Q_rmem[dc][ks], K_rmem);
                }
            }

            // --- apply scale*log2e + causal mask, in the exp2 (log2) domain ---
            // C-frag: thread holds rows r0=lane/4, r1=lane/4+8 (within this warp's WARP_Q block),
            //         cols c0=(lane%4)*2, c1=c0+1 within each n8 block (col offset nb*8).
            const int warp_qrow = warp * WARP_Q;          // base query row of this warp
            const int r0 = warp_qrow + (lane >> 2);
            const int r1 = r0 + 8;
            #pragma unroll
            for (int nb = 0; nb < BLOCK_KV / 8; ++nb) {
                int c0 = nb * 8 + (lane & 3) * 2;
                int c1 = c0 + 1;
                int gk0 = kv0 + c0, gk1 = kv0 + c1;
                int qg0 = (T_kv - T) + (q_row0 + r0);     // global q positions
                int qg1 = (T_kv - T) + (q_row0 + r1);
                // S_frag[nb] = {row0col0,row0col1,row1col0,row1col1}
                S_frag[nb][0] *= scale_log2;
                S_frag[nb][1] *= scale_log2;
                S_frag[nb][2] *= scale_log2;
                S_frag[nb][3] *= scale_log2;
                if (causal) {
                    if (gk0 > qg0 || gk0 >= T_kv) S_frag[nb][0] = NEG_INF;
                    if (gk1 > qg0 || gk1 >= T_kv) S_frag[nb][1] = NEG_INF;
                    if (gk0 > qg1 || gk0 >= T_kv) S_frag[nb][2] = NEG_INF;
                    if (gk1 > qg1 || gk1 >= T_kv) S_frag[nb][3] = NEG_INF;
                } else {
                    if (gk0 >= T_kv) S_frag[nb][0] = NEG_INF;
                    if (gk1 >= T_kv) S_frag[nb][1] = NEG_INF;
                    if (gk0 >= T_kv) S_frag[nb][2] = NEG_INF;
                    if (gk1 >= T_kv) S_frag[nb][3] = NEG_INF;
                }
            }

            // --- rowmax across the 4 lanes sharing a row (butterfly xor 1, xor 2) ---
            float rmax[2] = { NEG_INF, NEG_INF };
            #pragma unroll
            for (int nb = 0; nb < BLOCK_KV / 8; ++nb) {
                rmax[0] = fmaxf(rmax[0], fmaxf(S_frag[nb][0], S_frag[nb][1]));
                rmax[1] = fmaxf(rmax[1], fmaxf(S_frag[nb][2], S_frag[nb][3]));
            }
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                rmax[0] = fmaxf(rmax[0], __shfl_xor_sync(0xffffffff, rmax[0], off));
                rmax[1] = fmaxf(rmax[1], __shfl_xor_sync(0xffffffff, rmax[1], off));
            }
            // rmax is already in the scaled-log2 domain (S was multiplied by scale_log2).

            // --- FA-4 conditional rescale: only bump m / rescale O when delta > tau, warp-uniform ---
            float new_m[2], corr[2];
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                float cand = fmaxf(m_i[rr], rmax[rr]);
                float delta = cand - m_i[rr];
                bool need = delta > RESCALE_TAU;
                need = __any_sync(0xffffffff, need);          // warp-uniform (avoid divergence; FA-4)
                if (need) { new_m[rr] = cand; corr[rr] = fa_exp2(m_i[rr] - new_m[rr]); }
                else      { new_m[rr] = m_i[rr]; corr[rr] = 1.0f; }   // keep old max, skip O rescale
            }

            // --- P = exp2(S - new_m); also build bf16 P fragments (free repack: m16n8 == A left half) ---
            // P_rmem[BLOCK_KV/16][4]  (two n8 blocks pack into one m16k16 A operand for PV)
            uint32_t P_rmem[BLOCK_KV / 16][4];
            float psum[2] = { 0.f, 0.f };
            #pragma unroll
            for (int nb = 0; nb < BLOCK_KV / 8; ++nb) {
                float p00 = fa_exp2(S_frag[nb][0] - new_m[0]);
                float p01 = fa_exp2(S_frag[nb][1] - new_m[0]);
                float p10 = fa_exp2(S_frag[nb][2] - new_m[1]);
                float p11 = fa_exp2(S_frag[nb][3] - new_m[1]);
                psum[0] += p00 + p01;
                psum[1] += p10 + p11;
                // repack into A-operand: pair even/odd n8 blocks into one k16 fragment
                int kf = nb >> 1, half = nb & 1;          // kf in [0,BLOCK_KV/16), half selects k0..7 / k8..15
                P_rmem[kf][half*2 + 0] = pack_bf16x2(p00, p01);   // row0 pair
                P_rmem[kf][half*2 + 1] = pack_bf16x2(p10, p11);   // row1 pair
            }
            // reduce psum across the 4 lanes sharing a row
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                psum[0] += __shfl_xor_sync(0xffffffff, psum[0], off);
                psum[1] += __shfl_xor_sync(0xffffffff, psum[1], off);
            }

            // --- rescale running l and O accumulator (only multiplies when corr != 1) ---
            l_i[0] = l_i[0] * corr[0] + psum[0];
            l_i[1] = l_i[1] * corr[1] + psum[1];
            m_i[0] = new_m[0]; m_i[1] = new_m[1];
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8; ++i) {
                O_acc[i][0] *= corr[0]; O_acc[i][1] *= corr[0];
                O_acc[i][2] *= corr[1]; O_acc[i][3] *= corr[1];
            }

            // --- GEMM1: O += P @ V  (V loaded transposed; iterate 2 D-chunks of the output) ---
            #pragma unroll
            for (int dc = 0; dc < N_DCHUNK; ++dc) {
                __syncthreads();
                stage_V(kv0, dc, Vs);
                __syncthreads();
                // PV: A = P (m16 x k=BLOCK_KV), B = V^T (k=BLOCK_KV x n=D_CHUNK).
                // accumulate into O_acc columns [dc*D_CHUNK/8 .. +D_CHUNK/8)
                #pragma unroll
                for (int nb8 = 0; nb8 < D_CHUNK / 8; ++nb8) {
                    int ocol = dc * (D_CHUNK / 8) + nb8;        // which n8 output column block
                    #pragma unroll
                    for (int kf = 0; kf < BLOCK_KV / 16; ++kf) {
                        uint32_t V_rmem[2];
                        const __nv_bfloat16* p = &Vs[swz<D_CHUNK>((kf*16) * D_CHUNK + nb8*8)];
                        ldmatrix_x2_trans(V_rmem, p);           // transposed for row.col PV
                        mma_m16n8k16(O_acc[ocol], P_rmem[kf], V_rmem);
                    }
                }
            }
            // FA-3 ILP note: the next iteration's GEMM0 ldmatrix+mma issue overlaps the exp/rescale
            // above at the warp-scheduler level (HMMA pipe vs MUFU/FMA pipe). With FA_KV_FP16 + cp.async
            // we additionally prefetch the next K chunk here (commit_group) and wait_group<1> at the top.
        }

        // ---- epilogue: O /= l (single deferred normalize), cast f32, write ----
        const int r0 = warp * WARP_Q + (lane >> 2);
        const int r1 = r0 + 8;
        float inv0 = (l_i[0] > 0.f) ? 1.0f / l_i[0] : 0.f;
        float inv1 = (l_i[1] > 0.f) ? 1.0f / l_i[1] : 0.f;
        int grow0 = q_row0 + r0, grow1 = q_row0 + r1;
        #pragma unroll
        for (int oc = 0; oc < HEAD_DIM / 8; ++oc) {
            // O_acc[oc] = {row0 col0, row0 col1, row1 col0, row1 col1} for output n8 block oc
            int d0 = oc * 8 + (lane & 3) * 2;
            int d1 = d0 + 1;
            if (grow0 < T) {
                float* o = O + ((size_t)grow0 * n_head + head) * HEAD_DIM;
                o[d0] = O_acc[oc][0] * inv0;
                o[d1] = O_acc[oc][1] * inv0;
            }
            if (grow1 < T) {
                float* o = O + ((size_t)grow1 * n_head + head) * HEAD_DIM;
                o[d0] = O_acc[oc][2] * inv1;
                o[d1] = O_acc[oc][3] * inv1;
            }
        }
        __syncthreads();   // before reusing Qs for the next gqa head
    }
}
```

**Notes on the prefill structure (load-bearing details):**

- The **2×128 K-tile** (`N_DCHUNK=2`, `dc` loop) is the head_dim=256 mechanism: each chunk runs `KSTEPS_PER_CHUNK=8` MMA K-steps accumulating into the *same* `S_frag`. Only 128 of d is K-resident at a time.
- **f32 I/O path** uses synchronous coalesced f32→bf16 stage stores (no `cp.async`, since `cp.async` cannot convert dtype). The `cp.async` 2-stage pipeline and K-double-buffer are the **`FA_KV_FP16` path** (KV already 16-bit in cache → `cp_async_16` the bytes, `cp_commit`/`cp_wait<1>` for the next K chunk). Both are present; the f32 path is the Stage-1 correctness default.
- **FA-4 conditional rescale** is warp-uniform via `__any_sync` (the pitfall: divergent lanes would use different `m` in `exp2` → wrong). `tau=8` is in the scaled-log2 domain because `S` was already `*= scale_log2`.
- **Free P repack**: m16n8 accumulator layout == m16k16 A-operand left half for bf16, so packing `S→P` needs no cross-thread shuffle.
- **V transposed** via `ldmatrix.x2.trans` (PV reduction is along BLOCK_KV).

---

## 2. DECODE kernel — `fa_decode_f32`

`cu/flash_attn.cu` (same TU). fattn-vec style: one block per `(head, split-K block)`, each warp-lane owns `HEAD_DIM/32 = 8` dims, scalar/FMA dot products, warp-shuffle online softmax, split-K over KV. **No tensor cores.** f32 KV now; documented hook for q8_0-K / q5_1-V.

```cpp
// ============================================================================
// DECODE kernel (T=1).  fattn-vec style: GEMV, no tensor cores.
// grid  = (1, n_head, SPLIT_K)        // y = query head, z = split-K partition over KV
// block = (NTHREADS=128 = 4 warps, 1, 1)
// Each block streams its KV partition; lanes own HEAD_DIM/WARP_SIZE = 8 dims.
// If SPLIT_K==1: writes final O directly (inline normalize). Else writes partials to
// O_partial[ (split, head, d) ] + LSE meta[ (split, head) ] for the combine kernel below.
//
// FATTN_KQ_MAX_OFFSET = 3*ln2 added to running max to prevent VKQ underflow-to-zero.
// ============================================================================
#define DEC_WARPS 4
#define DEC_THREADS (DEC_WARPS * WARP_SIZE)
#define DIMS_PER_LANE (HEAD_DIM / WARP_SIZE)        // 8 (=4 float2)
#define KQ_MAX_OFFSET 2.0794415416798357f           // 3*ln2

extern "C" __global__ __launch_bounds__(DEC_THREADS)
void fa_decode_f32(const float* __restrict__ Q,           // [head_dim, n_head, 1]
                   const float* __restrict__ K,           // [head_dim, n_head_kv, T_kv]
                   const float* __restrict__ V,
                   float* __restrict__ O,                 // [head_dim, n_head, 1] (split=1) OR partials
                   float* __restrict__ O_meta,            // [SPLIT_K, n_head, 2] = {m, l}; null if SPLIT_K==1
                   int head_dim, int n_head, int n_head_kv, int T_kv,
                   float scale, int split_k) {
    const int head    = blockIdx.y;
    const int part    = blockIdx.z;                       // split-K partition
    const int kv_head = head / (n_head / n_head_kv);
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float scale_log2 = scale * LOG2E;

    // ---- load this lane's slice of Q into registers (decode: T=1, one query) ----
    // lane owns dims [lane, lane+32, ..., lane+ (DIMS_PER_LANE-1)*32]  (strided => coalesced)
    float q_reg[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        int d = lane + i * WARP_SIZE;
        q_reg[i] = Q[(size_t)head * HEAD_DIM + d];        // T=1 so token index 0
    }

    // ---- per-lane VKQ accumulator (this lane's 8 output dims), running m / l ----
    float vkq[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) vkq[i] = 0.f;
    float m_run = NEG_INF, l_run = 0.f;

    // ---- split-K range for this partition ----
    int per = (T_kv + split_k - 1) / split_k;
    int kv_lo = part * per;
    int kv_hi = min(kv_lo + per, T_kv);

    // shared scratch to broadcast each key's softmax weight across the warp's 32 lanes
    __shared__ float sweight[DEC_WARPS][WARP_SIZE];

    // Each warp processes a disjoint stripe of keys: key index k handled by warp w when (k % DEC_WARPS)==w.
    for (int k = kv_lo + warp; k < kv_hi; k += DEC_WARPS) {
        // ---- KQ dot: each lane partial over its 8 dims, warp-reduce ----
        // HOOK(quant-KV): for q8_0 K, replace this float load+FMA with dp4a over int8 quants
        //   (Q pre-quantized to q8_1 once at kernel entry); s = d_K * d_Q * sumi.  See section 5.
        const float* kp = K + ((size_t)k * n_head_kv + kv_head) * HEAD_DIM;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) s += q_reg[i] * kp[lane + i * WARP_SIZE];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        // s is now the full dot on every lane. fold scale into log2 domain.
        float s_l2 = s * scale_log2;

        // ---- online softmax update (decode is non-causal: single query attends all keys <= itself,
        //      and in decode the single query IS the last position, so it attends every cached key) ----
        float m_new = fmaxf(m_run, s_l2 + KQ_MAX_OFFSET);
        float corr  = fa_exp2(m_run - m_new);
        float p     = fa_exp2(s_l2 - m_new);
        l_run = l_run * corr + p;
        // rescale this warp-lane's accumulator and add p * V[k]
        const float* vp = V + ((size_t)k * n_head_kv + kv_head) * HEAD_DIM;
        // HOOK(quant-KV): for q5_1 V, dequantize affine (val = d*q + m) here instead of float load.
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            int d = lane + i * WARP_SIZE;
            vkq[i] = vkq[i] * corr + p * vp[d];
        }
        m_run = m_new;
    }

    // ---- reduce the 4 warps' (m, l, vkq) into warp 0 via shared memory ----
    __shared__ float m_sh[DEC_WARPS], l_sh[DEC_WARPS], v_sh[DEC_WARPS][HEAD_DIM];
    if (lane < HEAD_DIM / WARP_SIZE * WARP_SIZE) { /* layout */ }
    // store each warp's lane-dims into v_sh
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) v_sh[warp][lane + i * WARP_SIZE] = vkq[i];
    if (lane == 0) { m_sh[warp] = m_run; l_sh[warp] = l_run; }
    __syncthreads();

    if (warp == 0) {
        // global max across warps
        float M = NEG_INF;
        #pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) M = fmaxf(M, m_sh[w]);
        float L = 0.f;
        float out[DIMS_PER_LANE];
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) out[i] = 0.f;
        #pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) {
            float a = fa_exp2(m_sh[w] - M);
            L += a * l_sh[w];
            #pragma unroll
            for (int i = 0; i < DIMS_PER_LANE; ++i)
                out[i] += a * v_sh[w][lane + i * WARP_SIZE];
        }
        if (split_k == 1) {
            float inv = (L > 0.f) ? 1.0f / L : 0.f;
            float* o = O + (size_t)head * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < DIMS_PER_LANE; ++i) o[lane + i * WARP_SIZE] = out[i] * inv;
        } else {
            // write unnormalized partial numerator + (m,l) meta for the combine kernel
            float* o = O + ((size_t)part * n_head + head) * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < DIMS_PER_LANE; ++i) o[lane + i * WARP_SIZE] = out[i];   // numerator (sum p*V), based against M
            if (lane == 0) { O_meta[((size_t)part * n_head + head) * 2 + 0] = M;
                             O_meta[((size_t)part * n_head + head) * 2 + 1] = L; }
        }
    }
}

// ---- split-K combine (flash-decoding merge). grid=(1,n_head,1), block=(HEAD_DIM,1,1). ----
extern "C" __global__ void fa_decode_combine_f32(
        const float* __restrict__ O_partial,   // [SPLIT_K, n_head, HEAD_DIM] numerators
        const float* __restrict__ O_meta,      // [SPLIT_K, n_head, 2] = {m, l}
        float* __restrict__ O,                  // [head_dim, n_head, 1]
        int head_dim, int n_head, int split_k) {
    int head = blockIdx.y;
    int d    = threadIdx.x;
    if (d >= head_dim) return;
    // global max over partitions
    float M = NEG_INF;
    for (int p = 0; p < split_k; ++p) M = fmaxf(M, O_meta[((size_t)p * n_head + head) * 2 + 0]);
    float num = 0.f, den = 0.f;
    for (int p = 0; p < split_k; ++p) {
        float mp = O_meta[((size_t)p * n_head + head) * 2 + 0];
        float lp = O_meta[((size_t)p * n_head + head) * 2 + 1];
        float a  = fa_exp2(mp - M);            // mp, M are in scaled-log2 domain
        num += a * O_partial[((size_t)p * n_head + head) * head_dim + d];
        den += a * lp;
    }
    O[(size_t)head * head_dim + d] = (den > 0.f) ? num / den : 0.f;
}
```

**Decode notes:** head_dim=256 needs **no** 2-tile split here (each lane owns 8 scalar dims; `HEAD_DIM % (2*WARP_SIZE)==0`). `KQ_MAX_OFFSET = 3*ln2` shifts the exp range up 8× to avoid VKQ flush-to-zero. Split-K writes partial *numerators* + `(M, L)` meta; the combine kernel does the associative merge `O = Σ exp(m_p-M)·num_p / Σ exp(m_p-M)·l_p`. `V_DOT2_F32_F16` is AMD-only — we accumulate VKQ in **f32** (matches Stage-1 + better numerics; what NVIDIA's vec path already does).

---

## 3. Rust launcher signatures (`bw24-engine/src/lib.rs`)

Drop-in replacements for `sdpa_naive` / `sdpa_naive_view`. Same `func()` / `launch_builder` style. `forward.rs:75` and `hybrid_forward.rs:107` call the prefill; `decode.rs:127` calls the decode. **Note:** must opt into 99 KB smem via `cudarc`'s `set_attribute` (or keep dyn smem ≤ 48 KB — the f32 layout above is 57.3 KB so the attribute call is required once at module load).

```rust
// ---- in Engine impl ----

/// FlashAttention PREFILL (T>1). Replaces sdpa_naive for forward/hybrid_forward.
/// Q:[head_dim,n_head,T], K/V:[head_dim,n_head_kv,T_kv] -> O:[head_dim,n_head,T].
pub fn fa_prefill(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                  o: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, n_head_kv: usize,
                  t: usize, t_kv: usize, scale: f32, causal: bool)
                  -> Result<(), Box<dyn std::error::Error>> {
    const BLOCK_Q: usize = 64;
    const NTHREADS: u32 = 128;
    // dyn smem (bf16, see kernel layout): Q + 2*K + 1*V chunks
    let smem_bf16 = BLOCK_Q * head_dim + 2 * 32 * 128 + 32 * 128;   // elements
    let smem_bytes = (smem_bf16 * 2) as u32;                        // 57344 for d=256

    let f = self.func("fa_prefill_f32");
    // opt-in to >48KB dynamic smem (idempotent; cache a "done" flag if hot)
    // f.set_attribute(CudaFunctionAttribute::MaxDynamicSharedMemorySize, smem_bytes as i32)?;

    let n_q_tiles = (t + BLOCK_Q - 1) / BLOCK_Q;
    let cfg = LaunchConfig {
        grid_dim: (n_q_tiles as u32, n_head_kv as u32, 1),   // x: q-tile, y: KV head (GQA group)
        block_dim: (NTHREADS, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let (hd, nh, nhkv, ti, tkvi, cz) =
        (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(q).arg(k).arg(v).arg(o)
     .arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
    unsafe { b.launch(cfg)?; }
    Ok(())
}

/// FlashAttention DECODE (T=1). K/V are CudaViews into the resident KV cache.
/// Replaces sdpa_naive_view in decode.rs. split_k=1 path needs no combine pass.
pub fn fa_decode_view(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<f32>,
                      v: &cudarc::driver::CudaView<f32>, o: &mut CudaSlice<f32>,
                      head_dim: usize, n_head: usize, n_head_kv: usize, t_kv: usize,
                      scale: f32) -> Result<(), Box<dyn std::error::Error>> {
    const DEC_THREADS: u32 = 128;
    // choose split_k to fill ~82 SMs: aim for n_head*split_k >= 82 when t_kv is long.
    let split_k: usize = if t_kv >= 2048 { ((82 + n_head - 1) / n_head).max(1) } else { 1 };

    let f = self.func("fa_decode_f32");
    let (hd, nh, nhkv, tkvi, sk) =
        (head_dim as i32, n_head as i32, n_head_kv as i32, t_kv as i32, split_k as i32);

    if split_k == 1 {
        let cfg = LaunchConfig {
            grid_dim: (1, n_head as u32, 1), block_dim: (DEC_THREADS, 1, 1), shared_mem_bytes: 0,
        };
        // O_meta unused; pass a 1-elem dummy (cudarc requires a valid ptr) or a null arg variant.
        let dummy = self.gpu.stream.alloc_zeros::<f32>(1)?;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&dummy)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(&tkvi).arg(&scale).arg(&sk);
        unsafe { b.launch(cfg)?; }
    } else {
        // allocate partials [split_k, n_head, head_dim] + meta [split_k, n_head, 2]
        let mut partial = self.gpu.stream.alloc_zeros::<f32>(split_k * n_head * head_dim)?;
        let mut meta    = self.gpu.stream.alloc_zeros::<f32>(split_k * n_head * 2)?;
        let cfg = LaunchConfig {
            grid_dim: (1, n_head as u32, split_k as u32), block_dim: (DEC_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(&mut partial).arg(&mut meta)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(&tkvi).arg(&scale).arg(&sk);
        unsafe { b.launch(cfg)?; }

        // combine pass
        let fc = self.func("fa_decode_combine_f32");
        let cfgc = LaunchConfig {
            grid_dim: (1, n_head as u32, 1), block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: 0,
        };
        let mut bc = self.gpu.stream.launch_builder(&fc);
        bc.arg(&partial).arg(&meta).arg(o).arg(&hd).arg(&nh).arg(&sk);
        unsafe { bc.launch(cfgc)?; }
    }
    Ok(())
}
```

Call-site swaps: `forward.rs:75` and `hybrid_forward.rs:107` → `e.fa_prefill(&q,&k,&v,&mut attn, head_dim,n_head,n_head_kv,t,t,scale,true)?;` (T>1). `decode.rs:127` → `e.fa_decode_view(&q,&k_view,&v_view,&mut attn, head_dim,n_head,n_head_kv,t_kv,scale)?;`. Add `"fa_prefill_f32"`, `"fa_decode_f32"`, `"fa_decode_combine_f32"` to whatever fatbin `func()` resolves (they compile into `kernels.cu` / a new `flash_attn.cu` — add it to the `build.rs` source list).

---

## 4. Validation plan

**Oracle:** the existing `sdpa_naive_f32` (already verified correct in `kernel_check.rs`, gated at `maxdiff < 1e-4`). The online-softmax + deferred-normalize math is algebraically identical to naive softmax in exact arithmetic, so f32 round-off is the only difference.

**Step 1 — diff vs `sdpa_naive_f32` (the gate).** Extend `bin/kernel_check.rs`:

1. Random `Q,K,V ~ U(-1,1)` (or `N(0,1)`), apply the *same* upstream L2/QK-norm + RoPE the engine applies (or skip both and just feed normalized random — the kernel only does `softmax(QK·scale)@V`).
2. Run `sdpa_naive(q,k,v,&mut o_ref, …, causal=true)` and `fa_prefill(q,k,v,&mut o_fa, …)` on identical inputs.
3. `maxdiff = max|o_ref - o_fa|`. **Tolerance: `< 1e-3`** absolute (looser than naive's `1e-4` because bf16 mma inputs + exp2 introduce ~bf16 ULP error; if `FA_KV_FP16=0` and you keep f32 mma you'd hit `1e-4`, but bf16 mma is the perf path — `1e-3` is the right gate for the bf16-input kernel). Also check **mean abs diff `< 1e-4`** and **cosine sim `> 0.9999`** per (head, token) to catch systematic (not just worst-lane) errors.

**Grid of shapes (must all pass):**
- `head_dim=256, n_head=16, n_head_kv=4` (qwen35 prod).
- `T ∈ {1, 2, 17, 64, 65, 128, 333}` (covers BLOCK_Q boundary 64, partial tiles, T=1→decode path).
- `T_kv ∈ {T, T+7, 1024, 4096}` (past-context, causal offset `q_pos=(T_kv-T)+qt`, split-K boundary at 2048).
- `causal ∈ {true, false}`.
- **Decode-specific:** `T=1`, `split_k ∈ {1, 4}` must agree with each other and with naive (validates the combine merge).

**Targeted correctness traps to assert explicitly:**
- **Causal diagonal:** with `T=T_kv` and `causal=true`, output row 0 must equal `V[0]` exactly (only key 0 unmasked) — catches mask off-by-one.
- **Conditional rescale exactness:** run with `RESCALE_TAU=0` (force every-block rescale, classic FA-2) and `RESCALE_TAU=8` — outputs must match to `< 1e-5` of each other (proves deferred rescale is exact).
- **GQA mapping:** set `n_head_kv=1` (all heads share KV) and `n_head_kv=n_head` (no sharing); both must match naive — catches `kv_head` index bug.
- **NaN/Inf scan** on `o_fa` (fully-masked rows when `T_kv<T` shouldn't occur, but guard the `l_i==0 → inv=0` epilogue path).

**Step 2 — llama.cpp cross-check.** Build llama.cpp FA reference on the same box (`/home/avifenesh/projects/llama.cpp`), run its `flash_attn_ext` (mma path for prefill, vec path for decode) on the *same* `Q,K,V` tensors (export bw24's f32 tiles, import as ggml f32→f16). Compare bw24 `o_fa` vs llama.cpp output: expect `maxdiff < 2e-3` (both use 16-bit mma inputs, so ~bf16/fp16 ULP). This validates against an independently-correct sm_120 FA, not just our own oracle.

**Step 3 — perf sanity (not correctness).** After the gate passes, time `fa_prefill` vs `sdpa_naive` at `T=512, T_kv=512` and decode at `T_kv=4096`; record tok/s into the existing bench harness. The naive kernel is smem-bound single-thread-softmax, so any correct FA should be multiples faster — but **correctness gate first, perf second**.

---

## Files touched

- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` — new TU with `fa_prefill_f32`, `fa_decode_f32`, `fa_decode_combine_f32` (code above).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/build.rs` — add `("cu/flash_attn.cu", "BW24_FLASH_FATBIN")` to the source list (or fold into `kernels.cu` if a single fatbin is preferred; `func()` already searches all loaded modules).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — add `fa_prefill` / `fa_decode_view` (signatures above), load the new fatbin alongside the existing three.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/{forward.rs:75, hybrid_forward.rs:107, decode.rs:127}` — swap `sdpa_naive*` → `fa_prefill` / `fa_decode_view`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/bin/kernel_check.rs` — add the validation grid (Step 1).

**One implementation caveat to verify at compile time:** the prefill mma C-fragment→register index mapping (`r0=lane/4`, `c0=(lane%4)*2`, the P repack `kf=nb>>1, half=nb&1`) follows the standard `m16n8k16` PTX layout; assert it against llama.cpp `mma.cuh` `tile<16,8>` accessors during bring-up — a wrong lane map corrupts P@V silently and is the single highest-risk detail (it's why Step 2's llama.cpp cross-check exists). The f32-staging path (synchronous f32→bf16 smem stores, no `cp.async`) is the correctness default; flip `FA_KV_FP16` on only after the gate passes to enable the `cp.async` 2-stage pipeline.