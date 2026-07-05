// kv_align_probe.cu — ARC A candidate 2 microbench: 32B-ALIGNED SPLIT-PLANE K layout
// for the quantized KV cache, measured against the shipped interleaved layout BEFORE
// any engine wiring (measure-the-win-first law).
//
// THEORY UNDER TEST: K q8_0 blocks are 34B (2B half scale + 32B qs) INTERLEAVED, so
// every 32B qs read straddles a 32B transaction boundary (~2 sectors per request
// instead of 1). A split-plane layout — qs plane [max_ctx][nblk][32B] (perfectly
// 32B-aligned) + scale plane [max_ctx][nblk][2B] — should cut K qs sector traffic
// ~2x at DRAM-bound depths. V q5_1 (24B/blk) stays interleaved here (K first; V only
// if K wins).
//
// KERNEL BODIES are verbatim copies of the engine's fa_decode_vec_q (register path)
// and fa_decode_vec_q_smem (deep-ctx path) from crates/bw24-engine/cu/flash_attn.cu,
// with ONLY the K addressing remapped in the *_al twins. Dequant math (half scale x
// int8 q, bf16 round-trip) is UNCHANGED — outputs must be BIT-IDENTICAL between
// layouts (this probe verifies partO/partM/partL bytewise before timing).
//
// GEOMETRY = 27B daily target: n_head=24, n_head_kv=4 (GQA 6), head_dim=256.
// Launch mirror of the engine: grid=(n_head_kv, n_splits), block=(32, gqa);
// split_keys ladder 32 (<=8k) / 64 (<=16k) / 128 (>16k), n_splits=ceil(t_kv/sp).
// NLAYERS independent KV buffers rotate per timing iteration so L2 behaves like the
// real 17-full-attn-layer walk (single-buffer looping would fake L2 residency).
//
// DECISION GATE (task spec): aligned must beat interleaved by >5% at 32k+ on the
// path actually dispatched there (smem twin; register also reported) or the probe
// records NEGATIVE and no engine wiring happens.
//
// Build: nvcc -O3 -arch=compute_120a -code=sm_120a probe/kv_align_probe.cu -o probe/kv_align_probe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CK(x) do{auto e=(x); if(e){printf("CUDA err %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);}}while(0)

#define WARP_SZ 32
#define NEG_INF (-1e30f)
#define LOG2E 1.4426950408889634f
#define FA_DEC_TILE 32
#define FA_DEC_MAX_DPL 8

// 27B geometry
#define HEAD_DIM 256
#define N_HEAD 24
#define N_HEAD_KV 4
#define GQA (N_HEAD/N_HEAD_KV)
#define KV_DIM (N_HEAD_KV*HEAD_DIM)          // 1024
#define NBLK (KV_DIM/32)                      // 32 blocks/token
#define K_TOK_BYTES ((long)NBLK*34)           // 1088
#define V_TOK_BYTES ((long)NBLK*24)           // 768

#define MAXCTX 65536
#define NLAYERS 8

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------ //
// synthetic KV init (interleaved layout, valid halves, deterministic) //
// grid-stride over tokens x blocks; one warp per block               //
// ------------------------------------------------------------------ //
__global__ void init_kv(uint8_t* K, uint8_t* V, int t_kv, int seed)
{
    const int lane = threadIdx.x & 31;
    const long warps = ((long)gridDim.x * blockDim.x) >> 5;
    const long wid0  = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long total = (long)t_kv * NBLK;
    for (long w = wid0; w < total; w += warps) {
        const long t = w / NBLK; const int b = (int)(w % NBLK);
        uint32_t h = (uint32_t)(t*2654435761u) ^ (uint32_t)(b*40503u) ^ (uint32_t)(seed*97u);
        // K q8_0
        uint8_t* kb = K + t*K_TOK_BYTES + (long)b*34;
        if (lane == 0) *(half*)kb = __float2half(0.002f + 0.00001f*(float)(h % 997));
        uint32_t hb = h ^ (lane*0x9e3779b9u); hb ^= hb >> 13; hb *= 0x85ebca6bu; hb ^= hb >> 16;
        ((int8_t*)(kb + 2))[lane] = (int8_t)((int)(hb % 255) - 127);
        // V q5_1
        uint8_t* vb = V + t*V_TOK_BYTES + (long)b*24;
        if (lane == 0) {
            *(half*)vb       = __float2half(0.003f + 0.00001f*(float)((h>>3) % 883));
            *(half*)(vb + 2) = __float2half(-0.05f + 0.0001f*(float)((h>>5) % 331));
            *(uint32_t*)(vb + 4) = h * 0x27d4eb2fu;
        }
        if (lane < 16) vb[8 + lane] = (uint8_t)((hb >> 3) & 0xFF);
    }
}

// repack interleaved K -> split-plane (qs plane at 0, scale plane at MAXCTX*NBLK*32)
__global__ void repack_k_split(const uint8_t* __restrict__ K, uint8_t* __restrict__ Ka, int t_kv)
{
    const int lane = threadIdx.x & 31;
    const long warps = ((long)gridDim.x * blockDim.x) >> 5;
    const long wid0  = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long total = (long)t_kv * NBLK;
    uint8_t* Kq = Ka;                                      // qs plane
    half*    Ks = (half*)(Ka + (long)MAXCTX*NBLK*32);      // scale plane
    for (long w = wid0; w < total; w += warps) {
        const long t = w / NBLK; const int b = (int)(w % NBLK);
        const uint8_t* kb = K + t*K_TOK_BYTES + (long)b*34;
        Kq[(t*NBLK + b)*32 + lane] = kb[2 + lane];
        if (lane == 0) Ks[t*NBLK + b] = *(const half*)kb;
    }
}

// ------------------------------------------------------------------ //
//  REGISTER path: interleaved (verbatim engine fa_decode_vec_q)       //
// ------------------------------------------------------------------ //
__global__ void fa_decode_vec_q_ref(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits, long k_tok_bytes, long v_tok_bytes)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * 34;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * 34;
                    const float d = __half2float(*(const half*)blk);
                    const int8_t q = ((const int8_t*)(blk + 2))[lane];
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(d * (float)q));
                }
            }
            float score = warp_reduce_sum(part);
            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * 24;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * 24;
                    const float d = __half2float(*(const half*)blk);
                    const float m = __half2float(*(const half*)(blk + 2));
                    const uint32_t qh = *(const uint32_t*)(blk + 4);
                    const uint8_t* qs = blk + 8;
                    const int lo = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
                    const int q5 = lo | (int)(((qh >> lane) & 1u) << 4);
                    acc[i] = acc[i] * alpha + p * __bfloat162float(__float2bfloat16(d * (float)q5 + m));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i]; }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ------------------------------------------------------------------ //
//  REGISTER path: ALIGNED split-plane K twin (only K addressing new)  //
// ------------------------------------------------------------------ //
__global__ void fa_decode_vec_q_al(
        const float* __restrict__ Q, const uint8_t* __restrict__ Ka, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits, long v_tok_bytes)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    const uint8_t* Kq = Ka;                                    // qs plane (32B/blk, aligned)
    const half*    Ks = (const half*)(Ka + (long)MAXCTX*NBLK*32);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const long tb = (long)t * NBLK + kblk0;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const float d = __half2float(Ks[tb + i]);
                    const int8_t q = ((const int8_t*)Kq)[(tb + i) * 32 + lane];
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(d * (float)q));
                }
            }
            float score = warp_reduce_sum(part);
            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * 24;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * 24;
                    const float d = __half2float(*(const half*)blk);
                    const float m = __half2float(*(const half*)(blk + 2));
                    const uint32_t qh = *(const uint32_t*)(blk + 4);
                    const uint8_t* qs = blk + 8;
                    const int lo = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
                    const int q5 = lo | (int)(((qh >> lane) & 1u) << 4);
                    acc[i] = acc[i] * alpha + p * __bfloat162float(__float2bfloat16(d * (float)q5 + m));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i]; }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ------------------------------------------------------------------ //
//  SMEM path: interleaved (verbatim engine fa_decode_vec_q_smem)      //
// ------------------------------------------------------------------ //
__global__ void fa_decode_vec_q_smem_ref(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits, long k_tok_bytes, long v_tok_bytes)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ __nv_bfloat16 ssh_vec[];
    __nv_bfloat16* sK = ssh_vec;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        for (int idx = bt; idx < nt * head_dim; idx += bsz) {
            int j = idx / head_dim;
            int d = idx - j * head_dim;
            const uint8_t* kb = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + (d >> 5)) * 34;
            const float kd = __half2float(*(const half*)kb);
            const int8_t kq = ((const int8_t*)(kb + 2))[d & 31];
            sK[idx] = __float2bfloat16(kd * (float)kq);
            const uint8_t* vb = V + (size_t)(t0 + j) * v_tok_bytes + (size_t)(kblk0 + (d >> 5)) * 24;
            const float vd = __half2float(*(const half*)vb);
            const float vm = __half2float(*(const half*)(vb + 2));
            const uint32_t qh = *(const uint32_t*)(vb + 4);
            const uint8_t* qs = vb + 8;
            const int dl = d & 31;
            const int lo = (dl < 16) ? (qs[dl] & 0x0F) : (qs[dl - 16] >> 4);
            const int q5 = lo | (int)(((qh >> dl) & 1u) << 4);
            sV[idx] = __float2bfloat16(vd * (float)q5 + vm);
        }
        __syncthreads();
        for (int j = 0; j < nt; ++j) {
            const __nv_bfloat16* kj = sK + (size_t)j * head_dim;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) part += q_reg[i] * __bfloat162float(kj[lane + (i << 5)]);
            float score = warp_reduce_sum(part);
            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const __nv_bfloat16* vj = sV + (size_t)j * head_dim;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) acc[i] = acc[i] * alpha + p * __bfloat162float(vj[lane + (i << 5)]);
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i]; }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ------------------------------------------------------------------ //
//  SMEM path: ALIGNED split-plane K twin (only the K staging remapped)//
// ------------------------------------------------------------------ //
__global__ void fa_decode_vec_q_smem_al(
        const float* __restrict__ Q, const uint8_t* __restrict__ Ka, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits, long v_tok_bytes)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ __nv_bfloat16 ssh_vec[];
    __nv_bfloat16* sK = ssh_vec;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;

    const uint8_t* Kq = Ka;
    const half*    Ks = (const half*)(Ka + (long)MAXCTX*NBLK*32);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        for (int idx = bt; idx < nt * head_dim; idx += bsz) {
            int j = idx / head_dim;
            int d = idx - j * head_dim;
            const long tb = (long)(t0 + j) * NBLK + kblk0 + (d >> 5);
            const float kd = __half2float(Ks[tb]);
            const int8_t kq = ((const int8_t*)Kq)[tb * 32 + (d & 31)];
            sK[idx] = __float2bfloat16(kd * (float)kq);
            const uint8_t* vb = V + (size_t)(t0 + j) * v_tok_bytes + (size_t)(kblk0 + (d >> 5)) * 24;
            const float vd = __half2float(*(const half*)vb);
            const float vm = __half2float(*(const half*)(vb + 2));
            const uint32_t qh = *(const uint32_t*)(vb + 4);
            const uint8_t* qs = vb + 8;
            const int dl = d & 31;
            const int lo = (dl < 16) ? (qs[dl] & 0x0F) : (qs[dl - 16] >> 4);
            const int q5 = lo | (int)(((qh >> dl) & 1u) << 4);
            sV[idx] = __float2bfloat16(vd * (float)q5 + vm);
        }
        __syncthreads();
        for (int j = 0; j < nt; ++j) {
            const __nv_bfloat16* kj = sK + (size_t)j * head_dim;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) part += q_reg[i] * __bfloat162float(kj[lane + (i << 5)]);
            float score = warp_reduce_sum(part);
            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const __nv_bfloat16* vj = sV + (size_t)j * head_dim;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) acc[i] = acc[i] * alpha + p * __bfloat162float(vj[lane + (i << 5)]);
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) { int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i]; }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ------------------------------------------------------------------ //

static int split_keys(int t_kv) { return t_kv <= 8192 ? 32 : (t_kv <= 16384 ? 64 : 128); }

int main(int argc, char** argv)
{
    int iters = (argc > 1) ? atoi(argv[1]) : 60;
    printf("kv_align_probe: 27B geom nh=%d nhkv=%d hd=%d nblk=%d ktb=%ld vtb=%ld maxctx=%d layers=%d iters=%d\n",
           N_HEAD, N_HEAD_KV, HEAD_DIM, NBLK, K_TOK_BYTES, V_TOK_BYTES, MAXCTX, NLAYERS, iters);

    // per-layer buffers
    uint8_t *K[NLAYERS], *V[NLAYERS], *Ka[NLAYERS];
    const long ksz = (long)MAXCTX * K_TOK_BYTES;
    const long vsz = (long)MAXCTX * V_TOK_BYTES;
    const long kasz = (long)MAXCTX * NBLK * 32 + (long)MAXCTX * NBLK * 2;  // qs plane + scale plane
    for (int l = 0; l < NLAYERS; ++l) {
        CK(cudaMalloc(&K[l], ksz)); CK(cudaMalloc(&V[l], vsz)); CK(cudaMalloc(&Ka[l], kasz));
        init_kv<<<1024, 256>>>(K[l], V[l], MAXCTX, l + 1);
        repack_k_split<<<1024, 256>>>(K[l], Ka[l], MAXCTX);
    }
    CK(cudaDeviceSynchronize());

    // Q + partials (sized for worst case n_splits at 64k sp=128 -> 512)
    float *Q; CK(cudaMalloc(&Q, N_HEAD*HEAD_DIM*sizeof(float)));
    {
        float* hq = (float*)malloc(N_HEAD*HEAD_DIM*sizeof(float));
        for (int i = 0; i < N_HEAD*HEAD_DIM; ++i) hq[i] = 0.01f * (float)((i*2654435761u % 200) - 100) / 100.0f;
        CK(cudaMemcpy(Q, hq, N_HEAD*HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice)); free(hq);
    }
    const int max_splits = (MAXCTX + 127) / 128;
    float *pO_a, *pM_a, *pL_a, *pO_b, *pM_b, *pL_b;
    CK(cudaMalloc(&pO_a, (long)N_HEAD*max_splits*HEAD_DIM*sizeof(float)));
    CK(cudaMalloc(&pM_a, (long)N_HEAD*max_splits*sizeof(float)));
    CK(cudaMalloc(&pL_a, (long)N_HEAD*max_splits*sizeof(float)));
    CK(cudaMalloc(&pO_b, (long)N_HEAD*max_splits*HEAD_DIM*sizeof(float)));
    CK(cudaMalloc(&pM_b, (long)N_HEAD*max_splits*sizeof(float)));
    CK(cudaMalloc(&pL_b, (long)N_HEAD*max_splits*sizeof(float)));

    const float scale = 1.0f/16.0f;
    const int shmem = 2 * FA_DEC_TILE * HEAD_DIM * 2;   // sK+sV bf16
    cudaEvent_t ev_a, ev_b; CK(cudaEventCreate(&ev_a)); CK(cudaEventCreate(&ev_b));

    int tkvs[4] = {8192, 16384, 32768, 65536};
    for (int ci = 0; ci < 4; ++ci) {
        const int t_kv = tkvs[ci];
        const int sp = split_keys(t_kv);
        const int n_splits = (t_kv + sp - 1) / sp;
        dim3 grid(N_HEAD_KV, n_splits, 1), block(32, GQA, 1);
        const double uniq_mb = (double)t_kv * (K_TOK_BYTES + V_TOK_BYTES) / 1e6;

        // ---- exactness: aligned vs interleaved partials must be BIT-IDENTICAL ----
        long osz = (long)N_HEAD*n_splits*HEAD_DIM*sizeof(float), msz = (long)N_HEAD*n_splits*sizeof(float);
        // register pair
        fa_decode_vec_q_ref<<<grid, block>>>(Q, K[0], V[0], pO_a, pM_a, pL_a,
            HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES);
        fa_decode_vec_q_al<<<grid, block>>>(Q, Ka[0], V[0], pO_b, pM_b, pL_b,
            HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES);
        CK(cudaDeviceSynchronize());
        {
            void *ha = malloc(osz), *hb = malloc(osz);
            CK(cudaMemcpy(ha, pO_a, osz, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb, pO_b, osz, cudaMemcpyDeviceToHost));
            int bad = memcmp(ha, hb, osz);
            CK(cudaMemcpy(ha, pM_a, msz, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb, pM_b, msz, cudaMemcpyDeviceToHost));
            bad |= memcmp(ha, hb, msz);
            CK(cudaMemcpy(ha, pL_a, msz, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb, pL_b, msz, cudaMemcpyDeviceToHost));
            bad |= memcmp(ha, hb, msz);
            printf("t_kv=%5d reg  exactness: %s\n", t_kv, bad ? "MISMATCH <-- FAIL" : "bit-identical");
            if (bad) return 1;
            free(ha); free(hb);
        }
        // smem pair
        fa_decode_vec_q_smem_ref<<<grid, block, shmem>>>(Q, K[0], V[0], pO_a, pM_a, pL_a,
            HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES);
        fa_decode_vec_q_smem_al<<<grid, block, shmem>>>(Q, Ka[0], V[0], pO_b, pM_b, pL_b,
            HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES);
        CK(cudaDeviceSynchronize());
        {
            void *ha = malloc(osz), *hb = malloc(osz);
            CK(cudaMemcpy(ha, pO_a, osz, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb, pO_b, osz, cudaMemcpyDeviceToHost));
            int bad = memcmp(ha, hb, osz);
            printf("t_kv=%5d smem exactness: %s\n", t_kv, bad ? "MISMATCH <-- FAIL" : "bit-identical");
            if (bad) return 1;
            free(ha); free(hb);
        }

        // ---- timing: rotate layers so L2 behaves like the multi-layer walk ----
        struct { const char* name; int kind; } variants[4] = {
            {"reg-interleaved ", 0}, {"reg-aligned     ", 1},
            {"smem-interleaved", 2}, {"smem-aligned    ", 3} };
        float ms_res[4];
        for (int vi = 0; vi < 4; ++vi) {
            // warm
            for (int l = 0; l < NLAYERS; ++l) {
                switch (variants[vi].kind) {
                case 0: fa_decode_vec_q_ref<<<grid, block>>>(Q, K[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES); break;
                case 1: fa_decode_vec_q_al<<<grid, block>>>(Q, Ka[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES); break;
                case 2: fa_decode_vec_q_smem_ref<<<grid, block, shmem>>>(Q, K[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES); break;
                case 3: fa_decode_vec_q_smem_al<<<grid, block, shmem>>>(Q, Ka[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES); break;
                }
            }
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev_a));
            for (int it = 0; it < iters; ++it) {
                int l = it % NLAYERS;
                switch (variants[vi].kind) {
                case 0: fa_decode_vec_q_ref<<<grid, block>>>(Q, K[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES); break;
                case 1: fa_decode_vec_q_al<<<grid, block>>>(Q, Ka[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES); break;
                case 2: fa_decode_vec_q_smem_ref<<<grid, block, shmem>>>(Q, K[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, K_TOK_BYTES, V_TOK_BYTES); break;
                case 3: fa_decode_vec_q_smem_al<<<grid, block, shmem>>>(Q, Ka[l], V[l], pO_a, pM_a, pL_a,
                        HEAD_DIM, N_HEAD, N_HEAD_KV, t_kv, scale, n_splits, V_TOK_BYTES); break;
                }
            }
            CK(cudaEventRecord(ev_b));
            CK(cudaDeviceSynchronize());
            float ms; CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            ms_res[vi] = ms / iters;
            double gbps = uniq_mb / 1e3 / (ms_res[vi] / 1e3);
            printf("t_kv=%5d sp=%3d nsplit=%4d %s : %8.2f us  (%7.1f GB/s unique)\n",
                   t_kv, sp, n_splits, variants[vi].name, ms_res[vi]*1e3, gbps);
        }
        printf("t_kv=%5d  REG   aligned/interleaved = %.4f (%+.1f%%)\n",
               t_kv, ms_res[1]/ms_res[0], (ms_res[0]/ms_res[1]-1.0)*100.0);
        printf("t_kv=%5d  SMEM  aligned/interleaved = %.4f (%+.1f%%)\n\n",
               t_kv, ms_res[3]/ms_res[2], (ms_res[2]/ms_res[3]-1.0)*100.0);
    }
    return 0;
}
