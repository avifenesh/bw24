// flash_attn.cu — bw24 hand-written FlashAttention for RTX 5090 (sm_120a).
//
// Built ENTIRELY on the validated m16n8k16 bf16 mma primitives from /tmp/qkpv_test.cu
// (qk_test rel=6.33e-7, pv_test rel=8.10e-8 on the 5090; compute-sanitizer clean).
// Those two kernels ARE the inner GEMMs here, unchanged. This file wires them into
// the FA-2 online-softmax loop and adds GQA + causal + the decode split-K path.
//
// LAYOUT (matches sdpa_naive_f32 oracle, kernels.cu:99):
//   Q : [head_dim, n_head,    T   ]  head_dim fastest  -> element (qt,head,d) at ((qt*n_head+head)*head_dim + d)
//   K : [head_dim, n_head_kv, T_kv]  head_dim fastest  -> element (t, kvh, d)  at ((t *n_head_kv+kvh)*head_dim + d)
//   V : same shape as K
//   O : [head_dim, n_head,    T   ]  head_dim fastest (same as Q)
//   GQA   : kv_head = head / (n_head / n_head_kv)
//   causal: q_pos = (T_kv - T) + qt ; key t is masked when t > q_pos
//   head_dim = 256 (qwen35), scale = 1/16.
//
// WHY THIS IS CORRECT BY CONSTRUCTION (the 6 FA-v1 review bugs, all addressed):
//   C1 per-lane ldmatrix address      : the ported ld_A/ld_B/ld_A_trans bake the
//                                        per-LANE offset in (mma.cuh:834/790/891). VALIDATED.
//   C2 register pressure (>200/thread) : O accumulator (256 f32 / q-row) lives in
//                                        SHARED MEMORY (sO), NOT registers. The QK
//                                        score tile S (16x Bk) is the only big tile
//                                        and it is consumed immediately. Q is re-read
//                                        from smem via ldmatrix each KV tile (never
//                                        held in 64 regs). Footprint stays small.
//   C3 PV V-transpose                  : V is fed to PV's B operand via ld_A_trans
//                                        (the x4.trans loader + the {x0,x2}/{x1,x3}
//                                        register pairing). VALIDATED in pv_test.
//   C4 P->A repack is NOT free         : after softmax we WRITE P back to shared
//                                        memory (sP, bf16) and RE-LDMATRIX it for PV
//                                        via ld_A. This is the SMEM ROUND-TRIP the
//                                        review demands — no movmatrix games, the PV
//                                        operand layout is produced by ld_A reading
//                                        sP exactly as the validated pv_test does.
//   C5 K B-operand layout              : K is stored [key][d] head_dim-fastest which
//                                        is exactly ld_B's [n=key][k=d] source. VALIDATED.
//   C6 decode log2 offset              : exp2f used for fast-exp, exp(x)=exp2(x*LOG2E).
//                                        FA-v1's bug was adding a 2.079*ln2 constant in the
//                                        log2 domain. Here NO such bias is ever added: the
//                                        online-softmax recurrence (m_new = max, alpha =
//                                        exp2((m_prev-m_new)*LOG2E), p = exp2((s-m_new)*
//                                        LOG2E)) is exact and self-normalizing — any base
//                                        offset would cancel in the l_i ratio. If one ever
//                                        re-introduces a per-reduction-width bias it must be
//                                        log2(width) (e.g. log2(8)=3.0), NEVER 2.079.
//
// PERF NOTE: this is the CORRECTNESS-FIRST FA assembly (one warp / q-tile, smem O).
// It removes the O(T*T_kv) smem scores of sdpa_naive and uses tensor cores for both
// GEMMs. Throughput tuning (multi-warp, ping-pong cp.async, register O) is a follow-up;
// the primitives and the FA-2 recurrence here are the proven base to tune on.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define WARP_SZ 32
#define HEAD_DIM 256
#define M_ROWS  16     // query rows per warp tile
#define N_KEYS  8      // one mma N-step = 8 keys (QK) / 8 d-cols (PV)
#define K_STEP  16     // m16n8k16 contraction width (logical bf16)
#define BK      64     // KV tile width (keys processed per FA step); multiple of 16
#define NEG_INF (-1e30f)

// ===================================================================== //
//  PORTED + VALIDATED PRIMITIVES (verbatim from /tmp/qkpv_test.cu)       //
//  Lane maps are the mma.cuh non-AMD DATA_LAYOUT_I_MAJOR specializations.//
//  ALL `stride_pairs` args are in bf16-PAIR (u32) units = bf16_stride/2. //
// ===================================================================== //

// f32 accumulator C tile<16,8,float> (mma.cuh:245,262). ne=4 f32/lane.
struct CTile { float x[4];
    static __device__ __forceinline__ int get_i(int l){ return ((l/2)*8) + (threadIdx.x/4); }
    static __device__ __forceinline__ int get_j(int l){ return ((threadIdx.x%4)*2) + (l%2); }
};
// bf16 A operand tile<16,8,bf162> (mma.cuh:485,498). ne=4 u32/lane.
struct ATile { nv_bfloat162 x[4];
    static __device__ __forceinline__ int get_i(int l){ return ((l%2)*8) + (threadIdx.x/4); }
    static __device__ __forceinline__ int get_j(int l){ return ((l/2)*4) + (threadIdx.x%4); }
};
// bf16 B operand tile<8,8,bf162> (mma.cuh:481,493). ne=2 u32/lane.
struct BTile { nv_bfloat162 x[2];
    static __device__ __forceinline__ int get_i(int l){ return threadIdx.x/4; }
    static __device__ __forceinline__ int get_j(int l){ return (l*4) + (threadIdx.x%4); }
};

// load_ldmatrix tile<16,8> x4 (mma.cuh:829-837). addr = (tid%16)*stride + (tid/16)*4.
// FIX C1 (proven in mma_validate.cu): the address operand MUST be a 32-bit .shared
// address built via (uint32_t)__cvta_generic_to_shared(...) and passed with "r".
// Passing a 64-bit generic pointer via "l" yields a runtime "misaligned address".
static __device__ __forceinline__ void ld_A(ATile& t, const __nv_bfloat16* xs0, int stride_pairs){
    int* xi = (int*)t.x;
    const uint32_t* xs = (const uint32_t*)xs0 + (threadIdx.x % 16)*stride_pairs + (threadIdx.x / 16)*4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[1]),"=r"(xi[2]),"=r"(xi[3]) : "r"(addr));
}
// load_ldmatrix_trans tile<16,8> x4.trans (mma.cuh:884-894). OUTPUT reorder x0,x2,x1,x3.
// Same 32-bit .shared address as ld_A (FIX C1/C3, proven in mma_validate.cu pv_test).
static __device__ __forceinline__ void ld_A_trans(ATile& t, const __nv_bfloat16* xs0, int stride_pairs){
    int* xi = (int*)t.x;
    const uint32_t* xs = (const uint32_t*)xs0 + (threadIdx.x % 16)*stride_pairs + (threadIdx.x / 16)*4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[2]),"=r"(xi[1]),"=r"(xi[3]) : "r"(addr));
}
// mma m16n8k16 .f32.bf16.bf16.f32 (mma.cuh:1187). D[16x8] += A[16x16] @ B[8x16]^T.
static __device__ __forceinline__ void mma_bf16(CTile& D, const ATile& A, const BTile& B){
    const int* Ax=(const int*)A.x; const int* Bx=(const int*)B.x; float* Dx=D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(Dx[0]),"+f"(Dx[1]),"+f"(Dx[2]),"+f"(Dx[3])
        : "r"(Ax[0]),"r"(Ax[1]),"r"(Ax[2]),"r"(Ax[3]),"r"(Bx[0]),"r"(Bx[1]));
}

// log2(e) for the exp2 fast-exp (exp(x) = exp2(x*LOG2E)).
#define LOG2E 1.4426950408889634f

// ===================================================================== //
//  KV-CACHE QUANTIZATION  (q8_0 for K, q5_1 for V)                      //
//  Block layouts (ggml-common.h, verified byte-for-byte):              //
//    q8_0 : 34 B/32elem  = f16 d (2B) + int8 qs[32] (32B)              //
//           x[j] = f16_to_f32(d) * (float)qs[j]                         //
//    q5_1 : 24 B/32elem  = f16 d (2B) + f16 m (2B) + u32 qh (4B)        //
//                          + u8 qs[16] (16B)                            //
//           lo = (j<16)? (qs[j]&0xF) : (qs[j-16]>>4)                    //
//           hi = ((qh>>j)&1)<<4 ; q5 = lo|hi ; x[j] = d*q5 + m          //
//  Cache element-within-token index = kv_head*head_dim + d. block =     //
//  idx/32, lane = idx%32. head_dim%32==0 so a 32-block never straddles  //
//  heads. K/V token strides differ (k_tok_bytes vs v_tok_bytes).        //
// ===================================================================== //

// q8_0 dequant of one element. `K` is the cache base, `t` the token,
// `kv_dim` element-within-token index `eidx = kv_head*head_dim + d`.
static __device__ __forceinline__ float dq_q8_0_elem(
        const uint8_t* __restrict__ K, long t, long k_tok_bytes, int eidx)
{
    const uint8_t* blk = K + (size_t)t * k_tok_bytes + (size_t)(eidx >> 5) * 34;
    const half d = *(const half*)blk;
    const int8_t q = ((const int8_t*)(blk + 2))[eidx & 31];
    return __half2float(d) * (float)q;
}

// q5_1 dequant of one element (affine).
static __device__ __forceinline__ float dq_q5_1_elem(
        const uint8_t* __restrict__ V, long t, long v_tok_bytes, int eidx)
{
    const uint8_t* blk = V + (size_t)t * v_tok_bytes + (size_t)(eidx >> 5) * 24;
    const half d = *(const half*)blk;            // dm.x
    const half m = *(const half*)(blk + 2);      // dm.y
    const uint32_t qh = *(const uint32_t*)(blk + 4);
    const uint8_t* qs = blk + 8;
    const int j = eidx & 31;
    const int lo = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
    const int q5 = lo | (int)(((qh >> j) & 1u) << 4);
    return __half2float(d) * (float)q5 + __half2float(m);
}

// ---- warp reductions over a 32-lane block (one warp per 32-elem block) ----
static __device__ __forceinline__ float warp_amax(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
static __device__ __forceinline__ float warp_min(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
static __device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// full-warp sum (butterfly): every lane ends with the 32-lane sum (used by fa_decode_vec_q QK dot).
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Append-quantize one token's K (q8_0) and V (q5_1) into the resident cache.
//   grid  = (max(kv_dim_k, kv_dim_v)/32, 1, 1)  -- one CTA per 32-elem block
//   block = (32,1,1)                            -- one thread per element (one warp)
// Thread `lane` owns element `b*32+lane`. k_row/v_row are the post-RoPE f32
// K/V rows for the single new token (element order kv_head*head_dim + d).
extern "C" __global__ void append_quantize_kv_q8_0_q5_1(
        const float* __restrict__ k_row,   // [kv_dim_k]
        const float* __restrict__ v_row,   // [kv_dim_v]
        uint8_t* __restrict__ K,           // cache base (q8_0)
        uint8_t* __restrict__ V,           // cache base (q5_1)
        int t, int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes)
{
    const int b    = blockIdx.x;           // 32-elem block index within the token
    const int lane = threadIdx.x;          // 0..31
    const int eidx = b * 32 + lane;        // element index within token

    // ---- K block b -> q8_0 (symmetric) ----
    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_row[eidx] : 0.0f;
        float amax = warp_amax(x);
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int q = (int)lrintf(x * id);
        q = max(-127, min(127, q));
        uint8_t* blk = K + (size_t)t * k_tok_bytes + (size_t)b * 34;
        if (lane == 0) *(half*)blk = __float2half(d);
        ((int8_t*)(blk + 2))[lane] = (int8_t)q;
    }

    // ---- V block b -> q5_1 (affine) ----
    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_row[eidx] : 0.0f;
        float mn = warp_min(x);
        float mx = warp_max(x);
        float d = (mx - mn) / 31.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int q5 = (int)lrintf((x - mn) * id);
        q5 = max(0, min(31, q5));
        // qh bit j set iff element j has its 5th bit (bit 4) set. __ballot_sync
        // over all 32 lanes yields EXACTLY the little-endian qh u32 (bit j = lane j).
        uint32_t qh = __ballot_sync(0xffffffffu, (q5 >> 4) & 1);
        uint8_t* blk = V + (size_t)t * v_tok_bytes + (size_t)b * 24;
        if (lane == 0) {
            *(half*)blk        = __float2half(d);          // dm.x
            *(half*)(blk + 2)  = __float2half(mn);         // dm.y (min)
            *(uint32_t*)(blk + 4) = qh;                    // 5th bits
        }
        // qs nibble packing: lanes 0..15 own the LOW nibble of byte (lane),
        // lanes 16..31 own the HIGH nibble of byte (lane-16). Exchange the low
        // nibble of the partner lane (lane+16) via shuffle so each of bytes
        // 0..15 is written exactly once by lane in [0,16).
        uint8_t* qs = blk + 8;
        int nib = q5 & 0x0F;
        int partner_nib = __shfl_sync(0xffffffffu, nib, lane + 16) & 0x0F;  // lane+16's low nibble
        if (lane < 16) qs[lane] = (uint8_t)(nib | (partner_nib << 4));
    }
}

// ----- DEVICE-COUNTER variant (CUDA-GRAPH-PLAN Phase 2): identical math to
// append_quantize_kv_q8_0_q5_1, but the per-step WRITE OFFSET `t` is read from a
// device int[1] counter (t_dev[0]) instead of a host int arg. This is the only
// per-step varying scalar in KV-append; reading it from device makes the kernel's
// args FIXED across decode steps (the prerequisite for graph capture). The original
// (host-int) kernel stays for the non-graph eager path.
extern "C" __global__ void append_quantize_kv_q8_0_q5_1_dc(
        const float* __restrict__ k_row,   // [kv_dim_k]
        const float* __restrict__ v_row,   // [kv_dim_v]
        uint8_t* __restrict__ K,           // cache base (q8_0)
        uint8_t* __restrict__ V,           // cache base (q5_1)
        const int* __restrict__ t_dev,     // write slot (device counter, t_dev[0])
        int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes)
{
    const int t    = t_dev[0];             // <-- the ONLY change vs the host-int kernel
    const int b    = blockIdx.x;
    const int lane = threadIdx.x;
    const int eidx = b * 32 + lane;

    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_row[eidx] : 0.0f;
        float amax = warp_amax(x);
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int q = (int)lrintf(x * id);
        q = max(-127, min(127, q));
        uint8_t* blk = K + (size_t)t * k_tok_bytes + (size_t)b * 34;
        if (lane == 0) *(half*)blk = __float2half(d);
        ((int8_t*)(blk + 2))[lane] = (int8_t)q;
    }

    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_row[eidx] : 0.0f;
        float mn = warp_min(x);
        float mx = warp_max(x);
        float d = (mx - mn) / 31.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int q5 = (int)lrintf((x - mn) * id);
        q5 = max(0, min(31, q5));
        uint32_t qh = __ballot_sync(0xffffffffu, (q5 >> 4) & 1);
        uint8_t* blk = V + (size_t)t * v_tok_bytes + (size_t)b * 24;
        if (lane == 0) {
            *(half*)blk        = __float2half(d);
            *(half*)(blk + 2)  = __float2half(mn);
            *(uint32_t*)(blk + 4) = qh;
        }
        uint8_t* qs = blk + 8;
        int nib = q5 & 0x0F;
        int partner_nib = __shfl_sync(0xffffffffu, nib, lane + 16) & 0x0F;
        if (lane < 16) qs[lane] = (uint8_t)(nib | (partner_nib << 4));
    }
}

// ===================================================================== //
//  KERNEL 1 : fa_prefill_f32                                            //
//  One WARP per (head, q-tile of 16 rows). FA-2 online softmax.         //
//  grid = (ceil(T/16), n_head, 1) ; block = (32,1,1).                   //
//                                                                       //
//  Per-warp shared memory (bf16 unless noted):                         //
//    sQ : [16][HEAD_DIM]      query tile (staged once, re-ldmatrix'd)   //
//    sK : [BK][HEAD_DIM]      current KV key tile                       //
//    sV : [BK][HEAD_DIM]      current KV value tile                     //
//    sP : [16][BK]            softmax probs P (bf16) for the SMEM       //
//                             round-trip into PV's A operand (C4 fix)   //
//    sO : [16][HEAD_DIM] f32  output accumulator (in SMEM, NOT regs;C2) //
//    sS : [16][BK] f32        QK scores staged for the row softmax      //
//    sM : [16] f32            running max m_i per query row             //
//    sL : [16] f32            running sum  l_i per query row            //
// ===================================================================== //
extern "C" __global__ void fa_prefill_f32(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    const int q_tile = blockIdx.x;            // which 16-row block of queries
    const int head   = blockIdx.y;
    const int lane   = threadIdx.x;           // 0..31 (single warp)
    const int q_base = q_tile * M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int kv_head = head / (n_head / n_head_kv);
    const int nq = min(M_ROWS, T - q_base);   // valid query rows in this tile

    // ----- dynamic shared memory layout -----
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem_raw;                  // 16*HEAD_DIM
    __nv_bfloat16* sK = sQ + M_ROWS*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;                          // BK*HEAD_DIM
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;                          // 16*BK
    float* sO = (float*)(sP + M_ROWS*BK);                         // 16*HEAD_DIM f32
    float* sS = sO + M_ROWS*HEAD_DIM;                             // 16*BK f32
    float* sM = sS + M_ROWS*BK;                                   // 16 f32
    float* sL = sM + M_ROWS;                                       // 16 f32

    // ----- stage Q tile, init accumulators -----
    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        float qv = (r < nq) ? Q[((size_t)(q_base + r) * n_head + head) * head_dim + d] : 0.0f;
        sQ[i] = __float2bfloat16(qv);
    }
    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) sO[i] = 0.0f;
    for (int i = lane; i < M_ROWS; i += WARP_SZ) { sM[i] = NEG_INF; sL[i] = 0.0f; }
    __syncwarp();

    // absolute query position of row r (for causal mask): q_pos(r) = (T_kv - T) + q_base + r
    const int q_pos0 = (T_kv - T) + q_base;

    // ===== FA-2 loop over KV in tiles of BK keys =====
    for (int k0 = 0; k0 < T_kv; k0 += BK) {
        const int nk = min(BK, T_kv - k0);    // valid keys this tile

        // causal early-out: if every key in this tile is strictly past the max query
        // position in this q-tile, the whole tile is masked — skip it.
        if (causal && k0 > (q_pos0 + nq - 1)) break;

        // ---- stage K,V tiles to smem (pad invalid keys with 0) ----
        for (int i = lane; i < BK*HEAD_DIM; i += WARP_SZ) {
            int kk = i / HEAD_DIM, d = i % HEAD_DIM;
            float kv = (kk < nk) ? K[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
            float vv = (kk < nk) ? V[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
            sK[i] = __float2bfloat16(kv);
            sV[i] = __float2bfloat16(vv);
        }
        __syncwarp();

        // ---- GEMM0: S[16 q][BK key] = Q @ K^T  (tensor cores) ----
        // FIX C5 (proven in mma_validate.cu qk_test): K's B-operand is loaded with the
        // SAME 16x8 x4 ld_A loader (always 16B-aligned) over a 16-KEY block, then the 4
        // registers are split into two 8-key N-blocks: n0={B[0],B[2]}, n1={B[1],B[3]}
        // (mma.cuh:1204-1209). The fragile 8x8 x2 ld_B (8B-misaligned for 16-wide rows)
        // is NOT used. Each 16-key group accumulates over the 256/16=16 head_dim k-steps.
        for (int kg = 0; kg < BK; kg += 2*N_KEYS) {           // 16 keys per group
            CTile C0, C1;                                     // C0: keys kg+0..7 ; C1: kg+8..15
            C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
            C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < HEAD_DIM; kk += K_STEP) {
                ATile A, Kt;
                ld_A(A,  sQ + kk,                  HEAD_DIM/2);  // Q[16][kk..kk+16]
                ld_A(Kt, sK + kg*HEAD_DIM + kk,    HEAD_DIM/2);  // K[kg..kg+16][kk..kk+16]
                BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];   // keys kg+0..7
                BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];   // keys kg+8..15
                mma_bf16(C0, A, Blo);
                mma_bf16(C1, A, Bhi);
            }
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int m = CTile::get_i(l), c8 = CTile::get_j(l);
                sS[m*BK + kg + 0      + c8] = C0.x[l];
                sS[m*BK + kg + N_KEYS + c8] = C1.x[l];
            }
        }
        __syncwarp();

        // ---- row softmax update (FA-2 online): one query row per lane (16 rows <= 32) ----
        if (lane < M_ROWS) {
            int r = lane;
            float* srow = sS + r*BK;
            int q_pos = q_pos0 + r;
            // tile-local max over valid+unmasked keys
            float m_tile = NEG_INF;
            for (int j = 0; j < nk; ++j) {
                float s = srow[j] * scale;
                if (causal && (k0 + j) > q_pos) s = NEG_INF;
                srow[j] = s;
                m_tile = fmaxf(m_tile, s);
            }
            float m_prev = sM[r];
            float m_new  = fmaxf(m_prev, m_tile);
            // rescale factor for the previous accumulator: exp(m_prev - m_new)
            float alpha = (m_prev == NEG_INF) ? 0.0f : exp2f((m_prev - m_new) * LOG2E);
            // P_ij = exp(s_ij - m_new); accumulate l_i
            float l_tile = 0.0f;
            for (int j = 0; j < nk; ++j) {
                float p = (srow[j] == NEG_INF) ? 0.0f : exp2f((srow[j] - m_new) * LOG2E);
                sP[r*BK + j] = __float2bfloat16(p);
                l_tile += p;
            }
            // zero-pad the rest of the sP row (BK-nk) so PV sees clean zeros
            for (int j = nk; j < BK; ++j) sP[r*BK + j] = __float2bfloat16(0.0f);
            sL[r] = sL[r] * alpha + l_tile;
            sM[r] = m_new;
            // Broadcast alpha to the other lanes via smem so the O-rescale (which is
            // strided over all 32 lanes) can read it. The sS scores for this row are
            // already fully consumed into sP, so col 0 is free scratch.
            sS[r*BK + 0] = alpha;
        }
        __syncwarp();

        // ---- rescale existing O by alpha (per row), in SMEM (C2: O is not in regs) ----
        // sS[r*BK+0] holds alpha for row r (written above).
        for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
            int r = i / HEAD_DIM;
            if (r < nq) sO[i] *= sS[r*BK + 0];
        }
        __syncwarp();

        // ---- GEMM1: O += P @ V  (tensor cores; P re-ldmatrix'd from sP = C4 round-trip) ----
        // O[16 q][HEAD_DIM d] += P[16 q][BK key] @ V[BK key][HEAD_DIM d].
        // mma: D[m=q,n=d] += A[m=q,k=key] * B[n=d,k=key]; B = V^T via ld_A_trans.
        // d in steps of 16 (one trans-load = TWO N=8 blocks); keys in steps of 16.
        for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
            CTile Clo, Chi;
            Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
            Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < BK; kk += K_STEP) {
                ATile A; ATile Bt;
                ld_A(A, sP + kk, BK/2);                          // P[16][kk..kk+16 keys]
                ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2); // V^T [key][d]
                BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];   // d in [d0,    d0+8)
                BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];   // d in [d0+8,  d0+16)
                mma_bf16(Clo, A, Blo);
                mma_bf16(Chi, A, Bhi);
            }
            // accumulate the PV contribution INTO sO (already rescaled by alpha above)
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int m   = CTile::get_i(l);
                int nlo = d0 +          CTile::get_j(l);
                int nhi = d0 + N_KEYS + CTile::get_j(l);
                sO[m*HEAD_DIM + nlo] += Clo.x[l];
                sO[m*HEAD_DIM + nhi] += Chi.x[l];
            }
        }
        __syncwarp();
    }

    // ===== deferred final normalize: O = sO / l_i ; write to global =====
    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        if (r < nq) {
            float linv = (sL[r] > 0.0f) ? (1.0f / sL[r]) : 0.0f;
            O[((size_t)(q_base + r) * n_head + head) * head_dim + d] = sO[i] * linv;
        }
    }
}

// ===================================================================== //
//  KERNEL 1b : fa_prefill_q  (quantized-cache prefill: q8_0 K / q5_1 V) //
//  Identical to fa_prefill_f32 EXCEPT the stage-to-smem copy dequants    //
//  the resident quantized KV cache. MMA / softmax / PV are byte-identical //
//  to the f32 kernel. Used by the MTP verify path (fa_prefill_view).     //
//  K/V token strides differ (k_tok_bytes vs v_tok_bytes).                //
// ===================================================================== //
extern "C" __global__ void fa_prefill_q(
        const float* __restrict__ Q, const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, long k_tok_bytes, long v_tok_bytes)
{
    const int q_tile = blockIdx.x;
    const int head   = blockIdx.y;
    const int lane   = threadIdx.x;
    const int q_base = q_tile * M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int kv_head = head / (n_head / n_head_kv);
    const int nq = min(M_ROWS, T - q_base);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* sK = sQ + M_ROWS*HEAD_DIM;
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;
    float* sO = (float*)(sP + M_ROWS*BK);
    float* sS = sO + M_ROWS*HEAD_DIM;
    float* sM = sS + M_ROWS*BK;
    float* sL = sM + M_ROWS;

    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        float qv = (r < nq) ? Q[((size_t)(q_base + r) * n_head + head) * head_dim + d] : 0.0f;
        sQ[i] = __float2bfloat16(qv);
    }
    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) sO[i] = 0.0f;
    for (int i = lane; i < M_ROWS; i += WARP_SZ) { sM[i] = NEG_INF; sL[i] = 0.0f; }
    __syncwarp();

    const int q_pos0 = (T_kv - T) + q_base;

    for (int k0 = 0; k0 < T_kv; k0 += BK) {
        const int nk = min(BK, T_kv - k0);
        if (causal && k0 > (q_pos0 + nq - 1)) break;

        // ---- stage K,V tiles to smem with INLINE DEQUANT (pad invalid keys with 0) ----
        for (int i = lane; i < BK*HEAD_DIM; i += WARP_SZ) {
            int kk = i / HEAD_DIM, d = i % HEAD_DIM;
            int eidx = kv_head * head_dim + d;       // element-within-token index
            float kv = (kk < nk) ? dq_q8_0_elem(K, (long)(k0 + kk), k_tok_bytes, eidx) : 0.0f;
            float vv = (kk < nk) ? dq_q5_1_elem(V, (long)(k0 + kk), v_tok_bytes, eidx) : 0.0f;
            sK[i] = __float2bfloat16(kv);
            sV[i] = __float2bfloat16(vv);
        }
        __syncwarp();

        for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
            CTile C0, C1;
            C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
            C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < HEAD_DIM; kk += K_STEP) {
                ATile A, Kt;
                ld_A(A,  sQ + kk,                  HEAD_DIM/2);
                ld_A(Kt, sK + kg*HEAD_DIM + kk,    HEAD_DIM/2);
                BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                mma_bf16(C0, A, Blo);
                mma_bf16(C1, A, Bhi);
            }
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int m = CTile::get_i(l), c8 = CTile::get_j(l);
                sS[m*BK + kg + 0      + c8] = C0.x[l];
                sS[m*BK + kg + N_KEYS + c8] = C1.x[l];
            }
        }
        __syncwarp();

        if (lane < M_ROWS) {
            int r = lane;
            float* srow = sS + r*BK;
            int q_pos = q_pos0 + r;
            float m_tile = NEG_INF;
            for (int j = 0; j < nk; ++j) {
                float s = srow[j] * scale;
                if (causal && (k0 + j) > q_pos) s = NEG_INF;
                srow[j] = s;
                m_tile = fmaxf(m_tile, s);
            }
            float m_prev = sM[r];
            float m_new  = fmaxf(m_prev, m_tile);
            float alpha = (m_prev == NEG_INF) ? 0.0f : exp2f((m_prev - m_new) * LOG2E);
            float l_tile = 0.0f;
            for (int j = 0; j < nk; ++j) {
                float p = (srow[j] == NEG_INF) ? 0.0f : exp2f((srow[j] - m_new) * LOG2E);
                sP[r*BK + j] = __float2bfloat16(p);
                l_tile += p;
            }
            for (int j = nk; j < BK; ++j) sP[r*BK + j] = __float2bfloat16(0.0f);
            sL[r] = sL[r] * alpha + l_tile;
            sM[r] = m_new;
            sS[r*BK + 0] = alpha;
        }
        __syncwarp();

        for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
            int r = i / HEAD_DIM;
            if (r < nq) sO[i] *= sS[r*BK + 0];
        }
        __syncwarp();

        for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
            CTile Clo, Chi;
            Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
            Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < BK; kk += K_STEP) {
                ATile A; ATile Bt;
                ld_A(A, sP + kk, BK/2);
                ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                mma_bf16(Clo, A, Blo);
                mma_bf16(Chi, A, Bhi);
            }
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int m   = CTile::get_i(l);
                int nlo = d0 +          CTile::get_j(l);
                int nhi = d0 + N_KEYS + CTile::get_j(l);
                sO[m*HEAD_DIM + nlo] += Clo.x[l];
                sO[m*HEAD_DIM + nhi] += Chi.x[l];
            }
        }
        __syncwarp();
    }

    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        if (r < nq) {
            float linv = (sL[r] > 0.0f) ? (1.0f / sL[r]) : 0.0f;
            O[((size_t)(q_base + r) * n_head + head) * head_dim + d] = sO[i] * linv;
        }
    }
}

// ===================================================================== //
//  KERNEL 2 : fa_decode_f32                                             //
//  T == 1 vector decode with flash-decoding split-K over the KV axis.   //
//  grid = (n_head, n_splits, 1) ; block = (HEAD_DIM/?, 1, 1) -> use 256  //
//  threads (one per head_dim element) for the simple, correct path.     //
//                                                                       //
//  Each block handles ONE (head, kv-split) and writes a PARTIAL:        //
//    partial O[head, split][d]  (f32, head_dim)                         //
//    partial m[head, split], l[head, split]  (the split's max & sum)    //
//  A second pass (fa_decode_combine_f32) merges splits with the         //
//  log-sum-exp rule. If n_splits==1 the combine is a trivial divide.    //
//                                                                       //
//  This is the scalar (CUDA-core) decode: for T=1 the QK and PV are     //
//  matrix-vector, where tensor cores give no win and add lane-map cost. //
//  Correctness-first; q8_0-K / q5_1-V dequant hooks are marked below.   //
//                                                                       //
//  C6: exp uses exp2f (exp(x)=exp2(x*LOG2E)). The split-combine uses the //
//  standard log-sum-exp merge; if a base bias on the running sum were    //
//  introduced it would be log2(N) with N the reduction width — for the   //
//  8-wide warp reductions that is log2(8)=3.0, NOT 2.079 (the FA-v1 bug).//
// ===================================================================== //

// Partials buffers are laid out [head][split][...]; caller sizes them n_head*n_splits.
extern "C" __global__ void fa_decode_f32(
        const float* __restrict__ Q,    // [head_dim, n_head, 1]
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,    // [n_head, n_splits, head_dim]
        float* __restrict__ partM,    // [n_head, n_splits]
        float* __restrict__ partL,    // [n_head, n_splits]
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits,
        long k_tok_bytes, long v_tok_bytes)
{
    const int head  = blockIdx.x;
    const int split = blockIdx.y;
    if (head >= n_head || split >= n_splits) return;
    const int kv_head = head / (n_head / n_head_kv);
    const int tid = threadIdx.x;                 // 0..head_dim-1 (block = head_dim threads)

    // this split owns keys [t_lo, t_hi)
    const int per = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ float ssh[];               // [head_dim] for q, + [32] reduction scratch
    float* sq = ssh;                             // head_dim
    float* red = sq + head_dim;                  // up to head_dim/32 partial sums

    // load q into smem (one element per thread)
    if (tid < head_dim) sq[tid] = Q[((size_t)0 * n_head + head) * head_dim + tid];
    __syncthreads();

    // running online softmax over this split's keys; accumulate o[d] in a register
    // (one thread owns one output dim d == tid).
    float m_i = NEG_INF;
    float l_i = 0.0f;
    float acc = 0.0f;                            // o[tid] partial (unnormalized, rescaled online)

    for (int t = t_lo; t < t_hi; ++t) {
        // score_t = scale * dot(q, K[:,kv_head,t])
        // ---- q8_0-K dequant: thread tid owns element kv_head*head_dim + tid ----
        // The dot reduction (warp+block) and online-softmax math are UNCHANGED.
        const int kidx = kv_head * head_dim + tid;       // element-within-token index
        float ktv = (tid < head_dim) ? dq_q8_0_elem(K, t, k_tok_bytes, kidx) : 0.0f;
        float prod = (tid < head_dim) ? sq[tid] * ktv : 0.0f;
        // block reduce prod -> score (warp shuffle + smem across warps)
        for (int o = 16; o > 0; o >>= 1) prod += __shfl_down_sync(0xffffffff, prod, o);
        if ((tid & 31) == 0) red[tid >> 5] = prod;
        __syncthreads();
        float score = 0.0f;
        if (tid == 0) {
            float s = 0.0f;
            int nwarp = (blockDim.x + 31) / 32;
            for (int w = 0; w < nwarp; ++w) s += red[w];
            red[0] = s * scale;
        }
        __syncthreads();
        score = red[0];
        __syncthreads();

        // online softmax merge of this single key
        float m_new = fmaxf(m_i, score);
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float p     = exp2f((score - m_new) * LOG2E);
        // ---- q5_1-V dequant: thread tid owns element kv_head*head_dim + tid ----
        const int vidx = kv_head * head_dim + tid;
        float vtv = (tid < head_dim) ? dq_q5_1_elem(V, t, v_tok_bytes, vidx) : 0.0f;
        if (tid < head_dim) acc = acc * alpha + p * vtv;
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // write this split's partial (UNNORMALIZED o, plus m_i and l_i for the combine)
    if (tid < head_dim) partO[((size_t)head * n_splits + split) * head_dim + tid] = acc;
    if (tid == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ===================================================================== //
//  KERNEL 2b : fa_decode_vec_q  (warp-per-token decode + GQA broadcast)  //
//  Replaces the element-per-thread fa_decode_f32 on the hot decode path  //
//  (T=1, split-K). BANDWIDTH lever (XQA/fattn-vec): each block owns ONE  //
//  KV head and dequants its KV tile ONCE into smem, broadcasting it to    //
//  all GQA_RATIO Q-head warps -> each KV byte leaves HBM/L2 ~1x/group     //
//  instead of GQA_RATIO x (was: grid.x=n_head, each Q-head re-dequants).  //
//                                                                         //
//  grid  = (n_head_kv, n_splits, 1)                                       //
//  block = (32, GQA_RATIO, 1)   warp y serves Q head kv_head*GQA + y      //
//                                                                         //
//  Per-warp register state (head_dim=256): each lane owns DPL=head_dim/32 //
//  = 8 Q elements (pre-scaled) and 8 output accumulators acc[8]. Online   //
//  softmax recurrence is BYTE-IDENTICAL to the validated prefill/decode   //
//  (exp2f + LOG2E, C6: no 2.079 bias). Writes the SAME [head][split][d]   //
//  partials -> fa_decode_combine_f32 merges (UNCHANGED).                  //
//                                                                         //
//  smem: sK[TILE][head_dim] + sV[TILE][head_dim] (f32), dequanted once    //
//  per block (all 32*GQA threads cooperate). TILE keys per FA step.       //
// ===================================================================== //
#define FA_DEC_TILE 32          // KV keys dequanted per step (one q8_0/q5_1 block row)
#define FA_DEC_MAX_DPL 8        // head_dim/32 ceiling (head_dim<=256). acc lives in regs.
extern "C" __global__ void fa_decode_vec_q(
        const float* __restrict__ Q,    // [head_dim, n_head, 1]
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [n_head, n_splits, head_dim]
        float* __restrict__ partM,      // [n_head, n_splits]
        float* __restrict__ partL,      // [n_head, n_splits]
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits,
        long k_tok_bytes, long v_tok_bytes)
{
    const int kv_head = blockIdx.x;              // ONE KV head per block (was per Q head)
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa     = n_head / n_head_kv;      // GQA_RATIO (4 for qwen35)
    const int wy      = threadIdx.y;             // 0..gqa-1: which Q head in the group
    const int lane    = threadIdx.x;             // 0..31
    if (wy >= gqa) return;
    const int head    = kv_head * gqa + wy;      // this warp's Q head
    const int dpl     = head_dim >> 5;           // dims-per-lane = head_dim/32 (==8 for 256)

    // this split owns keys [t_lo, t_hi)
    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    // ---- shared KV tile, dequanted ONCE per block and broadcast to all gqa warps ----
    // bf16 tiles (NOT f32): 2*32*256*2 = 32 KB, vs 64 KB f32 — doubles achievable occupancy.
    extern __shared__ __nv_bfloat16 ssh_vec[];   // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_vec;                 // [FA_DEC_TILE][head_dim]
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim; // [FA_DEC_TILE][head_dim]

    // stage this warp's Q row (one Q head, head_dim) into registers, PRE-SCALED by `scale`.
    // lane owns dims { lane, lane+32, ..., lane+32*(dpl-1) }.
    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    // per-warp online-softmax state + register accumulator (acc[i] is dim lane+32*i).
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    // cooperative dequant uses ALL block threads (32*gqa). Flat thread id over the block.
    const int bt   = wy * WARP_SZ + lane;        // 0 .. 32*gqa-1
    const int bsz  = WARP_SZ * gqa;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);    // valid keys this tile

        // ---- dequant K & V tile ONCE into smem (the GQA broadcast). All block threads
        //      stride over the nt*head_dim elements; eidx = kv_head*head_dim + d. ----
        for (int idx = bt; idx < nt * head_dim; idx += bsz) {
            int j = idx / head_dim;              // key within tile
            int d = idx - j * head_dim;          // head_dim element
            int eidx = kv_head * head_dim + d;
            sK[idx] = __float2bfloat16(dq_q8_0_elem(K, (long)(t0 + j), k_tok_bytes, eidx));
            sV[idx] = __float2bfloat16(dq_q5_1_elem(V, (long)(t0 + j), v_tok_bytes, eidx));
        }
        __syncthreads();

        // ---- per-warp: for each key in the tile, dot(q, K_j) -> online softmax -> acc += p*V_j ----
        for (int j = 0; j < nt; ++j) {
            const __nv_bfloat16* kj = sK + (size_t)j * head_dim;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) part += q_reg[i] * __bfloat162float(kj[lane + (i << 5)]);
            float score = warp_reduce_sum(part);     // every lane gets the full QK score (already *scale)

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
        __syncthreads();   // tile fully consumed before the next dequant overwrites sK/sV
    }

    // write this Q head's split partial (UNNORMALIZED acc, + m_i/l_i for the combine).
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// Combine flash-decoding splits with the log-sum-exp rule -> final O[head_dim, n_head, 1].
// grid = (n_head, 1, 1); block = (head_dim, 1, 1).
extern "C" __global__ void fa_decode_combine_f32(
        const float* __restrict__ partO, const float* __restrict__ partM,
        const float* __restrict__ partL, float* __restrict__ O,
        int head_dim, int n_head, int n_splits)
{
    const int head = blockIdx.x;
    const int tid  = threadIdx.x;
    if (head >= n_head || tid >= head_dim) return;

    // global max over splits
    float m = NEG_INF;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, partM[head * n_splits + s]);
    // combined sum and o
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = partM[head * n_splits + s];
        if (ms == NEG_INF) continue;
        float w = exp2f((ms - m) * LOG2E);                 // rescale this split to the global max
        l += partL[head * n_splits + s] * w;
        o += partO[((size_t)head * n_splits + s) * head_dim + tid] * w;
    }
    float linv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[((size_t)0 * n_head + head) * head_dim + tid] = o * linv;
}

// ===================================================================== //
//  DEVICE-COUNTER decode variants (CUDA-GRAPH-PLAN Phase 2)             //
//  Identical math to fa_decode_f32 / fa_decode_vec_q, but the sequence  //
//  length T_kv is read from a device int[1] counter (t_kv_dev[0]) for   //
//  the attention loop bound + per-split key range, NOT a host int arg.  //
//  The GRID is sized for a BUCKET-MAX n_splits at launch (baked at      //
//  capture). Splits whose key range [t_lo,t_hi) is EMPTY (t_lo>=T_kv,   //
//  so t_lo>=t_hi after per=ceil(T_kv/n_splits)) run the loop 0 times    //
//  and write the EMPTY partial (m_i=NEG_INF,l_i=0,acc=0) -> the combine //
//  skips them (ms==NEG_INF). So a graph captured for bucket-max T_kv    //
//  stays bit-correct for ANY actual T_kv <= bucket_max. The combine is  //
//  the SAME fa_decode_combine_f32 (it already skips NEG_INF splits).    //
// ===================================================================== //
extern "C" __global__ void fa_decode_f32_dc(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,
        float* __restrict__ partO,
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_dev,
        float scale, int n_splits,
        long k_tok_bytes, long v_tok_bytes)
{
    const int T_kv  = t_kv_dev[0];               // <-- device-resident sequence length
    const int head  = blockIdx.x;
    const int split = blockIdx.y;
    if (head >= n_head || split >= n_splits) return;
    const int kv_head = head / (n_head / n_head_kv);
    const int tid = threadIdx.x;

    const int per = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ float ssh[];
    float* sq = ssh;
    float* red = sq + head_dim;

    if (tid < head_dim) sq[tid] = Q[((size_t)0 * n_head + head) * head_dim + tid];
    __syncthreads();

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int t = t_lo; t < t_hi; ++t) {
        const int kidx = kv_head * head_dim + tid;
        float ktv = (tid < head_dim) ? dq_q8_0_elem(K, t, k_tok_bytes, kidx) : 0.0f;
        float prod = (tid < head_dim) ? sq[tid] * ktv : 0.0f;
        for (int o = 16; o > 0; o >>= 1) prod += __shfl_down_sync(0xffffffff, prod, o);
        if ((tid & 31) == 0) red[tid >> 5] = prod;
        __syncthreads();
        float score = 0.0f;
        if (tid == 0) {
            float s = 0.0f;
            int nwarp = (blockDim.x + 31) / 32;
            for (int w = 0; w < nwarp; ++w) s += red[w];
            red[0] = s * scale;
        }
        __syncthreads();
        score = red[0];
        __syncthreads();

        float m_new = fmaxf(m_i, score);
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float p     = exp2f((score - m_new) * LOG2E);
        const int vidx = kv_head * head_dim + tid;
        float vtv = (tid < head_dim) ? dq_q5_1_elem(V, t, v_tok_bytes, vidx) : 0.0f;
        if (tid < head_dim) acc = acc * alpha + p * vtv;
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    if (tid < head_dim) partO[((size_t)head * n_splits + split) * head_dim + tid] = acc;
    if (tid == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

extern "C" __global__ void fa_decode_vec_q_dc(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,
        float* __restrict__ partO,
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_dev,
        float scale, int n_splits,
        long k_tok_bytes, long v_tok_bytes)
{
    const int T_kv    = t_kv_dev[0];             // <-- device-resident sequence length
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa     = n_head / n_head_kv;
    const int wy      = threadIdx.y;
    const int lane    = threadIdx.x;
    if (wy >= gqa) return;
    const int head    = kv_head * gqa + wy;
    const int dpl     = head_dim >> 5;

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ __nv_bfloat16 ssh_vec[];
    __nv_bfloat16* sK = ssh_vec;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt   = wy * WARP_SZ + lane;
    const int bsz  = WARP_SZ * gqa;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);

        for (int idx = bt; idx < nt * head_dim; idx += bsz) {
            int j = idx / head_dim;
            int d = idx - j * head_dim;
            int eidx = kv_head * head_dim + d;
            sK[idx] = __float2bfloat16(dq_q8_0_elem(K, (long)(t0 + j), k_tok_bytes, eidx));
            sV[idx] = __float2bfloat16(dq_q5_1_elem(V, (long)(t0 + j), v_tok_bytes, eidx));
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
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}
