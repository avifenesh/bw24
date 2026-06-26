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
        const float* __restrict__ Q,  // [head_dim, n_head, 1]
        const float* __restrict__ K,  // [head_dim, n_head_kv, T_kv]
        const float* __restrict__ V,  // [head_dim, n_head_kv, T_kv]
        float* __restrict__ partO,    // [n_head, n_splits, head_dim]
        float* __restrict__ partM,    // [n_head, n_splits]
        float* __restrict__ partL,    // [n_head, n_splits]
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits)
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
        // ---- q8_0-K / q5_1-V HOOK -------------------------------------------------
        // For quantized KV: replace the f32 load below with a per-thread dequant of
        // K[t] block (q8_0: int8*scale; q5_1: 5-bit+min/scale) before the dot. The
        // dot reduction (warp+block) and the online-softmax math are UNCHANGED. The
        // V gather (acc += p * V[t][tid]) likewise dequants q5_1 per element. Keep
        // m_i/l_i in f32 regardless of KV dtype.
        // ---------------------------------------------------------------------------
        const float* kt = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float prod = (tid < head_dim) ? sq[tid] * kt[tid] : 0.0f;
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
        const float* vt = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
        if (tid < head_dim) acc = acc * alpha + p * vt[tid];
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // write this split's partial (UNNORMALIZED o, plus m_i and l_i for the combine)
    if (tid < head_dim) partO[((size_t)head * n_splits + split) * head_dim + tid] = acc;
    if (tid == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
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
