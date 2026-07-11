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
#include <cuda_fp8.h>
#include <cstdint>

#define WARP_SZ 32
// HEAD_DIM (2026-07-07): no longer a global #define — the FA prefill kernels are
// template<int HD> bodies stamped at BOTH 256 (qwen35 class, the original names) and
// 128 (MiniMax-M3 class, `_hd128` suffix). Each body opens with
//   constexpr int HEAD_DIM = HD;  HD_KTILES = HD/K_STEP;  O_NBLK = HD/N_KEYS;
// so the 256 instantiation compiles to the exact pre-template code (bit-identity
// pinned by the standard argmax/spec battery). Launchers (src/lib.rs fa_prefill*)
// pick the kernel by head_dim; other dims fall back to sdpa_naive at the callers.
#define M_ROWS  16     // query rows per warp tile
#define N_WARPS 4      // warps per CTA (P2 multi-warp) -> block (32,4,1)
#define BLOCK_Q (M_ROWS*N_WARPS) // query rows per CTA = 64 (= llama ncols)
#define N_KEYS  8      // one mma N-step = 8 keys (QK) / 8 d-cols (PV)
#define K_STEP  16     // m16n8k16 contraction width (logical bf16)
#define BK      32     // KV tile width (keys processed per FA step); = llama nbatch_fa
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

// ===================================================================== //
//  KV FORMAT SELECTION (kvbytes lane, 2026-07-08)                        //
//  Compile-time cache-format variants — build.rs compiles this file     //
//  once per (K,V) format pair into separate fatbins; lib.rs picks the   //
//  fatbin at Engine::new from env BW24_KV_K / BW24_KV_V. Default (no    //
//  -D flags) = the validated q8_0-K / q5_1-V daily config, and every    //
//  baseline code path below is the pre-refactor instruction sequence   //
//  verbatim (bit-identity pinned by the gate battery).                  //
//    -DBW24_KV_KFMT: 0 = q8_0 (34 B/32elem, default) | 1 = fp8-e4m3     //
//                        raw cast, NO block scale (32 B/32elem)         //
//    -DBW24_KV_VFMT: 0 = q5_1 (24 B/32elem, default) | 1 = q4_0         //
//                        (18 B/32elem) | 2 = fp8-e4m3 raw (32 B)        //
//  A non-default format is a NEW NUMERIC CONFIG: own argmax baseline,   //
//  gate battery must pass WITHIN it (exactness law binds per config).   //
//  Kernel entry names keep the historical q8_0_q5_1 suffix in ALL      //
//  variants — the format is a property of the loaded fatbin, not the   //
//  name (naming: no silent format switch; the env flag is the switch). //
// ===================================================================== //
#ifndef BW24_KV_KFMT
#define BW24_KV_KFMT 0
#endif
#ifndef BW24_KV_VFMT
#define BW24_KV_VFMT 0
#endif

// fp8-e4m3 RAW dequant of one element (scale-free: 1 B/elem, tok stride == kv_dim).
// sm_120 has native cvt.f32.e4m3 — one instruction, no d-scale load (the "cheaper
// dequant" arm). Works for K or V (no block structure; eidx addresses the byte).
static __device__ __forceinline__ float dq_fp8_elem(
        const uint8_t* __restrict__ P, long t, long tok_bytes, int eidx)
{
    return (float)((const __nv_fp8_e4m3*)(P + (size_t)t * tok_bytes))[eidx];
}

// q4_0 dequant of one element (symmetric, ggml layout: f16 d + 16 nibble bytes,
// elem j<16 = low nibble of byte j, elem j+16 = high nibble; x = d*(q-8)).
static __device__ __forceinline__ float dq_q4_0_elem(
        const uint8_t* __restrict__ V, long t, long v_tok_bytes, int eidx)
{
    const uint8_t* blk = V + (size_t)t * v_tok_bytes + (size_t)(eidx >> 5) * 18;
    const half d = *(const half*)blk;
    const uint8_t* qs = blk + 2;
    const int j = eidx & 31;
    const int q = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
    return __half2float(d) * (float)(q - 8);
}

#if BW24_KV_KFMT == 1
#define K_BLK_B 32
#define DQ_K_ELEM dq_fp8_elem
#else
#define K_BLK_B 34
#define DQ_K_ELEM dq_q8_0_elem
#endif
#if BW24_KV_VFMT == 1
#define V_BLK_B 18
#define DQ_V_ELEM dq_q4_0_elem
#elif BW24_KV_VFMT == 2
#define V_BLK_B 32
#define DQ_V_ELEM dq_fp8_elem
#else
#define V_BLK_B 24
#define DQ_V_ELEM dq_q5_1_elem
#endif

// Per-(dim-block, lane) dequant for the register-walk vec kernels: `blk` points at ONE
// 32-elem block's bytes for one token; lane owns element (block*32 + lane). The 32 lanes
// of a warp read consecutive bytes = coalesced, same as the inlined originals. The
// BASELINE bodies are the exact instruction sequences the validated kernels inlined.
static __device__ __forceinline__ float dq_K_lane(const uint8_t* __restrict__ blk, int lane)
{
#if BW24_KV_KFMT == 1
    return (float)((const __nv_fp8_e4m3*)blk)[lane];
#else
    const float d = __half2float(*(const half*)blk);
    const int8_t q = ((const int8_t*)(blk + 2))[lane];
    return d * (float)q;
#endif
}
static __device__ __forceinline__ float dq_V_lane(const uint8_t* __restrict__ blk, int lane)
{
#if BW24_KV_VFMT == 1
    const float d = __half2float(*(const half*)blk);
    const uint8_t* qs = blk + 2;
    const int lo = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
    return d * (float)(lo - 8);
#elif BW24_KV_VFMT == 2
    return (float)((const __nv_fp8_e4m3*)blk)[lane];
#else
    const float d = __half2float(*(const half*)blk);
    const float m = __half2float(*(const half*)(blk + 2));
    const uint32_t qh = *(const uint32_t*)(blk + 4);
    const uint8_t* qs = blk + 8;
    const int lo = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
    const int q5 = lo | (int)(((qh >> lane) & 1u) << 4);
    return d * (float)q5 + m;
#endif
}

// Append-quantize ONE 32-elem block (whole warp participates; `x` is this lane's element,
// caller zero-pads past kv_dim). `blk` = this block's cache bytes. The BASELINE bodies are
// the validated append kernels' warp programs verbatim (rows/dc bit-identity holds because
// all three appenders call the SAME function).
static __device__ __forceinline__ void quant_K_block(float x, int lane, uint8_t* __restrict__ blk)
{
#if BW24_KV_KFMT == 1
    ((__nv_fp8_e4m3*)blk)[lane] = __nv_fp8_e4m3(x);   // native cvt, satfinite (clamps ±448)
#else
    float amax = warp_amax(x);
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(x * id);
    q = max(-127, min(127, q));
    if (lane == 0) *(half*)blk = __float2half(d);
    ((int8_t*)(blk + 2))[lane] = (int8_t)q;
#endif
}
static __device__ __forceinline__ void quant_V_block(float x, int lane, uint8_t* __restrict__ blk)
{
#if BW24_KV_VFMT == 1
    // q4_0 (ggml quantize_row_q4_0 semantics): d = signed-max/-8, q = trunc(x*id + 8.5)
    // clamped to 15. |max| tie resolves to the LOWEST lane (= ggml's first-index scan).
    float amax = warp_amax(x);
    unsigned mm = __ballot_sync(0xffffffffu, fabsf(x) == amax);
    float mx = __shfl_sync(0xffffffffu, x, __ffs(mm) - 1);
    float d = mx / -8.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int q4 = min(15, (int)(x * id + 8.5f));
    if (lane == 0) *(half*)blk = __float2half(d);
    uint8_t* qs = blk + 2;
    int nib = q4 & 0x0F;
    int partner_nib = __shfl_sync(0xffffffffu, nib, lane + 16) & 0x0F;
    if (lane < 16) qs[lane] = (uint8_t)(nib | (partner_nib << 4));
#elif BW24_KV_VFMT == 2
    ((__nv_fp8_e4m3*)blk)[lane] = __nv_fp8_e4m3(x);
#else
    float mn = warp_min(x);
    float mx = warp_max(x);
    float d = (mx - mn) / 31.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int q5 = (int)lrintf((x - mn) * id);
    q5 = max(0, min(31, q5));
    // qh bit j set iff element j has its 5th bit (bit 4) set. __ballot_sync
    // over all 32 lanes yields EXACTLY the little-endian qh u32 (bit j = lane j).
    uint32_t qh = __ballot_sync(0xffffffffu, (q5 >> 4) & 1);
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
    int partner_nib = __shfl_sync(0xffffffffu, nib, lane + 16) & 0x0F;
    if (lane < 16) qs[lane] = (uint8_t)(nib | (partner_nib << 4));
#endif
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

    // ---- K block b (format via quant_K_block; baseline q8_0 symmetric) ----
    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_row[eidx] : 0.0f;
        quant_K_block(x, lane, K + (size_t)t * k_tok_bytes + (size_t)b * K_BLK_B);
    }

    // ---- V block b (format via quant_V_block; baseline q5_1 affine) ----
    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_row[eidx] : 0.0f;
        quant_V_block(x, lane, V + (size_t)t * v_tok_bytes + (size_t)b * V_BLK_B);
    }
}

// ----- BATCHED-ROWS variant (BATCHED PROMPT PRIME): appends T token rows in ONE
// launch. grid = (max(kv_dim_k,kv_dim_v)/32, T); block = (32,1,1). Each (b, tt)
// warp executes EXACTLY the per-token kernel's warp program on token row tt of the
// token-major k_rows/v_rows ([T, kv_dim]) writing at slot t0+tt — so every written
// cache row is BIT-IDENTICAL to T sequential append_quantize_kv_q8_0_q5_1 calls
// (kernel_check pins this bytewise). Replaces the T-launch loop (~T*n_layers*3us
// of launch overhead per prime).
extern "C" __global__ void append_quantize_kv_q8_0_q5_1_rows(
        const float* __restrict__ k_rows,  // [T, kv_dim_k] token-major
        const float* __restrict__ v_rows,  // [T, kv_dim_v] token-major
        uint8_t* __restrict__ K,           // cache base (q8_0)
        uint8_t* __restrict__ V,           // cache base (q5_1)
        int t0, int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes)
{
    const int b    = blockIdx.x;           // 32-elem block index within the token
    const int tt   = blockIdx.y;           // token index within the batch
    const int lane = threadIdx.x;          // 0..31
    const int eidx = b * 32 + lane;        // element index within token
    const int t    = t0 + tt;              // cache write slot

    // ---- K block b; identical math to the per-token kernel (same quant_K_block) ----
    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_rows[(size_t)tt * kv_dim_k + eidx] : 0.0f;
        quant_K_block(x, lane, K + (size_t)t * k_tok_bytes + (size_t)b * K_BLK_B);
    }

    // ---- V block b; identical math to the per-token kernel (same quant_V_block) ----
    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_rows[(size_t)tt * kv_dim_v + eidx] : 0.0f;
        quant_V_block(x, lane, V + (size_t)t * v_tok_bytes + (size_t)b * V_BLK_B);
    }
}

// ROUND-STREAM stage (c) 2: rows append with the write offset from a DEVICE counter (the
// pre-issued verify's t0 = len_d, unknown to the host at issue time). Body identical to
// append_quantize_kv_q8_0_q5_1_rows.
extern "C" __global__ void append_quantize_kv_q8_0_q5_1_rows_dc(
        const float* __restrict__ k_rows, const float* __restrict__ v_rows,
        uint8_t* __restrict__ K, uint8_t* __restrict__ V,
        const int* __restrict__ t0_dev, int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes)
{
    const int b    = blockIdx.x;
    const int tt   = blockIdx.y;
    const int lane = threadIdx.x;
    const int eidx = b * 32 + lane;
    const int t    = t0_dev[0] + tt;
    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_rows[(size_t)tt * kv_dim_k + eidx] : 0.0f;
        quant_K_block(x, lane, K + (size_t)t * k_tok_bytes + (size_t)b * K_BLK_B);
    }
    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_rows[(size_t)tt * kv_dim_v + eidx] : 0.0f;
        quant_V_block(x, lane, V + (size_t)t * v_tok_bytes + (size_t)b * V_BLK_B);
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
        quant_K_block(x, lane, K + (size_t)t * k_tok_bytes + (size_t)b * K_BLK_B);
    }

    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_row[eidx] : 0.0f;
        quant_V_block(x, lane, V + (size_t)t * v_tok_bytes + (size_t)b * V_BLK_B);
    }
}

// ===================================================================== //
//  KERNEL 1 : fa_prefill_f32  — FLOOR PORT (matches llama MMA-f16)       //
//  4 WARPS / CTA (block (32,4,1)); each warp owns 16 query rows of the   //
//  64-row CTA tile (BLOCK_Q=64 = llama ncols). FA-2 online softmax.      //
//  grid = (ceil(T/64), n_head_kv, 1).  GQA: 4 Q-heads share staged K/V   //
//  (P1, = llama ncols2=4) via an inner gq loop — K/V dequant/stage once. //
//                                                                        //
//  P0a Q-in-reg : each warp's 16x256 Q lives in HD_KTILES=16 A-fragments //
//                 (registers), staged through reused sK∪sV smem once per //
//                 (gq) — NO persistent sQ (was the 32KB occupancy block).//
//  P0b register-O: O[16][256] lives in O_NBLK=32 CTiles (128 f32/lane),  //
//                 NOT smem. Per-KV-block alpha rescale is a register FMA  //
//                 broadcast via __shfl_sync. No sO smem RMW.             //
//                                                                        //
//  Persistent shared memory (bf16 unless noted), shared by all 4 warps:  //
//    sK : [BK][HEAD_DIM]      current KV key tile (shared across gq)      //
//    sV : [BK][HEAD_DIM]      current KV value tile (shared across gq)    //
//    sP : [BLOCK_Q][BK]       softmax probs P (bf16) SMEM round-trip (C4) //
//    sS : [BLOCK_Q][BK] f32   QK scores staged for the row softmax        //
//    sM : [BLOCK_Q] f32       running max m_i per query row               //
//    sL : [BLOCK_Q] f32       running sum  l_i per query row              //
//  (sK∪sV doubles as the transient Q staging buffer before the KV loop.) //
// ===================================================================== //

// Load this warp's 16xHD Q tile into HD/K_STEP A-fragments (Q-in-reg, P0a).
// Q is staged into `stage` smem (reused sK∪sV) cooperatively by the warp, then
// ldmatrix'd. `qrow_base`/`nqw` give the warp's global Q rows; pads with 0.
template<int HD>
static __device__ __forceinline__ void load_q_frags(
        ATile* Qf, const float* __restrict__ Q, __nv_bfloat16* stage,
        int qrow_base, int nqw, int head, int n_head, int head_dim, int lane)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    // stage 16 rows x HEAD_DIM into `stage` (row-major, HEAD_DIM-fastest)
    for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        float qv = (r < nqw) ? Q[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] : 0.0f;
        stage[i] = __float2bfloat16(qv);
    }
    __syncwarp();
    #pragma unroll
    for (int kt = 0; kt < HD_KTILES; ++kt)
        ld_A(Qf[kt], stage + kt*K_STEP, HEAD_DIM/2);   // Q[16][kt*16 .. kt*16+16]
    __syncwarp();
}

template<int HD>
static __device__ __forceinline__ void fa_prefill_f32_body(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    const int warp = threadIdx.y;             // 0..N_WARPS-1
    const int lane = threadIdx.x;             // 0..31
    // grid.y = n_head (one Q-head per CTA). P1 GQA reuse via the inner gq loop is NOT
    // used for pp512: collapsing grid.y to n_head_kv (4) starves the 82-SM GPU (only
    // 8*4=32 CTAs << 82 SMs). Keeping grid.y=n_head gives 8*16=128 CTAs > 82 SMs ->
    // every SM gets work. KV is re-staged per head, but pp512 is COMPUTE-bound so the
    // KV-byte re-read is a wash (FA-MATCH-THEN-EXCEED §1) and full SM coverage wins.
    const int head    = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BLOCK_Q;       // CTA's first query row
    const int qrow_base = q_base + warp*M_ROWS;     // this warp's first query row
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);     // valid query rows for this warp (>=0)

    // ----- persistent dynamic shared memory layout (shared across 4 warps) -----
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem_raw;                 // BK*HEAD_DIM
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;                         // BK*HEAD_DIM
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;                         // BLOCK_Q*BK
    float* sS = (float*)(sP + BLOCK_Q*BK);                        // BLOCK_Q*BK f32
    float* sM = sS + BLOCK_Q*BK;                                  // BLOCK_Q f32
    float* sL = sM + BLOCK_Q;                                     // BLOCK_Q f32
    // this warp's sub-slices (16 rows starting at warp*M_ROWS)
    __nv_bfloat16* sPw = sP + warp*M_ROWS*BK;
    float* sSw = sS + warp*M_ROWS*BK;
    float* sMw = sM + warp*M_ROWS;
    float* sLw = sL + warp*M_ROWS;
    // transient Q staging area for THIS warp (reuse sK∪sV: 4 warps x 16*HEAD_DIM
    // = 64*HEAD_DIM = (sK+sV) capacity, one 16-row slab per warp, no overlap).
    __nv_bfloat16* sQstage = sK + warp*M_ROWS*HEAD_DIM;

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;  // abs q-pos of this warp's row 0

        // --- P0a: load this warp's Q into A-fragments (registers) via reused sK∪sV ---
        ATile Qf[HD_KTILES];
        load_q_frags<HD>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        __syncthreads();   // all warps done reading their Q slab before sK/sV is overwritten

        // --- P0b: O accumulator in registers (CTiles), running m_i/l_i per row ---
        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        if (lane < M_ROWS) { sMw[lane] = NEG_INF; sLw[lane] = 0.0f; }
        __syncthreads();

        // ===== FA-2 loop over KV in tiles of BK keys =====
        for (int k0 = 0; k0 < T_kv; k0 += BK) {
            const int nk = min(BK, T_kv - k0);
            // causal early-out: whole tile past the CTA's max query position -> done.
            const int q_pos_max = (T_kv - T) + q_base + (BLOCK_Q - 1);
            if (causal_i && k0 > q_pos_max) break;

            // ---- stage K,V tile to smem ONCE per gq (block-cooperative, 128 threads) ----
            const int bt = warp*WARP_SZ + lane;       // flat thread id 0..127
            for (int i = bt; i < BK*HEAD_DIM; i += N_WARPS*WARP_SZ) {
                int kk = i / HEAD_DIM, d = i % HEAD_DIM;
                float kv = (kk < nk) ? K[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
                float vv = (kk < nk) ? V[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
                sK[i] = __float2bfloat16(kv);
                sV[i] = __float2bfloat16(vv);
            }
            __syncthreads();

            // ---- GEMM0: S[16 q][BK key] = Q @ K^T (Q from registers Qf) ----
            for (int kg = 0; kg < BK; kg += 2*N_KEYS) {           // 16 keys per group
                CTile C0, C1;                                     // C0: keys kg+0..7 ; C1: kg+8..15
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Kt;
                    ld_A(Kt, sK + kg*HEAD_DIM + kt*K_STEP, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qf[kt], Blo);
                    mma_bf16(C1, Qf[kt], Bhi);
                }
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int m = CTile::get_i(l), c8 = CTile::get_j(l);
                    sSw[m*BK + kg + 0      + c8] = C0.x[l];
                    sSw[m*BK + kg + N_KEYS + c8] = C1.x[l];
                }
            }
            __syncwarp();

            // ---- row softmax update (one query row per lane; 16 rows <= 32) ----
            // alpha[r] is written to sSw[r*BK+0] AFTER the row's scores are fully consumed,
            // for the register-O rescale broadcast below.
            float alpha_self = 1.0f;   // alpha for the row this lane will rescale (lane->row map)
            if (lane < M_ROWS) {
                int r = lane;
                float* srow = sSw + r*BK;
                int q_pos = q_pos0w + r;
                float m_tile = NEG_INF;
                for (int j = 0; j < nk; ++j) {
                    float s = srow[j] * scale;
                    if (causal_i && (k0 + j) > q_pos) s = NEG_INF;
                    srow[j] = s;
                    m_tile = fmaxf(m_tile, s);
                }
                float m_prev = sMw[r];
                float m_new  = fmaxf(m_prev, m_tile);
                float alpha = (m_prev == NEG_INF) ? 0.0f : exp2f((m_prev - m_new) * LOG2E);
                float l_tile = 0.0f;
                for (int j = 0; j < nk; ++j) {
                    float p = (srow[j] == NEG_INF) ? 0.0f : exp2f((srow[j] - m_new) * LOG2E);
                    sPw[r*BK + j] = __float2bfloat16(p);
                    l_tile += p;
                }
                for (int j = nk; j < BK; ++j) sPw[r*BK + j] = __float2bfloat16(0.0f);
                sLw[r] = sLw[r] * alpha + l_tile;
                sMw[r] = m_new;
                sSw[r*BK + 0] = alpha;   // broadcast slot (scores consumed into sPw above)
            }
            __syncwarp();

            // ---- P0b: rescale register-O by alpha (per row), via __shfl broadcast ----
            // CTile lane->row map: lane holds rows {lane/4, lane/4+8}. alpha for row r lives
            // in lane r's sSw[r*BK+0]; read each row's alpha by shuffling from the owning lane.
            int r_lo = lane / 4;          // CTile get_i(l) for l in {0,1}
            int r_hi = r_lo + 8;          // CTile get_i(l) for l in {2,3}
            float a_lo = sSw[r_lo*BK + 0];
            float a_hi = sSw[r_hi*BK + 0];
            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= a_lo; O_acc[c].x[1] *= a_lo;   // rows r_lo (l=0,1)
                O_acc[c].x[2] *= a_hi; O_acc[c].x[3] *= a_hi;   // rows r_hi (l=2,3)
            }

            // ---- GEMM1: O += P @ V (P re-ldmatrix'd from sPw; accumulate INTO O_acc) ----
            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    ld_A(A, sPw + kk, BK/2);
                    ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            __syncthreads();   // all warps done with sK/sV/sPw before next tile overwrites
        }

        // ===== deferred final normalize: O = O_acc / l_i ; write to global =====
        // CTile lane map: O_acc[c].x[l] is row CTile::get_i(l), col c*8 + CTile::get_j(l).
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    float linv = (sLw[r] > 0.0f) ? (1.0f / sLw[r]) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();   // ensure all warps finish writing O / reading sLw before next gq
    }
}

// extern-C stamps: hd256 keeps the ORIGINAL name (qwen35-class dispatch unchanged);
// `_hd128` is the MiniMax-M3 twin.
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_f32(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    fa_prefill_f32_body<256>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal);
}
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_f32_hd128(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    fa_prefill_f32_body<128>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal);
}

// ===================================================================== //
//  KERNEL 1c : fa_prefill_f32_pp  — Edge 5a (FA3 softmax-GEMM overlap)   //
//  PURE REORDER of fa_prefill_f32: the QK scores of a tile are kept in   //
//  REGISTERS (4 CTiles / warp = the 16x32 score tile) and the online     //
//  softmax (max/sum reduce + exp2 + alpha) runs on those registers via   //
//  4-lane __shfl_xor butterflies — eliminating the sSw smem write+read   //
//  ROUND-TRIP that is the dominant short_scoreboard stall in the floor.  //
//  This lets the softmax transcendental+reduce latency hide behind the   //
//  tensor-issue/ldmatrix pipe instead of serializing on a smem dep.      //
//                                                                        //
//  Score CTile layout (per warp, BK=32 cols = 4 CTiles of 8 cols):       //
//    Sc[g].x[l] = row CTile::get_i(l), col g*8 + CTile::get_j(l).        //
//    For a fixed lane: x[0],x[1] -> row r_lo=lane/4, cols c0,c0+1;        //
//                      x[2],x[3] -> row r_hi=r_lo+8,  cols c0,c0+1;       //
//                      c0=(lane%4)*2 ; the 4 lanes {lane/4*4 .. +3} hold  //
//    the 4 col-pairs (8 cols) of one CTile -> a row's 32-col reduce is a  //
//    butterfly over __shfl_xor offsets {1,2} (the 4 lanes sharing r).    //
//  exp2/LOG2E, m_i/l_i recurrence: BYTE-IDENTICAL to fa_prefill_f32 (the  //
//  only float-order change is the per-row sum becomes a 4-lane tree add   //
//  vs the serial smem add -> rel drift ~1e-7, immaterial; argmax-safe).  //
// ===================================================================== //
// Per-row reduce of the 4 lanes that share a CTile row (lanes differ in
// lane%4 only). offset 1 then 2 covers {0,1,2,3} within the row's quad.
static __device__ __forceinline__ float row_max4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;   // all 4 lanes of the quad hold the row max
}
static __device__ __forceinline__ float row_sum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;   // all 4 lanes of the quad hold the row sum
}

template<int HD>
static __device__ __forceinline__ void fa_prefill_f32_pp_body(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head    = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BLOCK_Q;
    const int qrow_base = q_base + warp*M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem_raw;                 // BK*HEAD_DIM
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;                         // BK*HEAD_DIM
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;                         // BLOCK_Q*BK
    // sS retained ONLY as the alpha broadcast slot (BLOCK_Q f32 is enough but
    // keep the same layout offsets so the launcher smem calc is unchanged).
    float* sS = (float*)(sP + BLOCK_Q*BK);                        // BLOCK_Q*BK f32
    float* sM = sS + BLOCK_Q*BK;                                  // BLOCK_Q f32
    float* sL = sM + BLOCK_Q;                                     // BLOCK_Q f32
    __nv_bfloat16* sPw = sP + warp*M_ROWS*BK;
    float* sMw = sM + warp*M_ROWS;
    float* sLw = sL + warp*M_ROWS;
    __nv_bfloat16* sQstage = sK + warp*M_ROWS*HEAD_DIM;

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;

        ATile Qf[HD_KTILES];
        load_q_frags<HD>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        __syncthreads();

        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        // running m_i / l_i held in REGISTERS (per the two rows this lane owns).
        float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
        const int r_lo = lane / 4;          // CTile get_i(l=0,1)
        const int r_hi = r_lo + 8;          // CTile get_i(l=2,3)
        const int c0   = (lane % 4) * 2;    // CTile get_j base for this lane

        for (int k0 = 0; k0 < T_kv; k0 += BK) {
            const int nk = min(BK, T_kv - k0);
            const int q_pos_max = (T_kv - T) + q_base + (BLOCK_Q - 1);
            if (causal_i && k0 > q_pos_max) break;

            const int bt = warp*WARP_SZ + lane;
            for (int i = bt; i < BK*HEAD_DIM; i += N_WARPS*WARP_SZ) {
                int kk = i / HEAD_DIM, d = i % HEAD_DIM;
                float kv = (kk < nk) ? K[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
                float vv = (kk < nk) ? V[((size_t)(k0 + kk) * n_head_kv + kv_head) * head_dim + d] : 0.0f;
                sK[i] = __float2bfloat16(kv);
                sV[i] = __float2bfloat16(vv);
            }
            __syncthreads();

            // ---- GEMM0: QK^T -> 4 score CTiles HELD IN REGISTERS (no sSw write) ----
            CTile Sc[BK/N_KEYS];                 // BK/8 = 4 CTiles, 8 cols each
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
            for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                CTile C0, C1;
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Kt;
                    ld_A(Kt, sK + kg*HEAD_DIM + kt*K_STEP, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qf[kt], Blo);
                    mma_bf16(C1, Qf[kt], Bhi);
                }
                Sc[kg/N_KEYS + 0] = C0;          // cols kg+0..7
                Sc[kg/N_KEYS + 1] = C1;          // cols kg+8..15
            }

            // ---- SOFTMAX on registers: scale + causal mask, then 4-lane reduce ----
            // Sc[g].x[l]: row (l<2?r_lo:r_hi), col g*8 + c0 + (l&1).
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int col = g*N_KEYS + c0 + (l & 1);
                    int row = (l < 2) ? r_lo : r_hi;
                    int q_pos = q_pos0w + row;
                    float s = Sc[g].x[l] * scale;
                    if (col >= nk) s = NEG_INF;
                    if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                    Sc[g].x[l] = s;
                    if (l < 2) s_tile_max_lo = fmaxf(s_tile_max_lo, s);
                    else       s_tile_max_hi = fmaxf(s_tile_max_hi, s);
                }
            }
            s_tile_max_lo = row_max4(s_tile_max_lo);   // 4-lane reduce -> row max
            s_tile_max_hi = row_max4(s_tile_max_hi);

            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
            float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
            float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);

            // exp2 each score against its row's m_new; partial l per lane, then 4-lane sum.
            float l_part_lo = 0.0f, l_part_hi = 0.0f;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float mn = (l < 2) ? m_new_lo : m_new_hi;
                    float s  = Sc[g].x[l];
                    float p  = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
                    Sc[g].x[l] = p;                          // P now in the score regs
                    if (l < 2) l_part_lo += p; else l_part_hi += p;
                }
            }
            l_part_lo = row_sum4(l_part_lo);
            l_part_hi = row_sum4(l_part_hi);
            l_lo = l_lo * alpha_lo + l_part_lo;
            l_hi = l_hi * alpha_hi + l_part_hi;
            m_lo = m_new_lo; m_hi = m_new_hi;

            // ---- write P to sPw (MANDATORY smem round-trip for PV's A-operand layout) ----
            // Sc[g].x[l] -> sPw[row*BK + g*8 + c0 + (l&1)].
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                sPw[r_lo*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[0]);
                sPw[r_lo*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[1]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[2]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[3]);
            }
            __syncwarp();

            // ---- rescale register-O by alpha (alpha already per-row in regs, no smem) ----
            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
                O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
            }

            // ---- GEMM1: O += P @ V ----
            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    ld_A(A, sPw + kk, BK/2);
                    ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            __syncthreads();
        }

        // store l_i for the two rows this lane owns (col-pair lanes agree after row_sum4),
        // only the lane that owns the canonical write does it -> use sLw, lane c0==0 writes.
        if (c0 == 0) { sLw[r_lo] = l_lo; sLw[r_hi] = l_hi; }
        __syncwarp();

        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    float linv = (sLw[r] > 0.0f) ? (1.0f / sLw[r]) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_f32_pp(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    fa_prefill_f32_pp_body<256>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal);
}
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_f32_pp_hd128(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal)
{
    fa_prefill_f32_pp_body<128>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal);
}

// ===================================================================== //
//  KERNEL 1b : fa_prefill_q  (quantized-cache prefill: q8_0 K / q5_1 V) //
//  Identical to fa_prefill_f32 EXCEPT the stage-to-smem copy dequants    //
//  the resident quantized KV cache. MMA / softmax / PV are byte-identical //
//  to the f32 kernel. Used by the MTP verify path (fa_prefill_view).     //
//  K/V token strides differ (k_tok_bytes vs v_tok_bytes).                //
// ===================================================================== //
template<int HD>
static __device__ __forceinline__ void fa_prefill_q_body(
        const float* __restrict__ Q, const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, long k_tok_bytes, long v_tok_bytes)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head    = blockIdx.y;           // grid.y = n_head (full SM subscription)
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BLOCK_Q;
    const int qrow_base = q_base + warp*M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;
    float* sS = (float*)(sP + BLOCK_Q*BK);
    float* sM = sS + BLOCK_Q*BK;
    float* sL = sM + BLOCK_Q;
    __nv_bfloat16* sPw = sP + warp*M_ROWS*BK;
    float* sSw = sS + warp*M_ROWS*BK;
    float* sMw = sM + warp*M_ROWS;
    float* sLw = sL + warp*M_ROWS;
    __nv_bfloat16* sQstage = sK + warp*M_ROWS*HEAD_DIM;

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;

        ATile Qf[HD_KTILES];
        load_q_frags<HD>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        __syncthreads();

        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        // Edge 5a: register-resident online-softmax state (no sSw round-trip).
        float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
        const int r_lo = lane / 4;          // CTile get_i(l=0,1)
        const int r_hi = r_lo + 8;          // CTile get_i(l=2,3)
        const int c0   = (lane % 4) * 2;    // CTile get_j base for this lane

        for (int k0 = 0; k0 < T_kv; k0 += BK) {
            const int nk = min(BK, T_kv - k0);
            const int q_pos_max = (T_kv - T) + q_base + (BLOCK_Q - 1);
            if (causal_i && k0 > q_pos_max) break;

            // ---- stage K,V tile to smem with INLINE DEQUANT, ONCE per gq (128 threads) ----
            const int bt = warp*WARP_SZ + lane;
            for (int i = bt; i < BK*HEAD_DIM; i += N_WARPS*WARP_SZ) {
                int kk = i / HEAD_DIM, d = i % HEAD_DIM;
                int eidx = kv_head * head_dim + d;
                float kv = (kk < nk) ? DQ_K_ELEM(K, (long)(k0 + kk), k_tok_bytes, eidx) : 0.0f;
                float vv = (kk < nk) ? DQ_V_ELEM(V, (long)(k0 + kk), v_tok_bytes, eidx) : 0.0f;
                sK[i] = __float2bfloat16(kv);
                sV[i] = __float2bfloat16(vv);
            }
            __syncthreads();

            // ---- GEMM0: QK^T -> 4 score CTiles HELD IN REGISTERS (no sSw write) ----
            CTile Sc[BK/N_KEYS];
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
            for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                CTile C0, C1;
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Kt;
                    ld_A(Kt, sK + kg*HEAD_DIM + kt*K_STEP, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qf[kt], Blo);
                    mma_bf16(C1, Qf[kt], Bhi);
                }
                Sc[kg/N_KEYS + 0] = C0;
                Sc[kg/N_KEYS + 1] = C1;
            }

            // ---- SOFTMAX on registers (scale + causal mask + 4-lane reduce) ----
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int col = g*N_KEYS + c0 + (l & 1);
                    int row = (l < 2) ? r_lo : r_hi;
                    int q_pos = q_pos0w + row;
                    float s = Sc[g].x[l] * scale;
                    if (col >= nk) s = NEG_INF;
                    if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                    Sc[g].x[l] = s;
                    if (l < 2) s_tile_max_lo = fmaxf(s_tile_max_lo, s);
                    else       s_tile_max_hi = fmaxf(s_tile_max_hi, s);
                }
            }
            s_tile_max_lo = row_max4(s_tile_max_lo);
            s_tile_max_hi = row_max4(s_tile_max_hi);

            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
            float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
            float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);

            float l_part_lo = 0.0f, l_part_hi = 0.0f;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float mn = (l < 2) ? m_new_lo : m_new_hi;
                    float s  = Sc[g].x[l];
                    float p  = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
                    Sc[g].x[l] = p;
                    if (l < 2) l_part_lo += p; else l_part_hi += p;
                }
            }
            l_part_lo = row_sum4(l_part_lo);
            l_part_hi = row_sum4(l_part_hi);
            l_lo = l_lo * alpha_lo + l_part_lo;
            l_hi = l_hi * alpha_hi + l_part_hi;
            m_lo = m_new_lo; m_hi = m_new_hi;

            // ---- write P to sPw (MANDATORY for PV's A-operand ldmatrix layout) ----
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                sPw[r_lo*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[0]);
                sPw[r_lo*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[1]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[2]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[3]);
            }
            __syncwarp();

            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
                O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
            }

            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    ld_A(A, sPw + kk, BK/2);
                    ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            __syncthreads();
        }

        if (c0 == 0) { sLw[r_lo] = l_lo; sLw[r_hi] = l_hi; }
        __syncwarp();

        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    float linv = (sLw[r] > 0.0f) ? (1.0f / sLw[r]) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_q(
        const float* __restrict__ Q, const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, long k_tok_bytes, long v_tok_bytes)
{
    fa_prefill_q_body<256>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv,
                           scale, causal, k_tok_bytes, v_tok_bytes);
}
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_q_hd128(
        const float* __restrict__ Q, const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, long k_tok_bytes, long v_tok_bytes)
{
    fa_prefill_q_body<128>(Q, K, V, O, head_dim, n_head, n_head_kv, T, T_kv,
                           scale, causal, k_tok_bytes, v_tok_bytes);
}

// ===================================================================== //
//  KERNEL 1b-ws : dequant-once chunk-prime workspace (ARC B, 2026-07-05)//
//  fa_prefill_q's inline dequant is 64x-redundant at chunk prime: each  //
//  of the T/BLOCK_Q q-block CTAs (x n_head/n_head_kv GQA CTAs) re-      //
//  dequants the SAME up-to-40k-token quantized KV stream (30.5% of the  //
//  32k prime wall). Fix: dequant the full [T_kv, kv_dim] K and V ONCE   //
//  per (layer, chunk-prime call) into a bf16 workspace, then run        //
//  fa_prefill_qw (below) over it. EXACTNESS: the workspace stores       //
//  __float2bfloat16(dq_*_elem(...)) — the IDENTICAL value fa_prefill_q  //
//  writes to smem — so the MMA sees bit-identical inputs and the output //
//  is bit-identical (kernel_check pins ws-vs-inline bitdiff=0).         //
//  One thread per element, grid-stride over K elems then V elems.      //
// ===================================================================== //
extern "C" __global__ void fa_dequant_kv_ws_bf16(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        __nv_bfloat16* __restrict__ Kw, __nv_bfloat16* __restrict__ Vw,
        int kv_dim_k, int kv_dim_v, int t_kv,
        long k_tok_bytes, long v_tok_bytes)
{
    const long nk = (long)t_kv * kv_dim_k;
    const long nv = (long)t_kv * kv_dim_v;
    const long total = nk + nv;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        if (idx < nk) {
            const long t = idx / kv_dim_k; const int e = (int)(idx % kv_dim_k);
            Kw[idx] = __float2bfloat16(DQ_K_ELEM(K, t, k_tok_bytes, e));
        } else {
            const long j = idx - nk;
            const long t = j / kv_dim_v; const int e = (int)(j % kv_dim_v);
            Vw[j] = __float2bfloat16(DQ_V_ELEM(V, t, v_tok_bytes, e));
        }
    }
}

// ===================================================================== //
//  KERNEL 1b-qw : fa_prefill_qw  (bf16-workspace prefill twin)          //
//  VERBATIM copy of fa_prefill_q except the stage-to-smem loop reads    //
//  the pre-dequanted bf16 workspace (plain copy, no dequant ALU, no     //
//  scattered 34B/24B block reads). Workspace element (t, kv_head, d) at //
//  t*kv_dim + kv_head*head_dim + d — same element order as the cache.   //
//  All MMA / softmax / PV code is byte-identical to fa_prefill_q, and   //
//  the staged bf16 values are bit-identical (see fa_dequant_kv_ws_bf16) //
//  -> bit-identical O. Keep the two kernels in lockstep on any edit.    //
// ===================================================================== //
template<int HD>
static __device__ __forceinline__ void fa_prefill_qw_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head    = blockIdx.y;           // grid.y = n_head (full SM subscription)
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BLOCK_Q;
    const int qrow_base = q_base + warp*M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* sV = sK + BK*HEAD_DIM;
    __nv_bfloat16* sP = sV + BK*HEAD_DIM;
    float* sS = (float*)(sP + BLOCK_Q*BK);
    float* sM = sS + BLOCK_Q*BK;
    float* sL = sM + BLOCK_Q;
    __nv_bfloat16* sPw = sP + warp*M_ROWS*BK;
    float* sSw = sS + warp*M_ROWS*BK;
    float* sMw = sM + warp*M_ROWS;
    float* sLw = sL + warp*M_ROWS;
    __nv_bfloat16* sQstage = sK + warp*M_ROWS*HEAD_DIM;
    (void)sSw; (void)sMw;

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;

        ATile Qf[HD_KTILES];
        load_q_frags<HD>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        __syncthreads();

        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        // Edge 5a: register-resident online-softmax state (no sSw round-trip).
        float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
        const int r_lo = lane / 4;          // CTile get_i(l=0,1)
        const int r_hi = r_lo + 8;          // CTile get_i(l=2,3)
        const int c0   = (lane % 4) * 2;    // CTile get_j base for this lane

        const size_t kv_off = (size_t)kv_head * head_dim;

        for (int k0 = 0; k0 < T_kv; k0 += BK) {
            const int nk = min(BK, T_kv - k0);
            const int q_pos_max = (T_kv - T) + q_base + (BLOCK_Q - 1);
            if (causal_i && k0 > q_pos_max) break;

            // ---- stage K,V tile to smem: VECTORIZED bf16 COPY from the workspace ----
            // 16B (8xbf16) uint4 copies — pure byte copy, bit-identical smem contents to the
            // scalar loop. Alignment: workspace rows are kv_dim*2B (512B-mult) apart, kv_off*2
            // is 512B-mult, dv*16 is 16B-mult; smem rows are HEAD_DIM*2=512B apart. All 16B-ok.
            const int bt = warp*WARP_SZ + lane;
            {
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int i = bt; i < BK*(HEAD_DIM/8); i += N_WARPS*WARP_SZ) {
                    int kk = i / (HEAD_DIM/8), dv = i % (HEAD_DIM/8);
                    uint4 kx = z4, vx = z4;
                    if (kk < nk) {
                        kx = *(const uint4*)(Kw + (size_t)(k0 + kk) * kv_dim_k + kv_off + dv*8);
                        vx = *(const uint4*)(Vw + (size_t)(k0 + kk) * kv_dim_v + kv_off + dv*8);
                    }
                    *(uint4*)(sK + kk*HEAD_DIM + dv*8) = kx;
                    *(uint4*)(sV + kk*HEAD_DIM + dv*8) = vx;
                }
            }
            __syncthreads();

            // ---- GEMM0: QK^T -> 4 score CTiles HELD IN REGISTERS (no sSw write) ----
            CTile Sc[BK/N_KEYS];
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
            for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                CTile C0, C1;
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Kt;
                    ld_A(Kt, sK + kg*HEAD_DIM + kt*K_STEP, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qf[kt], Blo);
                    mma_bf16(C1, Qf[kt], Bhi);
                }
                Sc[kg/N_KEYS + 0] = C0;
                Sc[kg/N_KEYS + 1] = C1;
            }

            // ---- SOFTMAX on registers (scale + causal mask + 4-lane reduce) ----
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int col = g*N_KEYS + c0 + (l & 1);
                    int row = (l < 2) ? r_lo : r_hi;
                    int q_pos = q_pos0w + row;
                    float s = Sc[g].x[l] * scale;
                    if (col >= nk) s = NEG_INF;
                    if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                    Sc[g].x[l] = s;
                    if (l < 2) s_tile_max_lo = fmaxf(s_tile_max_lo, s);
                    else       s_tile_max_hi = fmaxf(s_tile_max_hi, s);
                }
            }
            s_tile_max_lo = row_max4(s_tile_max_lo);
            s_tile_max_hi = row_max4(s_tile_max_hi);

            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
            float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
            float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);

            float l_part_lo = 0.0f, l_part_hi = 0.0f;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float mn = (l < 2) ? m_new_lo : m_new_hi;
                    float s  = Sc[g].x[l];
                    float p  = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
                    Sc[g].x[l] = p;
                    if (l < 2) l_part_lo += p; else l_part_hi += p;
                }
            }
            l_part_lo = row_sum4(l_part_lo);
            l_part_hi = row_sum4(l_part_hi);
            l_lo = l_lo * alpha_lo + l_part_lo;
            l_hi = l_hi * alpha_hi + l_part_hi;
            m_lo = m_new_lo; m_hi = m_new_hi;

            // ---- write P to sPw (MANDATORY for PV's A-operand ldmatrix layout) ----
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                sPw[r_lo*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[0]);
                sPw[r_lo*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[1]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[2]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[3]);
            }
            __syncwarp();

            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
                O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
            }

            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    ld_A(A, sPw + kk, BK/2);
                    ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            __syncthreads();
        }

        if (c0 == 0) { sLw[r_lo] = l_lo; sLw[r_hi] = l_hi; }
        __syncwarp();

        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    float linv = (sLw[r] > 0.0f) ? (1.0f / sLw[r]) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_qw(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    fa_prefill_qw_body<256>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv,
                            scale, causal, kv_dim_k, kv_dim_v);
}
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 2) fa_prefill_qw_hd128(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    fa_prefill_qw_body<128>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv,
                            scale, causal, kv_dim_k, kv_dim_v);
}

// ===================================================================== //
//  KERNEL 1b-qwdb : fa_prefill_qw_db  (cp.async double-buffered twin)   //
//  fa_prefill_qw with the K/V workspace staging DOUBLE-BUFFERED via     //
//  cp.async: tile n+1's L2->smem copy is issued before tile n's compute //
//  so the staging latency hides behind the MMA pipe (ncu on the single- //
//  buffer twin: mem 66% / compute 15% / DRAM 0.6% — staging-stalled).   //
//  Costs a second sK+sV pair (+32KB smem -> 1 CTA/SM vs 2); the A/B     //
//  measurement arbitrates the default. EXACT: staging is a pure byte    //
//  copy and the compute code is byte-identical to fa_prefill_qw ->      //
//  bit-identical O (kernel_check pins db-vs-inline bitdiff=0).          //
// ===================================================================== //
static __device__ __forceinline__ void cp_async_16(__nv_bfloat16* smem_dst, const __nv_bfloat16* gsrc) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(gsrc));
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
static __device__ __forceinline__ void cp_async_wait_1() { asm volatile("cp.async.wait_group 1;\n"); }
static __device__ __forceinline__ void cp_async_wait_0() { asm volatile("cp.async.wait_group 0;\n"); }

// Issue one KV tile's staging into buffer `sKb`/`sVb` (cp.async 16B lines; tail rows
// past nk zero-filled with plain stores — visible after the same __syncthreads).
template<int HD>
static __device__ __forceinline__ void stage_kv_tile_async(
        __nv_bfloat16* sKb, __nv_bfloat16* sVb,
        const __nv_bfloat16* __restrict__ Kw, const __nv_bfloat16* __restrict__ Vw,
        int k0, int nk, int kv_dim_k, int kv_dim_v, size_t kv_off, int bt)
{
    constexpr int HEAD_DIM = HD;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    for (int i = bt; i < BK*(HEAD_DIM/8); i += N_WARPS*WARP_SZ) {
        int kk = i / (HEAD_DIM/8), dv = i % (HEAD_DIM/8);
        if (kk < nk) {
            cp_async_16(sKb + kk*HEAD_DIM + dv*8, Kw + (size_t)(k0 + kk) * kv_dim_k + kv_off + dv*8);
            cp_async_16(sVb + kk*HEAD_DIM + dv*8, Vw + (size_t)(k0 + kk) * kv_dim_v + kv_off + dv*8);
        } else {
            *(uint4*)(sKb + kk*HEAD_DIM + dv*8) = z4;
            *(uint4*)(sVb + kk*HEAD_DIM + dv*8) = z4;
        }
    }
    cp_async_commit();
}

template<int HD>
static __device__ __forceinline__ void fa_prefill_qw_db_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    constexpr int HEAD_DIM  = HD;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head    = blockIdx.y;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BLOCK_Q;
    const int qrow_base = q_base + warp*M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    // smem: DOUBLE K/V tile buffers + sP + sM/sL (no sS — register softmax needs no
    // score staging; sLw is the only cross-warp slot).
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK0 = (__nv_bfloat16*)smem_raw;                // BK*HEAD_DIM
    __nv_bfloat16* sK1 = sK0 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sV0 = sK1 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sV1 = sV0 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sP  = sV1 + BK*HEAD_DIM;                      // BLOCK_Q*BK
    float* sL = (float*)(sP + BLOCK_Q*BK);                        // BLOCK_Q f32
    __nv_bfloat16* sPw = sP + warp*M_ROWS*BK;
    float* sLw = sL + warp*M_ROWS;
    // transient Q staging: sK0∪sK1 = 32KB = 4 warps x 16*HEAD_DIM bf16, one slab per warp.
    __nv_bfloat16* sQstage = sK0 + warp*M_ROWS*HEAD_DIM;

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;

        ATile Qf[HD_KTILES];
        load_q_frags<HD>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        __syncthreads();   // all warps done with sK0∪sK1 before prefetch overwrites

        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
        const int r_lo = lane / 4;
        const int r_hi = r_lo + 8;
        const int c0   = (lane % 4) * 2;

        const size_t kv_off = (size_t)kv_head * head_dim;
        const int bt = warp*WARP_SZ + lane;

        // tile count, folding the causal early-out into the bound (same tiles as the
        // single-buffer twin's `break`).
        const int q_pos_max = (T_kv - T) + q_base + (BLOCK_Q - 1);
        int nt = (T_kv + BK - 1) / BK;
        if (causal_i) { int ntc = q_pos_max / BK + 1; nt = min(nt, ntc); }

        if (nt > 0)
            stage_kv_tile_async<HD>(sK0, sV0, Kw, Vw, 0, min(BK, T_kv), kv_dim_k, kv_dim_v, kv_off, bt);

        for (int ti = 0; ti < nt; ++ti) {
            const int k0 = ti * BK;
            const int nk = min(BK, T_kv - k0);
            __nv_bfloat16* sK = (ti & 1) ? sK1 : sK0;
            __nv_bfloat16* sV = (ti & 1) ? sV1 : sV0;
            // prefetch tile ti+1 into the OTHER buffer (its compute finished last iter)
            if (ti + 1 < nt) {
                const int k1 = (ti + 1) * BK;
                stage_kv_tile_async<HD>((ti & 1) ? sK0 : sK1, (ti & 1) ? sV0 : sV1,
                                    Kw, Vw, k1, min(BK, T_kv - k1), kv_dim_k, kv_dim_v, kv_off, bt);
                cp_async_wait_1();   // tile ti's group done; ti+1 may still be in flight
            } else {
                cp_async_wait_0();
            }
            __syncthreads();

            // ---- GEMM0: QK^T -> 4 score CTiles HELD IN REGISTERS ----
            CTile Sc[BK/N_KEYS];
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
            for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                CTile C0, C1;
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Kt;
                    ld_A(Kt, sK + kg*HEAD_DIM + kt*K_STEP, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qf[kt], Blo);
                    mma_bf16(C1, Qf[kt], Bhi);
                }
                Sc[kg/N_KEYS + 0] = C0;
                Sc[kg/N_KEYS + 1] = C1;
            }

            // ---- SOFTMAX on registers (scale + causal mask + 4-lane reduce) ----
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int col = g*N_KEYS + c0 + (l & 1);
                    int row = (l < 2) ? r_lo : r_hi;
                    int q_pos = q_pos0w + row;
                    float s = Sc[g].x[l] * scale;
                    if (col >= nk) s = NEG_INF;
                    if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                    Sc[g].x[l] = s;
                    if (l < 2) s_tile_max_lo = fmaxf(s_tile_max_lo, s);
                    else       s_tile_max_hi = fmaxf(s_tile_max_hi, s);
                }
            }
            s_tile_max_lo = row_max4(s_tile_max_lo);
            s_tile_max_hi = row_max4(s_tile_max_hi);

            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
            float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
            float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);

            float l_part_lo = 0.0f, l_part_hi = 0.0f;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float mn = (l < 2) ? m_new_lo : m_new_hi;
                    float s  = Sc[g].x[l];
                    float p  = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
                    Sc[g].x[l] = p;
                    if (l < 2) l_part_lo += p; else l_part_hi += p;
                }
            }
            l_part_lo = row_sum4(l_part_lo);
            l_part_hi = row_sum4(l_part_hi);
            l_lo = l_lo * alpha_lo + l_part_lo;
            l_hi = l_hi * alpha_hi + l_part_hi;
            m_lo = m_new_lo; m_hi = m_new_hi;

            // ---- write P to sPw (MANDATORY for PV's A-operand ldmatrix layout) ----
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                sPw[r_lo*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[0]);
                sPw[r_lo*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[1]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 0] = __float2bfloat16(Sc[g].x[2]);
                sPw[r_hi*BK + g*N_KEYS + c0 + 1] = __float2bfloat16(Sc[g].x[3]);
            }
            __syncwarp();

            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
                O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
            }

            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    ld_A(A, sPw + kk, BK/2);
                    ld_A_trans(Bt, sV + kk*HEAD_DIM + d0, HEAD_DIM/2);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            __syncthreads();   // compute on this buffer done before it is re-prefetched
        }

        if (c0 == 0) { sLw[r_lo] = l_lo; sLw[r_hi] = l_hi; }
        __syncwarp();

        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    float linv = (sLw[r] > 0.0f) ? (1.0f / sLw[r]) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 1) fa_prefill_qw_db(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    fa_prefill_qw_db_body<256>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv,
                               scale, causal, kv_dim_k, kv_dim_v);
}
extern "C" __global__ void __launch_bounds__(N_WARPS*WARP_SZ, 1) fa_prefill_qw_db_hd128(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    fa_prefill_qw_db_body<128>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv,
                               scale, causal, kv_dim_k, kv_dim_v);
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
        float ktv = (tid < head_dim) ? DQ_K_ELEM(K, t, k_tok_bytes, kidx) : 0.0f;
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
        float vtv = (tid < head_dim) ? DQ_V_ELEM(V, t, v_tok_bytes, vidx) : 0.0f;
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

    // REGISTER-DEQUANT REWRITE (2026-07-03, the fattn-vec structural port): no smem staging, no
    // block syncs, no bf16 round-trip. Each warp walks its split's keys directly; lane owns dims
    // {lane, lane+32, ...} — its K element of dim-block i is byte `lane` of q8_0 block
    // (kv_head*hd/32 + i), so the 32 lanes read 32 CONSECUTIVE bytes per block = coalesced. The
    // 4 GQA warps re-read the same KV bytes; L2 serves the reuse (KV @2048 ctx = 2.2MB << 64MB L2)
    // — the old cross-warp smem broadcast bought nothing and cost 2 __syncthreads per 32-key tile
    // + a full bf16 smem round-trip (measured 126us vs the reference engine's 10.4us structure).
    // Same per-lane ascending-i accumulation + same warp butterfly as before; only numeric change
    // is REMOVING the bf16 rounding of dequanted K/V (more accurate; gate battery is the arbiter).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;      // first q8_0/q5_1 block of this kv head
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to the old smem-staged path (which stored
                    // dequanted K as bf16). Pure ALU on a DRAM-bound kernel — keeps every gate
                    // (incl. run-spec exactness) exactly where the validated kernel had it.
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);     // every lane gets the full QK score (already *scale)

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
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

// hd-512 twin (gemma4 globals): FA_DEC_MAX_DPL16=16 register accumulators (dpl = 512/32).
// Body = fa_decode_vec_q VERBATIM modulo the ceiling.
#define FA_DEC_MAX_DPL16 16
extern "C" __global__ void fa_decode_vec_q_dpl16(
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

    // stage this warp's Q row (one Q head, head_dim) into registers, PRE-SCALED by `scale`.
    // lane owns dims { lane, lane+32, ..., lane+32*(dpl-1) }.
    float q_reg[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    // per-warp online-softmax state + register accumulator (acc[i] is dim lane+32*i).
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) acc[i] = 0.0f;

    // REGISTER-DEQUANT REWRITE (2026-07-03, the fattn-vec structural port): no smem staging, no
    // block syncs, no bf16 round-trip. Each warp walks its split's keys directly; lane owns dims
    // {lane, lane+32, ...} — its K element of dim-block i is byte `lane` of q8_0 block
    // (kv_head*hd/32 + i), so the 32 lanes read 32 CONSECUTIVE bytes per block = coalesced. The
    // 4 GQA warps re-read the same KV bytes; L2 serves the reuse (KV @2048 ctx = 2.2MB << 64MB L2)
    // — the old cross-warp smem broadcast bought nothing and cost 2 __syncthreads per 32-key tile
    // + a full bf16 smem round-trip (measured 126us vs the reference engine's 10.4us structure).
    // Same per-lane ascending-i accumulation + same warp butterfly as before; only numeric change
    // is REMOVING the bf16 rounding of dequanted K/V (more accurate; gate battery is the arbiter).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;      // first q8_0/q5_1 block of this kv head
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to the old smem-staged path (which stored
                    // dequanted K as bf16). Pure ALU on a DRAM-bound kernel — keeps every gate
                    // (incl. run-spec exactness) exactly where the validated kernel had it.
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);     // every lane gets the full QK score (already *scale)

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    // write this Q head's split partial (UNNORMALIZED acc, + m_i/l_i for the combine).
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}


extern "C" __global__ void fa_decode_vec_q_dpl16_dc(
        const float* __restrict__ Q,    // [head_dim, n_head, 1]
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [n_head, n_splits, head_dim]
        float* __restrict__ partM,      // [n_head, n_splits]
        float* __restrict__ partL,      // [n_head, n_splits]
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_dev,
        float scale, int n_splits,
        long k_tok_bytes, long v_tok_bytes)
{
    const int T_kv    = t_kv_dev[0];             // device-resident sequence length
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

    // stage this warp's Q row (one Q head, head_dim) into registers, PRE-SCALED by `scale`.
    // lane owns dims { lane, lane+32, ..., lane+32*(dpl-1) }.
    float q_reg[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)0 * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    // per-warp online-softmax state + register accumulator (acc[i] is dim lane+32*i).
    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) acc[i] = 0.0f;

    // REGISTER-DEQUANT REWRITE (2026-07-03, the fattn-vec structural port): no smem staging, no
    // block syncs, no bf16 round-trip. Each warp walks its split's keys directly; lane owns dims
    // {lane, lane+32, ...} — its K element of dim-block i is byte `lane` of q8_0 block
    // (kv_head*hd/32 + i), so the 32 lanes read 32 CONSECUTIVE bytes per block = coalesced. The
    // 4 GQA warps re-read the same KV bytes; L2 serves the reuse (KV @2048 ctx = 2.2MB << 64MB L2)
    // — the old cross-warp smem broadcast bought nothing and cost 2 __syncthreads per 32-key tile
    // + a full bf16 smem round-trip (measured 126us vs the reference engine's 10.4us structure).
    // Same per-lane ascending-i accumulation + same warp butterfly as before; only numeric change
    // is REMOVING the bf16 rounding of dequanted K/V (more accurate; gate battery is the arbiter).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;      // first q8_0/q5_1 block of this kv head
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to the old smem-staged path (which stored
                    // dequanted K as bf16). Pure ALU on a DRAM-bound kernel — keeps every gate
                    // (incl. run-spec exactness) exactly where the validated kernel had it.
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);     // every lane gets the full QK score (already *scale)

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    // write this Q head's split partial (UNNORMALIZED acc, + m_i/l_i for the combine).
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}



// ===================================================================== //
//  KERNEL 2b-smem : fa_decode_vec_q_smem (LONG-CTX twin, 2026-07-05)     //
//  The pre-register-rewrite smem-broadcast body, resurrected for DEEP    //
//  ctx: the register path's premise ("L2 serves the GQA reuse") holds   //
//  at 2-8k (layer KV ~2-8MB) but DIES at 40k (~37MB/layer >> L2 share)  //
//  where the 4 GQA warps' re-read = 4x DRAM traffic. This twin dequants  //
//  each KV tile ONCE into smem per block. BIT-IDENTICAL per (token,     //
//  split) to the register path: same bf16 round-trip of dequanted K/V,  //
//  same ascending-i per-lane accumulation, same warp butterfly, same    //
//  partial layout -> same combine. Dispatch: t_kv >= BW24_FA_SMEM_TKV   //
//  (default 16384; 0 = never).                                          //
// ===================================================================== //
extern "C" __global__ void fa_decode_vec_q_smem(
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
            sK[idx] = __float2bfloat16(DQ_K_ELEM(K, (long)(t0 + j), k_tok_bytes, eidx));
            sV[idx] = __float2bfloat16(DQ_V_ELEM(V, (long)(t0 + j), v_tok_bytes, eidx));
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

// ===================================================================== //
//  KERNEL 2c-smem : fa_decode_vec_q_rows_smem (deep-ctx verify twin)     //
//  Rows kernel with the smem KV-tile broadcast (see fa_decode_vec_q_smem //
//  rationale — the register path's L2-reuse premise dies at deep ctx;    //
//  the verify multiplies the loss by T rows). Bit-identical per          //
//  (row, token, split); host dispatches above BW24_FA_SMEM_TKV.          //
// ===================================================================== //
extern "C" __global__ void fa_decode_vec_q_rows_smem(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, int t_kv_base,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base + r + 1;      // this row's causal key bound
    const int n_splits = (T_kv + split_keys - 1) / split_keys;  // == host fa_split_keys sizing
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    // SMEM-BROADCAST walk (deep-ctx twin of the register walk below in _rows): dequant each
    // 32-key tile ONCE per CTA into smem, all gqa warps consume it. BIT-IDENTICAL per (token,
    // split) to the register path: same bf16 round-trip of dequanted K/V, same ascending-i
    // accumulation, same warp butterfly (the smem value IS the bf16-rounded dequant the register
    // path computes inline). Dispatched by the host above BW24_FA_SMEM_TKV, mirroring fa_decode.
    extern __shared__ __nv_bfloat16 ssh_rows[];   // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_rows;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
            const int nt = min(FA_DEC_TILE, t_hi - t0);
            for (int idx = bt; idx < nt * head_dim; idx += bsz) {
                int j = idx / head_dim;
                int d = idx - j * head_dim;
                const uint8_t* kb = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + (d >> 5)) * K_BLK_B;
                sK[idx] = __float2bfloat16(dq_K_lane(kb, d & 31));
                const uint8_t* vb = V + (size_t)(t0 + j) * v_tok_bytes + (size_t)(kblk0 + (d >> 5)) * V_BLK_B;
                sV[idx] = __float2bfloat16(dq_V_lane(vb, d & 31));
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
    }
    // ---- ORIGINAL register walk removed in this twin; tail below unchanged ----
    if (false) {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to fa_decode_vec_q (see comment there).
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// ===================================================================== //
//  KERNEL 2b-v2 : fa_decode_vec_q_v2  (FAVENDOR lane, 2026-07-08)        //
//  llama.cpp fattn-vec MECHANISM vendored into OUR frame (split          //
//  partition, partial layout, combine kernel all unchanged).             //
//                                                                        //
//  What is vendored (ggml/src/ggml-cuda/fattn-vec.cuh, flash_attn_ext_   //
//  vec<D,1,q8_0,q5_1>): TILE-BATCHED online softmax. llama's warp        //
//  processes a tile of keys with INDEPENDENT row dots (lane j keeps row  //
//  j's score, every lane tracks the tile max from the butterfly result), //
//  then does the softmax bookkeeping ONCE per tile: one m update, one    //
//  alpha, ONE VKQ rescale — vs our per-key serial chain (per key: fmaxf  //
//  + 2 exp2f + dpl-FMA rescale, each iteration data-dependent on the     //
//  last). At d6257/sp64 that chain is 64 deep per warp; llama's is 2     //
//  deep per 32-key tile. llama also streams quantized K/V bytes straight //
//  from global into registers (NO smem staging, NO __syncthreads — our   //
//  smem twin pays 2 block syncs per 32-key tile across 8 warps).         //
//                                                                        //
//  What is KEPT ours (the frame): grid=(n_head_kv, n_splits), block=     //
//  (32, gqa); contiguous [t_lo,t_hi) split partition (llama strides      //
//  interleaved); per-lane dim ownership {lane, lane+32,...}; the per-row //
//  DOT accumulation order (ascending dim-block i, bf16 round-trip of     //
//  the dequanted element, full 32-lane butterfly) — so each individual   //
//  ROW SCORE is bit-identical to fa_decode_vec_q's; ascending-t V walk;  //
//  f32 accumulators (llama uses half2 — not vendored, exactness first);  //
//  [head][split] partial layout -> the UNCHANGED fa_decode_combine_f32.  //
//                                                                        //
//  NUMERIC CONFIG: the tile-level regrouping changes WHEN alpha rescales //
//  land (exp(score - tile_max) vs exp(score - running_max)) => partials  //
//  differ in FP order from fa_decode_vec_q => BW24_FA_V2=1 is its own    //
//  numeric config with its own argmax baseline (env-flagged, default     //
//  OFF, never silent). Within the flag it is fully deterministic: the    //
//  rows twin below calls the SAME walk body -> rows-vs-loop bitdiff==0   //
//  (kernel-check), run-gen argmax + run-spec self-consistency arbitrate. //
// ===================================================================== //

// REVISION 2 (same day): the first v2 cut vendored llama's REGISTER STREAMING too
// (each warp re-reads quantized K/V straight from global, no smem) — measured 2x
// WORSE at depth (125.7 vs 65.1 us at d6257 on the fa_v2_bench probe): with gqa=8
// warps per CTA the 8x redundant global walk loses to our stage-once smem broadcast
// even from L1/L2. KEPT ours: the smem KV-tile broadcast (dequant once per CTA).
// VENDORED: (a) the tile-batched online softmax, (b) llama's WIDE-LOAD dequant shape
// for the staging phase — one thread dequants one whole 32-elem quant BLOCK from
// 4-byte int loads (llama reads q4/q8 quants as ints and unpacks with shifts;
// dq_*_elem re-loads d/m/qh per ELEMENT and reads qs one BYTE at a time — ~8x the
// load instructions for the same bytes). The staged bf16 VALUES are bit-identical
// to dq_q8_0_elem/dq_q5_1_elem (same per-element math on the same bytes).

// The shared per-warp split walk over the staged smem tile (called by BOTH the T=1
// kernel and the rows twin => per-(row,split) bit identity by construction).
// Block-cooperative: stages [t0, t0+nt) into sK/sV (bit-identical values to the
// smem twin's staging), then each warp runs the vendored tile-batched softmax.
static __device__ __forceinline__ void fa_dec_v2_walk(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        __nv_bfloat16* sK, __nv_bfloat16* sV, int bt, int bsz,
        const float* q_reg, int dpl, int lane, int head_dim,
        int t_lo, int t_hi, int kblk0, long k_tok_bytes, long v_tok_bytes,
        float& m_i, float& l_i, float* acc)
{
    const int blocks_per_key = head_dim >> 5;    // 32-elem quant blocks per key (this kv head)
    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);

        // ---- Phase A (staging, WIDE LOADS): one thread = one 32-elem quant block.
        //      Values bit-identical to dq_q8_0_elem / dq_q5_1_elem. ----
        for (int b = bt; b < nt * blocks_per_key; b += bsz) {
            const int j     = b / blocks_per_key;        // key within tile
            const int blk_i = b - j * blocks_per_key;    // block within key
            // K block: q8_0 = f16 d + 32x int8, read qs as 8 aligned-4B words.
            {
                const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes
                                       + (size_t)(kblk0 + blk_i) * 34;
                const float d = __half2float(*(const half*)blk);
                __nv_bfloat16* out = sK + (size_t)j * head_dim + (blk_i << 5);
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    int v; memcpy(&v, blk + 2 + 4 * w, 4);   // 34B stride -> unaligned-safe
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int8_t q = (int8_t)(v >> (8 * l));
                        out[4 * w + l] = __float2bfloat16(d * (float)q);
                    }
                }
            }
            // V block: q5_1 = f16 d + f16 m + u32 qh + 16B nibbles, read qs as 4x 4B words.
            {
                const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                       + (size_t)(kblk0 + blk_i) * 24;
                const float d = __half2float(*(const half*)blk);
                const float m = __half2float(*(const half*)(blk + 2));
                uint32_t qh; memcpy(&qh, blk + 4, 4);
                uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
                __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    const int byte = (e < 16) ? e : e - 16;
                    const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                    const int lo   = (e < 16) ? (nib & 0x0F) : (nib >> 4);
                    const int q5   = lo | (int)(((qh >> e) & 1u) << 4);
                    out[e] = __float2bfloat16(d * (float)q5 + m);
                }
            }
        }
        __syncthreads();

        // ---- Phase B1 (vendored): nt INDEPENDENT row dots from smem. Lane j keeps
        //      row j's score; every lane tracks the tile max (the butterfly gives
        //      every lane the full sum). Per-row dot order (ascending dim-block i,
        //      full 32-lane butterfly) = fa_decode_vec_q_smem exactly. ----
        float my_score = NEG_INF;          // this lane's key score (key t0+lane)
        float tile_max = m_i;              // seeded with the running max (llama KQ_max_new)
        #pragma unroll 4
        for (int j = 0; j < nt; ++j) {
            const __nv_bfloat16* kj = sK + (size_t)j * head_dim;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) part += q_reg[i] * __bfloat162float(kj[lane + (i << 5)]);
            float score = warp_reduce_sum(part);   // every lane gets the full QK score
            if (lane == j) my_score = score;
            tile_max = fmaxf(tile_max, score);
        }

        // ---- Phase B2 (vendored): softmax bookkeeping ONCE per tile ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;          // ONE rescale per tile (was per key)
        }
        m_i = m_new;

        // ---- Phase B3: ascending-t V accumulation from smem, p broadcast by ONE
        //      shfl/key (llama round-trips p through smem; the shfl is the 1-warp
        //      equivalent). V element order = fa_decode_vec_q_smem exactly. ----
        #pragma unroll 2
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            const __nv_bfloat16* vj = sV + (size_t)j * head_dim;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i)
                if (i < dpl) acc[i] += p * __bfloat162float(vj[lane + (i << 5)]);
        }
        __syncthreads();   // tile fully consumed before the next staging overwrites sK/sV
    }
}

// T=1 decode twin. Same signature/grid/block/partial-layout as fa_decode_vec_q.
extern "C" __global__ void fa_decode_vec_q_rows_smem_w(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;      // this row's causal key bound
    // WINDOWED twin (gemma R6): every row attends exactly `window` keys; split geometry/key
    // order mirror the decode window-VIEW chain (start+j absolute; host gates full-window rows).
    const int start    = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa     = n_head / n_head_kv;
    const int wy      = threadIdx.y;
    const int lane    = threadIdx.x;
    if (wy >= gqa) return;
    const int head    = kv_head * gqa + wy;
    const int dpl     = head_dim >> 5;

    const int per  = (window + n_splits - 1) / n_splits;
    const int t_lo = start + split * per;
    const int t_hi = start + min(window, split * per + per);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    // SMEM-BROADCAST walk (deep-ctx twin of the register walk below in _rows): dequant each
    // 32-key tile ONCE per CTA into smem, all gqa warps consume it. BIT-IDENTICAL per (token,
    // split) to the register path: same bf16 round-trip of dequanted K/V, same ascending-i
    // accumulation, same warp butterfly (the smem value IS the bf16-rounded dequant the register
    // path computes inline). Dispatched by the host above BW24_FA_SMEM_TKV, mirroring fa_decode.
    extern __shared__ __nv_bfloat16 ssh_rows[];   // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_rows;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
            const int nt = min(FA_DEC_TILE, t_hi - t0);
            for (int idx = bt; idx < nt * head_dim; idx += bsz) {
                int j = idx / head_dim;
                int d = idx - j * head_dim;
                const uint8_t* kb = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + (d >> 5)) * K_BLK_B;
                sK[idx] = __float2bfloat16(dq_K_lane(kb, d & 31));
                const uint8_t* vb = V + (size_t)(t0 + j) * v_tok_bytes + (size_t)(kblk0 + (d >> 5)) * V_BLK_B;
                sV[idx] = __float2bfloat16(dq_V_lane(vb, d & 31));
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
    }
    // ---- ORIGINAL register walk removed in this twin; tail below unchanged ----
    if (false) {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to fa_decode_vec_q (see comment there).
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// ===================================================================== //
//  KERNEL 2b-v2 : fa_decode_vec_q_v2  (FAVENDOR lane, 2026-07-08)        //
//  llama.cpp fattn-vec MECHANISM vendored into OUR frame (split          //
//  partition, partial layout, combine kernel all unchanged).             //
//                                                                        //
//  What is vendored (ggml/src/ggml-cuda/fattn-vec.cuh, flash_attn_ext_   //
//  vec<D,1,q8_0,q5_1>): TILE-BATCHED online softmax. llama's warp        //
//  processes a tile of keys with INDEPENDENT row dots (lane j keeps row  //
//  j's score, every lane tracks the tile max from the butterfly result), //
//  then does the softmax bookkeeping ONCE per tile: one m update, one    //
//  alpha, ONE VKQ rescale — vs our per-key serial chain (per key: fmaxf  //
//  + 2 exp2f + dpl-FMA rescale, each iteration data-dependent on the     //
//  last). At d6257/sp64 that chain is 64 deep per warp; llama's is 2     //
//  deep per 32-key tile. llama also streams quantized K/V bytes straight //
//  from global into registers (NO smem staging, NO __syncthreads — our   //
//  smem twin pays 2 block syncs per 32-key tile across 8 warps).         //
//                                                                        //
//  What is KEPT ours (the frame): grid=(n_head_kv, n_splits), block=     //
//  (32, gqa); contiguous [t_lo,t_hi) split partition (llama strides      //
//  interleaved); per-lane dim ownership {lane, lane+32,...}; the per-row //
//  DOT accumulation order (ascending dim-block i, bf16 round-trip of     //
//  the dequanted element, full 32-lane butterfly) — so each individual   //
//  ROW SCORE is bit-identical to fa_decode_vec_q's; ascending-t V walk;  //
//  f32 accumulators (llama uses half2 — not vendored, exactness first);  //
//  [head][split] partial layout -> the UNCHANGED fa_decode_combine_f32.  //
//                                                                        //
//  NUMERIC CONFIG: the tile-level regrouping changes WHEN alpha rescales //
//  land (exp(score - tile_max) vs exp(score - running_max)) => partials  //
//  differ in FP order from fa_decode_vec_q => BW24_FA_V2=1 is its own    //
//  numeric config with its own argmax baseline (env-flagged, default     //
//  OFF, never silent). Within the flag it is fully deterministic: the    //
//  rows twin below calls the SAME walk body -> rows-vs-loop bitdiff==0   //
//  (kernel-check), run-gen argmax + run-spec self-consistency arbitrate. //
// ===================================================================== //

// REVISION 2 (same day): the first v2 cut vendored llama's REGISTER STREAMING too
// (each warp re-reads quantized K/V straight from global, no smem) — measured 2x
// WORSE at depth (125.7 vs 65.1 us at d6257 on the fa_v2_bench probe): with gqa=8
// warps per CTA the 8x redundant global walk loses to our stage-once smem broadcast
// even from L1/L2. KEPT ours: the smem KV-tile broadcast (dequant once per CTA).
// VENDORED: (a) the tile-batched online softmax, (b) llama's WIDE-LOAD dequant shape
// for the staging phase — one thread dequants one whole 32-elem quant BLOCK from
// 4-byte int loads (llama reads q4/q8 quants as ints and unpacks with shifts;
// dq_*_elem re-loads d/m/qh per ELEMENT and reads qs one BYTE at a time — ~8x the
// load instructions for the same bytes). The staged bf16 VALUES are bit-identical
// to dq_q8_0_elem/dq_q5_1_elem (same per-element math on the same bytes).

// The shared per-warp split walk over the staged smem tile (called by BOTH the T=1
// kernel and the rows twin => per-(row,split) bit identity by construction).
// Block-cooperative: stages [t0, t0+nt) into sK/sV (bit-identical values to the
extern "C" __global__ void fa_decode_vec_q_v2(
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

    extern __shared__ __nv_bfloat16 ssh_v2[];        // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_v2;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v2_walk(K, V, sK, sV, bt, bsz, q_reg, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

static __device__ __forceinline__ void fa_rows_v3_body(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int r);

// ROUND-STREAM stage (c) 2: rows FA with the causal base from a DEVICE counter (pre-issued
// verify: t_kv_base = len_d value at execution time, unknown at issue). Same body.
extern "C" __global__ void fa_decode_vec_q_rows_v3_dc(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r    = blockIdx.z;
    const int T_kv = t_kv_base_dev[0] + r + 1;
    fa_rows_v3_body(Q, K, V, partO, partM, partL, head_dim, n_head, n_head_kv, T_kv,
                    scale, n_splits_max, split_keys, k_tok_bytes, v_tok_bytes, r);
}

// Multi-row (spec-verify) twin: grid.z = query row, causal bound per row —
// same frame as fa_decode_vec_q_rows/_smem, same walk body as the T=1 twin
// above (the spec-exactness law: eager decode and verify must never diverge).
extern "C" __global__ void fa_decode_vec_q_rows_v2(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, int t_kv_base,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base + r + 1;      // this row's causal key bound
    const int n_splits = (T_kv + split_keys - 1) / split_keys;  // == host fa_split_keys sizing
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    extern __shared__ __nv_bfloat16 ssh_rows_v2[];   // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_rows_v2;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v2_walk(K, V, sK, sV, bt, bsz, q_reg, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// _dc (graph-capture) twin of fa_decode_vec_q_v2: T_kv comes from a device counter, n_splits is
// sized from bucket_max at capture (same contract as fa_decode_vec_q_dc). Calls the SAME
// fa_dec_v2_walk body -> bit-identical to the eager v2 kernel for equal (t_kv, n_splits), which is
// the graph-vs-eager identity the graph_decode_gate pins. Without this twin, BW24_FA_V2=1 would
// silently diverge the captured graph (per-key _dc walk) from eager (tile-batched v2 walk).
extern "C" __global__ void fa_decode_vec_q_v2_dc(
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

    extern __shared__ __nv_bfloat16 ssh_v2_dc[];     // sK[FA_DEC_TILE*head_dim] then sV[...]
    __nv_bfloat16* sK = ssh_v2_dc;
    __nv_bfloat16* sV = sK + FA_DEC_TILE * head_dim;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v2_walk(K, V, sK, sV, bt, bsz, q_reg, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// ===================================================================== //
//  KERNEL 2b-v3 : fa_decode_vec_q_v3  (FA v3 lane, 2026-07-09)           //
//  The HYBRID from research/fa/fa_v3_design.md: llama's int8-dp4a K.Q    //
//  with register-quantized Q (fattn-vec.cuh mechanism, their depth       //
//  lever) + OUR CTA-shared staged V + OUR split partition/combine.       //
//                                                                        //
//  What changes vs v2 (KERNEL 2b-v2 above):                              //
//  - K path VENDORED (fattn-common.cuh:304-329 vec_dot_q8_0_q8_1_impl):  //
//    Q is quantized to int8 in registers ONCE per warp (scale folded in  //
//    first, one shared f32 scale per 32-elem block via group-amax), K    //
//    rows ride RAW q8_0 bytes from global (L2-resident; the 8x GQA       //
//    re-read is what llama already proves affordable) dotted via dp4a.   //
//    Kills Phase A's K half entirely (no K dequant, no bf16 convert, no  //
//    K smem write/read), halves smem 32->16KB @hd256.                    //
//  - V path KEPT ours: smem-staged bf16 V tile, dequant ONCE per CTA     //
//    shared by all gqa warps — the REVISION-2 lesson (naive full         //
//    register streaming measured 2x WORSE at depth; V has no int8       //
//    shortcut, its dequant is the expensive part).                       //
//  - Softmax KEPT v2's tile-batched bookkeeping (once per 32-key tile).  //
//  - The first __syncthreads moves AFTER the K dot: B1 never touches     //
//    smem, so V staging latency hides behind the dp4a work.              //
//                                                                        //
//  NUMERIC CONFIG: int8-dp4a scores != v2's bf16-roundtrip FMA scores    //
//  => BW24_FA_V3=1 is its OWN numeric config (default OFF, own argmax    //
//  baseline; eager + rows + dc twins flip TOGETHER — the FA_V2 lane's    //
//  law). Within the flag it is fully deterministic: all three twins      //
//  call the SAME walk body -> rows-vs-loop and graph-vs-eager bitdiff    //
//  == 0 (kernel-check + graph_decode_gate arbitrate).                    //
//                                                                        //
//  CONSTRAINTS (host-gated in lib.rs fa_v3_usable): q8_0 K / q5_1 V      //
//  default formats ONLY (the dp4a path reads raw q8_0 bytes; V staging   //
//  is the v2 q5_1 recipe verbatim), head_dim % 128 == 0 (dp4a needs      //
//  dpl % 4 == 0 consecutive quants per lane; both daily models are       //
//  hd256).                                                               //
// ===================================================================== //

// 2-byte-aligned int load: q8_0 qs sit at +2 inside the 34B block, so every
// int-sized chunk is 2-aligned but only alternately 4-aligned — two u16 loads
// beat memcpy's byte-wise fallback and are always safe.
static __device__ __forceinline__ int fa_ld_int_2a(const uint8_t* p) {
    unsigned short lo, hi;
    memcpy(&lo, p, 2); memcpy(&hi, p + 2, 2);
    return (int)((unsigned)lo | ((unsigned)hi << 16));
}

// Per-warp register Q quantization (llama's Q->q8 mechanism in OUR layout).
// DOT-phase ownership is CONSECUTIVE: lane l owns Q elements [l*dpl,(l+1)*dpl)
// of this head's row (dp4a needs consecutive bytes; the strided {lane,lane+32,..}
// ownership survives only in acc / the V phase). Quant block b = (l*dpl)>>5
// shares ONE scale across its 32/dpl lanes (aligned-group amax via xor-shuffle).
// `scale` is folded into Q BEFORE quantization (llama-style). Deterministic:
// registers only, fixed shuffle order.
//
// (REVISION 2, same day: a multi-key B1 with one WHOLE block per lane — llama's
// exact warp shape: 32/dpl keys in flight, log2(dpl) group reduce — was tried
// and measured EQUAL at depth but 12-19% worse at t_kv 512-2048: the 8-deep
// dp4a chain + 17 serial loads per key lose to this layout's cross-key ILP
// when the grid is small and latency rules. Reverted.)
static __device__ __forceinline__ void fa_dec_v3_qquant(
        const float* __restrict__ Q, size_t qoff, float scale,
        int dpl, int lane, int* qq, float& dQ)
{
    float qf[FA_DEC_MAX_DPL];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < FA_DEC_MAX_DPL; ++j) {
        if (j < dpl) {
            qf[j] = Q[qoff + (size_t)lane * dpl + j] * scale;
            amax = fmaxf(amax, fabsf(qf[j]));
        } else qf[j] = 0.0f;
    }
    // group amax over the 32/dpl lanes sharing this quant block (groups are
    // lane-aligned: dpl in {4,8} -> group size 8 or 4, both powers of two).
    for (int off = (32 / dpl) >> 1; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    dQ = amax * (1.0f / 127.0f);
    const float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    #pragma unroll
    for (int w = 0; w < FA_DEC_MAX_DPL / 4; ++w) {
        int packed = 0;
        if (4 * w < dpl) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qi = (int)rintf(qf[4 * w + j] * id);
                packed |= (qi & 0xFF) << (8 * j);
            }
        }
        qq[w] = packed;
    }
}

// The shared per-warp v3 split walk (called by ALL THREE twins => per-(row,split)
// bit identity by construction, the same contract as fa_dec_v2_walk).
static __device__ __forceinline__ void fa_dec_v3_walk(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        __nv_bfloat16* sV, int bt, int bsz,
        const int* qq, float dQ, int dpl, int lane, int head_dim,
        int t_lo, int t_hi, int kblk0, long k_tok_bytes, long v_tok_bytes,
        float& m_i, float& l_i, float* acc)
{
    const int blocks_per_key = head_dim >> 5;
    // dp4a K addressing: lane l covers elements [l*dpl,(l+1)*dpl) of this kv
    // head's row -> quant block bK = (l*dpl)>>5, byte offset (l*dpl)&31 in qs.
    const int bK   = (lane * dpl) >> 5;
    const int koff = (lane * dpl) & 31;
    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);

        // ---- Phase A: stage ONLY V (q5_1 -> bf16, once per CTA — v2's recipe
        //      verbatim, bit-identical staged values). NO sync yet: B1/B2 never
        //      touch sV, so the later warps' staging latency hides behind the
        //      dp4a dots. (REVISION 3: an A1-loads/A2-unpack split around B1 —
        //      prefetch the 24B block to registers, unpack after B2 — measured
        //      WORSE at every depth (59.0 vs 55.2 us @6257): +24 regs and the
        //      unpack lands on the pre-sync critical path. Reverted.) ----
        for (int b = bt; b < nt * blocks_per_key; b += bsz) {
            const int j     = b / blocks_per_key;        // key within tile
            const int blk_i = b - j * blocks_per_key;    // block within key
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);          // d|m in one aligned word
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int byte = (e < 16) ? e : e - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e) & 1u) << 4);
                out[e] = __float2bfloat16(d * (float)q5 + m);
            }
        }

        // ---- Phase B1 (vendored dp4a): nt independent row dots on RAW q8_0
        //      bytes from global. part = (dK*dQ) * sumi, pinned with __fmul_rn
        //      so all three twins compile the identical FP chain; the butterfly
        //      gives every lane the full score (v2's reduce shape). ----
        float my_score = NEG_INF;
        float tile_max = m_i;
        #pragma unroll 4
        for (int j = 0; j < nt; ++j) {
            const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes
                                   + (size_t)(kblk0 + bK) * 34;
            const float dK = __half2float(*(const half*)blk);
            // ALIGNED-WORD K loads (REVISION 4b): the qs pointer's alignment class
            // ((34*blk + 2 + koff) & 3, 0 or 2) is CONSTANT per lane across keys
            // (k_tok_bytes % 4 == 0), so read aligned u32s and funnel-shift the
            // lane's bytes out — 2-3 L1 transactions/key vs 4 u16 (L1 was 64%
            // utilized, top stall long_scoreboard). Extracted ints bit-identical
            // to fa_ld_int_2a's. The trailing word is loaded ONLY when the class
            // needs it (misaligned lanes) — never reads past the last block.
            const uint8_t* qsp = blk + 2 + koff;
            const unsigned sh8 = ((unsigned)(size_t)qsp & 3u) * 8u;
            const uint8_t* ap  = (const uint8_t*)((size_t)qsp & ~(size_t)3);
            uint32_t w0, w1 = 0, w2 = 0;
            memcpy(&w0, ap, 4);
            if (dpl > 4) { memcpy(&w1, ap + 4, 4); if (sh8) memcpy(&w2, ap + 8, 4); }
            else if (sh8) memcpy(&w1, ap + 4, 4);
            int sumi = __dp4a((int)__funnelshift_r(w0, w1, sh8), qq[0], 0);
            if (dpl > 4)
                sumi = __dp4a((int)__funnelshift_r(w1, w2, sh8), qq[1], sumi);
            const float part = __fmul_rn(__fmul_rn(dK, dQ), (float)sumi);
            const float score = warp_reduce_sum(part);
            if (lane == j) my_score = score;
            tile_max = fmaxf(tile_max, score);
        }

        // ---- Phase B2: softmax bookkeeping ONCE per tile (v2 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        __syncthreads();   // sV staged by ALL warps before any warp reads it

        // ---- Phase B3: ascending-t V accumulation from smem. PAIRED loads
        //      (bf16x2): acc register i holds dim 2*lane + 64*(i/2) + (i&1) so a
        //      lane reads dpl/2 aligned 4B words instead of dpl 2B ones — half
        //      the LDS transactions. Element values and each dim's j-ascending
        //      accumulation chain are unchanged (bit-identical partials; only
        //      the register->dim mapping moved, and the partial store maps it
        //      back). ----
        #pragma unroll 2
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        __syncthreads();   // tile fully consumed before the next staging overwrites sV
    }
}

// T=1 decode twin. Same signature/grid/block/partial-layout as fa_decode_vec_q_v2;
// smem is sV ONLY (half of v2's).
extern "C" __global__ void fa_decode_vec_q_v3(
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

    int qq[8]; float dQ;   // one full 32-elem Q block per lane (multi-key B1)
    fa_dec_v3_qquant(Q, (size_t)head * head_dim, scale, dpl, lane, qq, dQ);

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    extern __shared__ __nv_bfloat16 ssh_v3[];        // sV[FA_DEC_TILE*head_dim] only
    __nv_bfloat16* sV = ssh_v3;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v3_walk(K, V, sV, bt, bsz, qq, dQ, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// Multi-row (spec-verify) twin: grid.z = query row, causal bound per row —
// same frame as fa_decode_vec_q_rows_v2, same walk body as the T=1 twin
// above (the spec-exactness law: eager decode and verify must never diverge).
extern "C" __global__ void fa_decode_vec_q_rows_v3(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, int t_kv_base,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base + r + 1;      // this row's causal key bound
    fa_rows_v3_body(Q, K, V, partO, partM, partL, head_dim, n_head, n_head_kv, T_kv,
                    scale, n_splits_max, split_keys, k_tok_bytes, v_tok_bytes, r);
}
// shared body: everything below the causal bound is row-local (extracted so the _dc twin is
// call-site-identical; the original kernel's remaining body was moved here VERBATIM).
static __device__ __forceinline__ void fa_rows_v3_body(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int T_kv,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int r)
{
    const int n_splits = (T_kv + split_keys - 1) / split_keys;  // == host fa_split_keys sizing
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    int qq[8]; float dQ;   // one full 32-elem Q block per lane (multi-key B1)
    fa_dec_v3_qquant(Q, ((size_t)r * n_head + head) * head_dim, scale, dpl, lane, qq, dQ);

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    extern __shared__ __nv_bfloat16 ssh_rows_v3[];   // sV[FA_DEC_TILE*head_dim] only
    __nv_bfloat16* sV = ssh_rows_v3;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v3_walk(K, V, sV, bt, bsz, qq, dQ, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// _dc (graph-capture) twin of fa_decode_vec_q_v3: T_kv from a device counter,
// n_splits sized from bucket_max at capture (same contract as _v2_dc). Calls the
// SAME fa_dec_v3_walk body -> bit-identical to eager v3 for equal (t_kv, n_splits).
extern "C" __global__ void fa_decode_vec_q_v3_dc(
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

    int qq[8]; float dQ;   // one full 32-elem Q block per lane (multi-key B1)
    fa_dec_v3_qquant(Q, (size_t)head * head_dim, scale, dpl, lane, qq, dQ);

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    extern __shared__ __nv_bfloat16 ssh_v3_dc[];     // sV[FA_DEC_TILE*head_dim] only
    __nv_bfloat16* sV = ssh_v3_dc;
    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;
    fa_dec_v3_walk(K, V, sV, bt, bsz, qq, dQ, dpl, lane, head_dim,
                   t_lo, t_hi, kblk0, k_tok_bytes, v_tok_bytes, m_i, l_i, acc);

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map
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
        float ktv = (tid < head_dim) ? DQ_K_ELEM(K, t, k_tok_bytes, kidx) : 0.0f;
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
        float vtv = (tid < head_dim) ? DQ_V_ELEM(V, t, v_tok_bytes, vidx) : 0.0f;
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

    // REGISTER-DEQUANT (2026-07-03): same rewrite as fa_decode_vec_q — see comment there. The _dc
    // twin must stay numerically IDENTICAL to the eager kernel (graph bit-identity contract).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to the old smem-staged path (which stored
                    // dequanted K as bf16). Pure ALU on a DRAM-bound kernel — keeps every gate
                    // (incl. run-spec exactness) exactly where the validated kernel had it.
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
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

// ===================================================================== //
//  MULTI-ROW VERIFY decode (spec verify, T=K+1 causal rows).            //
//  ONE launch replaces the T separate per-row fa_decode_vec_q calls of  //
//  full_attn_verify: grid.z = query row r; each z-slice executes        //
//  fa_decode_vec_q's EXACT program for its OWN causal bound             //
//    t_kv_r     = t_kv_base + r + 1                                     //
//    n_splits_r = ceil(t_kv_r / split_keys)   (== the host fa_split_keys//
//                 sizing formula; split_keys passed from the launcher)  //
//    per_r      = ceil(t_kv_r / n_splits_r)   (same in-kernel formula)  //
//  so every row's split partition, key-walk order and online-softmax    //
//  accumulation are BIT-IDENTICAL to the eager per-row call it          //
//  replaces (the spec-exactness law: same kernel body, same blockDim,   //
//  same split boundaries, same reduce shape — kernel-check pins the     //
//  rows-vs-loop byte identity). grid.y is sized for the LAST row's      //
//  n_splits; blocks with split >= n_splits_r exit without writing and   //
//  the row combine below never reads those slots (no empty split can    //
//  exist below n_splits_r: per<=split_keys ==> (n_splits_r-1)*per<t_kv). //
//  WHY: the single-row launch is latency-bound and underfills the SMs   //
//  (measured 392 CTAs = 4.8/SM vs 12 resident achievable, 201us/row at  //
//  6.3k ctx); fusing T rows multiplies resident CTAs by T and shares    //
//  the KV prefix across rows through L2 within ONE launch.              //
//  partO layout: [row, n_head, n_splits_max, head_dim]; M/L analogous.  //
// ===================================================================== //

extern "C" __global__ void fa_decode_vec_q_rows(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, int t_kv_base,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base + r + 1;      // this row's causal key bound
    const int n_splits = (T_kv + split_keys - 1) / split_keys;  // == host fa_split_keys sizing
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    // REGISTER-DEQUANT walk: byte-for-byte the fa_decode_vec_q body (see comment there); only the
    // Q read and partial writes carry the row offset. Any change HERE must be mirrored in
    // fa_decode_vec_q/_dc and re-gated (kernel-check rows-vs-loop bit identity + run-spec battery).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to fa_decode_vec_q (see comment there).
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// Row-batched combine: grid = (n_head, T). Row r merges its OWN n_splits_r
// (same ceil(t_kv_r/split_keys) formula) in the SAME ascending-split order as
// fa_decode_combine_f32 — identical values, identical fmax/sum order; only the
// partial STRIDE differs (n_splits_max vs n_splits_r) and slots >= n_splits_r
// are never read. Writes O[row, n_head, head_dim] (the verify attn stack).
// ROUND-STREAM stage (c): combine twin with the causal base from a device counter — body
// identical to fa_decode_combine_rows (per-row n_splits derived the same way from T_kv).
extern "C" __global__ void fa_decode_combine_rows_dc(
        const float* __restrict__ partO, const float* __restrict__ partM,
        const float* __restrict__ partL, float* __restrict__ O,
        int head_dim, int n_head, const int* __restrict__ t_kv_base_dev, int base_plus,
        int n_splits_max, int split_keys)
{
    const int head     = blockIdx.x;
    const int r        = blockIdx.y;
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int tid      = threadIdx.x;
    if (head >= n_head || tid >= head_dim) return;
    const float* pM = partM + ((size_t)r * n_head + head) * n_splits_max;
    const float* pL = partL + ((size_t)r * n_head + head) * n_splits_max;
    const float* pO = partO + ((size_t)r * n_head + head) * n_splits_max * head_dim;
    float m = NEG_INF;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, pM[s]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = pM[s];
        if (ms == NEG_INF) continue;
        float w = exp2f((ms - m) * LOG2E);
        l += pL[s] * w;
        o += pO[(size_t)s * head_dim + tid] * w;
    }
    float linv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[((size_t)r * n_head + head) * head_dim + tid] = o * linv;
}

// windowed combine twin: n_splits constant (every row folds exactly `window` keys).
extern "C" __global__ void fa_decode_combine_rows_w(
        const float* __restrict__ partO, const float* __restrict__ partM,
        const float* __restrict__ partL, float* __restrict__ O,
        int head_dim, int n_head, int n_splits_max, int split_keys, int window)
{
    const int head     = blockIdx.x;
    const int r        = blockIdx.y;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int tid      = threadIdx.x;
    if (head >= n_head || tid >= head_dim) return;
    const float* pM = partM + ((size_t)r * n_head + head) * n_splits_max;
    const float* pL = partL + ((size_t)r * n_head + head) * n_splits_max;
    const float* pO = partO + ((size_t)r * n_head + head) * n_splits_max * head_dim;
    float m = NEG_INF;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, pM[s]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = pM[s];
        if (ms == NEG_INF) continue;
        float w = exp2f((ms - m) * LOG2E);
        l += pL[s] * w;
        o += pO[(size_t)s * head_dim + tid] * w;
    }
    float linv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[((size_t)r * n_head + head) * head_dim + tid] = o * linv;
}

extern "C" __global__ void fa_decode_combine_rows(
        const float* __restrict__ partO, const float* __restrict__ partM,
        const float* __restrict__ partL, float* __restrict__ O,
        int head_dim, int n_head, int t_kv_base, int n_splits_max, int split_keys)
{
    const int head     = blockIdx.x;
    const int r        = blockIdx.y;
    const int T_kv     = t_kv_base + r + 1;
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int tid      = threadIdx.x;
    if (head >= n_head || tid >= head_dim) return;
    const float* pM = partM + ((size_t)r * n_head + head) * n_splits_max;
    const float* pL = partL + ((size_t)r * n_head + head) * n_splits_max;
    const float* pO = partO + ((size_t)r * n_head + head) * n_splits_max * head_dim;

    float m = NEG_INF;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, pM[s]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = pM[s];
        if (ms == NEG_INF) continue;
        float w = exp2f((ms - m) * LOG2E);
        l += pL[s] * w;
        o += pO[(size_t)s * head_dim + tid] * w;
    }
    float linv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[((size_t)r * n_head + head) * head_dim + tid] = o * linv;
}


// ===================== FA V4: KEY-PER-LANE SCORE PHASE (2026-07-10) =====================
// fa_v3 at d6257 runs at 14% of bytes-wall — latency-bound on the reduce-per-key structure:
// per 32-key tile, 32 x (8 dp4a + 5-shfl warp_reduce) ≈ 416 warp-serial steps. V4 stages the
// K tile INT-REPACKED to smem (qs as aligned ints + d halves separated) and each LANE computes
// the FULL q·k_lane dot chunk-serially (8 chunks x 8 dp4a, all 32 keys in PARALLEL, zero
// shuffles in the score phase). B2 softmax bookkeeping + B3 V-accumulation are the v3 bodies
// verbatim (my_score lands in lane j exactly as v3's butterfly left it).
// NEW NUMERIC CONFIG: the per-key dot accumulates chunk-serial in ONE lane (v3: lane-parallel
// + tree reduce) — decode and verify flip TOGETHER (dispatch parity keeps self-consistency);
// the battery + acceptance-shift check arbitrate per model. q8_0 K / q5_1 V / hd256 only.
// smem: sQ 64 ints + 8 dQ, sK 32x(64 ints + 8 halves) ≈ 8.7KB, sV 32xhd bf16 = 16KB -> ~25KB.
struct fa_v4_smem {
    int   q_ints[8][64];            // [gqa<=8][64] per-warp quantized Q (8 chunks x 8 ints)
    float q_d[8][8];                // [gqa][8] per-chunk Q scales
    int   k_ints[FA_DEC_TILE][64];  // repacked K tile
    float k_d[FA_DEC_TILE][8];      // per-chunk K scales
    // sV follows in dynamic smem (v3 layout; element type = fa_v4_sv_t below)
};

#if BW24_KV_VFMT == 2
// e4m3 sV tile stages the RAW BYTE, cvt at use: every e4m3 value is exactly representable
// in bf16, so this is BIT-IDENTICAL to the bf16 tile at HALF the smem — the 27.9KB/block
// footprint capped residency at 3 blocks/SM (12.5% theoretical occupancy, ncu 2026-07-12);
// 19.7KB lifts the cap. Host shmem sizing mirrors this (g-module: 32*head_dim*1).
typedef uint8_t fa_v4_sv_t;
#else
typedef __nv_bfloat16 fa_v4_sv_t;
#endif

static __device__ __forceinline__ void fa_v4_stage_q(
        const float* __restrict__ Q, size_t qoff, float scale, int lane, int wy,
        fa_v4_smem* sm) {
    // v3's qquant grouping verbatim (dpl=8: 4-lane groups of 32 elems), then smem publish.
    int qq[FA_DEC_MAX_DPL / 4];
    float dQ;
    fa_dec_v3_qquant(Q, qoff, scale, 8, lane, qq, dQ);
    // lane holds elems [lane*8, lane*8+8) as 2 ints; chunk c = elems [c*32,(c+1)*32) = lanes 4c..4c+3
    sm->q_ints[wy][lane * 2]     = qq[0];
    sm->q_ints[wy][lane * 2 + 1] = qq[1];
    if ((lane & 3) == 0) sm->q_d[wy][lane >> 2] = dQ;
}

static __device__ __forceinline__ void fa_v4_stage_k(
        const uint8_t* __restrict__ K, int t0, int nt, int bt, int bsz,
        int kblk0, long k_tok_bytes, fa_v4_smem* sm) {
#if BW24_KV_KFMT == 1
    // fp8-e4m3 K (KFMT==1): 32 raw bytes per chunk — cvt to f32, per-chunk absmax requant
    // to int8 so the dp4a score phase is format-agnostic (k_d = absmax/127; absmax==0 ->
    // zeros). NEW NUMERIC CONFIG for the fp8 module only; the default arm below is verbatim.
    for (int task = bt; task < nt * 8; task += bsz) {
        int j = task >> 3, c = task & 7;
        const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + c) * K_BLK_B;
        float vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < 32; e++) {
            vals[e] = (float)((const __nv_fp8_e4m3*)blk)[e];
            amax = fmaxf(amax, fabsf(vals[e]));
        }
        const float kd = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        sm->k_d[j][c] = kd;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            int packed = 0;
            #pragma unroll
            for (int b8 = 0; b8 < 4; b8++) {
                const int e = w * 4 + b8;
                const int q = __float2int_rn(vals[e] * inv);
                packed |= (q & 0xFF) << (8 * b8);
            }
            sm->k_ints[j][c * 8 + w] = packed;
        }
    }
#else
    // task = (key j, chunk c): unpack q8_0 block (2B d + 32 int8) into aligned ints + half.
    for (int task = bt; task < nt * 8; task += bsz) {
        int j = task >> 3, c = task & 7;
        const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + c) * K_BLK_B;
        sm->k_d[j][c] = __half2float(*(const half*)blk);
        // aligned-word + funnelshift extraction (REVISION 4b recipe) — same ints as the byte
        // path, 9 aligned LDG.32 instead of ~32 byte loads.
        const uint8_t* qs = blk + 2;
        const unsigned sh8 = ((unsigned)(size_t)qs & 3u) * 8u;
        const uint32_t* ap = (const uint32_t*)((size_t)qs & ~(size_t)3);
        uint32_t w0 = ap[0];
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t w1 = ap[w + 1];
            sm->k_ints[j][c * 8 + w] = (int)__funnelshift_r(w0, w1, sh8);
            w0 = w1;
        }
    }
#endif
}

extern "C" __global__ void fa_decode_vec_q_v4(
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
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, (size_t)head * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        // ---- B3 (v3 body; unroll 8 — the MACs are independent across j, ILP hides LDS) ----
        #pragma unroll 8
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// PROBES (wall-arc phase isolation; bench-only, never dispatched in prod)
extern "C" __global__ void fa_decode_vec_q_v4_dc(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_dev,
        float scale, int n_splits, long k_tok_bytes, long v_tok_bytes)
{
    const int T_kv    = t_kv_dev[0];             // device-resident sequence length
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, (size_t)head * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        // ---- B3 (v3 body; unroll 8 — the MACs are independent across j, ILP hides LDS) ----
        #pragma unroll 8
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

// PROBES (wall-arc phase isolation; bench-only, never dispatched in prod)

extern "C" __global__ void fa_decode_vec_q_v4_noB3(
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
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, (size_t)head * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}

extern "C" __global__ void fa_decode_vec_q_v4_stage(
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
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, (size_t)head * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // probe: consume one staged byte per lane so staging is not eliminated
        #if BW24_KV_VFMT == 2
        m_i += (float)sm->k_ints[lane % FA_DEC_TILE][0] * 1e-30f + (float)*(const __nv_fp8_e4m3*)&sV[lane] * 1e-30f;
        #else
        m_i += (float)sm->k_ints[lane % FA_DEC_TILE][0] * 1e-30f + __bfloat162float(sV[lane]) * 1e-30f;
        #endif
        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[((size_t)head * n_splits + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) { partM[head * n_splits + split] = m_i; partL[head * n_splits + split] = l_i; }
}


// V4 rows twin (spec verify): grid.z = query row, per-row causal bound — same v4 body, so
// verify and decode share the numeric config when BW24_FA_V4=1 flips both (dispatch parity).
extern "C" __global__ void fa_decode_vec_q_rows_v4(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, int t_kv_base,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base + r + 1;      // per-row causal bound
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (T_kv + n_splits - 1) / n_splits;
    const int t_lo = split * per;
    const int t_hi = min(T_kv, t_lo + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, ((size_t)r * n_head + head) * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        // ---- B3 (v3 body; unroll 8 — the MACs are independent across j, ILP hides LDS) ----
        #pragma unroll 8
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

extern "C" __global__ void fa_decode_vec_q_rows_v4_w(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;      // per-row causal bound
    // WINDOWED twin (gemma R6): every row attends exactly `window` keys; split geometry and
    // key order mirror the T=1 decode's fa_decode-over-window-VIEW bit-for-bit (start+j
    // absolute mapping; host gates base_len+1 >= window so no row is under-window).
    const int start    = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    if (wy >= gqa) return;
    const int head = kv_head * gqa + wy;
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (window + n_splits - 1) / n_splits;
    const int t_lo = start + split * per;
    const int t_hi = start + min(window, split * per + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    fa_v4_stage_q(Q, ((size_t)r * n_head + head) * head_dim, scale, lane, wy, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * gqa;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        // ---- B3 (v3 body; unroll 8 — the MACs are independent across j, ILP hides LDS) ----
        #pragma unroll 8
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        __syncthreads();   // tile fully consumed before restaging
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

extern "C" __global__ void fa_decode_vec_q_rows_v4_w_sp(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;      // per-row causal bound
    // WINDOWED twin (gemma R6): every row attends exactly `window` keys; split geometry and
    // key order mirror the T=1 decode's fa_decode-over-window-VIEW bit-for-bit (start+j
    // absolute mapping; host gates base_len+1 >= window so no row is under-window).
    const int start    = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    // STAGING-PARALLEL twin (2026-07-11): gqa==1 host-gated; warp 1 is a STAGING HELPER
    // (the v4 phase probe: staging = 61% of the kernel) — bsz doubles, score/softmax/value
    // phases run on warp 0 only. Numerics identical to v4_w (same stage layout, same chains).
    const int gqa  = n_head / n_head_kv;
    const int wy   = threadIdx.y;
    const int lane = threadIdx.x;
    (void)gqa;
    const int head = kv_head;
    const int dpl  = head_dim >> 5;           // == 8 (host-gated hd256)

    const int per  = (window + n_splits - 1) / n_splits;
    const int t_lo = start + split * per;
    const int t_hi = start + min(window, split * per + per);

    extern __shared__ unsigned char sm_raw_v4[];
    fa_v4_smem* sm = (fa_v4_smem*)sm_raw_v4;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_v4 + sizeof(fa_v4_smem));

    if (wy == 0) fa_v4_stage_q(Q, ((size_t)r * n_head + head) * head_dim, scale, lane, 0, sm);
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = wy * WARP_SZ + lane;
    const int bsz = WARP_SZ * 2;
    const int kblk0 = (kv_head * head_dim) >> 5;

    for (int t0 = t_lo; t0 < t_hi; t0 += FA_DEC_TILE) {
        const int nt = min(FA_DEC_TILE, t_hi - t0);
        // stage V (v3/v2 recipe verbatim, all warps) + K repack (all warps)
        for (int b = bt; b < nt * dpl * 4; b += bsz) {
            // 4x-finer task split (8 elems/task): the 32-elem scalar unpack chain was the
            // staging critical path (phase probe: staging = 61% of the kernel).
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k(K, t0, nt, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        if (wy == 0) {
        // ---- V4 SCORE PHASE: lane j owns key j; full dot chunk-serial, zero shuffles ----
        float my_score = NEG_INF;
        if (lane < nt) {
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int sumi = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    sumi = __dp4a(sm->k_ints[lane][c * 8 + w], sm->q_ints[wy][c * 8 + w], sumi);
                s = __fmaf_rn(__fmul_rn(sm->k_d[lane][c], sm->q_d[wy][c]), (float)sumi, s);
            }
            my_score = s;
        }
        // tile max across lanes (one 5-shfl tree per TILE, not per key)
        float tile_max = m_i;
        {
            float v = my_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
            tile_max = fmaxf(tile_max, v);
        }

        // ---- B2 (v3 verbatim) ----
        const float m_new = tile_max;
        const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        const float p_lane = (lane < nt) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
        l_i = l_i * alpha + warp_reduce_sum(p_lane);
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) acc[i] *= alpha;
        }
        m_i = m_new;

        // ---- B3 (v3 body; unroll 8 — the MACs are independent across j, ILP hides LDS) ----
        #pragma unroll 8
        for (int j = 0; j < nt; ++j) {
            const float p = __shfl_sync(0xffffffffu, p_lane, j);
            #if BW24_KV_VFMT == 2
            const uchar2* vj2 = (const uchar2*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const uchar2 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                    acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                }
            }
            #else
            const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)j * head_dim);
            #pragma unroll
            for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                if (2 * i2 < dpl) {
                    const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                    acc[2 * i2]     += p * __bfloat162float(vv.x);
                    acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                }
            }
            #endif
        }
        }
        __syncthreads();   // tile fully consumed before restaging
    }

    if (wy == 0) {
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = (lane << 1) + ((i >> 1) << 6) + (i & 1);   // paired-B3 dim map (v3)
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
    }
}



// MULTI-ROW v4 windowed kernel (2026-07-11, the windowed-lane occupancy/staging fix): gemma
// SWA has gqa==1 (nh==nkv), so v4_w ran ONE warp per block and staged the k-tile once PER
// (row, split). Here warp wy = verify row over a WIDENED shared tile (FA_DEC_TILE + R - 1
// staged keys, cooperative across all R warps): row r's k-th tile is staged slots
// [wy + 32k, wy + 32k + 32), which is EXACTLY its own t_lo_r-aligned grouping — per-row tile
// boundaries (and therefore the online-softmax order) match the single-row v4_w. This kernel
// REPLACES v4_w at every t (R=1 decode .. R=8 gates) so parity stays structural (one symbol).
// gqa==1 is host-gated.
#define FA_MR_MAX 8
struct fa_v4_smem_mr {
    int   q_ints[FA_MR_MAX][64];
    float q_d[FA_MR_MAX][8];
    int   k_ints[FA_DEC_TILE + FA_MR_MAX - 1][64];
    float k_d[FA_DEC_TILE + FA_MR_MAX - 1][8];
    // sV bf16 [(FA_DEC_TILE + FA_MR_MAX - 1) x head_dim] follows in dynamic smem
};
static __device__ __forceinline__ void fa_v4_stage_q_mr(
        const float* __restrict__ Q, size_t qoff, float scale, int lane, int wy,
        fa_v4_smem_mr* sm) {
    int qq[FA_DEC_MAX_DPL / 4];
    float dQ;
    fa_dec_v3_qquant(Q, qoff, scale, 8, lane, qq, dQ);
    sm->q_ints[wy][lane * 2]     = qq[0];
    sm->q_ints[wy][lane * 2 + 1] = qq[1];
    if ((lane & 3) == 0) sm->q_d[wy][lane >> 2] = dQ;
}
static __device__ __forceinline__ void fa_v4_stage_k_mr(
        const uint8_t* __restrict__ K, int t0, int nt, int bt, int bsz,
        int kblk0, long k_tok_bytes, fa_v4_smem_mr* sm) {
#if BW24_KV_KFMT == 1
    // fp8-e4m3 K (KFMT==1): 32 raw bytes per chunk — cvt to f32, per-chunk absmax requant
    // to int8 so the dp4a score phase is format-agnostic (k_d = absmax/127; absmax==0 ->
    // zeros). NEW NUMERIC CONFIG for the fp8 module only; the default arm below is verbatim.
    for (int task = bt; task < nt * 8; task += bsz) {
        int j = task >> 3, c = task & 7;
        const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + c) * K_BLK_B;
        float vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < 32; e++) {
            vals[e] = (float)((const __nv_fp8_e4m3*)blk)[e];
            amax = fmaxf(amax, fabsf(vals[e]));
        }
        const float kd = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        sm->k_d[j][c] = kd;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            int packed = 0;
            #pragma unroll
            for (int b8 = 0; b8 < 4; b8++) {
                const int e = w * 4 + b8;
                const int q = __float2int_rn(vals[e] * inv);
                packed |= (q & 0xFF) << (8 * b8);
            }
            sm->k_ints[j][c * 8 + w] = packed;
        }
    }
#else
    for (int task = bt; task < nt * 8; task += bsz) {
        int j = task >> 3, c = task & 7;
        const uint8_t* blk = K + (size_t)(t0 + j) * k_tok_bytes + (size_t)(kblk0 + c) * K_BLK_B;
        sm->k_d[j][c] = __half2float(*(const half*)blk);
        const uint8_t* qs = blk + 2;
        const unsigned sh8 = ((unsigned)(size_t)qs & 3u) * 8u;
        const uint32_t* ap = (const uint32_t*)((size_t)qs & ~(size_t)3);
        uint32_t w0 = ap[0];
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t w1 = ap[w + 1];
            sm->k_ints[j][c * 8 + w] = (int)__funnelshift_r(w0, w1, sh8);
            w0 = w1;
        }
    }
#endif
}
extern "C" __global__ void fa_decode_vec_q_rows_v4_w_mr(
        const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ partO, float* __restrict__ partM, float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window, int n_rows)
{
    const int r     = threadIdx.y;               // this warp's verify row
    const int lane  = threadIdx.x;
    const int T_kv  = t_kv_base_dev[0] + base_plus + r + 1;
    const int start = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int head = kv_head;                    // gqa == 1 (host-gated)
    const int dpl  = head_dim >> 5;

    const int per   = (window + n_splits - 1) / n_splits;
    const int t_lo  = start + split * per;                          // this ROW's range
    const int t_hi  = start + min(window, split * per + per);
    const int u0    = (t_kv_base_dev[0] + base_plus + 1) - window + split * per;  // row 0's t_lo
    const int nkeys = (t_hi - t_lo);                                // == per for every row

    extern __shared__ unsigned char sm_raw_mr[];
    fa_v4_smem_mr* sm = (fa_v4_smem_mr*)sm_raw_mr;
    fa_v4_sv_t* sV = (fa_v4_sv_t*)(sm_raw_mr + sizeof(fa_v4_smem_mr));

    if (r < n_rows) {
        fa_v4_stage_q_mr(Q, ((size_t)r * n_head + head) * head_dim, scale, lane, r, sm);
    }
    __syncthreads();

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int bt  = threadIdx.y * WARP_SZ + lane;
    const int bsz = WARP_SZ * blockDim.y;
    const int kblk0 = (kv_head * head_dim) >> 5;

    // tiles walk row 0's grid; the shared stage covers [u0 + 32k, u0 + 32k + 32 + n_rows - 1)
    // so every row's own 32-key window [u0 + r + 32k, ...+32) is resident.
    for (int k0 = 0; k0 < nkeys; k0 += FA_DEC_TILE) {
        const int t0  = u0 + k0;
        const int nt_row = min(FA_DEC_TILE, nkeys - k0);            // keys this tile, per row
        const int nt_stage = nt_row + n_rows - 1;                   // union staged
        // stage V (q5_1 -> bf16) for the union
        for (int b = bt; b < nt_stage * dpl * 4; b += bsz) {
            const int sub   = b & 3;
            const int b32   = b >> 2;
            const int j     = b32 / dpl;
            const int blk_i = b32 - j * dpl;
            #if BW24_KV_VFMT == 2
            // fp8-e4m3 V: raw bytes, one cvt per element (V_BLK_B = 32; no scales).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            fa_v4_sv_t* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                out[e2] = blk[e2];   // raw e4m3 byte; cvt at use (bit-identical, half smem)
            }
            #elif BW24_KV_VFMT == 1
            // q4_0 V: f16 d + nibbles (V_BLK_B = 18; x = d*(q-8)).
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * V_BLK_B;
            const float d40 = __half2float(*(const half*)blk);
            const uint8_t* qs4 = blk + 2;
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2 = sub * 8 + e0;
                const int q40 = (e2 < 16) ? (qs4[e2] & 0x0F) : (qs4[e2 - 16] >> 4);
                out[e2] = __float2bfloat16(d40 * (float)(q40 - 8));
            }
            #else
            const uint8_t* blk = V + (size_t)(t0 + j) * v_tok_bytes
                                   + (size_t)(kblk0 + blk_i) * 24;
            uint32_t wdm; memcpy(&wdm, blk, 4);
            const float d = __half2float(__ushort_as_half((unsigned short)(wdm & 0xFFFFu)));
            const float m = __half2float(__ushort_as_half((unsigned short)(wdm >> 16)));
            uint32_t qh; memcpy(&qh, blk + 4, 4);
            uint32_t qsw[4]; memcpy(qsw, blk + 8, 16);
            __nv_bfloat16* out = sV + (size_t)j * head_dim + (blk_i << 5);
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e2   = sub * 8 + e0;
                const int byte = (e2 < 16) ? e2 : e2 - 16;
                const int nib  = (uint8_t)(qsw[byte >> 2] >> (8 * (byte & 3)));
                const int lo   = (e2 < 16) ? (nib & 0x0F) : (nib >> 4);
                const int q5   = lo | (int)(((qh >> e2) & 1u) << 4);
                out[e2] = __float2bfloat16(d * (float)q5 + m);
            }
            #endif
        }
        fa_v4_stage_k_mr(K, t0, nt_stage, bt, bsz, kblk0, k_tok_bytes, sm);
        __syncthreads();

        if (r < n_rows) {
            // ---- score: lane j owns THIS ROW's key j = staged slot j + r ----
            float my_score = NEG_INF;
            if (lane < nt_row) {
                float s = 0.0f;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int sumi = 0;
                    #pragma unroll
                    for (int w = 0; w < 8; w++)
                        sumi = __dp4a(sm->k_ints[lane + r][c * 8 + w], sm->q_ints[r][c * 8 + w], sumi);
                    s = __fmaf_rn(__fmul_rn(sm->k_d[lane + r][c], sm->q_d[r][c]), (float)sumi, s);
                }
                my_score = s;
            }
            float tile_max = m_i;
            {
                float v = my_score;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
                tile_max = fmaxf(tile_max, v);
            }
            const float m_new = tile_max;
            const float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            const float p_lane = (lane < nt_row) ? exp2f((my_score - m_new) * LOG2E) : 0.0f;
            l_i = l_i * alpha + warp_reduce_sum(p_lane);
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) acc[i] *= alpha;
            }
            m_i = m_new;
            #pragma unroll 8
            for (int j = 0; j < nt_row; ++j) {
                const float p = __shfl_sync(0xffffffffu, p_lane, j);
                #if BW24_KV_VFMT == 2
                const uchar2* vj2 = (const uchar2*)(sV + (size_t)(j + r) * head_dim);
                #pragma unroll
                for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                    if (2 * i2 < dpl) {
                        const uchar2 vv = vj2[lane + (i2 << 5)];
                        acc[2 * i2]     += p * (float)*(const __nv_fp8_e4m3*)&vv.x;
                        acc[2 * i2 + 1] += p * (float)*(const __nv_fp8_e4m3*)&vv.y;
                    }
                }
                #else
                const __nv_bfloat162* vj2 = (const __nv_bfloat162*)(sV + (size_t)(j + r) * head_dim);
                #pragma unroll
                for (int i2 = 0; i2 < FA_DEC_MAX_DPL / 2; ++i2) {
                    if (2 * i2 < dpl) {
                        const __nv_bfloat162 vv = vj2[lane + (i2 << 5)];
                        acc[2 * i2]     += p * __bfloat162float(vv.x);
                        acc[2 * i2 + 1] += p * __bfloat162float(vv.y);
                    }
                }
                #endif
            }
        }
        __syncthreads();
    }

    if (r < n_rows) {
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) {
                int d = lane + (i << 5);
                partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
            }
        }
        if (lane == 0) {
            partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
            partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
        }
    }
}

// windowed REGISTER rows twin (gemma R6 non-v4 lane; body = fa_decode_vec_q_rows verbatim).
extern "C" __global__ void fa_decode_vec_q_rows_reg_w(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major (verify q stack)
        const uint8_t* __restrict__ K,  // q8_0 cache [token, kv_dim_k bytes]
        const uint8_t* __restrict__ V,  // q5_1 cache [token, kv_dim_v bytes]
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,      // [T, n_head, n_splits_max]
        float* __restrict__ partL,      // [T, n_head, n_splits_max]
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window)
{
    const int r        = blockIdx.z;             // query row (verify column)
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;      // this row's causal key bound
    const int start    = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;  // == host fa_split_keys sizing
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa     = n_head / n_head_kv;
    const int wy      = threadIdx.y;
    const int lane    = threadIdx.x;
    if (wy >= gqa) return;
    const int head    = kv_head * gqa + wy;
    const int dpl     = head_dim >> 5;

    const int per  = (window + n_splits - 1) / n_splits;
    const int t_lo = start + split * per;
    const int t_hi = start + min(window, split * per + per);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    // REGISTER-DEQUANT walk: byte-for-byte the fa_decode_vec_q body (see comment there); only the
    // Q read and partial writes carry the row offset. Any change HERE must be mirrored in
    // fa_decode_vec_q/_dc and re-gated (kernel-check rows-vs-loop bit identity + run-spec battery).
    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    // bf16 round-trip: BIT-IDENTICAL to fa_decode_vec_q (see comment there).
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    // bf16 round-trip: see K above.
                    // PINNED FP association (kvbytes refactor): FMUL(p,vv) then FFMA(acc,alpha,prod) —
                    // the exact pre-refactor SASS. Without intrinsics ptxas flipped which product
                    // fuses (rounds acc*alpha instead of p*vv) = silent numeric-config change.
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// 2-KEY INTERLEAVE windowed register twin (FP8-WINDOWED lane, 2026-07-11): on q8_0 the
// register walk lost to v4_w on dq-chain latency; e4m3 dq is a byte cvt, so this is the
// windowed kernel for fp8 layers (launched from the kf8vf8 module only — dq_K/V_lane are
// format macros). Two dq chains in flight + fused paired softmax update.
extern "C" __global__ void fa_decode_vec_q_rows_reg_w_i2(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,
        float* __restrict__ partO,
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes, int window)
{
    const int r        = blockIdx.z;
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;
    const int start    = T_kv - window;
    const int n_splits = (window + split_keys - 1) / split_keys;
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
    if (kv_head >= n_head_kv || split >= n_splits) return;
    const int gqa     = n_head / n_head_kv;
    const int wy      = threadIdx.y;
    const int lane    = threadIdx.x;
    if (wy >= gqa) return;
    const int head    = kv_head * gqa + wy;
    const int dpl     = head_dim >> 5;

    const int per  = (window + n_splits - 1) / n_splits;
    const int t_lo = start + split * per;
    const int t_hi = start + min(window, split * per + per);

    float q_reg[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) acc[i] = 0.0f;

    const int kblk0 = (kv_head * head_dim) >> 5;
    int t = t_lo;
    for (; t + 1 < t_hi; t += 2) {
        const uint8_t* ka = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        const uint8_t* kb = K + (size_t)(t + 1) * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        float pa = 0.0f, pb = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) {
                pa += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(ka + i * K_BLK_B, lane)));
                pb += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(kb + i * K_BLK_B, lane)));
            }
        }
        float sA = warp_reduce_sum(pa);
        float sB = warp_reduce_sum(pb);
        float m_new = fmaxf(m_i, fmaxf(sA, sB));
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float wA = exp2f((sA - m_new) * LOG2E);
        float wB = exp2f((sB - m_new) * LOG2E);
        const uint8_t* va = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        const uint8_t* vb = V + (size_t)(t + 1) * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) {
                float vva = __bfloat162float(__float2bfloat16(dq_V_lane(va + i * V_BLK_B, lane)));
                float vvb = __bfloat162float(__float2bfloat16(dq_V_lane(vb + i * V_BLK_B, lane)));
                acc[i] = __fmaf_rn(acc[i], alpha, __fmaf_rn(wA, vva, __fmul_rn(wB, vvb)));
            }
        }
        l_i = l_i * alpha + wA + wB;
        m_i = m_new;
    }
    for (; t < t_hi; ++t) {
        const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) {
                part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(kt + i * K_BLK_B, lane)));
            }
        }
        float score = warp_reduce_sum(part);
        float m_new = fmaxf(m_i, score);
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float p     = exp2f((score - m_new) * LOG2E);
        const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
            if (i < dpl) {
                acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(vt + i * V_BLK_B, lane)))));
            }
        }
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}



// hd512 ROWS twin (gemma globals verify + decode, parity law): fa_decode_vec_q_dpl16's
// EXACT walk with the rows frame (r = blockIdx.z causal bound, [T,...] partials). Decode
// passes t=1 — decode and verify share THIS symbol in the hd512 vec regime, so parity does
// not depend on codegen luck (the 2026-07-10 SASS lesson: identical source != identical SASS).
extern "C" __global__ void fa_decode_vec_q_rows_dpl16(
        const float* __restrict__ Q,    // [T, n_head, head_dim] token-major
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,
        float* __restrict__ partO,      // [T, n_head, n_splits_max, head_dim]
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) acc[i] = 0.0f;

    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                }
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = vt + i * V_BLK_B;
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(blk, lane)))));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// K=V twin (gemma globals: wv:=wk — the K and V planes hold the same VALUES, but K is q8_0
// and V is q5_1): reuse the dequantized+bf16-rounded q8_0 key chunk as the value chunk. The
// separate V walk disappears (the V plane is never read — ~40% less KV traffic + half the dq
// ALU) and the value vector carries q8_0 precision instead of q5_1 (a strictly finer numeric
// config — NEW CONFIG, battery-arbitrated). Parity is structural: every hd512 caller shares
// this symbol via fa_decode_rows. Callers pass kv_shared only when K-values == V-values.
extern "C" __global__ void fa_decode_vec_q_rows_dpl16_kv(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,   // unused (kept for launch-arg symmetry)
        float* __restrict__ partO,
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    (void)V; (void)v_tok_bytes;
    const int r        = blockIdx.z;
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) acc[i] = 0.0f;

    {
        const int kblk0 = (kv_head * head_dim) >> 5;
        for (int t = t_lo; t < t_hi; ++t) {
            const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
            float kv[FA_DEC_MAX_DPL16];
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    const uint8_t* blk = kt + i * K_BLK_B;
                    kv[i] = __bfloat162float(__float2bfloat16(dq_K_lane(blk, lane)));
                    part += q_reg[i] * kv[i];
                } else kv[i] = 0.0f;
            }
            float score = warp_reduce_sum(part);

            float m_new = fmaxf(m_i, score);
            float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
            float p     = exp2f((score - m_new) * LOG2E);
            #pragma unroll
            for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
                if (i < dpl) {
                    acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, kv[i]));
                }
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}

// 2-KEY INTERLEAVE twin (2026-07-11, register-frugal ILP for the 30x-off-floor global lane):
// each iteration scores TWO keys with interleaved dq chains (2 loads in flight instead of 1),
// does one fused softmax update (m_new = max(m, sA, sB)), then accumulates both values with
// interleaved V chains. +6 registers vs the serial walk (the two-pass rewrite's +32 collapsed
// occupancy — jsonl). NEW NUMERIC CONFIG (paired max/update order); every caller shares this
// symbol via fa_decode_rows — battery + depth run-gen arbitrate.
extern "C" __global__ void fa_decode_vec_q_rows_dpl16_i2(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K,
        const uint8_t* __restrict__ V,
        float* __restrict__ partO,
        float* __restrict__ partM,
        float* __restrict__ partL,
        int head_dim, int n_head, int n_head_kv, const int* __restrict__ t_kv_base_dev, int base_plus,
        float scale, int n_splits_max, int split_keys,
        long k_tok_bytes, long v_tok_bytes)
{
    const int r        = blockIdx.z;
    const int T_kv     = t_kv_base_dev[0] + base_plus + r + 1;
    const int n_splits = (T_kv + split_keys - 1) / split_keys;
    const int kv_head  = blockIdx.x;
    const int split    = blockIdx.y;
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

    float q_reg[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            q_reg[i] = Q[((size_t)r * n_head + head) * head_dim + d] * scale;
        } else q_reg[i] = 0.0f;
    }

    float m_i = NEG_INF, l_i = 0.0f;
    float acc[FA_DEC_MAX_DPL16];
    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) acc[i] = 0.0f;

    const int kblk0 = (kv_head * head_dim) >> 5;
    int t = t_lo;
    for (; t + 1 < t_hi; t += 2) {
        const uint8_t* ka = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        const uint8_t* kb = K + (size_t)(t + 1) * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        float pa = 0.0f, pb = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
            if (i < dpl) {
                pa += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(ka + i * K_BLK_B, lane)));
                pb += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(kb + i * K_BLK_B, lane)));
            }
        }
        float sA = warp_reduce_sum(pa);
        float sB = warp_reduce_sum(pb);
        float m_new = fmaxf(m_i, fmaxf(sA, sB));
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float wA = exp2f((sA - m_new) * LOG2E);
        float wB = exp2f((sB - m_new) * LOG2E);
        const uint8_t* va = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        const uint8_t* vb = V + (size_t)(t + 1) * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
            if (i < dpl) {
                float vva = __bfloat162float(__float2bfloat16(dq_V_lane(va + i * V_BLK_B, lane)));
                float vvb = __bfloat162float(__float2bfloat16(dq_V_lane(vb + i * V_BLK_B, lane)));
                acc[i] = __fmaf_rn(acc[i], alpha, __fmaf_rn(wA, vva, __fmul_rn(wB, vvb)));
            }
        }
        l_i = l_i * alpha + wA + wB;
        m_i = m_new;
    }
    for (; t < t_hi; ++t) {   // odd tail: the serial walk body
        const uint8_t* kt = K + (size_t)t * k_tok_bytes + (size_t)kblk0 * K_BLK_B;
        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
            if (i < dpl) {
                part += q_reg[i] * __bfloat162float(__float2bfloat16(dq_K_lane(kt + i * K_BLK_B, lane)));
            }
        }
        float score = warp_reduce_sum(part);
        float m_new = fmaxf(m_i, score);
        float alpha = (m_i == NEG_INF) ? 0.0f : exp2f((m_i - m_new) * LOG2E);
        float p     = exp2f((score - m_new) * LOG2E);
        const uint8_t* vt = V + (size_t)t * v_tok_bytes + (size_t)kblk0 * V_BLK_B;
        #pragma unroll
        for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
            if (i < dpl) {
                acc[i] = __fmaf_rn(acc[i], alpha, __fmul_rn(p, __bfloat162float(__float2bfloat16(dq_V_lane(vt + i * V_BLK_B, lane)))));
            }
        }
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    #pragma unroll
    for (int i = 0; i < FA_DEC_MAX_DPL16; ++i) {
        if (i < dpl) {
            int d = lane + (i << 5);
            partO[(((size_t)r * n_head + head) * n_splits_max + split) * head_dim + d] = acc[i];
        }
    }
    if (lane == 0) {
        partM[((size_t)r * n_head + head) * n_splits_max + split] = m_i;
        partL[((size_t)r * n_head + head) * n_splits_max + split] = l_i;
    }
}




