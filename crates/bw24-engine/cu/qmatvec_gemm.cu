// qmatvec_gemm.cu — batched tensor-core int8 quant GEMM for the bw24 PREFILL path (sm_120a).
//
// THE 43x FIX. The dp4a matvec kernels (qmatvec.cu) index `wrow = W + o*row_bytes` once per
// (out-row o, token t): at T=512 every weight row is re-read & re-decoded 512x. That structural
// 512x weight re-read is the entire prefill gap (143 vs 6240 pp512). Here we tile so a weight
// block is DECODED-TO-INT8 ONCE into shared memory and reused across all N tokens via the int8
// tensor-core mma — amortizing the weight read/decode N-fold.
//
//   y[T, out_f] = aq[T, in_f](int8 q8_1) · W[out_f, in_f](quant)^T
//   scaled by activation block scales ad[T, in_f/32] x weight block scales dw[out_f, in_f/32].
//
// PRIMITIVE: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (int8, s32 accumulate). Chosen over
// bf16 (219 vs 117 TFLOP/s sm_120; keeps weights quantized = VRAM; reuses the q8_1 activation
// format quantize_q8_1 already produces). s32 accumulate is EXACT vs dp4a; only the final f32
// scale rounding differs. BK=32 == the q8_1/quant 32-block, so each K-step's s32 partial is
// scaled by exactly one (dw, da) pair and summed in f32 — bit-equivalent to the dp4a path's
// `acc += dw*ad*(float)sumi` (qmatvec.cu:407).
//
// FRAGMENT LOADING: we do NOT use ldmatrix. Instead each lane reads its required int8 operands
// straight from the smem tile per the canonical PTX m16n8k32.s8 fragment layout. This avoids the
// fragile int8 ldmatrix addressing entirely; the int8 mma is the throughput engine and ldmatrix
// is only a load optimization. Correctness is gated bit-exactly vs dp4a in kernel_check.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---- quant type codes (must match qmatvec.cu QType + lib.rs QT_*) ----
#define GQT_Q8_0  0
#define GQT_Q4_K  1
#define GQT_Q6_K  2
#define GQT_Q5_K  3
#define GQT_NVFP4 7

#define WARP_SZ 32
// CTA tile: BM output rows x BN tokens x BK contraction. 4 warps, each owns a 32x32 output quad
// (2 m16-row-frags x 4 n8-col-frags). One mma K-step = BK=32 (one quant/q8_1 block).
#define BM 64      // output rows per CTA  (4 warps x 16 rows)
#define BN 128     // tokens   per CTA (each warp covers all BN tokens; reduces weight re-decode)
#define BK 32      // contraction per K-step (== quant 32-block)
#define NWARP 4
#define WARP_M 16  // each warp's M rows (one m16 frag)

__device__ __forceinline__ float ghalf2float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ===================================================================== //
//  int8 mma m16n8k32: D[16x8] s32 += A[16x32] s8 * B[8x32](col) s8       //
//  A: 4 x .b32 regs/lane (16 int8). B: 2 x .b32/lane (8 int8). C/D: 4 s32/lane.
// ===================================================================== //
__device__ __forceinline__ void mma_s8_m16n8k32(int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ===================================================================== //
//  per-dtype: DECODE one weight 32-block to int8[32] + its f32 block scale, into smem.          //
//  `wrow` = W + o*row_bytes (the o-th out-row base). `g` = global 32-block index along in_f.     //
//  Returns dw (the f32 scale for this 32-block); writes 32 int8 weights to out[0..31].           //
//  Lifted from the dp4a inner-loop decode (qmatvec.cu) so the int weights MATCH bit-for-bit.     //
//  For min-offset quants (Q4_K) the block min*scale is folded into a per-block bias `bias` that  //
//  is applied (× activation block-sum) at scale time — exactly like the dp4a sumi_sum term.      //
// ===================================================================== //

// Q8_0: weight already int8. 34 B/block = fp16 d + int8[32].
__device__ __forceinline__ float decode_q8_0(const unsigned char* wrow, int g, int8_t* out, float* bias) {
    const unsigned char* b = wrow + (long)g * 34;
    *bias = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) out[j] = (int8_t)b[2 + j];
    return ghalf2float(*(const unsigned short*)b);
}

// Q4_K: superblock 256 / 144 B. group g&7 of 32; 6-bit sub scale/min. int weight = nibble (0..15).
// y_block = d*sc * dp4a(nibble, a) - dmin*mn * sum(a). We return the int = nibble, the scale =
// d*sc, and bias = -(dmin*mn) so scale-time does acc += scale*sumi + bias*sumA  (dp4a-identical).
__device__ __forceinline__ float decode_q4_k(const unsigned char* wrow, int g, int8_t* out, float* bias) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 144;
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs     = b + 16;
    unsigned char sc, mn;
    if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
    else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
           mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
    int chunk = grp >> 1;
    const unsigned char* q = qs + chunk * 32;
    bool hi = (grp & 1);
    #pragma unroll
    for (int j = 0; j < 32; j++) out[j] = (int8_t)(hi ? (q[j] >> 4) : (q[j] & 0xF));
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}

// Q5_K: superblock 256 / 176 B. group g&7 of 32 has ONE (sc, mn) 6-bit pair (like Q4_K).
// int weight = nibble | (qh-bit << 4) in [0,31]. scale = d*sc, bias = -dmin*mn (same fold as Q4_K).
__device__ __forceinline__ float decode_q5_k(const unsigned char* wrow, int g, int8_t* out, float* bias) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 176;
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* qs = b + 48;
    unsigned char sc, mn;
    if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
    else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
           mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
    int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
    const unsigned char* q = qs + g64 * 32;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int lowbits = hi ? (q[j] >> 4) : (q[j] & 0x0F);
        int h = (qh[j] >> hbit) & 1;
        out[j] = (int8_t)(lowbits | (h << 4));   // 0..31
    }
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}

// Q6_K: superblock 256 / 210 B. symmetric, no min. int weight = (ql|qh<<4)-32 in [-32,31].
// Per-32 block g&7 spans TWO 16-elem scale groups (is0/is1). We can't fold two scales into one
// f32-per-block, so we PRE-MULTIPLY the int weight by its 16-group scale ratio... no — instead we
// bake the per-element scale into int? No. Q6_K uses a single fp16 d * int8 scale per 16. The two
// halves of the 32-block have different int8 scales scn[is0], scn[is1]. To keep the s32 mma exact
// we scale by d at block level and absorb the per-16 int scale into the WEIGHT: out = w*scn (w in
// [-32,31], scn int8 -> product fits int16, but mma needs int8). That overflows int8. So Q6_K
// cannot be a single-scale 32-block. Handle by splitting: we store w (the -32..31 int) and return
// scale=d, and pass scn via a SECOND path — see decode_q6_k_split below; the kernel uses 16-wide
// sub-accumulation for Q6_K. For the unified path we approximate NOT allowed. (kept for ref.)

// NVFP4: 64-elem block / 36 B. per-16 sub UE4M3 scale. int weight = mxfp4 codebook value in
// [-12,12]. The 32-block g spans TWO 16-elem sub-blocks (own scale each) -> same two-scale issue
// as Q6_K. NVFP4 also has a per-TENSOR macro-scale applied post-matmul (scale_inplace), like dp4a.

// ===================================================================== //
//  GEMM kernel template (single-scale-per-32-block dtypes: Q8_0, Q4_K).  //
//  grid = (out_f/BM, ceil(T/BN), 1) ; block = (32, NWARP, 1) = 128 thr.   //
//  Each warp w owns output rows [BM-tile + w*16 .. +16), all BN tokens.   //
// ===================================================================== //
template<int QT>
__device__ __forceinline__ float decode_block(const unsigned char* wrow, int g, int8_t* out, float* bias);
template<> __device__ __forceinline__ float decode_block<GQT_Q8_0>(const unsigned char* w, int g, int8_t* o, float* b){ return decode_q8_0(w,g,o,b); }
template<> __device__ __forceinline__ float decode_block<GQT_Q4_K>(const unsigned char* w, int g, int8_t* o, float* b){ return decode_q4_k(w,g,o,b); }
template<> __device__ __forceinline__ float decode_block<GQT_Q5_K>(const unsigned char* w, int g, int8_t* o, float* b){ return decode_q5_k(w,g,o,b); }

// smem layout per CTA (double NOT buffered; correctness-first):
//   sW   : int8 [BM][BK]      weight tile (decoded once per K-step)
//   sA   : int8 [BN][BK]      activation tile (already int8 from quantize_q8_1)
//   sWd  : f32  [BM]          weight block scale (dw*sc)
//   sWb  : f32  [BM]          weight block bias  (-dmin*mn) for min-offset quants, else 0
//   sAd  : f32  [BN]          activation block scale (ad)
//   sAsum: f32  [BN]          activation block sum (sum of the 32 int8) for the bias term
// row-major; sW[r*BK + k], sA[n*BK + k].
template<int QT>
__device__ void qmatvec_gemm_kernel(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes)
{
    const int rowtile = blockIdx.x * BM;     // first out-row of this CTA
    const int toktile = blockIdx.y * BN;     // first token of this CTA
    const int warp = threadIdx.y;            // 0..NWARP-1
    const int lane = threadIdx.x;            // 0..31
    const int tid  = warp * WARP_SZ + lane;  // 0..127
    const int nblk = in_f / 32;

    __shared__ int8_t sW[BM][BK];
    __shared__ int8_t sA[BN][BK];
    __shared__ float  sWd[BM];
    __shared__ float  sWb[BM];
    __shared__ float  sAd[BN];
    __shared__ float  sAsum[BN];

    // each warp accumulates its 16 rows x BN(=64) tokens = 16x8 frags over the 8 n-tiles.
    // acc[ntile][4] s32. We apply scales per K-step into f32 (because dw/da vary per 32-block).
    float facc[BN / 8][4];   // BN/8 = 8 n-tiles per warp
    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++)
        #pragma unroll
        for (int i = 0; i < 4; i++) facc[nt][i] = 0.0f;

    // ===== K loop over 32-blocks =====
    for (int g = 0; g < nblk; g++) {
        // ---- decode weight tile: BM rows x 32, ONE 32-block per row. cooperative over 128 thr. ----
        // each thread decodes a subset of the BM=64 rows fully (row r owns 32 int8).
        for (int r = tid; r < BM; r += NWARP * WARP_SZ) {
            int o = rowtile + r;
            float bias = 0.0f, dw;
            if (o < out_f) {
                const unsigned char* wrow = W + (long)o * row_bytes;
                dw = decode_block<QT>(wrow, g, &sW[r][0], &bias);
            } else {
                dw = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) sW[r][k] = 0;
            }
            sWd[r] = dw; sWb[r] = bias;
        }
        // ---- load activation tile: BN tokens x 32 (already int8). + block scale + block sum. ----
        for (int n = tid; n < BN; n += NWARP * WARP_SZ) {
            int t = toktile + n;
            float as = 0.0f; int ssum = 0;
            if (t < T) {
                const signed char* arow = aq + (size_t)t * in_f + (size_t)g * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) { int8_t v = arow[k]; sA[n][k] = v; ssum += v; }
                as = ad[(size_t)t * nblk + g];
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) sA[n][k] = 0;
            }
            sAd[n] = as; sAsum[n] = (float)ssum;
        }
        __syncthreads();

        // ---- build A fragment for this warp's 16 rows (rows rowtile+warp*16 .. +16) ----
        // A is 16x32 s8: lane holds 4 .b32. Per the m16n8k32 layout:
        //   reg ai (0..3): for byte bj (0..3): row = lane/4 + (ai&1)*8; col = (lane%4)*4 + (ai>>1)*16 + bj
        int afrag[4];
        {
            const int8_t (*Wsub)[BK] = (const int8_t(*)[BK]) &sW[warp * WARP_M][0];
            #pragma unroll
            for (int ai = 0; ai < 4; ai++) {
                int row = lane / 4 + (ai & 1) * 8;
                int col0 = (lane % 4) * 4 + (ai >> 1) * 16;
                const int8_t* p = &Wsub[row][col0];
                afrag[ai] = (int)(uint8_t)p[0] | ((int)(uint8_t)p[1] << 8)
                          | ((int)(uint8_t)p[2] << 16) | ((int)(uint8_t)p[3] << 24);
            }
        }
        // ---- per 8-token n-tile: build B fragment + mma, then scale s32 -> f32 ----
        #pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            int bfrag[2];
            // B is 8x32 s8, col-major (.col): lane holds 2 .b32.
            //   reg bi (0..1): for byte bj (0..3): k = (lane%4)*4 + bi*16 + bj; n = lane/4
            #pragma unroll
            for (int bi = 0; bi < 2; bi++) {
                int ncol = nt * 8 + lane / 4;
                int k0 = (lane % 4) * 4 + bi * 16;
                const int8_t* p = &sA[ncol][k0];
                bfrag[bi] = (int)(uint8_t)p[0] | ((int)(uint8_t)p[1] << 8)
                          | ((int)(uint8_t)p[2] << 16) | ((int)(uint8_t)p[3] << 24);
            }
            int dacc[4] = {0, 0, 0, 0};
            mma_s8_m16n8k32(dacc, afrag, bfrag);
            // s32 partials -> f32 with this 32-block's (dw, da) scales (+ bias for min-quants).
            //   y += dw*da*sumi + bias*da*sumA   (exactly the dp4a fold)
            // C/D layout: reg ci (0..3): row = lane/4 + (ci>>1)*8 ; col = (lane%4)*2 + (ci&1)
            #pragma unroll
            for (int ci = 0; ci < 4; ci++) {
                int rr = warp * WARP_M + lane / 4 + (ci >> 1) * 8;  // 0..BM-1 GLOBAL tile row
                int nn = nt * 8 + (lane % 4) * 2 + (ci & 1);  // 0..63 token within tile
                float da = sAd[nn];
                facc[nt][ci] += sWd[rr] * da * (float)dacc[ci] + sWb[rr] * da * sAsum[nn];
            }
        }
        __syncthreads();
    }

    // ===== write out: y[t*out_f + o] (token-major). =====
    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++) {
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            int rr = lane / 4 + (ci >> 1) * 8;
            int nn = nt * 8 + (lane % 4) * 2 + (ci & 1);
            int o = rowtile + warp * WARP_M + rr;
            int t = toktile + nn;
            if (o < out_f && t < T) y[(size_t)t * out_f + o] = facc[nt][ci];
        }
    }
}

extern "C" __global__ void qmatvec_gemm_q8_0(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel<GQT_Q8_0>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
extern "C" __global__ void qmatvec_gemm_q4_K(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel<GQT_Q4_K>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
extern "C" __global__ void qmatvec_gemm_q5_K(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel<GQT_Q5_K>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}

// ===================================================================== //
//  Q6_K / NVFP4: the 32-block spans TWO 16-elem sub-blocks with distinct //
//  scales, so a single-scale-per-block mma cannot represent them. Use a  //
//  16-wide K-step: mma over BK=32 but split the 32-block into two 16-elem //
//  halves, each with its own (signed) scale. The int8 mma still consumes  //
//  32 k at once; we instead run TWO accumulators per 32-block — for each   //
//  half we ZERO the other 16 lanes of the A/B fragment so the s32 result   //
//  is the half-only dot, then scale each half by its own sub-scale.        //
//                                                                          //
//  Concretely: weight ints for the 32-block are stored once; we keep the   //
//  TWO sub-scales (s0,s1) per row. For the mma we run it once with all 32   //
//  (gives sum over both halves) — not separable. So instead we do the      //
//  half-split at the SCALE stage by running mma on each 16-half with the    //
//  other half zeroed in the WEIGHT tile (activation kept full; zeroed       //
//  weights contribute 0). Two mmas per 32-block, each k=32 with 16 zeros.   //
// ===================================================================== //

// Q6_K decode: writes int8[32] = (ql|qh<<4)-32 in [-32,31]; returns d (fp16) as block scale,
// and the TWO per-16 signed sub-scales via sc0/sc1 (int).  bias unused (symmetric).
__device__ __forceinline__ float decode_q6_k_2(const unsigned char* wrow, int g, int8_t* out,
                                               int* sc0, int* sc1) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 210;
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char*   scales = (const signed char*)(b + 192);
    float d = ghalf2float(*(const unsigned short*)(b + 208));
    int n   = grp >> 2;
    int run = grp & 3;
    const unsigned char* qlh = ql + n * 64;
    const unsigned char* qhh = qh + n * 32;
    const signed char*   scn = scales + n * 8;
    int ql_off = (run & 1) ? 32 : 0;
    int ql_hi  = (run >= 2);
    int qh_sh  = run * 2;
    #pragma unroll
    for (int il = 0; il < 32; il++) {
        int ql_bits = ql_hi ? (qlh[il + ql_off] >> 4) : (qlh[il + ql_off] & 0xF);
        int qh_bits = (qhh[il] >> qh_sh) & 3;
        out[il] = (int8_t)((ql_bits | (qh_bits << 4)) - 32);   // -32..31
    }
    *sc0 = (int)scn[run * 2 + 0];   // scale for k 0..15
    *sc1 = (int)scn[run * 2 + 1];   // scale for k 16..31
    return d;
}

// NVFP4 decode: writes int8[32] = mxfp4 codebook value in [-12,12]; returns 1.0 (block scale
// carried via the two UE4M3 sub-scales su0/su1, f32). per-TENSOR macro-scale applied post-matmul.
__device__ __constant__ signed char gkvalues_mxfp4[16] =
    {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ __forceinline__ float gue4m3_to_f32(unsigned char x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? ldexpf(man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}
// Fast 4-bit codebook lookup via __byte_perm (llama.cpp get_int_from_table_16). For 4 packed
// bytes (8 nibbles) returns .x = 4 codebook int8s of the LOW nibbles, .y = of the HIGH nibbles.
__device__ __forceinline__ int2 gtable16(int q4, const signed char* table) {
    const uint32_t* t = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t lhsel = (0x32103210u | ((q4 & 0x88888888u) >> 1));
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16u * i;
        const uint32_t lo = __byte_perm(t[0], t[1], (uint32_t)q4 >> sh);
        const uint32_t hi = __byte_perm(t[2], t[3], (uint32_t)q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, lhsel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
__device__ __forceinline__ void decode_nvfp4_2(const unsigned char* wrow, int g, int8_t* out,
                                              float* su0, float* su1) {
    int sblk = g >> 1;          // 64-elem block_nvfp4 (36 B)
    int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
    const unsigned char* b = wrow + (long)sblk * 36;
    const unsigned char* d_bytes = b;
    const unsigned char* qs = b + 4;
    int s0 = whichHalf * 2, s1 = s0 + 1;
    int* o32 = (int*)out;   // out is 4-byte aligned (int8_t[32] local/smem); write 4 int8 at once
    // sub s -> k 16*sl..; 8 qs bytes (low nibble = elem 0..7, high = elem 8..15).
    #pragma unroll
    for (int sl = 0; sl < 2; sl++) {
        const unsigned char* qss = qs + (s0 + sl) * 8;
        int q4a = (int)qss[0] | ((int)qss[1] << 8) | ((int)qss[2] << 16) | ((int)qss[3] << 24);
        int q4b = (int)qss[4] | ((int)qss[5] << 8) | ((int)qss[6] << 16) | ((int)qss[7] << 24);
        int2 va = gtable16(q4a, gkvalues_mxfp4);  // .x=elems0..3 (low) .y=elems8..11 (high)
        int2 vb = gtable16(q4b, gkvalues_mxfp4);  // .x=elems4..7        .y=elems12..15
        int base = sl * 4;   // 4 ints = 16 int8
        o32[base + 0] = va.x;  // elems 0..3
        o32[base + 1] = vb.x;  // elems 4..7
        o32[base + 2] = va.y;  // elems 8..11
        o32[base + 3] = vb.y;  // elems 12..15
    }
    *su0 = gue4m3_to_f32(d_bytes[s0]);
    *su1 = gue4m3_to_f32(d_bytes[s1]);
}

// Two-sub-scale GEMM (Q6_K, NVFP4). Splits each 32-block into two 16-halves with distinct scales.
// We run the mma TWICE per 32-block: once with the upper-16 weights zeroed (gives the lower-16
// dot) and once with the lower-16 zeroed (upper-16 dot), then scale each by its sub-scale.
// The activation full-32 is used both times; zeroed weight lanes contribute 0 to the s32 sum.
// Q6_K: sub-scale is int (scn) and overall block scale is d (f32) -> half_scale = d * scn * da.
// NVFP4: sub-scale is f32 (UE4M3) and block d is 1 -> half_scale = su * da. (macro-scale post.)
template<int QT>
__device__ void qmatvec_gemm_kernel2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes)
{
    const int rowtile = blockIdx.x * BM;
    const int toktile = blockIdx.y * BN;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp * WARP_SZ + lane;
    const int nblk = in_f / 32;

    __shared__ int8_t sWlo[BM][BK];   // lower-16 weights (upper zeroed)
    __shared__ int8_t sWhi[BM][BK];   // upper-16 weights (lower zeroed)
    __shared__ int8_t sA[BN][BK];
    __shared__ float  sS0[BM];        // lower-half scale factor (d*scn or su0)
    __shared__ float  sS1[BM];        // upper-half scale factor
    __shared__ float  sAd[BN];

    float facc[BN / 8][4];
    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++)
        #pragma unroll
        for (int i = 0; i < 4; i++) facc[nt][i] = 0.0f;

    for (int g = 0; g < nblk; g++) {
        for (int r = tid; r < BM; r += NWARP * WARP_SZ) {
            int o = rowtile + r;
            int8_t wq[32];
            float s0 = 0.0f, s1 = 0.0f;
            if (o < out_f) {
                const unsigned char* wrow = W + (long)o * row_bytes;
                if (QT == GQT_Q6_K) {
                    int sc0, sc1; float d = decode_q6_k_2(wrow, g, wq, &sc0, &sc1);
                    s0 = d * (float)sc0; s1 = d * (float)sc1;
                } else { // NVFP4
                    float su0, su1; decode_nvfp4_2(wrow, g, wq, &su0, &su1);
                    s0 = su0; s1 = su1;
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) wq[k] = 0;
            }
            #pragma unroll
            for (int k = 0; k < 16; k++) { sWlo[r][k] = wq[k]; sWlo[r][16 + k] = 0; }
            #pragma unroll
            for (int k = 0; k < 16; k++) { sWhi[r][k] = 0; sWhi[r][16 + k] = wq[16 + k]; }
            sS0[r] = s0; sS1[r] = s1;
        }
        for (int n = tid; n < BN; n += NWARP * WARP_SZ) {
            int t = toktile + n;
            float as = 0.0f;
            if (t < T) {
                const signed char* arow = aq + (size_t)t * in_f + (size_t)g * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) sA[n][k] = arow[k];
                as = ad[(size_t)t * nblk + g];
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) sA[n][k] = 0;
            }
            sAd[n] = as;
        }
        __syncthreads();

        // build two A fragments (lo / hi) for this warp's 16 rows
        int aflo[4], afhi[4];
        {
            const int8_t (*Wlo)[BK] = (const int8_t(*)[BK]) &sWlo[warp * WARP_M][0];
            const int8_t (*Whi)[BK] = (const int8_t(*)[BK]) &sWhi[warp * WARP_M][0];
            #pragma unroll
            for (int ai = 0; ai < 4; ai++) {
                int row = lane / 4 + (ai & 1) * 8;
                int col0 = (lane % 4) * 4 + (ai >> 1) * 16;
                const int8_t* pl = &Wlo[row][col0];
                const int8_t* ph = &Whi[row][col0];
                aflo[ai] = (int)(uint8_t)pl[0] | ((int)(uint8_t)pl[1] << 8)
                         | ((int)(uint8_t)pl[2] << 16) | ((int)(uint8_t)pl[3] << 24);
                afhi[ai] = (int)(uint8_t)ph[0] | ((int)(uint8_t)ph[1] << 8)
                         | ((int)(uint8_t)ph[2] << 16) | ((int)(uint8_t)ph[3] << 24);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            int bfrag[2];
            #pragma unroll
            for (int bi = 0; bi < 2; bi++) {
                int ncol = nt * 8 + lane / 4;
                int k0 = (lane % 4) * 4 + bi * 16;
                const int8_t* p = &sA[ncol][k0];
                bfrag[bi] = (int)(uint8_t)p[0] | ((int)(uint8_t)p[1] << 8)
                          | ((int)(uint8_t)p[2] << 16) | ((int)(uint8_t)p[3] << 24);
            }
            int dlo[4] = {0,0,0,0}, dhi[4] = {0,0,0,0};
            mma_s8_m16n8k32(dlo, aflo, bfrag);
            mma_s8_m16n8k32(dhi, afhi, bfrag);
            #pragma unroll
            for (int ci = 0; ci < 4; ci++) {
                int rr = warp * WARP_M + lane / 4 + (ci >> 1) * 8;  // GLOBAL tile row
                int nn = nt * 8 + (lane % 4) * 2 + (ci & 1);
                float da = sAd[nn];
                facc[nt][ci] += (sS0[rr] * (float)dlo[ci] + sS1[rr] * (float)dhi[ci]) * da;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++) {
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            int rr = lane / 4 + (ci >> 1) * 8;
            int nn = nt * 8 + (lane % 4) * 2 + (ci & 1);
            int o = rowtile + warp * WARP_M + rr;
            int t = toktile + nn;
            if (o < out_f && t < T) y[(size_t)t * out_f + o] = facc[nt][ci];
        }
    }
}

extern "C" __global__ void qmatvec_gemm_q6_K(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel2<GQT_Q6_K>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
extern "C" __global__ void qmatvec_gemm_nvfp4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel2<GQT_NVFP4>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
