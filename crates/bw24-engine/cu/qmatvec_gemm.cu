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
// FRAGMENT LOADING: ldmatrix.sync (x4.b16 for A weights, x2.b16 for B activations) reinterpreted
// for s8 — loads the EXACT bytes the old scalar byte-assembly produced (bit-identical mma input,
// gated vs dp4a in kernel_check). PIPELINE: NSTAGE=3 cp.async ring buffer overlaps the next K-step's
// activation global->smem copy behind the current mma. Single __syncthreads/K-step (the top barrier
// guards both cur's visibility and the WAR for the post-barrier prefetch). __launch_bounds__(128,4)
// caps regs so 4 CTAs/SM co-reside.
//
// FIX A (pre-decode, PERF-1d): for Q4_K/Q5_K the RAW quant superblock is cp.async'd into a smem ring
// (sWraw) one superblock ahead, then ALU-decoded from RESIDENT smem during prefetch — so the long-
// scoreboard global weight read leaves the mma chain (ncu: Q4_K 2.61->1.88) and weight DRAM traffic
// drops 8-fold. See StageMeta below: PREDEC is set PER-DTYPE by measured pp512 (Q4_K/Q5_K win; NVFP4/
// Q6_K/Q8_0 KEEP the inline-global decode — pre-decode's extra smem dropped their occupancy and
// REGRESSED them, same occupancy-bound tradeoff as the reverted swizzle-pad/tile-redesign). FIX B
// (barrier batching via deeper NSTAGE) was tested and REJECTED: NSTAGE=4 regressed pp512 1287->1220
// (more smem -> fewer CTAs/SM); this tile is occupancy-bound, not barrier-frequency-bound at depth.

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
// CTA tile: BM output rows x BN tokens x BK contraction. One mma K-step = BK=32 (one quant/q8_1 block).
//
// MMQ-PORT (llama mmq.cuh structure on this sm_120 silicon): the single-scale GEMM (kernel1, used by
// Q8_0/Q4_K/Q5_K) runs the PROVEN llama layout — 8 WARPS/CTA + a row-PADDED weight smem tile + a small
// per-warp accumulator — instead of the prior 4-warp / 64-reg big-accum that left this kernel
// occupancy-bound at 2 CTA/SM with ~20-24M smem bank conflicts (ncu, this box). The two structural
// levers ported (NOT a bw24 re-tile guess, which reverted twice as PERF-1c):
//   (1) 8 warps (NWARP=8, MMQ_NWARPS=8 in mmq.cuh:15) split as a 2D warp grid: NTX=2 token-groups x
//       NWY=4 row-groups. Each warp owns WARP_M=16 rows (one m16 frag) x BN/NTX=64 tokens (8 n-tiles).
//       facc[BN/NTX/8][4] = 8x4 = 32 s32->f32 accum regs/warp (HALF the old facc[16][4]=64) so 8 warps
//       co-reside (the granularity/rows_per_warp accumulator of vec_dot_q8_1_q8_1_mma, mmq.cuh:1384).
//   (2) the x_qs ROW PAD: llama's x_qs tile stride is MMQ_MMA_TILE_X_K_Q8_1 = 2*32+8+4 ints, the +4
//       making stride%8==4 to break smem bank conflicts BY CONSTRUCTION (mmq.cuh:219-227, why llama has
//       0.13M conflicts vs bw24 24M). bw24 analog: pad the int8 weight tile row from BK=32 to SW_STRIDE
//       =36 bytes (=9 ints, odd -> the row-to-row bank step is now coprime to 32, scattering ldmatrix +
//       decode-store accesses across banks). The decoded weights are still the SAME bytes (decode math
//       unchanged) -> bit-exact; only the storage stride differs. The mma atom is unchanged
//       (mma_s8_m16n8k32, == llama's tile<16,8> mma.cuh:942).
#define BM 64      // output rows per CTA  (4 row-groups x 16 rows)
// BN=256 (was 128): ncu showed the prefill GEMM is SMEM-BANDWIDTH-SATURATED (L1 70%, issue 28%) at
// 2 CTAs/SM. A bigger token tile (256) doubles weight REUSE per CTA (1 weight-decode serves 256
// tokens vs 128) = HALF the weight-smem traffic per output, and the larger smem pushes toward 1
// CTA/SM (llama's winning config: 1 CTA, big tile, owns all L1 bw). facc grows 32->64 regs/warp.
#define BN 256     // tokens per CTA (NTX token-groups; weight reused across all BN)
#define BK 32      // contraction per K-step (== quant 32-block)
#define NWARP 8    // MMQ-PORT (kernel1: Q8_0/Q4_K/Q5_K): 8 warps/CTA (was 4) — hide mma+decode latency
// ARITHMETIC-INTENSITY LEVER (2026-06-28): MFRAG m16-fragments per warp. ncu showed llama's 5451 GEMM
// wins via higher SM throughput (40% vs bw24 22%) = more mma per smem-load. MFRAG=2: each warp owns 2
// m16 row-frags (32 rows) and REUSES each loaded B-frag across BOTH -> per K-step a warp does 8 mma
// from 4 B-loads + 2 A-loads (6 smem loads) instead of 8 mma from 8 B-loads + 1 A-load (9 loads) =
// 33% fewer smem transactions per mma, NO extra smem tile, same 32 accum regs. With MFRAG=2 the grid
// is NTX=4 token-groups x NWY=2 row-groups; each row-group covers WARP_M*MFRAG=32 rows (NWY*32=BM=64).
#define MFRAG 2
#define NTX 4      // token-groups (MFRAG=2 -> NTX=4 so NWY=2 row-groups cover BM=64 at 32 rows each)
#define NWY (NWARP / NTX)  // row-groups (=2): each covers WARP_M*MFRAG=32 rows
// kernel2 (Q6_K/NVFP4 two-sub-scale) and the FP4 mxf4 kernel KEEP the original 4-warp / all-tokens-per-warp
// layout (out of scope for the MMQ k-quant port; their two-scale split + FP4 ldmatrix swizzle are tuned
// for it). They use NWARP2 and are launched with block (32, NWARP2). Only kernel1 takes the 8-warp port.
#define NWARP2 4
#define WARP_M 16  // each warp's M rows (one m16 frag)
#define WARP_N (BN / NTX)  // each warp's tokens (=64 = 8 n-tiles of 8)
// MMQ-PORT weight smem int8 row stride (bytes). KEPT at BK=32 (no pad). MEASURED FINDING: llama's
// x_qs %8==4 pad is conflict-free for ITS 76-int (2-K-block) row + load_ldmatrix k-offset addressing,
// but bw24's ld_A_s8 per-lane addr = (lane%16)*S4 + (lane/16)*4 has a (lane/16)*4 two-8-row-group split
// that, for ANY 16B-aligned stride S4 (the ldmatrix alignment constraint forces S4 % 4 == 0), keeps the
// two halves bank-aligned mod 32 -> >=4-way conflict regardless of pad (bank-model verified: strides 8/
// 12/16/20/24/28/36 ints ALL give >=4-way; the conflict-free 10/17-int strides break ldmatrix's 16B
// alignment). So a row pad cannot make THIS A-load conflict-free (matches bw24's prior no-op pad results)
// and only costs smem/occupancy. The MMQ structural win here is the 8-warp + small-accumulator, NOT the
// pad; the residual smem-LD conflicts are the same population the dp4a-fold already tolerates. (Killing
// them needs an XOR-swizzle of the ldmatrix chunk index like the FP4 path's SWZ_CHUNK, a separate lever.)
#define SW_STRIDE BK
// cp.async ring buffer depth (overlap next K-step's global->smem behind current mma).
#define NSTAGE 3

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

// ---- async-copy + ldmatrix helpers (sm_120a: cp.async.cg + ldmatrix.sync are native) ----
// 16-byte async global->smem copy (bypasses the register file). `smem` must be 16B-aligned in
// shared space; `g` is a generic global pointer. Caller commits + waits the group.
__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
// FIX A (pre-decode): cp.async a RAW weight-superblock window of N 16B chunks. Source `g16` MUST be
// 16B-aligned (we pass the 16B-FLOOR of the superblock byte offset); dst `smem` 16B-aligned. The
// decode later reads at the recorded phase = (off & 15) inside the staged window -> bit-identical
// bytes to a direct global read (the gate proves this). One lane copies the whole window (per-row);
// the windows are tiny (<=15 chunks) and amortized 1/superblock, so single-lane is fine.
__device__ __forceinline__ void cp_async_window(void* smem, const void* g16, int nchunk) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    const char* src = (const char*)g16;
    #pragma unroll 1
    for (int c = 0; c < nchunk; c++)
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s + (uint32_t)(c * 16)), "l"(src + c * 16));
}
// EMPIRICAL ldmatrix.x4.b16 lane->row map (probe/int8_swizzle_aload.cu dump_map, sm_120a, 2026-06-28):
// lane L's 4 regs hold A-rows by the m8n8 QUADRANT, NOT "row L%16": measured L0..2 -> rows {0,8},
// L16,17 -> rows {4,12}, i.e. row = (L/16)*4 + quadrant-row, regs[0..3]->R regs[4..7]->R+8 (repeat).
// The naive "lane reads its own row" swizzle = 32 frag mismatches (DISPROVED). A conflict-killing
// manual swizzled load must preserve THIS map. (de-risk lever for the int8 A-load 21M-conflict fix.)
// ldmatrix x4.b16: per-lane addr = (lane%16)*stride_b16 + (lane/16)*4 (in .b16 units), built as a
// 32-bit .shared address (proven in flash_attn.cu ld_A / mma_validate.cu). Loads 4x .b32 = 16
// int8 A-operands in the exact m16n8k32.s8 A-fragment layout the scalar byte-assembly produced.
// SWIZZLED int8 A-load (probe/int8_swizzle_aload.cu PROVEN: 0 frag mismatch vs ldmatrix, A-load
// conflict 4-way->2-way). Manual load matching ldmatrix's empirical map (lane L -> rows {L/4, L/4+8},
// k-word L%4) with physical word-col XORed by (row&7) — pairs with a decode-store that writes word c
// of row r at phys col (c^(r&7)). stride_bytes must be 32 (8 u32 words/row, the swizzle domain).
__device__ __forceinline__ void ld_A_s8_sw(int (&t)[4], const int8_t* base, int stride_bytes) {
    int L = threadIdx.x;
    int r0 = L / 4, r8 = r0 + 8, kw = L % 4;
    const uint32_t* p0 = (const uint32_t*)(base + r0 * stride_bytes);
    const uint32_t* p8 = (const uint32_t*)(base + r8 * stride_bytes);
    t[0] = p0[kw       ^ (r0 & 7)];
    t[1] = p8[kw       ^ (r8 & 7)];
    t[2] = p0[(kw + 4) ^ (r0 & 7)];
    t[3] = p8[(kw + 4) ^ (r8 & 7)];
}
__device__ __forceinline__ void ld_A_s8(int (&t)[4], const int8_t* base, int stride_bytes) {
    const uint32_t* xs = (const uint32_t*)base + (threadIdx.x % 16) * (stride_bytes / 4) + (threadIdx.x / 16) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(t[0]), "=r"(t[1]), "=r"(t[2]), "=r"(t[3]) : "r"(addr));
}
// ldmatrix x2.b16 for the m16n8k32.s8 B operand (.col): 8 tokens x 32 k, source sA[token][k]
// row-major (k contiguous, stride_bytes row pitch). NON-trans: reg0=k0..15, reg1=k16..31, no swap.
// Per-lane row base = (lane%8) rows of matrix (lane/8 %2). All offsets multiple of 16 -> aligned.
__device__ __forceinline__ void ld_B_s8(int (&t)[2], const int8_t* base, int stride_bytes) {
    const uint32_t* xs = (const uint32_t*)base + (threadIdx.x % 8) * (stride_bytes / 4) + ((threadIdx.x / 8) % 2) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
        : "=r"(t[0]), "=r"(t[1]) : "r"(addr));
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

// ===================================================================== //
//  FIX A — PRE-DECODE staging metadata + smem-source decodes.                                    //
//  Per dtype: SB_BYTES = bytes of the superblock that holds GPSB consecutive 32-blocks; GPSB =   //
//  32-blocks (K-steps) sharing one superblock. The pre-decode pipeline cp.async's the superblock //
//  (a 16B-floored window of SB_BYTES) into smem ONCE per GPSB K-steps, then ALU-decodes group     //
//  g&(GPSB-1) from the resident superblock each K-step -> the global read leaves the mma chain    //
//  AND superblock traffic drops GPSB-fold. The smem decodes below are byte-identical to the       //
//  global decodes above: same intra-superblock indexing, just a different (smem) base. The bit-   //
//  equivalence gate (kernel_check, rel<1e-3) proves identity.                                     //
//  RAW_W (per-row staged-window bytes) = roundup(15 + SB_BYTES, 16): max phase 15 + the block.    //
// ===================================================================== //
// FIX A (pre-decode): PREDEC=1 cp.async's the RAW quant superblock into smem one superblock ahead and
// ALU-decodes from RESIDENT smem -> the long-scoreboard global weight read leaves the mma chain (ncu:
// Q4_K long-scoreboard 2.61->1.88, NVFP4 2.61->0.47) AND weight DRAM traffic drops GPSB-fold. BUT the
// raw ring costs smem and drops CTAs/SM. PREDEC is set PER-DTYPE by MEASURED net pp512 (A/B on 9B):
//   Q4_K/Q5_K (PREDEC=1): +6% — the long-scoreboard win beats the occupancy loss (43.5/47.6KB -> 2 CTAs).
//   NVFP4    (PREDEC=0): pre-decode REGRESSED pp512 (~-3%, 1287->1240 alone) — its inline decode is
//                        light-ALU/L2-resident, so the +8KB raw ring (4->3 CTAs) costs more than it saves.
//   Q8_0     (PREDEC=0): GPSB=1, no superblock sharing; decode is a trivial int8 memcpy (mild stall).
//   Q6_K     (PREDEC=0): 210B 2-aligned window busts the 48KB static-smem cap at NSTAGE_RAW=2.
// GPSB = 32-blocks per superblock; SB_BYTES = superblock bytes; RAW_W = roundup(15 + SB_BYTES, 16).
template<int QT> struct StageMeta;
template<> struct StageMeta<GQT_Q8_0>{ enum{ SB_BYTES=34,  GPSB=1, RAW_W=48,  PREDEC=0 }; };
template<> struct StageMeta<GQT_Q4_K>{ enum{ SB_BYTES=144, GPSB=8, RAW_W=160, PREDEC=1 }; };
template<> struct StageMeta<GQT_Q5_K>{ enum{ SB_BYTES=176, GPSB=8, RAW_W=192, PREDEC=1 }; };
template<> struct StageMeta<GQT_Q6_K>{ enum{ SB_BYTES=210, GPSB=8, RAW_W=240, PREDEC=0 }; };
template<> struct StageMeta<GQT_NVFP4>{ enum{ SB_BYTES=36,  GPSB=2, RAW_W=64,  PREDEC=0 }; };  // 64-elem block
// superblock byte offset within a weight row for K-block g (== the `b - wrow` of the global decode)
template<int QT> __device__ __forceinline__ long sb_byte_off(int g);
template<> __device__ __forceinline__ long sb_byte_off<GQT_Q8_0>(int g){ return (long)g * 34; }
template<> __device__ __forceinline__ long sb_byte_off<GQT_Q4_K>(int g){ return (long)(g>>3) * 144; }
template<> __device__ __forceinline__ long sb_byte_off<GQT_Q5_K>(int g){ return (long)(g>>3) * 176; }
template<> __device__ __forceinline__ long sb_byte_off<GQT_Q6_K>(int g){ return (long)(g>>3) * 210; }
template<> __device__ __forceinline__ long sb_byte_off<GQT_NVFP4>(int g){ return (long)(g>>1) * 36; }

// --- smem-source decodes (kernel1 single-scale dtypes): `b` = staged superblock base in smem
//     (== sWraw_row + phase), `grp` = g & (GPSB-1). Bodies VECTORIZED (4-byte int LDS + SIMD nibble/
//     bit unpack + packed int store) to kill the per-byte LDS bank conflicts (the smem-pipe bound;
//     ncu: q5_K 72M->24M conflicts, q4_K 43M->20M, short_scoreboard q5_K 4.02->2.66, l1tex no longer
//     saturated). Math byte-identical to the scalar global decodes (kernel_check rel BIT-IDENTICAL).
//   ALIGNMENT FINDING (the int-LDS gotcha): the int reads (q4[]/qh4[]/scales-int) require `b` 4B-aligned.
//     `b = &sWraw[rs][r][phase]` with phase = (o*row_bytes + sb*SB_BYTES) & 15. For the PREDEC dtypes
//     Q4_K (SB_BYTES=144=16*9) and Q5_K (SB_BYTES=176=16*11), AND row_bytes = (in_f/256)*SB_BYTES is a
//     multiple of 16, so the byte offset is ALWAYS a multiple of 16 -> phase==0 -> b == sWraw row base.
//     sWraw is __align__(16) and RAW_W (160/192) is a multiple of 16, so the row base is 16B-aligned.
//     Therefore b, b+4 (scales), b+16 (qs/qh), b+48 (q5 qs) are all >=4B-aligned and the int LDS are
//     legal. (Q6_K SB_BYTES=210 is NOT 16-mult -> phase varies; but Q6_K is PREDEC=0/global LDG, not
//     a smem-conflict source and not 4B-aligned, so it KEEPS the scalar byte decode — left untouched.) ---
__device__ __forceinline__ float decode_q8_0_s(const unsigned char* b, int /*grp*/, int8_t* out, float* bias) {
    *bias = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) out[j] = (int8_t)b[2 + j];
    return ghalf2float(*(const unsigned short*)b);
}
// VECTORIZED smem-source scale unpack (Q4_K/Q5_K identical 6-bit scale/min layout). `scales` =
// b+4 (12 bytes, 4B-aligned since b is 16B-aligned -> phase==0 for these dtypes). Replaces the
// per-byte scales[] LDS.U8 reads with THREE 4-byte int LDS (scales[0..3], [4..7], [8..11]) + the
// SAME shift/mask math. Math BYTE-IDENTICAL to the scalar form: each `scales[i]` below is the i-th
// byte extracted from the int words via (>> (8*phase)) & 0xFF, so the produced (sc, mn) are bit-exact.
__device__ __forceinline__ void unpack_scales_q45_K_s(const unsigned char* scales, int grp,
                                                      unsigned char* sc, unsigned char* mn) {
    const int* si = (const int*)scales;                   // 4B-aligned: si[0]=B0..3, si[1]=B4..7, si[2]=B8..11
    int lo = si[0], md = si[1], hi32 = si[2];
    // byte extractor: word for byte i, shifted/masked to that byte's value
    auto byte = [&](int i) -> unsigned {
        int w = (i < 4) ? lo : ((i < 8) ? md : hi32);
        int p = i & 3;
        return ((unsigned)w >> (8 * p)) & 0xFFu;
    };
    if (grp < 4) { *sc = byte(grp) & 63; *mn = byte(grp + 4) & 63; }
    else { *sc = (byte(grp + 4) & 0xF) | ((byte(grp - 4) >> 6) << 4);
           *mn = (byte(grp + 4) >> 4) | ((byte(grp) >> 6) << 4); }
}
__device__ __forceinline__ float decode_q4_k_s(const unsigned char* b, int grp, int8_t* out, float* bias) {
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs     = b + 16;
    unsigned char sc, mn;
    unpack_scales_q45_K_s(scales, grp, &sc, &mn);
    int chunk = grp >> 1;
    const unsigned char* q = qs + chunk * 32;             // chunk*32 keeps 16B alignment (qs is 16B-aligned)
    bool hi = (grp & 1);
    // VECTORIZED: 8 int LDS (q4[0..7]) instead of 32 byte LDS; SIMD nibble unpack 4-at-a-time; packed
    // int store. (qw & 0x0F0F0F0F) = low nibbles, ((qw>>4)&0x0F0F0F0F) = high nibbles of 4 bytes ->
    // the 4 int8 outputs (0..15) for those 4 positions, byte-identical to the scalar (q[j]&0xF)/(q[j]>>4).
    const int* q4 = (const int*)q;                        // 16B-aligned
    int* o32 = (int*)out;                                 // out (sW row) is 16B-aligned
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        int qw = q4[w];
        o32[w] = hi ? ((qw >> 4) & 0x0F0F0F0F) : (qw & 0x0F0F0F0F);
    }
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}
__device__ __forceinline__ float decode_q5_k_s(const unsigned char* b, int grp, int8_t* out, float* bias) {
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* qs = b + 48;
    unsigned char sc, mn;
    unpack_scales_q45_K_s(scales, grp, &sc, &mn);
    int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
    const unsigned char* q = qs + g64 * 32;               // g64*32 keeps 16B alignment (qs is 16B-aligned)
    // VECTORIZED: 8 int LDS for q (low/high nibble) + 8 int LDS for qh (the 5th bit) instead of 32+32
    // byte LDS (this is the WORST conflict source, 72M). SIMD: low nibble = (qw & 0x0F0F0F0F) or
    // ((qw>>4)&...); 5th bit per byte = ((qhw >> hbit) & 0x01010101) << 4. out byte = lowbits | (h<<4),
    // byte-identical to the scalar (q[j]&0xF | ((qh[j]>>hbit)&1)<<4). qh is 16B-aligned (b+16).
    const int* q4  = (const int*)q;                       // 16B-aligned
    const int* qh4 = (const int*)qh;                      // 16B-aligned (b+16)
    int* o32 = (int*)out;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        int qw  = q4[w];
        int qhw = qh4[w];
        int low = hi ? ((qw >> 4) & 0x0F0F0F0F) : (qw & 0x0F0F0F0F);
        int h   = ((qhw >> hbit) & 0x01010101) << 4;      // 5th bit of each of the 4 bytes, placed in bit4
        o32[w] = low | h;                                 // 0..31 per byte
    }
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}
template<int QT>
__device__ __forceinline__ float decode_block_s(const unsigned char* b, int grp, int8_t* out, float* bias);
template<> __device__ __forceinline__ float decode_block_s<GQT_Q8_0>(const unsigned char* b,int grp,int8_t* o,float* bs){ return decode_q8_0_s(b,grp,o,bs); }
template<> __device__ __forceinline__ float decode_block_s<GQT_Q4_K>(const unsigned char* b,int grp,int8_t* o,float* bs){ return decode_q4_k_s(b,grp,o,bs); }
template<> __device__ __forceinline__ float decode_block_s<GQT_Q5_K>(const unsigned char* b,int grp,int8_t* o,float* bs){ return decode_q5_k_s(b,grp,o,bs); }

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
    const int warp = threadIdx.y;            // 0..NWARP-1 (8)
    const int lane = threadIdx.x;            // 0..31
    const int tid  = warp * WARP_SZ + lane;  // 0..255
    const int nblk = in_f / 32;
    // MMQ-PORT 2D warp grid (llama ntx split): wy = row-group (0..NWY-1), wx = token-group (0..NTX-1).
    // This warp owns rows [wy*WARP_M .. +WARP_M) of the BM tile and tokens [wx*WARP_N .. +WARP_N) of BN.
    const int wy = warp / NTX;               // 0..NWY-1 -> which row band (covers WARP_M*MFRAG rows)
    const int wx = warp % NTX;               // 0..NTX-1 -> which token group
    const int rowbase = wy * (WARP_M * MFRAG); // this warp's first tile row; owns MFRAG m16-frags
    const int tokbase = wx * WARP_N;         // this warp's first tile token (WARP_N = BN/NTX = 32)
    const int cta_threads = NWARP * WARP_SZ; // 8 warps = 256 (grid-stride over BM/BN fetch+decode)

    // NSTAGE-deep ring buffer: stage s holds the decoded weight tile, async-copied activation
    // tile, and per-tile scales for one K-step. The already-int8 activations are cp.async'd straight
    // from global. FIX A: the RAW quant superblock is cp.async'd into sWraw (a separate NSTAGE_RAW
    // ring keyed by SUPERBLOCK) one superblock ahead; the ALU decode reads that RESIDENT smem (not
    // global) during prefetch -> the long-scoreboard global weight read leaves the mma chain, and
    // superblock DRAM traffic drops GPSB-fold (8x for Q4_K/Q5_K).
    // MMQ-PORT: weight tile row-PADDED to SW_STRIDE (llama x_qs %8==4 pad) -> conflict-free decode-store +
    // ldmatrix. Decoded 32 int8 live in [0..31]; [32..47] is the bank-conflict-breaking pad.
    __shared__ __align__(16) int8_t sW[NSTAGE][BM][SW_STRIDE];
    __shared__ int8_t sA[NSTAGE][BN][BK];   // BK (16-aligned) so cp.async.cg 16B copies are legal
    __shared__ float  sWd[NSTAGE][BM];
    __shared__ float  sWb[NSTAGE][BM];
    __shared__ float  sAd[NSTAGE][BN];
    __shared__ float  sAsum[NSTAGE][BN];
    // raw superblock ring (FIX A). FIX A applies to GPSB>=2 dtypes (Q4_K/Q5_K, GPSB=8): a 16B-floored
    // superblock window cp.async'd ONE superblock ahead (lead = GPSB K-steps >= NSTAGE, so always landed
    // at the per-iter wait_group), decoded from RESIDENT smem -> global read off the mma chain + GPSB-fold
    // less weight DRAM. Keyed by superblock; depth NSTAGE_RAW=2 (decoding-sb + 1 prefetched). Q8_0 (GPSB=1)
    // has no superblock sharing AND a 1-step lead can't beat the wait_group, so it KEEPS the inline-global
    // decode (USE_PREDECODE=false): the dead raw ring compiles to nothing (NSTAGE_RAW*BM*1).
    enum { GPSB = StageMeta<QT>::GPSB, USE_PREDECODE = StageMeta<QT>::PREDEC,
           RAW_W = USE_PREDECODE ? (int)StageMeta<QT>::RAW_W : 1, NSTAGE_RAW = 2 };
    __shared__ __align__(16) unsigned char sWraw[NSTAGE_RAW][BM][RAW_W];

    // MMQ-PORT small accumulator: each warp owns WARP_M=16 rows x WARP_N=64 tokens = one m16 frag x
    // (WARP_N/8=8) n-tiles. facc[8][4] = 32 s32->f32 regs/warp (HALF the old facc[16][4]=64), the
    // granularity/rows-per-warp accumulator that lets 8 warps co-reside. Scales applied per K-step
    // into f32 (dw/da vary per 32-block).
    // MFRAG m16-frags x (WARP_N/8) n-tiles, 4 accum regs each = MFRAG*4*4 = 32 regs (same as before).
    float facc[MFRAG][WARP_N / 8][4];
    #pragma unroll
    for (int mf = 0; mf < MFRAG; mf++)
        #pragma unroll
        for (int nt = 0; nt < WARP_N / 8; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) facc[mf][nt][i] = 0.0f;

    // ---- FETCH: cp.async the RAW superblock `sb` (== g/GPSB) into the raw ring (one row per out-row),
    //      from the 16B-FLOOR of its byte offset; record nothing (phase recomputed at decode). Issued
    //      only at superblock boundaries (caller gates on g%GPSB==0). ----
    auto fetch_superblock = [&](int sb) {
        int rs = sb % NSTAGE_RAW;
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            if (o < out_f) {
                long off = (long)o * row_bytes + (long)sb * (long)StageMeta<QT>::SB_BYTES;
                long aoff = off & ~(long)15;          // 16B floor of the superblock
                int  phase = (int)(off - aoff);       // 0..15
                int  nchunk = (phase + (int)StageMeta<QT>::SB_BYTES + 15) >> 4;
                cp_async_window(&sWraw[rs][r][0], W + aoff, nchunk);
            }
        }
    };
    // ---- cp.async the activation tile for K-block g into stage `s` (unchanged from the inline path). ----
    auto fetch_activation = [&](int s, int g) {
        for (int n = tid; n < BN; n += cta_threads) {
            int t = toktile + n;
            if (t < T) {
                const signed char* arow = aq + (size_t)t * in_f + (size_t)g * 32;
                cp_async16(&sA[s][n][0],  arow);
                cp_async16(&sA[s][n][16], arow + 16);
                sAd[s][n] = ad[(size_t)t * nblk + g];
                const int* aw = (const int*)arow;     // 16B-aligned (in_f%16==0, g*32)
                int ssum = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++) ssum = __dp4a(aw[w], 0x01010101, ssum);
                sAsum[s][n] = (float)ssum;
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) sA[s][n][k] = 0;
                sAd[s][n] = 0.0f; sAsum[s][n] = 0.0f;
            }
        }
    };
    // ---- DECODE group g (off the mma chain): ALU-unpack the RESIDENT raw superblock smem -> sW[s].
    //      Reads sWraw[(g/GPSB)%NSTAGE_RAW] at phase = (rowbyteoff & 15); bit-identical to the global
    //      decode (same intra-superblock math, smem base). Result visible at the next barrier. ----
    auto decode_stage = [&](int s, int g) {
        int rs  = (g / GPSB) % NSTAGE_RAW;
        int grp = g & (GPSB - 1);
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            float bias = 0.0f, dw;
            if (o < out_f) {
                long off = (long)o * row_bytes + (long)(g / GPSB) * (long)StageMeta<QT>::SB_BYTES;
                int  phase = (int)(off & 15);
                const unsigned char* b = &sWraw[rs][r][phase];
                dw = decode_block_s<QT>(b, grp, &sW[s][r][0], &bias);
            } else {
                dw = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) sW[s][r][k] = 0;
            }
            sWd[s][r] = dw; sWb[s][r] = bias;
        }
    };
    // ---- INLINE decode (Q8_0 fallback): read global weight + decode straight into sW[s] (the original
    //      path). Used only when !USE_PREDECODE; the global read is on the chain but Q8_0's decode is a
    //      trivial memcpy of already-int8 weights, so the long-scoreboard is mild for it. ----
    auto decode_stage_inline = [&](int s, int g) {
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            float bias = 0.0f, dw;
            if (o < out_f) {
                const unsigned char* wrow = W + (long)o * row_bytes;
                dw = decode_block<QT>(wrow, g, &sW[s][r][0], &bias);
            } else {
                dw = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) sW[s][r][k] = 0;
            }
            sWd[s][r] = dw; sWb[s][r] = bias;
        }
    };

    const int nsb = (nblk + GPSB - 1) / GPSB;            // total superblocks along K
    if (USE_PREDECODE) {
        // ===== PROLOGUE (pre-decode): seed raw superblocks 0..1 (read by prologue stages + loop's
        //       first decode) + prologue activations; drain; decode prologue stages from resident smem.
        //       seed count (2) == NSTAGE_RAW so seeds land in distinct slots. =====
        #pragma unroll
        for (int sb = 0; sb < NSTAGE_RAW; sb++)
            if (sb < nsb) fetch_superblock(sb);
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++)
            if (s < nblk) fetch_activation(s, s);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");          // drain: raw superblocks + activations resident
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++)
            if (s < nblk) decode_stage(s, s);            // decode prologue stages from resident raw smem
        asm volatile("cp.async.commit_group;");          // keep per-iter commit cadence (empty group ok)
    } else {
        // ===== PROLOGUE (inline): original path — fetch activation + inline-global decode per stage. ==
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++) {
            if (s < nblk) { decode_stage_inline(s, s); fetch_activation(s, s); }
            asm volatile("cp.async.commit_group;");
        }
    }

    // ===== K loop over 32-blocks (SINGLE barrier/step: top __syncthreads guards both cur's
    //       visibility AND the WAR for the prefetch that follows it). =====
    for (int g = 0; g < nblk; g++) {
        int cur = g % NSTAGE;
        int nxt = (g + NSTAGE - 1) % NSTAGE;
        int gp  = g + NSTAGE - 1;             // the K-block prefetched/decoded this iter

        // wait until only NSTAGE-2 newest groups remain pending -> stage `cur`'s activation (committed
        // NSTAGE-1 iters ago) has landed; raw superblocks (fetched >=1 superblock ago) are long landed.
        asm volatile("cp.async.wait_group %0;" :: "n"(NSTAGE - 2));
        __syncthreads();   // cur visible (sA landed + sW decoded last iter); WAR-safe prefetch

        // (A) prefetch gp's activation; (pre-decode path) fetch superblock gp/GPSB+1 when gp enters a new
        //     superblock (keeps raw ring 1 superblock ahead) then DECODE gp from RESIDENT smem (no global
        //     read on chain). (inline path) inline-global decode gp straight into sW (original behavior).
        if (gp < nblk) {
            fetch_activation(nxt, gp);
            if (USE_PREDECODE) {
                if (gp % GPSB == 0) { int sbf = gp / GPSB + 1; if (sbf < nsb) fetch_superblock(sbf); }
                decode_stage(nxt, gp);        // reads raw smem (resident); NO global read on chain
            } else {
                decode_stage_inline(nxt, gp); // Q8_0: original inline-global decode
            }
        }
        asm volatile("cp.async.commit_group;");

        // ---- build A fragment for THIS warp's WARP_M=16 rows (rowbase) via ldmatrix.x4.b16 ----
        // The 16-row x 32-int8 weight subtile == 16x16 b16; ldmatrix loads the 4 .b32 A-operands in the
        // exact m16n8k32.s8 layout the scalar byte-assembly produced (bit-equivalent). MMQ-PORT: stride is
        // SW_STRIDE (the %8==4 row pad), row base &sW[cur][rowbase] (rowbase = wy*16).
        // load MFRAG A m16-frags (one per 16-row sub-band of this warp's 32-row span).
        int afrag[MFRAG][4];
        #pragma unroll
        for (int mf = 0; mf < MFRAG; mf++)
            ld_A_s8(afrag[mf], &sW[cur][rowbase + mf * WARP_M][0], SW_STRIDE);
        // ---- per 8-token n-tile: load B ONCE, mma against BOTH A-frags (B reuse = the AI lever) --
        #pragma unroll
        for (int nt = 0; nt < WARP_N / 8; nt++) {
            int ntok = tokbase + nt * 8;
            int bfrag[2];
            ld_B_s8(bfrag, &sA[cur][ntok][0], BK);   // ONE B-load reused across MFRAG m-frags
            #pragma unroll
            for (int mf = 0; mf < MFRAG; mf++) {
                int dacc[4] = {0, 0, 0, 0};
                mma_s8_m16n8k32(dacc, afrag[mf], bfrag);
                #pragma unroll
                for (int ci = 0; ci < 4; ci++) {
                    int rr = rowbase + mf * WARP_M + lane / 4 + (ci >> 1) * 8;   // CTA tile row
                    int nn = ntok + (lane % 4) * 2 + (ci & 1);                   // token within CTA
                    float da = sAd[cur][nn];
                    facc[mf][nt][ci] += sWd[cur][rr] * da * (float)dacc[ci] + sWb[cur][rr] * da * sAsum[cur][nn];
                }
            }
        }
    }

    // ===== write out: y[t*out_f + o] (token-major). MFRAG m-frags x (WARP_N/8) n-tiles per warp. =====
    #pragma unroll
    for (int mf = 0; mf < MFRAG; mf++) {
        #pragma unroll
        for (int nt = 0; nt < WARP_N / 8; nt++) {
            int ntok = tokbase + nt * 8;
            #pragma unroll
            for (int ci = 0; ci < 4; ci++) {
                int rr = rowbase + mf * WARP_M + lane / 4 + (ci >> 1) * 8;
                int nn = ntok + (lane % 4) * 2 + (ci & 1);
                int o = rowtile + rr;
                int t = toktile + nn;
                if (o < out_f && t < T) y[(size_t)t * out_f + o] = facc[mf][nt][ci];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) qmatvec_gemm_q8_0(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel<GQT_Q8_0>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
extern "C" __global__ void __launch_bounds__(256, 2) qmatvec_gemm_q4_K(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel<GQT_Q4_K>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
extern "C" __global__ void __launch_bounds__(256, 2) qmatvec_gemm_q5_K(
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
// FIX A smem-source NVFP4 decode: `b` = the resident 36B nvfp4 block base in smem (== sWraw_row+phase),
// `whichHalf` = g&1. Body copied verbatim from decode_nvfp4_2 with sblk already absorbed into `b`.
__device__ __forceinline__ void decode_nvfp4_2_s(const unsigned char* b, int whichHalf, int8_t* out,
                                                 float* su0, float* su1) {
    const unsigned char* d_bytes = b;
    const unsigned char* qs = b + 4;
    int s0 = whichHalf * 2, s1 = s0 + 1;
    int* o32 = (int*)out;
    #pragma unroll
    for (int sl = 0; sl < 2; sl++) {
        const unsigned char* qss = qs + (s0 + sl) * 8;
        int q4a = (int)qss[0] | ((int)qss[1] << 8) | ((int)qss[2] << 16) | ((int)qss[3] << 24);
        int q4b = (int)qss[4] | ((int)qss[5] << 8) | ((int)qss[6] << 16) | ((int)qss[7] << 24);
        int2 va = gtable16(q4a, gkvalues_mxfp4);
        int2 vb = gtable16(q4b, gkvalues_mxfp4);
        int base = sl * 4;
        o32[base + 0] = va.x; o32[base + 1] = vb.x; o32[base + 2] = va.y; o32[base + 3] = vb.y;
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
// kernel2 (two-sub-scale: Q6_K, NVFP4), generic over (NW warps, NTX_K token-groups). NVFP4 W4A8 runs
// the 8-warp 2D MMQ layout (NW=8, NTX_K=2: NWY=4 row-groups x 2 token-groups, each warp 16 rows x 64
// tokens, facc[8][4]=32 regs) — the SAME port that made kernel1 (Q4_K) 4.5x faster, applied to the
// two-scale split. (llama's NVFP4 MMQ is ALSO q8_1-activation/W4A8 at pp512=5451 — int8 W4A8 is NOT
// capped; this closes the 4-warp implementation gap.) Q6_K keeps NW=4, NTX_K=1 (wy=warp, wx=0,
// rowbase=warp*16, tokbase=0) -> reproduces the OLD kernel2 EXACTLY -> bit-identical.
// MF = m16-frags per warp (B-reuse arithmetic-intensity lever, same as kernel1 MFRAG). NVFP4: MF=2,
// NTX_K=4 -> NWY=2 row-groups x MF=2 frags = BM=64 rows; each warp reuses each B-frag across both
// m-frags (AND across the lo/hi two-scale mma) = 4 mma per B-load. Q6_K: MF=1, NTX_K=1 (bit-identical).
// SWZ: apply the proven A-load XOR-swizzle (probe 81e513e) on the sW store + ld_A_s8_sw load to
// halve the A-load bank conflicts. NVFP4 path only (SWZ=true); Q6_K SWZ=false (untouched, bit-id).
template<int QT, int NW, int NTX_K, int MF, bool SWZ = false>
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
    const int cta_threads = NW * WARP_SZ;        // NW warps grid-stride over BM/BN fetch+decode
    constexpr int NWY_K = NW / NTX_K;            // row-groups
    constexpr int WARP_N_K = BN / NTX_K;         // tokens per warp
    const int wy = warp / NTX_K;
    const int wx = warp % NTX_K;
    const int rowbase = wy * (WARP_M * MF);      // this warp's first row; owns MF m16-frags
    const int tokbase = wx * WARP_N_K;
    (void)NWY_K;

    // NSTAGE ring buffer. ONE full 32-wide weight tile (no separate lo/hi smem): the two-sub-scale
    // split is done at fragment level by zeroing the off-half A-registers (afrag[0,1]=k0..15 lo,
    // afrag[2,3]=k16..31 hi) — bit-identical to the old sWlo/sWhi zero-padding, half the weight smem.
    __shared__ int8_t sW[NSTAGE][BM][BK];
    __shared__ int8_t sA[NSTAGE][BN][BK];
    __shared__ float  sS0[NSTAGE][BM];        // lower-half scale factor (d*scn or su0)
    __shared__ float  sS1[NSTAGE][BM];        // upper-half scale factor
    __shared__ float  sAd[NSTAGE][BN];
    // raw superblock ring (FIX A). NVFP4 (PREDEC=1, GPSB=2) pre-decodes from a resident 36B-block window;
    // Q6_K (PREDEC=0) keeps the inline-global decode (210B 2-aligned window busts the 48KB smem cap).
    enum { GPSB = StageMeta<QT>::GPSB, USE_PREDECODE = StageMeta<QT>::PREDEC,
           RAW_W = USE_PREDECODE ? (int)StageMeta<QT>::RAW_W : 1, NSTAGE_RAW = 2 };
    __shared__ __align__(16) unsigned char sWraw[NSTAGE_RAW][BM][RAW_W];

    constexpr int NT_W = WARP_N_K / 8;           // n-tiles per warp
    float facc[MF][NT_W][4];
    #pragma unroll
    for (int mf = 0; mf < MF; mf++)
        #pragma unroll
        for (int nt = 0; nt < NT_W; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) facc[mf][nt][i] = 0.0f;

    // ---- FETCH raw superblock sb (16B-floored window) into the raw ring (FIX A; NVFP4 only). ----
    auto fetch_superblock = [&](int sb) {
        int rs = sb % NSTAGE_RAW;
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            if (o < out_f) {
                long off = (long)o * row_bytes + (long)sb * (long)StageMeta<QT>::SB_BYTES;
                long aoff = off & ~(long)15;
                int  phase = (int)(off - aoff);
                int  nchunk = (phase + (int)StageMeta<QT>::SB_BYTES + 15) >> 4;
                cp_async_window(&sWraw[rs][r][0], W + aoff, nchunk);
            }
        }
    };
    // ---- cp.async the activation tile for K-block g into stage s (unchanged). ----
    auto fetch_activation = [&](int s, int g) {
        for (int n = tid; n < BN; n += cta_threads) {
            int t = toktile + n;
            if (t < T) {
                const signed char* arow = aq + (size_t)t * in_f + (size_t)g * 32;
                cp_async16(&sA[s][n][0],  arow);
                cp_async16(&sA[s][n][16], arow + 16);
                sAd[s][n] = ad[(size_t)t * nblk + g];
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) sA[s][n][k] = 0;
                sAd[s][n] = 0.0f;
            }
        }
    };
    // ---- DECODE group g off the mma chain: ALU-unpack the RESIDENT raw block smem -> sW[s] + scales. --
    auto decode_stage = [&](int s, int g) {
        int rs = (g / GPSB) % NSTAGE_RAW;
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            int8_t wq[32];
            float s0 = 0.0f, s1 = 0.0f;
            if (o < out_f) {
                long off = (long)o * row_bytes + (long)(g / GPSB) * (long)StageMeta<QT>::SB_BYTES;
                int  phase = (int)(off & 15);
                const unsigned char* b = &sWraw[rs][r][phase];
                // PREDEC=1 in kernel2 is NVFP4 only (Q6_K is inline); decode the resident 36B block.
                float su0, su1; decode_nvfp4_2_s(b, g & 1, wq, &su0, &su1);
                s0 = su0; s1 = su1;
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) wq[k] = 0;
            }
            // SWZ store: word w (=k/4) of row r written at phys word (w^(r&7)) -> bank-scattered,
            // matched by ld_A_s8_sw. Non-SWZ: contiguous (Q6_K, bit-identical).
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                int kk = SWZ ? (((k >> 2) ^ (r & 7)) << 2) | (k & 3) : k;
                sW[s][r][kk] = wq[k];
            }
            sS0[s][r] = s0; sS1[s][r] = s1;
        }
    };
    // ---- INLINE-global decode (Q6_K, and NVFP4 fallback): the original path. ----
    auto decode_stage_inline = [&](int s, int g) {
        for (int r = tid; r < BM; r += cta_threads) {
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
            // SWZ store: word w (=k/4) of row r written at phys word (w^(r&7)) -> bank-scattered,
            // matched by ld_A_s8_sw. Non-SWZ: contiguous (Q6_K, bit-identical).
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                int kk = SWZ ? (((k >> 2) ^ (r & 7)) << 2) | (k & 3) : k;
                sW[s][r][kk] = wq[k];
            }
            sS0[s][r] = s0; sS1[s][r] = s1;
        }
    };

    const int nsb = (nblk + GPSB - 1) / GPSB;
    if (USE_PREDECODE) {
        // ===== PROLOGUE (pre-decode): seed raw superblocks 0..NSTAGE_RAW-1 + prologue activations;
        //       drain; decode prologue stages from resident raw smem. =====
        #pragma unroll
        for (int sb = 0; sb < NSTAGE_RAW; sb++)
            if (sb < nsb) fetch_superblock(sb);
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++)
            if (s < nblk) fetch_activation(s, s);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++)
            if (s < nblk) decode_stage(s, s);
        asm volatile("cp.async.commit_group;");
    } else {
        // ===== PROLOGUE (inline, Q6_K): original — inline-global decode + activation per stage. ==
        #pragma unroll
        for (int s = 0; s < NSTAGE - 1; s++) {
            if (s < nblk) { decode_stage_inline(s, s); fetch_activation(s, s); }
            asm volatile("cp.async.commit_group;");
        }
    }

    for (int g = 0; g < nblk; g++) {
        int cur = g % NSTAGE;
        int nxt = (g + NSTAGE - 1) % NSTAGE;
        int gp  = g + NSTAGE - 1;

        asm volatile("cp.async.wait_group %0;" :: "n"(NSTAGE - 2));
        __syncthreads();   // cur visible + WAR-safe prefetch (see kernel1 for the argument)

        if (gp < nblk) {
            fetch_activation(nxt, gp);
            if (USE_PREDECODE) {
                if (gp % GPSB == 0) { int sbf = gp / GPSB + 1; if (sbf < nsb) fetch_superblock(sbf); }
                decode_stage(nxt, gp);        // resident raw smem; NO global read on chain
            } else {
                decode_stage_inline(nxt, gp); // Q6_K: original inline-global decode
            }
        }
        asm volatile("cp.async.commit_group;");

        // load MF A m16-frags (one per 16-row sub-band). Each: ONE ldmatrix of 32 k, split lo (k0..15
        // = af[0,1]) / hi (k16..31 = af[2,3]) by zeroing the off-half regs (the two-sub-scale split).
        int aflo[MF][4], afhi[MF][4];
        #pragma unroll
        for (int mf = 0; mf < MF; mf++) {
            int af[4];
            if (SWZ) ld_A_s8_sw(af, &sW[cur][rowbase + mf * WARP_M][0], BK);
            else     ld_A_s8(af, &sW[cur][rowbase + mf * WARP_M][0], BK);
            aflo[mf][0] = af[0]; aflo[mf][1] = af[1]; aflo[mf][2] = 0; aflo[mf][3] = 0;
            afhi[mf][0] = 0; afhi[mf][1] = 0; afhi[mf][2] = af[2]; afhi[mf][3] = af[3];
        }
        #pragma unroll
        for (int nt = 0; nt < NT_W; nt++) {
            int ntok = tokbase + nt * 8;          // this warp's global n-tile token base
            int bfrag[2];
            ld_B_s8(bfrag, &sA[cur][ntok][0], BK);   // ONE B-load reused across MF m-frags x lo/hi
            #pragma unroll
            for (int mf = 0; mf < MF; mf++) {
                int dlo[4] = {0,0,0,0}, dhi[4] = {0,0,0,0};
                mma_s8_m16n8k32(dlo, aflo[mf], bfrag);
                mma_s8_m16n8k32(dhi, afhi[mf], bfrag);
                #pragma unroll
                for (int ci = 0; ci < 4; ci++) {
                    int rr = rowbase + mf * WARP_M + lane / 4 + (ci >> 1) * 8;   // GLOBAL tile row
                    int nn = ntok + (lane % 4) * 2 + (ci & 1);                   // GLOBAL tile token
                    float da = sAd[cur][nn];
                    facc[mf][nt][ci] += (sS0[cur][rr] * (float)dlo[ci] + sS1[cur][rr] * (float)dhi[ci]) * da;
                }
            }
        }
    }

    #pragma unroll
    for (int mf = 0; mf < MF; mf++) {
        #pragma unroll
        for (int nt = 0; nt < NT_W; nt++) {
            #pragma unroll
            for (int ci = 0; ci < 4; ci++) {
                int rr = rowbase + mf * WARP_M + lane / 4 + (ci >> 1) * 8;
                int nn = tokbase + nt * 8 + (lane % 4) * 2 + (ci & 1);
                int o = rowtile + rr;
                int t = toktile + nn;
                if (o < out_f && t < T) y[(size_t)t * out_f + o] = facc[mf][nt][ci];
            }
        }
    }
}

// Q6_K: MF=1, NTX_K=1, NW=4 (bit-identical to the original kernel2).
extern "C" __global__ void __launch_bounds__(128, 4) qmatvec_gemm_q6_K(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel2<GQT_Q6_K, 4, 1, 1>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}
// NVFP4 W4A8: 8-warp 2D MMQ + MF=2 B-reuse (NW=8, NTX_K=4 -> NWY=2 row-groups x MF=2 = BM=64).
extern "C" __global__ void __launch_bounds__(256, 2) qmatvec_gemm_nvfp4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_kernel2<GQT_NVFP4, 8, 4, 2, true>(W, aq, ad, y, in_f, out_f, T, row_bytes);
}

// ===================================================================== //
//  STAGE-C: native FP4 block-scale GEMM for NVFP4 weights (sm_120a).     //
//  mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X .ue4m3     //
//  — 762 TFLOP/s peak (vs int8 219). Feeds the RAW e2m1 weight nibbles + //
//  RAW UE4M3 micro-scales DIRECTLY to the tensor core: NO dequant-to-int8//
//  (drops gtable16). The GGUF e2m1 codebook value = 2x the HW e2m1, and  //
//  GGUF UE4M3 = 0.5x the HW UE4M3, so feeding both RAW reproduces the    //
//  GGUF dequant EXACTLY (factors cancel). Activations are FP4 e2m1 too   //
//  (quantize_fp4_act), per-16 UE4M3 scale. K-step = one 64-elem NVFP4    //
//  block (BK=64). All fragment/scale layouts verified on-device by       //
//  probe/fp4_4x_final.cu (maxrel=0 vs f32 oracle).                       //
//
//  Layout facts (probe-verified, sm_120a):
//   A-frag (lane L): reg0=row L/4 K[(L%4)*8..+7]; reg1=row L/4+8 same K;
//     reg2=row L/4 K[+32..+7]; reg3=row L/4+8 K[+32]. nibble n -> K base+n.
//   B-frag (lane L): col L/4; reg0=K[(L%4)*8], reg1=K[+32]; nibble n->K+n.
//   SFA(4X): 4 ue4m3 bytes = K16 blocks 0..3. Lane L%4==2 supplies row L/4;
//            L%4==3 supplies row L/4+8.
//   SFB(4X): 4 ue4m3 bytes = K16 blocks 0..3. Lane L%4==1 supplies col L/4.
// ===================================================================== //
__device__ __forceinline__ void mma_mxf4_m16n8k64(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        unsigned sa, unsigned sb) {
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}

// PASS 3 swizzle: each smem row is 8 u32 = two 16B (4-u32) chunks. ldmatrix reads one 4-u32 chunk
// per lane at chunk index `c`; 8 rows of one 8x8 sub-tile sit at byte base row*32 + c*16 -> banks
// (row*8 + c*4)%32, so rows differing by 4 collide 2-way (ncu: 60M/74M ld wavefronts are conflict
// replays). XOR the chunk index by bit2 of the (within-tile) row -> c_phys = c ^ ((row>>2)&1) makes
// all 8 rows of every sub-tile land in distinct bank groups (conflict-free), at the SAME 32B stride
// (no pad, no occupancy loss). MUST be applied identically at the repack/activation STORE and the
// ldmatrix LOAD address or the operands corrupt (caught by kernel_check rel + argmax).
#define SWZ_CHUNK(row) (((row) >> 2) & 1)

// PASS 1: ldmatrix.x4.b16 for the mxf4 A-operand (weight nibbles). Replaces the 4 scalar afrag
// smem loads/lane/K-step with ONE warp-cooperative matrix load. `sWq_row0` = &sWq[cur][warp*WARP_M][0]:
// 16 rows x 8 u32 (=16 b16 units) contiguous, 16B-aligned. Per-lane addr mirrors `ld_A_s8` (the
// device-proven int8 m16n8k32 A-load), widened to the FP4 A-tile (8 u32/row): in u32 units
// xs = base + (lane%16)*8 + (chunk^swz)*4 where chunk = lane/16, row = lane%16 (PASS 3 swizzle).
// ld_A_s8 (same addr form) feeds the int8 mma with NO remap; its raw output order
// (row,klo),(row+8,klo),(row,khi),(row+8,khi) == the FP4 scalar afrag order -> identity remap.
__device__ __forceinline__ void ld_A_mxf4(unsigned (&a)[4], const unsigned* sWq_row0) {
    unsigned row = threadIdx.x % 16, chunk = threadIdx.x / 16;
    const unsigned* xs = sWq_row0 + row * 8 + (chunk ^ SWZ_CHUNK(row)) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    unsigned t[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(t[0]),"=r"(t[1]),"=r"(t[2]),"=r"(t[3]) : "r"(addr));
    a[0] = t[0]; a[1] = t[1]; a[2] = t[2]; a[3] = t[3];
}

// PASS 2: ldmatrix.x2.b16 for the mxf4 B-operand (activation nibbles). Replaces the 2 scalar bfrag
// smem loads/n-tile/lane (x BN/8 n-tiles = the dominant 32 scalar loads/lane/K-step) with ONE
// ldmatrix per n-tile. `sAq_ntile0` = &sAq[cur][nt*8][0]: 8 tokens x 8 u32 (=16 b16 units) contiguous,
// 16B-aligned. Per-lane addr mirrors `ld_B_s8` (device-proven int8 m16n8k32 B-load): in u32 units
// xs = base + (lane%8)*8 + ((lane/8 %2)^swz)*4 where tok-in-tile = lane%8 (PASS 3 swizzle). ld_B_s8
// feeds the int8 mma with NO remap; raw output == FP4 scalar bfrag (tok,klo),(tok,khi) -> identity.
__device__ __forceinline__ void ld_B_mxf4(unsigned (&b)[2], const unsigned* sAq_ntile0) {
    unsigned row = threadIdx.x % 8, chunk = (threadIdx.x / 8) % 2;
    const unsigned* xs = sAq_ntile0 + row * 8 + (chunk ^ SWZ_CHUNK(row)) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
        : "=r"(b[0]),"=r"(b[1]) : "r"(addr));
}

// The GGUF NVFP4 36-byte block (4 ue4m3 scale bytes + 32 qs e2m1 bytes) is repacked into the
// A-fragment-friendly smem form INLINE in the kernel's `fetch` (from a register copy, not per-byte
// global loads). Layout: word gi (0..7) holds 8 e2m1 nibbles for K-group gi<4? gi*8 : (gi-4)*8+32
// (i.e. gi 0..3 -> K {0,8,16,24}; gi 4..7 -> K {32,40,48,56}; nibble n -> K base+n). The 4 ue4m3
// micro-scale bytes are fed RAW. GGUF qs: element k -> sub-block s=k/16, within=k%16, byte
// qs[s*8+(within&7)], low nibble if within<8 else high; a K-group g (g%8==0) is all-low (g%16==0) or
// all-high (g%16==8) of qs[s*8..s*8+7]. (Layout verified on-device by probe/fp4_4x_final.cu.)

// 4-byte async global->smem copy (for the 4 ue4m3 activation scale bytes / token). `smem` 4B-aligned.
__device__ __forceinline__ void cp_async4(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" :: "r"(s), "l"(g));
}

// y[T, out_f] = aq4[T, in_f](e2m1) . W[out_f, in_f](NVFP4 e2m1)^T, per-16 UE4M3 block scales applied
// inside the MMA. Token-major output. BK=64 (one NVFP4 block / K-step). FP4_NS-deep cp.async ring:
// the RAW 36-byte weight block + the activation words/scales are cp.async'd into the smem ring ONE
// block ahead (instead of synchronous u32 global loads on the mma chain), and the A-fragment e2m1
// nibble REPACK then runs from RESIDENT smem one block later — so the long-scoreboard global weight
// read leaves the mma critical path (same discipline as the int8 GEMM's FIX-A pre-decode). The repack
// is amortized 1/(BN tokens) since the staged tile feeds all 128 tokens' mma. SINGLE __syncthreads/
// K-step: the top barrier guards both `cur`'s visibility (raw landed, repacked last iter) AND the WAR
// for the post-barrier prefetch (cp.async wait_group + barrier, like the int8 kernel).
// FP4_NS=2 chosen by measured pp512: deepening to 3 dropped CTAs/SM 4->2 (smem-bound: 36KB/CTA, ncu
// Block Limit Shared Mem=2) and REGRESSED pp512 (~1871->1846) — this kernel is MIO/smem-throughput-
// bound (ncu: Mem 77% / Compute 44%, top stall = MIO queue), NOT cp.async-latency-bound, so the extra
// overlap from a deeper ring does not pay for the lost occupancy (same occupancy tradeoff that reverted
// the int8 NSTAGE=4 and tile-redesign experiments).
#define FP4_NS 2
// Raw-block ring depth. The repack of K-block `gp` reads a raw block that must have been cp.async'd
// FP4_NS-1 iters earlier (so it is landed under wait_group(FP4_NS-2)); the loop therefore fetches the
// raw block 2*(FP4_NS-1) iters ahead of consumption, keeping that many distinct blocks alive at once.
#define FP4_NS_RAW (2 * (FP4_NS - 1))
// sWraw row stride (bytes). The repack reads the resident raw block per-BYTE (LDS.U8 of qs[]/scale[]),
// 64+ scalar shared loads per K-step — these (NOT the ldmatrix operand loads, which the SWZ_CHUNK/x4.b16
// path already drives conflict-free) are the dominant shared-load population (ncu SASS: 72 LDS.U8 +
// 32 LDS vs only 17 LDSM). At the natural 64B row stride, row r maps to bank (r*16)%32, so all 32 rows
// of a warp alias just 2 bank groups -> 16-WAY conflict (ncu: 53.2M conflicts / 6.6-way avg, the whole
// FP4 GEMM conflict). Padding the stride to 80B (the %16-aligned analog of llama's %8==4 tile pad —
// 16B-aligned so cp.async.cg-16 stays legal) makes the per-row bank step (80/4)%32=20, scattering the
// 32 rows over 8 distinct bank groups -> 4-WAY (4x fewer conflicts). Bit-exact: storage stride only;
// the phase math (off&15) and nibble/scale decode are unchanged. +2KB/CTA smem (21.5->23.5KB), still
// below the FP4_NS occupancy threshold (the kernel is smem-bound only past ~36KB).
#define FP4_RAW_STRIDE 80
__device__ void qmatvec_gemm_mxf4_kernel(
        const unsigned char* __restrict__ W, const unsigned* __restrict__ aq4,
        const unsigned char* __restrict__ ad4, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes)
{
    const int rowtile = blockIdx.x * BM;
    const int toktile = blockIdx.y * BN;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp * WARP_SZ + lane;
    const int nblk64 = in_f / 64;                 // K-blocks (64-elem NVFP4 blocks)
    const int aw_per_tok = in_f / 8;              // u32 words per token in aq4
    const int as_per_tok = in_f / 16;             // ue4m3 scale bytes per token in ad4
    const int cta_threads = NWARP2 * WARP_SZ;     // FP4 kernel keeps the 4-warp grid-stride (launched 32x4)

    // Repacked staged tiles: A-fragment-ready weight nibbles (8 u32 groups/row) + 1 u32 of 4 packed
    // ue4m3 scales/row; activation 8 u32 groups + 4 scale bytes/token. The weight REPACK (e2m1 nibble
    // gather) is done ONCE per row at stage time (not per lane in the mma loop), amortized across all
    // BN tokens — the heavy ALU leaves the mma critical path. Activations land in their final mma form
    // (no repack), so they're cp.async'd straight into sAq/sAsc. The RAW 36B weight block is cp.async'd
    // into sWraw (a separate ring keyed by K-block, fetched 1 block ahead) and repacked from resident
    // smem -> the long-scoreboard global weight read leaves the mma chain.
    __shared__ __align__(16) unsigned sWq[FP4_NS][BM][8];   // 8 u32/row (A-frag-ready); 16B-aligned for int4
    __shared__ unsigned      sWsc[FP4_NS][BM];      // 4 ue4m3 packed into 1 u32 / row
    __shared__ __align__(16) unsigned sAq[FP4_NS][BN][8];   // 8 u32/token; 16B-aligned for cp.async.16
    __shared__ unsigned      sAsc[FP4_NS][BN];      // 4 ue4m3 packed into 1 u32 / token (single wide load
                                                    // for the SFB scale -> 1/4 the smem transactions of a
                                                    // per-byte read; LE u32 == the old s[0]|s[1]<<8|...).
    // raw weight ring: 36B block at a 16B-floored phase (max phase 15 + 36B = 51 -> round up to 64).
    __shared__ __align__(16) unsigned char sWraw[FP4_NS_RAW][BM][FP4_RAW_STRIDE];

    float facc[BN / 8][4];
    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++)
        #pragma unroll
        for (int i = 0; i < 4; i++) facc[nt][i] = 0.0f;

    // ---- FETCH_RAW (cp.async): raw 36B weight block for K-block g into the raw ring (16B-floored
    //      window). Async; nothing on the mma critical path. Kept >=1 block ahead of its repack. ----
    auto fetch_raw = [&](int g) {
        int rs = g % FP4_NS_RAW;
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            if (o < out_f) {
                long off  = (long)o * row_bytes + (long)g * 36;
                long aoff = off & ~(long)15;                 // 16B floor of the 36B block
                int  phase = (int)(off - aoff);              // 0..15
                int  nchunk = (phase + 36 + 15) >> 4;         // 16B chunks covering [phase, phase+36)
                cp_async_window(&sWraw[rs][r][0], W + aoff, nchunk);
            }
        }
    };
    // ---- FETCH_ACT (cp.async): the activation u32 groups + scale bytes for K-block g straight into
    //      sAq/sAsc (already the final mma form -> no repack needed). Async, off the mma chain. ----
    auto fetch_act = [&](int s, int g) {
        for (int n = tid; n < BN; n += cta_threads) {
            int t = toktile + n;
            if (t < T) {
                const unsigned* aw = aq4 + (size_t)t * aw_per_tok + (size_t)g * 8;   // 32B, 16B-aligned
                const unsigned char* asc = ad4 + (size_t)t * as_per_tok + (size_t)g * 4;  // 4B
                // PASS 3: place the two 16B chunks at swizzled offsets (^SWZ_CHUNK on tok-in-tile n%8 ==
                // n's bit2) -> conflict-free ldmatrix.x2 read (ld_B_mxf4 applies the identical XOR).
                int sw = SWZ_CHUNK(n);
                cp_async16(&sAq[s][n][(0 ^ sw) * 4], aw);
                cp_async16(&sAq[s][n][(1 ^ sw) * 4], aw + 4);
                cp_async4(&sAsc[s][n], asc);
            } else {
                #pragma unroll
                for (int gi = 0; gi < 8; gi++) sAq[s][n][gi] = 0;
                sAsc[s][n] = 0;
            }
        }
    };
    // ---- REPACK (off the mma chain): ALU-gather the e2m1 nibbles of K-block g from the RESIDENT raw
    //      smem block into the A-fragment-ready sWq/sWsc. Reads sWraw[g%FP4_NS_RAW] at phase=(off&15);
    //      byte-identical to the old register-source repack (same nibble math, smem base). ----
    // PASS 4 NOTE: a (row, group-half) split across all 128 threads (vs 64 active) was tried to cut the
    //   barrier stall, but it DOUBLED the sWraw smem-read traffic (each half re-reads the row's phase
    //   base + qs) and spiked mio_throttle 9.2->17.1, regressing T=512 1.15->1.67ms. Reverted: the
    //   repack is off the mma chain (hidden under wait_group) so its barrier cost does not pay for the
    //   extra MIO pressure on this smem-throughput-bound tile. Kept the P3 single-thread-per-row form.
    auto repack = [&](int s, int g) {
        int rs = g % FP4_NS_RAW;
        for (int r = tid; r < BM; r += cta_threads) {
            int o = rowtile + r;
            if (o < out_f) {
                long off = (long)o * row_bytes + (long)g * 36;
                int  phase = (int)(off & 15);
                const unsigned char* b = &sWraw[rs][r][phase];
                sWsc[s][r] = (unsigned)b[0] | ((unsigned)b[1] << 8)
                           | ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24);  // 4 ue4m3 scale bytes
                const unsigned char* qs = b + 4;                               // 32 qs bytes
                unsigned wq[8];
                #pragma unroll
                for (int gi = 0; gi < 8; gi++) {
                    int base = (gi < 4) ? (gi * 8) : ((gi - 4) * 8 + 32);
                    int sb = base >> 4, hinib = (base & 8) ? 4 : 0;
                    const unsigned char* q = qs + sb * 8;
                    // PASS 4b: gather 8 same-position nibbles (one per byte) into a u32 WITHOUT the
                    // 8-deep dependent shift/OR chain. Read the 8 bytes as 2 u32 (qs is 16B-aligned
                    // within the staged block at phase&15; use byte loads + assemble to stay phase-safe),
                    // mask the wanted nibble of each byte, then compact 4 nibbles-per-u32 in a 2-step
                    // parallel tree (>>4 &00FF00FF; >>8 &0000FFFF). Bit-identical to the scalar gather.
                    unsigned lo = (unsigned)q[0] | ((unsigned)q[1] << 8) | ((unsigned)q[2] << 16) | ((unsigned)q[3] << 24);
                    unsigned hi = (unsigned)q[4] | ((unsigned)q[5] << 8) | ((unsigned)q[6] << 16) | ((unsigned)q[7] << 24);
                    lo = (lo >> hinib) & 0x0F0F0F0Fu;  hi = (hi >> hinib) & 0x0F0F0F0Fu;
                    lo = (lo | (lo >> 4)) & 0x00FF00FFu; lo = (lo | (lo >> 8)) & 0x0000FFFFu;  // 4 nibbles -> low16
                    hi = (hi | (hi >> 4)) & 0x00FF00FFu; hi = (hi | (hi >> 8)) & 0x0000FFFFu;
                    wq[gi] = lo | (hi << 16);
                }
                // 2x 128-bit smem stores (vs 8 narrow u32) -> 1/4 the store transactions (ncu: "fewer
                // wider"). sWq rows are 32B (8 u32) contiguous & 16B-aligned -> int4 store legal.
                // PASS 3: store each 16B chunk at swizzled chunk index (^SWZ_CHUNK(r)) -> conflict-free
                // ldmatrix read (which applies the identical XOR on the load address).
                int sw = SWZ_CHUNK(r);
                reinterpret_cast<int4*>(&sWq[s][r][0])[0 ^ sw] = *reinterpret_cast<const int4*>(&wq[0]);
                reinterpret_cast<int4*>(&sWq[s][r][0])[1 ^ sw] = *reinterpret_cast<const int4*>(&wq[4]);
            } else {
                int4 z = make_int4(0, 0, 0, 0);
                reinterpret_cast<int4*>(&sWq[s][r][0])[0] = z;
                reinterpret_cast<int4*>(&sWq[s][r][0])[1] = z;
                sWsc[s][r] = 0;
            }
        }
    };

    // ===== PROLOGUE: seed the raw ring with blocks 0..FP4_NS_RAW-1 (blocks 0..FP4_NS-2 read by the
    //       prologue-stage repacks; the rest are the FP4_NS-1-ahead lead for the loop's first repacks)
    //       + the FP4_NS-1 prologue activation stages; drain; repack the prologue stages from resident
    //       raw smem. The loop then fetches raw block gp+(FP4_NS-1) each iter; the FP4_NS_RAW =
    //       2*(FP4_NS-1) consecutive in-flight blocks map to distinct slots mod FP4_NS_RAW. =====
    #pragma unroll
    for (int sb = 0; sb < FP4_NS_RAW; sb++)
        if (sb < nblk64) fetch_raw(sb);
    #pragma unroll
    for (int s = 0; s < FP4_NS - 1; s++)
        if (s < nblk64) fetch_act(s, s);               // FP4_NS-1 prologue activation stages
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");           // drain: raw blocks + prologue activations
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < FP4_NS - 1; s++)
        if (s < nblk64) repack(s, s);                  // repack prologue stages from resident raw smem
    asm volatile("cp.async.commit_group;");            // keep per-iter commit cadence (empty group ok)

    int r0 = lane / 4;          // 0..15 within the warp's 16-row tile
    int kg = lane % 4;          // K-group selector
    int q  = lane & 3;
    int srow = (q == 2) ? r0 : (q == 3 ? r0 + 8 : -1);   // SFA-supplying row (or none)

    for (int g = 0; g < nblk64; g++) {
        int cur = g % FP4_NS;
        int nxt = (g + FP4_NS - 1) % FP4_NS;
        int gp  = g + FP4_NS - 1;                  // the K-block prefetched(activation)/repacked this iter
        // Raw weight lead = FP4_NS-1 iters: a cp.async committed j iters ago is landed under
        // wait_group(FP4_NS-2) iff j >= FP4_NS-1. repack reads raw block `gp`; to guarantee it is
        // resident now, it must have been fetched FP4_NS-1 iters ago -> this iter fetches block
        // gp + (FP4_NS-1) = g + 2*(FP4_NS-1). The raw ring holds FP4_NS_RAW(=FP4_NS) consecutive
        // blocks [gp .. gp+FP4_NS-1] -> distinct slots mod FP4_NS_RAW.
        int gr  = gp + (FP4_NS - 1);               // raw block fetched this iter (FP4_NS-1 ahead of repack)

        // wait until only FP4_NS-2 newest groups remain pending -> stage `cur`'s activation (committed
        // FP4_NS-1 iters ago) has landed AND the raw block for `gp` (fetched FP4_NS-1 iters ago) is resident.
        asm volatile("cp.async.wait_group %0;" :: "n"(FP4_NS - 2));
        __syncthreads();   // cur visible (sA landed + sW repacked last iter); WAR-safe prefetch

        if (gp < nblk64) {
            fetch_act(nxt, gp);                    // cp.async stage `nxt`'s activations (FP4_NS-1 ahead)
            if (gr < nblk64) fetch_raw(gr);        // cp.async the raw weight FP4_NS-1 blocks ahead of repack
            repack(nxt, gp);                       // repack gp from RESIDENT raw smem (no global on chain)
        }
        asm volatile("cp.async.commit_group;");

        // A fragment: ONE ldmatrix.x4.b16 (PASS 1) replaces the 4 scalar afrag smem loads.
        unsigned afrag[4];
        ld_A_mxf4(afrag, &sWq[cur][warp * WARP_M][0]);
        unsigned sa = (srow >= 0) ? sWsc[cur][warp * WARP_M + srow] : 0u;

        #pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            int tok = nt * 8 + (lane / 4);
            // B fragment: ONE ldmatrix.x2.b16 (PASS 2) replaces the 2 scalar bfrag smem loads.
            unsigned bfrag[2];
            ld_B_mxf4(bfrag, &sAq[cur][nt * 8][0]);
            unsigned sb = (q == 1) ? sAsc[cur][tok] : 0u;   // single wide u32 load (LE == old byte gather)
            mma_mxf4_m16n8k64(facc[nt], afrag, bfrag, sa, sb);
        }
    }

    // write out: y[t*out_f + o]. D layout: reg ci -> row = lane/4 + (ci>>1)*8, col = (lane%4)*2 + (ci&1).
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
extern "C" __global__ void __launch_bounds__(128, 4) qmatvec_gemm_nvfp4_fp4(
        const unsigned char* __restrict__ W, const unsigned* __restrict__ aq4,
        const unsigned char* __restrict__ ad4, float* __restrict__ y,
        int in_f, int out_f, int T, long row_bytes) {
    qmatvec_gemm_mxf4_kernel(W, aq4, ad4, y, in_f, out_f, T, row_bytes);
}
