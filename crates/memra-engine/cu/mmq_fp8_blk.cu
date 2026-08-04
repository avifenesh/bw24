// mmq_fp8_blk.cu — PER-BLOCK FP8 MMQ prefill GEMM (P1 option (b), lane/fp8-mmq-v2, sm_120a).
//
// Consumes the Qwen-official block-scaled FP8 checkpoint DIRECTLY:
//   W        : [out_dim x in_dim] uint8 e4m3 codes, row-major, stride = in_dim bytes.
//   blk_scale: [ceil(out_dim/128) x ceil(in_dim/128)] f32, row-major. scales[(o>>7)*cols + (e>>7)]
//              scales element W[o][e] — the Fp8BlockScales / F8BlockGrid layout contract
//              (same grid cu/fp8_blk_dequant.cu reads).
//
// WHY THIS EXISTS (research/fp8st-20260803/P1-VERDICT.md):
//   * cuBLASLt on sm_120 exposes ONLY per-tensor SCALAR FP8 scales (BLK128x128 -> status 7/15,
//     nh=0 at every m and both D dtypes). It cannot consume this grid.
//   * ARM A folded the grid into one per-tensor scale: +18.4% pp but the 128-token greedy stream
//     diverges at generated pos 20 (102/128 differ) — a real re-quant, not shippable.
//   * ARM B' device-dequants to Q8_0: byte-exact, but then rides the Q8_0 MMQ = floor perf.
// This kernel keeps EVERY weight block's own scale, exactly, at tensor-core-class throughput.
//
// =================== V2 STRUCTURE (research/fp8st-20260804/mmq/LANE-VERDICT.jsonl §6) ============
// v1 was exact (kernel-check bit-identity ALL GREEN) and 0.81-0.94x the Q8_0 MMQ floor. Its profile
// said: NEITHER arm is MMA-bound (105-127 TF against the 381-TF f8f6f4 class) and the floor wins by
// deferring ALL scaling to one epilogue fold, while v1 paid mixed f32 scale work every 32 k-values
// and ran 1 CTA/SM on a 61 KB tile. v2 attacks exactly that:
//
//   (1) THE 128-WIDE SCALE BLOCK IS THE OUTER k LOOP. One tile iteration == one scale block == one
//       uniform (s_blk, dB) pair, so the four k32 MMAs of a block CHAIN into a SINGLE f32
//       tensor-core accumulator with no intermediate scaling, and (s_blk*dB) folds ONCE per block
//       per accumulator element. Per 128 k-values and j-tile that is 8 zero-inits + 2 scalar mults
//       + 8 FMAs, against v1's 32 + 8 + 32 — a 4x cut of the epilogue f32 work, which is the shape
//       the floor measured faster with.
//   (2) THE WEIGHT TILE HALVES with the iteration: MMQ_ITER_K 256 -> 128 k-values, x tile row 64 ->
//       32 value-ints, smem 61 KB -> 37 KB (mmq_y 128) / 28 KB (mmq_y 64). Each weight row is
//       still read exactly once per k, so this is not extra traffic — it is occupancy headroom,
//       which is what a latency-exposed non-MMA-bound kernel needs.
//   (3) UNIFORM dB IS WHAT LICENSES (1) — see the activation note below.
//
// ACTIVATIONS (v2's one arithmetic change vs v1, declared not smuggled): this kernel owns its
// quantizer, quantize_mmq_e4m3_d128_kernel, a per-128 twin of the W4A8-FP8 per-32 one. Block
// struct, output layout and byte footprint are IDENTICAL (block_e4m3_mmq: 4x f32 + 128 e4m3 bytes
// == block_q8_1_mmq's 144 B, so every y-tile smem/stride expression is the vendored q8_1 math
// unchanged); all four d4 slots simply carry the same per-128 amax/448. A per-32 dB CANNOT be
// hoisted out of the 128-k run — it multiplies the MMA result, so a varying dB forces v1's
// per-32 fold no matter how the loop is nested. WHY THE COARSER BLOCK IS CHEAP HERE: e4m3 is a
// FLOATING container (sign/4-exp/3-mantissa), so the block scale only has to bring the block onto
// the e4m3 grid; relative precision is then scale-invariant across ~15 binades, and widening the
// amax window 32 -> 128 costs mantissa only for values more than ~2^9 below the block amax, whose
// contribution to the sum is already negligible. This is the opposite of int8, where per-32 amax
// is load-bearing. The kernel-check RAND arm bounds the residual and the model battery measures
// it; it is NOT claimed to be v1's arithmetic.
//
// THE KEY PROPERTY IS UNCHANGED — THE WEIGHT SIDE IS NOT RE-QUANTIZED AT ALL:
// the checkpoint bytes ARE the A operand. e4m3 x e4m3 -> f32 is a native Blackwell MMA
// (mma.sync.aligned.kind::f8f6f4.m16n8k32, 381-TF class, the same op the W4A8-FP8 arm uses), so
// the tile loader is a pure global->smem COPY: no dequant, no LUT, no fold, zero weight-side
// precision loss. Every weight block keeps its own f32 scale.
//
// GEOMETRY (why the scale lookup is free):
//   FP8_MMQ_Y divides FP8_BLK and row tiles start at it*FP8_MMQ_Y, so every row in a CTA's tile
//   shares scale row (it*FP8_MMQ_Y)>>7 — one uniform grid row per CTA, hoisted out of the tile loop
//   entirely. MMQ_ITER_K == 128 == the scale block edge and iterations are 128-aligned, so a whole
//   tile iteration has ONE uniform scale scalar: a k-block boundary can never fall inside an MMA,
//   and per-iteration scale traffic is a single scalar load.
//
// ARITHMETIC CONTRACT (what the kernel-check host reference reproduces, in this order):
//   sum(i,j) = SUM over k-blocks kb (ascending, step 128) of (s_blk * dB) * C_kb ,
//   C_kb     = SUM over k01 in {0,8,16,24} (ascending, 32 k-values each) of      <- ONE accumulator,
//                SUM_{t=0..31} e4m3(W[i][g]) * e4m3(A[j][g])                        chained in HW
//   with g = kb + 4*k01 + t, s_blk = blk_scale[(it*mmq_y)>>7][min(kb>>7, cols-1)],
//   dB = act_d4[j][kb/128][0], and dst = sum * out_scale.
//   The only order the kernel does NOT define is the 32-product reduction INSIDE one MMA and the
//   chaining of the four MMAs (both hardware-internal). The kernel-check integer arm is
//   exact-by-construction (all products integers, |partial sums| < 2^24, scales powers of two) so
//   neither order can matter, giving a true BIT-IDENTITY gate; a second random-e4m3 arm bounds the
//   residual rounding.
//
// k TAIL: in_dim need not be a multiple of 128. k values >= in_dim are filled with 0x00 in smem —
// e4m3 0x00 is exactly 0.0, and the activation quantizer already zero-pads its side, so padded
// lanes contribute exact zeros. Requires in_dim % 16 == 0 (16B row alignment for the int4 tile
// copy; every real block-128 projection is a multiple of 128).
//
// NaN CODES: the hardware MMA treats magnitude 0x7F as NaN, while the host/ARM B' convention
// (nvfp4_repack::fp8_e4m3_to_f32, modelopt) decodes it to 0.0. modelopt-quantized weights do not
// contain it; memra_fp8_blk_count_nan() lets the dispatch PROVE that per tensor at load and refuse
// this kernel otherwise, instead of assuming.
//
// Seam: MEMRA_FP8_MMQ=1 (mmq_ffi.rs dispatch), default OFF.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>

// ======================= vendored ggml/MMQ constants (see mmq_nvfp4_w4a8.cu) =======================
#define WARP_SIZE 32
#define NO_DEVICE_CODE __trap()
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

#define QK8_1 32
#define QI8_1 8
#define MATRIX_ROW_PADDING 512

#define MMQ_TILE_NE_K 32                                        // value-INTS per tile row (128 k)
#define MMQ_ITER_K    128                                       // k-values per tile iteration ==
                                                                // one scale block (v2; v1 was 256)
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)   // 36 ints per y block
// x tile row stride, ints: 32 value-ints (128 e4m3 bytes == one scale block) + 4 pad ints.
// 36*4 = 144 B == 9*16, so every ldmatrix row address stays 16B aligned, and the +4 pad breaks the
// smem bank alignment the way MMQ_MMA_TILE_X_K_NVFP4 (84) does.
#define MMQ_MMA_TILE_X_K_FP8 (MMQ_TILE_NE_K + 4)

#define MMQ_WARP_SIZE  32
#define FP8_BLK        128        // scale block edge (both axes) — the checkpoint's grid
// Row tile == the scale block edge, so every row in a CTA's tile shares one scale-grid row (the
// hoist the vec_dot relies on). The v2 slice-2 Y/OCC seams (halved Y, minBlocks=2) concluded
// negative and were deleted (research/fp8st-20260804/mmq-v2/RESULTS.jsonl) — Y is fixed here.
#define FP8_MMQ_Y      128
#define FP8_MMQ_NWARPS (FP8_MMQ_Y / 16)
// Token tile. v2 default 256 (experiment A: the slice-1 weight-tile halving made X=256 affordable,
// and it beats X=128 on 5 of 6 real shapes at m=6257 and 4 of 6 at m=512). FP8_MMQ_X_SMALL is the
// fallback the launcher picks for grid-starved out_f — see the selection rule there.
#ifndef FP8_MMQ_X
#define FP8_MMQ_X      256
#endif
#ifndef FP8_MMQ_X_SMALL
#define FP8_MMQ_X_SMALL 128
#endif
// Token-tile selection margin, in percent: take the WIDE tile unless its wave-fill fraction falls
// below this percentage of the narrow tile's. 86 is the measured separator — see the TOKEN-TILE
// SELECTION note at the launcher for the cells it was fit against.
#ifndef FP8_MMQ_FILL_MARGIN_PCT
#define FP8_MMQ_FILL_MARGIN_PCT 86
#endif
static_assert(FP8_BLK % FP8_MMQ_Y == 0, "scale-row hoist requires mmq_y to divide the block edge");

// block_e4m3_mmq — footprint-identical to block_q8_1_mmq. v2 fills all four d4 slots with the same
// per-128 activation scale (see the ACTIVATIONS note).
struct block_e4m3_mmq {
    float   d4[4];
    uint8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_e4m3_mmq) == 4 * MMQ_TILE_Y_K, "y-tile stride contract");

// ======================= mma.cuh subset: tile<>, loads, f8f6f4 mma =======================
namespace memra_fp8_mma {
    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I, int J, typename T>
    static __device__ __forceinline__ void load_generic(
            tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
        }
    }

    // ldmatrix x4: the 16x8-int A tile (16 rows x 32 e4m3 bytes) in one instruction.
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }
} // namespace memra_fp8_mma

using namespace memra_fp8_mma;

// f32x2 -> packed e4m3x2 (Blackwell cvt; round-to-nearest-even, saturate to +-448).
static __device__ __forceinline__ uint16_t memra_fp8blk_cvt_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\n\tmov.b16 %0, t;\n}"
        : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

// plain-kind f8f6f4 MMA: D(f32 16x8) += A(e4m3 16x32) * B(e4m3 32x8). Same op as the W4A8-FP8 arm.
// v2 CHAINS this: c is the whole 128-k block accumulator, not a per-32 temporary.
static __device__ __forceinline__ void memra_fp8_mma_f8f4(
        float * __restrict__ c, const int * __restrict__ a, const int b0, const int b1) {
    asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ======================= tile loader: a COPY, not a dequant =======================
// [mmq_y rows x MMQ_ITER_K(128) k-values] of raw e4m3 -> smem, as int4 (16B) lines. k >= k_valid is
// zero-filled (e4m3 0x00 == exact 0.0). need_check clamps the SOURCE row to i_max exactly like the
// vendored loaders; the destination slot stays unclamped so no two threads alias.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_fp8_blk(
        const uint8_t * __restrict__ x, int * __restrict__ x_tile,
        const int kv0, const int i_max, const int stride_row, const int k_valid) {
    constexpr int nwarps        = mmq_y / 16;
    constexpr int nthreads      = nwarps * MMQ_WARP_SIZE;
    constexpr int lines_per_row = MMQ_ITER_K / 16;      // 16B lines per row (8)
    constexpr int nlines        = mmq_y * lines_per_row;
    const int t = threadIdx.y * MMQ_WARP_SIZE + threadIdx.x;

#pragma unroll
    for (int l0 = 0; l0 < nlines; l0 += nthreads) {
        const int L = l0 + t;
        if (nlines % nthreads != 0 && L >= nlines) { break; }
        const int r    = L / lines_per_row;
        const int line = L % lines_per_row;
        const int row  = need_check ? min(r, i_max) : r;
        const int kv   = kv0 + line * 16;

        int4 v = make_int4(0, 0, 0, 0);
        if (kv < k_valid) {   // in_dim % 16 == 0 (launcher-enforced) => never a partial 16B line
            v = *(const int4 *) (x + (size_t) row * (size_t) stride_row + (size_t) kv);
        }
        *(int4 *) (x_tile + r * MMQ_MMA_TILE_X_K_FP8 + line * 4) = v;
    }
}

// ======================= vec_dot: the whole 128-k scale block, ONE fold =======================
// s_blk AND dB are both uniform over this call — 128 k-values aligned to a 128 boundary, all rows
// of the CTA in one scale row, activation scale per 128 — so the four k32 MMAs chain into a single
// f32 accumulator and the epilogue folds (s_blk * dB) exactly once per element.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_fp8_blk_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum,
        const float s_blk) {
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity   = 16;                    // mmq_get_granularity_device(mmq_x>=48)
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int   * x_qs = x;
    const int   * y_qs = (const int   *) y + 4;          // skip the 4 f32 d4 slots
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx * tile_A_8::I);

    tile_A_8 A[ntx][MMQ_TILE_NE_K / 8];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            load_ldmatrix(A[n][k01 / 8],
                          x_qs + (i0 + n * tile_A_8::I) * MMQ_MMA_TILE_X_K_FP8 + k01,
                          MMQ_MMA_TILE_X_K_FP8);
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        // ONE unscaled accumulator per (j-tile, n) for the whole 128-k block.
        float C[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) { C[n][l] = 0.0f; }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            load_generic(B[0], y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0),           MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0 * MMQ_TILE_Y_K + (k01 + tile_B::J),   MMQ_TILE_Y_K);
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                memra_fp8_mma_f8f4(C[n], A[n][k01 / 8].x, B[0].x[0], B[1].x[0]);
            }
        }

        // Epilogue fold, ONCE per block: d4[0] is the per-128 activation scale (all four slots
        // carry it), s_blk the weight block scale.
        float sdB[tile_C::ne / 2];
#pragma unroll
        for (int l = 0; l < tile_C::ne / 2; ++l) {
            const int j = j0 + tile_C::get_j(l);
            sdB[l] = s_blk * y_df[j * MMQ_TILE_Y_K];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0 / tile_C::J + n) * tile_C::ne + l] += sdB[l % 2] * C[n][l];
            }
        }
    }
}

// ======================= write-back (mmq_write_back_mma) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_fp8_blk(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max, const float out_scale) {
    constexpr int granularity   = 16;
    constexpr int nwarps        = mmq_y / 16;
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;

    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
    static_assert(nwarps * tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                if (j > j_max) { continue; }
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (need_check && i > i_max) { continue; }
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l] * out_scale;
            }
        }
    }
}

// ======================= process_tile =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mul_mat_q_process_tile_fp8_blk(
        const uint8_t * __restrict__ x, const float * __restrict__ s_row,
        const int * __restrict__ y, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int k_valid, const int scale_cols,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = mmq_y / 16;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_e4m3_mmq) / sizeof(int);   // == MMQ_TILE_Y_K (36)
    const int k_iter_end = GGML_PAD(k_valid, MMQ_ITER_K);

    // OUTER LOOP == THE SCALE BLOCK (v2): one weight tile, one y chunk, one scalar scale load, one
    // chained-accumulator pass, one fold, two barriers. (The slice-3 cp.async double-buffer arm was
    // measured REFUTED and deleted — research/fp8st-20260804/mmq-v2/RESULTS.jsonl experiment B.)
    for (int kv0 = 0; kv0 < k_iter_end; kv0 += MMQ_ITER_K) {
        load_tiles_fp8_blk<mmq_y, need_check>(x, tile_x, kv0, tile_x_max_i, stride_row_x, k_valid);

        const int c0 = kv0 >> 7;                                  // y chunk == scale column
        // The clamp only guards a fully-padded tail chunk (its weight bytes are all 0x00, so
        // the scale value is irrelevant — the clamp keeps the grid read in bounds).
        const float sc0 = s_row[min(c0, scale_cols - 1)];

        {
            const int * by0 = y + ncols_y * c0 * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_fp8_blk_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, sc0);
        __syncthreads();
    }

    mmq_write_back_fp8_blk<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q (xy-tiling) =======================
template <int mmq_x, int mmq_y, bool need_check>
__launch_bounds__(MMQ_WARP_SIZE * (mmq_y / 16), 1)
static __global__ void mul_mat_q_fp8_blk(
        const uint8_t * __restrict__ x, const float * __restrict__ blk_scales,
        const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int k_valid, const int scale_cols, const float out_scale) {
    constexpr int nwarps    = mmq_y / 16;
    constexpr int warp_size = MMQ_WARP_SIZE;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) { break; }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int jt = blockIdx.y;   // token tile
    const int it = blockIdx.x;   // out-row tile; scale grid ROW == (it*mmq_y)>>7 (mmq_y | FP8_BLK)

    const int offset_y     = (jt * mmq_x) * (sizeof(block_e4m3_mmq) / sizeof(int));
    const int offset_dst   = jt * mmq_x * stride_col_dst + it * mmq_y;
    const int tile_x_max_i = nrows_x   - it * mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;

    mul_mat_q_process_tile_fp8_blk<mmq_x, mmq_y, need_check>(
        x + (size_t) it * mmq_y * (size_t) stride_row_x,
        blk_scales + (size_t) (((size_t) it * mmq_y) >> 7) * (size_t) scale_cols,
        y + offset_y, ids_dst_shared, dst + offset_dst,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, k_valid, scale_cols,
        out_scale);
}

// ======================= activation quantizer (v2: per-128 scale) =======================
// Twin of quantize_mmq_e4m3_d4_kernel (cu/mmq_nvfp4_f8f4.cu) with the amax reduction widened from
// 32 to 128 values — exactly one block_e4m3_mmq, which is exactly one warp's 32 lanes x 4 values.
// Output layout, block struct and byte footprint are IDENTICAL; all four d4 slots carry the same
// scale, so every consumer expression that reads d4[k01/8] still reads the right number.
static __global__ void quantize_mmq_e4m3_d128_kernel(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) { return; }   // ne0 % 512 == 0 and each CTA covers 512 values => never partial
    const int64_t i1 = blockIdx.x;

    const float4 * x4 = (const float4 *) x;
    block_e4m3_mmq * y = (block_e4m3_mmq *) vy;

    const int64_t ib  = (i0 / (4 * QK8_1)) * ne1 + i1;   // 128 values per block
    const int64_t iqs = i0 % (4 * QK8_1);

    const float4 xi = i0 < ne00 ? x4[(i1 * s01 + i0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
    // FULL-warp reduction: 32 lanes x 4 values == the 128-value block (v1 reduced over 8 lanes).
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    // e4m3 top-of-grid is 448; d maps the block amax onto it (mirror of 127/amax for int8).
    const float d_inv = amax == 0.0f ? 0.0f : 448.0f / amax;
    const uint16_t q01 = memra_fp8blk_cvt_e4m3x2(xi.x * d_inv, xi.y * d_inv);
    const uint16_t q23 = memra_fp8blk_cvt_e4m3x2(xi.z * d_inv, xi.w * d_inv);

    uint32_t * yqs4 = (uint32_t *) y[ib].qs;
    yqs4[iqs / 4] = (uint32_t) q01 | ((uint32_t) q23 << 16);

    if (iqs % 32 != 0) { return; }
    y[ib].d4[iqs / 32] = amax == 0.0f ? 0.0f : amax / 448.0f;
}

// ======================= NaN-code scan (dispatch guard) =======================
// Counts e4m3 bytes with magnitude 0x7F. Those decode to NaN in hardware but to 0.0 in the host /
// ARM B' convention, so a tensor containing them must NOT ride this kernel.
static __global__ void fp8_blk_count_nan_kernel(
        const uint8_t * __restrict__ x, const size_t n, unsigned int * __restrict__ out) {
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    unsigned int local = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        local += ((x[i] & 0x7Fu) == 0x7Fu) ? 1u : 0u;
    }
    if (local != 0u) { atomicAdd(out, local); }
}

// ======================= C-ABI host launcher =======================
// SM count of the current device, queried once and cached. The token-tile selection rule below needs
// it on every call; a cudaDeviceGetAttribute per prefill GEMM would put a driver round-trip in front
// of every launch. Cached per device ordinal (small fixed table, no allocation, no lock: two racing
// threads compute the same value).
static int memra_fp8_blk_nsm() {
    constexpr int MAX_DEV = 16;
    static int cache[MAX_DEV] = {0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= MAX_DEV) {
        // Unknown device: fall back to a query, and if that fails to 1 (which makes both candidate
        // tiles report full waves, so the rule prefers the wide tile — the m=6257 behaviour).
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev < 0 ? 0 : dev)
            != cudaSuccess || n <= 0) {
            return 1;
        }
        return n;
    }
    int n = cache[dev];
    if (n == 0) {
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
            || n <= 0) {
            n = 1;
        }
        cache[dev] = n;
    }
    return n;
}

extern "C" {

// Scratch sizing matches the W4A8-FP8 arm byte for byte (same block struct, same padding rule):
// v2 owns its quantizer but deliberately introduces no new activation FOOTPRINT.
size_t memra_mmq_fp8_blk_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    // Overread pad: the mul_mat_q y-tile loader always reads a FULL mmq_x-column tile; for the final
    // k-block with n_tokens % mmq_x != 0 that read runs past the last real column. Padding the
    // scratch keeps the overread mapped (values are garbage; write-back drops j > j_max). The pad
    // MUST be the widest tile the launcher can pick — v2 selects between FP8_MMQ_X (256) and
    // FP8_MMQ_X_SMALL (128) per shape, so it is the max of the two, not a hardcoded 128.
    constexpr int64_t max_tile_x =
        FP8_MMQ_X > FP8_MMQ_X_SMALL ? FP8_MMQ_X : FP8_MMQ_X_SMALL;
    return (size_t) (nblocks + max_tile_x) * sizeof(block_e4m3_mmq);
}

// Scale-grid dims for an [out_f x in_f] block-128 FP8 tensor.
int memra_mmq_fp8_blk_scale_rows(int out_f) { return (out_f + FP8_BLK - 1) / FP8_BLK; }
int memra_mmq_fp8_blk_scale_cols(int in_f)  { return (in_f  + FP8_BLK - 1) / FP8_BLK; }

// Run the per-block FP8 MMQ prefill GEMM.
//   y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T, scaled per [128x128] block.
//   W_e4m3      : uint8 e4m3 codes, row-major [out_f x in_f], row stride in_f bytes.
//   blk_scales  : f32 [ceil(out_f/128) x ceil(in_f/128)], row-major (device memory).
//   act_f32     : f32 activation [n_tokens, in_f].
//   act_scratch : >= memra_mmq_fp8_blk_act_bytes(in_f, n_tokens).
//   out_scale   : extra per-tensor factor folded into write-back (1.0 = none).
// Requires in_f % 16 == 0. Returns 0, 1 (bad dims), 1000+cudaError (quantize), 2000+cudaError.
int memra_mmq_fp8_blk(const void * W_e4m3, const float * blk_scales, const float * act_f32,
                      float * y, int in_f, int out_f, int n_tokens, void * act_scratch,
                      void * stream, float out_scale) {
    if (in_f <= 0 || out_f <= 0 || n_tokens <= 0 || (in_f % 16) != 0) { return 1; }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    {   // per-128 activation quantize (v2's own kernel; 128 threads x 4 values == 4 blocks per CTA,
        // one 128-value block per warp — see the ACTIVATIONS note in the header).
        const int64_t ne0 = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
        const int block_size = 128;
        const dim3 nb((unsigned) n_tokens, (unsigned) ((ne0 / 4 + block_size - 1) / block_size), 1);
        quantize_mmq_e4m3_d128_kernel<<<nb, block_size, 0, st>>>(
            act_f32, act_scratch, (int64_t) in_f, (int64_t) in_f, ne0, n_tokens);
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return 1000 + (int) e; } }

    const int scale_cols = (in_f + FP8_BLK - 1) / FP8_BLK;
    const bool need_check = (out_f % FP8_MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;

    // TOKEN-TILE SELECTION (measured; research/fp8st-20260804/mmq-v2/RESULTS.jsonl experiment A and
    // the 1.7B shape row). X=256 wins wherever the grid still fills the machine, and loses badly
    // wherever it does not: 5120->1024 is 8 row tiles at Y=128, so X=256 leaves 2 token tiles at
    // m=512 = 16 CTAs on an 82-SM part (0.522x floor) against X=128's 0.854x.
    //
    // The quantity that separates those cases is WAVE FILL, and an out_f-only threshold cannot
    // express it: n_tokens sets the token-tile count, so the same out_f starves at one m and fills at
    // another, and out_f thresholds calibrated on the 27B mis-picked EVERY qwen3-1.7B projection
    // (out_f 1024-6144 -> 0.535-0.865x at X=256 vs 0.847-0.890x at X=128). So compute both
    // candidates' fill directly: ctas = row_tiles * token_tiles, and fill = ctas / (waves * nsm),
    // i.e. how full the last wave is. Take the WIDE tile unless its fill is more than
    // FP8_MMQ_FILL_MARGIN_PCT percent below the narrow tile's — wide is preferred on ties because a
    // wider token tile amortizes each weight-tile read over twice the MMA work.
    //
    // The 86% separator classifies every measured cell correctly. The two pairs it has to split are
    // tight and both real: 27B q_proj at m=512 (fill ratio 0.833 -> narrow; X=128 wins 0.968 vs
    // 0.923) against 27B gate_up at m=512 (0.875 -> wide; X=256 wins 0.995 vs 0.973); and 27B
    // k/v_proj at m=6257, whose ratio is exactly 0.850 and which measured 0.913x at X=256 against
    // 0.951x at X=128 — i.e. the 1024-row shape stays narrow at every m tested, which is why the
    // margin sits at 86 and not 85. On the 1.7B at m=512 the rule picks X=128 everywhere, matching
    // that model's own GEMM sheet (0.847-0.890x at X=128 against 0.535-0.865x at X=256), and at
    // m=6257 it picks X=256 for the wide projections, which measured 0.998-1.064x.
    // SM count: a device property, so it is queried once per device and cached. This runs on every
    // prefill GEMM (25k+ calls in one 1.7B stream), so a driver round-trip per call would be charged
    // to the kernel it is selecting for.
    const int nsm = memra_fp8_blk_nsm();
    const int64_t nty64 = (out_f + FP8_MMQ_Y - 1) / FP8_MMQ_Y;
    auto fill_terms = [&](int mx, int64_t & ctas, int64_t & waves) {
        ctas  = nty64 * ((n_tokens + mx - 1) / mx);
        waves = (ctas + nsm - 1) / nsm;
    };
    int64_t ctas_w, waves_w, ctas_n, waves_n;
    fill_terms(FP8_MMQ_X,       ctas_w, waves_w);
    fill_terms(FP8_MMQ_X_SMALL, ctas_n, waves_n);
    // fill_w >= (margin/100) * fill_n, cross-multiplied (the common /nsm cancels).
    const bool use_wide = 100 * ctas_w * waves_n
                          >= (int64_t) FP8_MMQ_FILL_MARGIN_PCT * ctas_n * waves_w;
    #define MEMRA_FP8MMQ_LAUNCH(MX, NC) do {                                                      \
        const int    nty  = (out_f    + FP8_MMQ_Y - 1) / FP8_MMQ_Y;                                \
        const int    ntx  = (n_tokens + (MX) - 1) / (MX);                                          \
        const dim3   grid((unsigned) nty, (unsigned) ntx, 1);                                      \
        const dim3   blk3(MMQ_WARP_SIZE, FP8_MMQ_NWARPS, 1);                                       \
        const size_t nbs_ids = (size_t) (MX) * sizeof(int);                                        \
        const size_t nbs_y   = (size_t) (MX) * sizeof(block_e4m3_mmq);                             \
        const size_t pad     = (size_t) FP8_MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);              \
        const size_t nbs_x   = (size_t) FP8_MMQ_Y * MMQ_MMA_TILE_X_K_FP8 * sizeof(int);            \
        const size_t smem    = nbs_ids + GGML_PAD(nbs_y, pad) + nbs_x;                             \
        cudaFuncSetAttribute(mul_mat_q_fp8_blk<(MX), FP8_MMQ_Y, NC>,                              \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                   \
        mul_mat_q_fp8_blk<(MX), FP8_MMQ_Y, NC><<<grid, blk3, smem, st>>>(                          \
            (const uint8_t *) W_e4m3, blk_scales, y_q, y, out_f, n_tokens, in_f, n_tokens, out_f, \
            in_f, scale_cols, out_scale);                                                          \
    } while (0)
    if (use_wide) {
        if (need_check) { MEMRA_FP8MMQ_LAUNCH(FP8_MMQ_X, true); }
        else            { MEMRA_FP8MMQ_LAUNCH(FP8_MMQ_X, false); }
    } else {
        if (need_check) { MEMRA_FP8MMQ_LAUNCH(FP8_MMQ_X_SMALL, true); }
        else            { MEMRA_FP8MMQ_LAUNCH(FP8_MMQ_X_SMALL, false); }
    }
    #undef MEMRA_FP8MMQ_LAUNCH

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 2000 + (int) e; }
    return 0;
}

// Count e4m3 NaN codes in a device weight buffer. out_count must be a device u32 (zeroed by this
// call). Returns 0 or a cudaError_t.
int memra_fp8_blk_count_nan(const void * W_e4m3, size_t nbytes, unsigned int * out_count,
                            void * stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t e = cudaMemsetAsync(out_count, 0, sizeof(unsigned int), st);
    if (e != cudaSuccess) { return (int) e; }
    if (nbytes == 0) { return 0; }
    const unsigned int threads = 256;
    unsigned int blocks = (unsigned int) ((nbytes + threads - 1) / threads);
    if (blocks > 4096u) { blocks = 4096u; }
    fp8_blk_count_nan_kernel<<<blocks, threads, 0, st>>>((const uint8_t *) W_e4m3, nbytes, out_count);
    return (int) cudaGetLastError();
}

} // extern "C"
