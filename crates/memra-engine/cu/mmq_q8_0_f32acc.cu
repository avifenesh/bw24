// mmq_q8_0_f32acc.cu — THE Q1 INSTRUMENT for the FP8-ST v3 gate (research-only, lane/fp8-v3-gate).
//
// QUESTION (research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl §3, §6):
//   v2's ceiling claim is "what is left is the f32 accumulator itself against the floor's s32, and
//   that is structural to per-block FP8 as formulated". Its own stop rule then says a v3 (quantize
//   the e4m3 mantissa into an int8-compatible product per 128-block so the chain accumulates in s32)
//   "should not start without a receipted estimate that s32-vs-f32 accumulate is worth the >= 10pp
//   it would have to buy."
//
// THE INSTRUMENT: this TU is cu/mmq_q8_0.cu — the Q8_0 MMQ FLOOR — with the accumulator as its ONE
// free variable. Both arms live in this one file, share every loader, every smem expression, every
// tile constant, every launch parameter, and consume THE SAME device byte buffers. The entire diff
// between arm S32 and arm F32 is:
//
//   * the per-MMA accumulator register class: int[4] -> float[4]
//   * the MMA:  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//            -> mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// Nothing else. Same m16n8k32 shape, same A fragment (4x b32), same B fragment (2x b32), same D
// fragment (4 regs) — the s8 and e4m3 register ABIs for this shape are identical, which is why a
// one-instruction swap is a legal single-variable experiment and not a rewrite. Both arms keep the
// floor's f32 `sum[]` running total and the floor's per-k32 (dA*dB) fold, so the FOLD COUNT is
// controlled too: the only thing that moves is the MMA and the register it lands in.
//
// WHY THIS IS THE RIGHT ESTIMATOR, AND WHAT IT DOES *NOT* MEASURE:
//   v2 already chains FOUR k32 MMAs into one f32 accumulator and folds (s_blk*dB) ONCE per 128-k
//   block, while the floor folds every k32 — FOUR folds per 128 k-values. So v2's epilogue f32 work
//   is already strictly CHEAPER than the floor's, and the fold count cannot be the residual gap.
//   The only remaining named variable is the accumulator/MMA class itself, and that is exactly what
//   this file isolates: the floor at f32 vs the floor at s32, at byte-identical geometry, traffic
//   and instruction count. The measured delta is therefore an UPPER BOUND on what a v3's s32
//   conversion could recover, because a v3 would still pay per-128-block mantissa extraction on top
//   of whatever this delta shows.
//
// DATA VALIDITY (it is a TIMING instrument, not a numeric one): the weight and activation byte
// streams are synthesized host-side by the bench and shared bit-for-bit by both arms. The bench
// keeps them NaN-FREE in the e4m3 reading (no byte has magnitude 0x7F) and finite, so the F32 arm
// can neither NaN nor take a denormal/special-case path the S32 arm avoids. Neither arm's OUTPUT is
// meaningful and no exactness claim is made or implied: this TU has no dispatch seam, is never
// linked into a serving path, and the numeric contract for a real v3 kernel would need its own host
// reference (see the lane VERDICT).
//
// GEMM-ONLY BY DESIGN: both entry points take a PRE-QUANTIZED block_q8_1_mmq activation buffer and
// launch only mul_mat_q. The activation quantizer is deliberately outside the measurement — the
// accumulator lives in the GEMM, and the two arms must not differ by a quantizer.
//
// All symbols are static or _accprobe-suffixed so this TU cannot collide with cu/mmq_q8_0.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

// ======================= constants (verbatim from cu/mmq_q8_0.cu) =======================
#define WARP_SIZE 32
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

#define QK8_0 32
#define QI8_0 8
#define QK8_1 32
#define QI8_1 8
#define MATRIX_ROW_PADDING 512

#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K    256
#define MMQ_MMA_TILE_X_K_Q8_0 (2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4)  // 76
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)                        // 36

#define MMQ_WARP_SIZE 32
#define MMQ_NWARPS    8
#define MMQ_Y         128
#ifndef MMQ_X
#define MMQ_X         128
#endif

static __device__ __forceinline__ int get_int_b2_acc(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] <<  0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0_acc;
static_assert(sizeof(block_q8_0_acc) == 34, "wrong q8_0 block size/padding");

struct block_q8_1_mmq_acc {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq_acc) == 4 * MMQ_TILE_Y_K, "block_q8_1_mmq != MMQ_TILE_Y_K ints");

// ======================= mma.cuh subset: tile<>, loads, BOTH MMAs =======================
namespace memra_accprobe_mma {

    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else {
                __trap();
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else {
                __trap();
                return -1;
            }
        }
    };

    template <int I, int J, typename T>
    static __device__ __forceinline__ void load_generic(tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    // ---- ARM S32: the floor's MMA, verbatim. D(s32) += A(s8) * B(s8). ----
    // rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md
    //   16.06 cyc/warp-MMA, 309.7 TOP/s = fastest int8 form; nothing deeper exists (ptxas rejects
    //   m16n8k64.s8). OPTIMAL. This whole file is probe-only (bin accprobe_bench, never a serving
    //   path) -- see the MMA FORM block on the F32 arm below, which is where the rate mattered.
    static __device__ __forceinline__ void mma_s32(
            int * __restrict__ d, const int * __restrict__ a, const int b0, const int b1) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }

    // ---- ARM F32: the SAME m16n8k32 shape and the SAME A/B/D fragment ABI, f32 accumulate over
    // e4m3 operands. This is the exact op cu/mmq_fp8_blk.cu (v2) accumulates in. sm_100a+/sm_120a. ----
    //
    // MMA FORM — READ THIS BEFORE CITING THIS INSTRUMENT'S DELTA (added 2026-08-06, lane/rp-on-st).
    // The F32 arm's ONE instruction has two PTX spellings on sm_120a that compute the identical
    // e4m3xe4m3 product, at DIFFERENT issue intervals (research/w4a8-prefill-20260806 slices 3-4,
    // clock64 + full-GPU cudaEvent, NACC=1..16 ILP control):
    //
    //   kind::f8f6f4 (plain)                                32.02 cyc/warp-MMA
    //   kind::mxf8f6f4.block_scale.scale_vec::1X @ ue8m0     16.06 cyc/warp-MMA
    //   m16n8k32.s8.s8.s32   (the S32 arm below)             16.06 cyc/warp-MMA
    //
    // So the PLAIN form is 2x the interval of the S32 arm's MMA, at the same shape — meaning the
    // "f32 vs s32 accumulate" single-variable claim was NOT single-variable while this arm used the
    // plain form: the accumulator class and the MMA issue interval moved together, and the +19.8/+20.2
    // delta_pp the fp8-v3 gate published (research/fp8v3-gate-20260805/) is the SUM of both.
    // The block_scale form at the ue8m0 identity scale is bit-identical to the plain form (0/128
    // accumulator elements differ, live-operand controls at 2^1/2^-1 exact) and issues at the S32
    // arm's interval, so it is the form that makes this instrument actually single-variable.
    //
    // ACCPROBE_F32_PLAIN=1 (build-time, -DMEMRA_ACCPROBE_PLAIN_MMA) reproduces the ORIGINAL plain-form
    // arm — keep it: the published delta belongs to that arm and its receipts must stay reproducible.
    //
    // rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md — the rates quoted above
    // were re-measured independently by the repo-wide audit (12 forms, 3 reruns, SASS-census verified)
    // and CONFIRMED: plain 32.03, block_scale 16.06, s8-k32 16.06. Default arm = the fast one.
    // Verdict: OPTIMAL (default), and the plain arm behind MEMRA_ACCPROBE_PLAIN_MMA is a deliberate
    // receipt-reproduction door, not a rate defect.
#define MEMRA_ACCPROBE_UE8M0_ONE 0x7F7F7F7Fu   // four ue8m0 bytes, each 2^0
    static __device__ __forceinline__ void mma_f32(
            float * __restrict__ d, const int * __restrict__ a, const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#ifdef MEMRA_ACCPROBE_PLAIN_MMA
        asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
        asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
            ".f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
              "r"(MEMRA_ACCPROBE_UE8M0_ONE), "r"(MEMRA_ACCPROBE_UE8M0_ONE));
#endif
#else
        // Pre-Blackwell has no .kind::f8f6f4. The F32 arm fails closed rather than silently
        // measuring something else; the S32 arm still builds and runs everywhere.
        (void) a; (void) b0; (void) b1; (void) d;
        __trap();
#endif
    }
} // namespace memra_accprobe_mma

using namespace memra_accprobe_mma;

static constexpr __device__ int mmq_get_granularity_device_acc(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// ======================= load_tiles_q8_0 (verbatim; arm-independent) =======================
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q8_0_acc(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
        const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2 * MMQ_TILE_NE_K);

    const int txi  = threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) { i = min(i, i_max); }

        const block_q8_0_acc * bxi = (const block_q8_0_acc *) x + kbx0 + i * stride + kbx;

        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = get_int_b2_acc(bxi[0].qs, kqsx);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] =
            get_int_b2_acc(bxi[MMQ_TILE_NE_K / QI8_0].qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = 2 * MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }

        const block_q8_0_acc * bxi = (const block_q8_0_acc *) x + kbx0 + i * stride + kbxd;

        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
    }
}

// ======================= vec_dot — THE ONE VARIABLE =======================
// Identical to vec_dot_q8_0_q8_1_mma except that the per-MMA accumulator register class and the MMA
// instruction are selected by F32ACC. Same A loads, same B loads, same dA/dB gathers, same fold
// count, same loop nest, same unroll, same f32 sum[] running total.
template <int mmq_x, int mmq_y, bool F32ACC>
static __device__ __forceinline__ void vec_dot_q8_0_acc(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;   // shape/ne only; the accumulator storage class is below

    constexpr int granularity = mmq_get_granularity_device_acc(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2 * MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    tile_A A[ntx][MMQ_TILE_NE_K / QI8_0];
    float dA[ntx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_0];

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;
                dA[n][l][k01/QI8_0] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                if constexpr (F32ACC) {
                    float C[tile_C::ne] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_f32(C, A[n][k01/QI8_0].x, B.x[0], B.x[1]);
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        sum[(j0/tile_C::J + n)*tile_C::ne + l] += C[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                    }
                } else {
                    int C[tile_C::ne] = {0, 0, 0, 0};
                    mma_s32(C, A[n][k01/QI8_0].x, B.x[0], B.x[1]);
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        sum[(j0/tile_C::J + n)*tile_C::ne + l] += C[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                    }
                }
            }
        }
    }
}

// ======================= write-back (verbatim; arm-independent) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_acc(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    constexpr int granularity = mmq_get_granularity_device_acc(mmq_x);
    constexpr int nwarps = MMQ_NWARPS;
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I;

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
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

// ======================= process_tile / mul_mat_q (verbatim apart from the arm flag) ==============
template <int mmq_x, bool need_check, bool F32ACC>
static __device__ __forceinline__ void mul_mat_q_process_tile_acc(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = MMQ_NWARPS;
    constexpr int qk        = QK8_0;
    constexpr int mmq_y     = MMQ_Y;

    extern __shared__ int data_mul_mat_q_acc[];
    int * tile_y = data_mul_mat_q_acc + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block        = 4 * QK8_1;
    constexpr int ITER_K          = MMQ_ITER_K;
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq_acc) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_q8_0_acc<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_0_acc<mmq_x, mmq_y, F32ACC>(tile_x, tile_y, sum, 0);
        __syncthreads();
        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_0_acc<mmq_x, mmq_y, F32ACC>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_acc<mmq_x, mmq_y, need_check>(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

template <int mmq_x, bool need_check, bool F32ACC>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_acc(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int mmq_y = MMQ_Y;

    extern __shared__ int ids_dst_shared_acc[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) { break; }
        ids_dst_shared_acc[j] = j;
    }
    __syncthreads();

    const int jt = blockIdx.y;
    const int it = blockIdx.x;

    const int col_diff = ncols_dst;
    const int offset_y   = (jt * mmq_x) * (sizeof(block_q8_1_mmq_acc) / sizeof(int));
    // 64-bit offset_dst (audit Q7, 2026-08-05): wraps at n_tokens*out_f >= 2^31 — see mmq_q8_0.cu.
    const int64_t offset_dst = (int64_t) jt * mmq_x * stride_col_dst + (int64_t) it * mmq_y;

    const int tile_x_max_i = nrows_x  - it * mmq_y - 1;
    const int tile_y_max_j = col_diff - jt * mmq_x - 1;

    const int offset_x = it * mmq_y * stride_row_x;

    mul_mat_q_process_tile_acc<mmq_x, need_check, F32ACC>(
        x, offset_x, y + offset_y, ids_dst_shared_acc, dst + offset_dst,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00);
}

// ======================= host launcher =======================
static size_t mmq_acc_nbytes_shared() {
    const size_t nbs_ids = (size_t) MMQ_X * sizeof(int);
    const size_t nbs_x   = (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_Q8_0 * sizeof(int);
    const size_t nbs_y   = (size_t) MMQ_X * sizeof(block_q8_1_mmq_acc);
    const size_t pad     = (size_t) MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

template <bool F32ACC>
static int launch_acc(const void * W, const void * act_q, float * y,
                      int in_f, int out_f, int n_tokens, cudaStream_t st) {
    const int stride_row_x    = in_f / QK8_0;
    const int blocks_per_ne00 = in_f / QK8_0;
    const int stride_col_dst  = out_f;
    const int ncols_y         = n_tokens;

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    const size_t smem = mmq_acc_nbytes_shared();

    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_q;
    const char * Wc = (const char *) W;

    if (need_check) {
        cudaFuncSetAttribute(mul_mat_q_acc<MMQ_X, true, F32ACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_acc<MMQ_X, true, F32ACC><<<grid, block, smem, st>>>(
            Wc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00);
    } else {
        cudaFuncSetAttribute(mul_mat_q_acc<MMQ_X, false, F32ACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_acc<MMQ_X, false, F32ACC><<<grid, block, smem, st>>>(
            Wc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

extern "C" {

// Activation-scratch bytes for a pre-quantized block_q8_1_mmq stream (same rule as the floor).
size_t memra_accprobe_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    return (size_t) (nblocks + MMQ_X) * sizeof(block_q8_1_mmq_acc);
}

// ARM S32 — the Q8_0 MMQ floor's GEMM, s32 accumulate. GEMM only (act_q pre-quantized).
int memra_accprobe_gemm_s32(const void * W_q8_0_blocks, const void * act_q, float * y,
                            int in_f, int out_f, int n_tokens, void * stream) {
    if (in_f <= 0 || out_f <= 0 || n_tokens <= 0 || (in_f % 32) != 0) { return 1; }
    return launch_acc<false>(W_q8_0_blocks, act_q, y, in_f, out_f, n_tokens,
                             reinterpret_cast<cudaStream_t>(stream));
}

// ARM F32 — byte-identical kernel, f32 accumulate over the e4m3 reading of the same bytes.
int memra_accprobe_gemm_f32(const void * W_q8_0_blocks, const void * act_q, float * y,
                            int in_f, int out_f, int n_tokens, void * stream) {
    if (in_f <= 0 || out_f <= 0 || n_tokens <= 0 || (in_f % 32) != 0) { return 1; }
    return launch_acc<true>(W_q8_0_blocks, act_q, y, in_f, out_f, n_tokens,
                            reinterpret_cast<cudaStream_t>(stream));
}

} // extern "C"
