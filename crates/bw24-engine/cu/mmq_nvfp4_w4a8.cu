// mmq_nvfp4_w4a8.cu — NVFP4 W4A8 int8-MMA MMQ prefill GEMM (vendored floor, ggml-decoupled, sm_120a).
//
// STAGE 2 "accuracy-safe rung": the SAME fast MMQ tile as the W4A4 kernel (cu/mmq_fp4.cu), but the
// NON-Blackwell NVFP4 MMQ pair — weight FP4 is LUT-dequantized to int8 at tile-load (bit-exact, the
// weight side is never the accuracy problem) and the activation stays q8_1 int8 (the SAME quant the
// default int8 W4A8 GEMM uses, which passes ALL exactness gates). This keeps bw24's int8-W4A8
// accuracy class while running on the MMQ tensor-core tile. Source: llama.cpp ggml/src/ggml-cuda/
//   - quantize.cu  : quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4> (activation f32 -> block_q8_1_mmq,
//                    symmetric float scale d per 32, NO sum term — NVFP4 is symmetric, uses D4 not DS4)
//   - mmq.cuh      : load_tiles_nvfp4 (non-BLACKWELL arm, mmq.cuh:1069): FP4->int8 via
//                    get_int_from_table_16(src_qs, kvalues_mxfp4) into x_qs + per-16 UE4M3 float scale
//                    into x_df; vec_dot_q8_0_16_q8_1_mma (mmq.cuh:1495 TURING arm): int8 m16n8k16 mma
//                    pairs, per-4-col x scale * per-32 y scale; mmq_write_back_mma; mul_mat_q xy-tiling
//   - mma.cuh      : tile<>, load_ldmatrix, load_generic, mma.sync.m16n8k16.row.col.s32.s8.s8.s32
//
// DECOUPLING: no ggml headers, same treatment as mmq_fp4.cu / mmq_q45k.cu. Self-contained TU (all
// functions static/internal, no link collisions with the W4A4 nvfp4 kernel).
//
// KEY DIFFS vs the W4A4 file (mmq_fp4.cu) — this is the SAME weight format, DIFFERENT math:
//   - tile size MMQ_MMA_TILE_X_K_NVFP4 (84) not MMQ_MMA_TILE_X_K_FP4 (76): x_qs holds the FP4->int8
//     dequant (2*MMQ_TILE_NE_K ints) + x_df holds MMQ_TILE_NE_K/2 UE4M3-decoded FLOAT scales.
//   - MMQ_ITER_K = 256 (not 512): 4 NVFP4 blocks per iteration, two 128-value q8_1 y-chunks.
//   - Activation is q8_1 int8 D4 (float scale, symmetric), NOT block_fp4_mmq (2-level FP8 W4A4).
//   - MMA is plain int8 m16n8k16 with float epilogue, NOT the mxf4nvf4 block-scale op. -> no
//     Blackwell-only asm; the W4A8 accuracy class = bw24's default GEMM math.
//
// C-ABI launcher: bw24_mmq_nvfp4_w4a8 (+ shared bw24_mmq_nvfp4_w4a8_act_bytes). Compiled to the same
// libbw24_mmq.a static lib, called from Rust via FFI, dispatched behind BW24_MMQ_W4A8=1.
//
// BW24_PP_PIPE=1 (DEFAULT since 2026-07-09): cp.async multi-stage smem pipeline for the same tile —
// raw FP4 staged async + LUT-dequanted from smem, q8_1 y-chunks double-buffered. Bit-identical
// output (data movement only); see the pipeline section below for the schedule.
//
// BW24_PP_PIPE=2 (w4a8v2 lane, default OFF): 2-CTA/SM occupancy mode. ncu on g7e (2026-07-09,
// pp1845 27B) showed the pipelined kernel warp-starved, NOT latency-bound: tensor(INT) pipe 59%
// (top), 2.00 active warps/scheduler, 48% no-eligible cycles, dominant stall = math_pipe_throttle
// (1.20 of 3.85 cyc/issue) — one 8-warp CTA stalls in lockstep on the tensor pipe. A 64x64 tile
// probe (2 CTA/SM) halved math_pipe_throttle but doubled weight staging+dequant (X-halving tax) =
// net flat. Mode 2 = the config that gets 2 CTAs WITHOUT duplicating dequant: mmq_y=64 (4 warps),
// mmq_x kept at 128, and a slimmer pipe (SINGLE y slot, no ring) so smem fits 2 CTAs:
// ids 512B + ty 18432B + tile_x 21504B + xr 9216B = 48.5KB <= 50.2KB budget. Inter-CTA overlap
// replaces the intra-CTA y-ring; total weight bytes + dequant work UNCHANGED (Y-axis split), only
// y-chunk loads duplicate (DRAM was 6.8%). Bit-identical per (token,row): same dequant bytes/math,
// same per-output k-accumulation order — Y-tiling only partitions output rows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>

// ======================= ggml constants/macros (vendored, sm_120) =======================
#define TURING_MMA_AVAILABLE
#define WARP_SIZE 32
#define NO_DEVICE_CODE __trap()
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

// quant-format constants (ggml-common.h)
#define QK_K 256
#define QK8_1 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16           // 16-element sub-block (one UE4M3 micro-scale each)
#define QI8_1 8                   // QK8_1 / (4*QR8_1), QR8_1 == 1
#define MATRIX_ROW_PADDING 512

// MMQ tile constants (mmq.cuh) — NVFP4 GENERIC (non-Blackwell) path.
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K 256
#define MMQ_MMA_TILE_X_K_NVFP4 (2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4)   // 84, %8==4 padded
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)                 // 36

// sm_120 launch constants (same shape as the W4A4 nvfp4 / q45k kernels: 8 warps, 128x128 tile).
#define MMQ_WARP_SIZE 32
// MMQ_Y seam (2026-07-06): mmq_y = nwarps * 16 (write-back static_assert), so Y and NWARPS move
// together. Default 128x8 = 42KB tile_x = 1 CTA/SM (warps_active 16.7% of 48). Y=64/NWARPS=4
// halves tile_x -> 2 CTA/SM candidate; total weight/act bytes UNCHANGED (unlike the MMQ_X axis,
// which re-reads weights per token tile — that's why X=32 lost 28% while this axis is free).
#ifndef MMQ_Y
#define MMQ_Y         128
#endif
#define MMQ_NWARPS    (MMQ_Y / 16)
#ifndef MMQ_X
#define MMQ_X         128
#endif
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// FP4 e2m1 reconstruction LUT (ggml-common.h kvalues_mxfp4 == kvalues_fp4).
__constant__ int8_t kvalues_mxfp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume >= 4 byte alignment
}

// UE4M3 (FP8 e4m3, divided by 2 to match ggml CPU impl) -> fp32.
static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // NaN -> 0.0f to match CPU impl
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}

// get_int_from_table_16 (vecdotq.cuh:34, CUDA branch): 4-bit LUT gather of 8 e2m1 codes -> two ints
// of 4 packed int8 each (even nibbles in .x, odd nibbles in .y).
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ======================= weight / activation block structs =======================
// llama block_nvfp4 (ggml-common.h): 36 bytes = 4 UE4M3 scales (per 16) + 32 packed e2m1 (64 vals).
typedef struct {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // UE4M3 scales (4 bytes)
    uint8_t qs[QK_NVFP4 / 2];           // packed 4-bit e2m1 (32 bytes)
} block_nvfp4;

// block_q8_1_mmq (mmq.cuh): 4x float scale (D4) + 128 int8 quants.
struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4 * QK8_1];               // 128 values
};
static_assert(sizeof(block_q8_1_mmq) == 4 * MMQ_TILE_Y_K, "block_q8_1_mmq != MMQ_TILE_Y_K ints");

// ======================= mma.cuh: tile<>, loads, int8 m16n8k16 mma =======================
namespace ggml_cuda_mma {
    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
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
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
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

    // ldmatrix x2 for the 16x4 int tile (A minitile, mma.cuh:801).
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
    }

    // ldmatrix x4 for the 16x8 int tile (A_8 load, two 16x4 minitiles at once; mma.cuh:830).
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    // int8 MMA (mma.cuh:920, Ampere+): D(s32) += A(16x4 s8) * B(8x4 s8), m16n8k16.
    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
    }
} // namespace ggml_cuda_mma

using namespace ggml_cuda_mma;

// Turing+ granularity (mmq_get_granularity_device): mmq_x>=48 -> 16.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// ======================= load_tiles_nvfp4 (mmq.cuh:1069, NON-Blackwell arm) =======================
// FP4->int8 dequant via the kvalues LUT + per-16 UE4M3 float scale. NVFP4 is symmetric (no min-offset).
//
// TWO weight layouts behind ONE loader (is_rp template arm — pure ADDRESS remap, the dequant math,
// smem write order, and FP ops are token-for-token IDENTICAL, so the kernel result is bit-identical):
//   is_rp=false: GGUF 36B blocks ([4B UE4M3 d][32B qs] interleaved per 64 values), base = x.
//   is_rp=true : A6 split planes (model.rs repack_nvfp4_split): quant plane out_f x nsb64 x 32B at x,
//                scale plane out_f x nsb64 x 4B at x_sc (= x + out_f*nsb64*32). The repack copies the
//                same 32 qs bytes / 4 d bytes verbatim, and the flat block index ib = row*stride + kbx
//                indexes BOTH planes directly (quant at ib*32, scale at ib*4).
template <int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void load_tiles_nvfp4_w4a8(
        const char * __restrict__ x, const char * __restrict__ x_sc, int * __restrict__ x_tile,
        const int kb0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_y / 16;     // warp count rides the row tile (write-back mapping)
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K * 2);

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;   // 4 blocks per row per iter
    constexpr int rows_per_warp = warp_size / threads_per_row; // 8
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;
        if constexpr (need_check) { i = min(i, i_max); }

        const int ib = kb0 + i * stride + kbx; // flat NVFP4 block index (row-major, both layouts)
        const uint32_t * __restrict__ src_qs;
        const uint8_t  * __restrict__ src_d;
        if constexpr (is_rp) {
            src_qs = reinterpret_cast<const uint32_t *>(x    + (size_t) ib * 32);
            src_d  = reinterpret_cast<const uint8_t  *>(x_sc + (size_t) ib * 4);
        } else {
            const block_nvfp4 * bxi = (const block_nvfp4 *) x + ib;
            src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
            src_d  = bxi->d;
        }
        const int kqs = 16 * kbx;
        const int ksc = 4 * kbx;

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], kvalues_mxfp4);
            const int2 q1 = get_int_from_table_16(src_qs[2 * sub + 1], kvalues_mxfp4);
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 0] = q0.x;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 1] = q1.x;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 2] = q0.y;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 3] = q1.y;
            x_df[i * MMQ_MMA_TILE_X_K_NVFP4 + ksc + sub] = ggml_cuda_ue4m3_to_fp32(src_d[sub]);
        }
    }
}

// ======================= cp.async multi-stage pipeline (BW24_PP_PIPE=1) =======================
// Marlin-style cross-iteration smem pipeline for the SAME tile math above. cp.async changes WHEN
// bytes arrive, never WHAT is computed: the weight FP4 bytes are staged RAW into a smem buffer and
// the LUT dequant (dequant_tiles_* below) reads the identical bytes with identical math, thread
// mapping, and smem write order as load_tiles_nvfp4_w4a8 — bit-identical tile_x. The q8_1 y-chunks
// go into a 2-slot smem ring, each chunk's copy issued one compute phase ahead of use.
//
// Steady-state per k-iteration i (3 commit groups, uniform wait counts, 4 syncthreads = baseline):
//   wait<1>  -> X_i raw complete (G2 of iter i-1)          | sync (a)
//   dequant xr -> tile_x (smem->smem, off critical mem path)
//   G1: cp.async Y[2i+1] -> ty1                            | wait<1> -> Y[2i] (G3 of i-1) | sync(bc)
//   G2: cp.async X_{i+1} raw -> xr (single slot: free after dequant; empty group on last iter)
//   vec_dot(ty0, k00=0)                                    | sync (d)
//   G3: cp.async Y[2i+2] -> ty0 (empty group on last iter)
//   wait<2>  -> Y[2i+1] (G1) complete                      | sync (e)
//   vec_dot(ty1, k00=32)
// X_{i+1} is in flight across BOTH vec_dots (a full iteration); each Y chunk across one vec_dot.

static __device__ __forceinline__ void pipe_cp_async_16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(gsrc));
}
static __device__ __forceinline__ void pipe_cp_async_4(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(gsrc));
}
static __device__ __forceinline__ void pipe_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int n>
static __device__ __forceinline__ void pipe_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// Stage one q8_1 y-chunk (mmq_x * MMQ_TILE_Y_K ints, contiguous in global) as 16B lines. Global
// offsets are all multiples of 144B (36-int block_q8_1_mmq strides) -> 16B alignment holds.
template <int mmq_x, int nwarps>
static __device__ __forceinline__ void pipe_stage_y(int * __restrict__ dst, const int * __restrict__ src) {
    constexpr int nthreads = nwarps * MMQ_WARP_SIZE;
    constexpr int nlines   = mmq_x * MMQ_TILE_Y_K / 4; // 16B lines (TILE_Y_K=36 -> always integral)
    const int t = threadIdx.y * MMQ_WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int l0 = 0; l0 < nlines; l0 += nthreads) {
        const int L = l0 + t;
        if (nlines % nthreads != 0 && L >= nlines) { break; }
        pipe_cp_async_16(dst + 4 * L, src + 4 * L);
    }
}

// Raw x-stage buffer layout (18KB at mmq_y=128, both weight layouts):
//   is_rp=true : [mmq_y*4 blocks x 32B qs][mmq_y*4 blocks x 4B d]  (local block li = i*4 + kbx)
//   is_rp=false: [mmq_y*4 blocks x 36B block_nvfp4]                (li*36, rows contiguous 144B)
// need_check: source row clamps to i_max (same rows the plain loader reads); dest slot stays
// UNclamped so no two cp.async writes alias — dequant reads slot min(i,i_max), which holds row
// min(i,i_max)'s bytes either way.
static constexpr size_t pipe_xr_bytes(int mmq_y) { return (size_t) mmq_y * 4 * 36; }

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void pipe_stage_x_rp(
        char * __restrict__ xr, const char * __restrict__ x, const char * __restrict__ x_sc,
        const int kb0, const int i_max, const int stride) {
    constexpr int nthreads = (mmq_y / 16) * MMQ_WARP_SIZE;
    const int t = threadIdx.y * MMQ_WARP_SIZE + threadIdx.x;
    // qs plane: mmq_y rows x 128B (4 consecutive blocks x 32B, 32B-aligned) as 16B lines.
    constexpr int qlines = mmq_y * 8;
#pragma unroll
    for (int l0 = 0; l0 < qlines; l0 += nthreads) {
        const int L = l0 + t;
        const int r = L / 8, line = L % 8;
        const int row = need_check ? min(r, i_max) : r;
        pipe_cp_async_16(xr + (size_t) r * 128 + line * 16,
                         x + (size_t) (kb0 + row * stride) * 32 + line * 16);
    }
    // d plane: mmq_y rows x 16B (4 blocks x 4B). Global only 4B-aligned in general -> 4B copies.
    char * xr_d = xr + (size_t) mmq_y * 128;
    constexpr int dops = mmq_y * 4;
#pragma unroll
    for (int l0 = 0; l0 < dops; l0 += nthreads) {
        const int L = l0 + t;
        const int r = L / 4, w = L % 4;
        const int row = need_check ? min(r, i_max) : r;
        pipe_cp_async_4(xr_d + (size_t) r * 16 + w * 4,
                        x_sc + (size_t) (kb0 + row * stride) * 4 + w * 4);
    }
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void pipe_stage_x_gguf(
        char * __restrict__ xr, const char * __restrict__ x,
        const int kb0, const int i_max, const int stride) {
    constexpr int nthreads = (mmq_y / 16) * MMQ_WARP_SIZE;
    const int t = threadIdx.y * MMQ_WARP_SIZE + threadIdx.x;
    // mmq_y rows x 144B (4 consecutive 36B blocks). 36B blocks are only 4B-aligned -> 4B copies.
    constexpr int ops = mmq_y * 36;
#pragma unroll
    for (int l0 = 0; l0 < ops; l0 += nthreads) {
        const int L = l0 + t;
        const int r = L / 36, o = L % 36;
        const int row = need_check ? min(r, i_max) : r;
        pipe_cp_async_4(xr + (size_t) r * 144 + o * 4,
                        x + (size_t) (kb0 + row * stride) * 36 + o * 4);
    }
}

// Dequant the staged raw bytes -> tile_x. Token-for-token the same math, thread mapping, and smem
// write order as load_tiles_nvfp4_w4a8 — only the source pointers move from global to the xr stage.
template <int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void dequant_tiles_nvfp4_w4a8(
        const char * __restrict__ xr, int * __restrict__ x_tile, const int i_max) {
    constexpr int nwarps = mmq_y / 16;
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K * 2);

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;     // 4 blocks per row per iter
    constexpr int rows_per_warp = warp_size / threads_per_row; // 8
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;
        if constexpr (need_check) { i = min(i, i_max); }

        const int li = i * threads_per_row + kbx; // staged local block index
        const uint32_t * __restrict__ src_qs;
        const uint8_t  * __restrict__ src_d;
        if constexpr (is_rp) {
            src_qs = reinterpret_cast<const uint32_t *>(xr + (size_t) li * 32);
            src_d  = reinterpret_cast<const uint8_t  *>(xr + (size_t) mmq_y * 128 + (size_t) li * 4);
        } else {
            const block_nvfp4 * bxi = (const block_nvfp4 *) xr + li;
            src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
            src_d  = bxi->d;
        }
        const int kqs = 16 * kbx;
        const int ksc = 4 * kbx;

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], kvalues_mxfp4);
            const int2 q1 = get_int_from_table_16(src_qs[2 * sub + 1], kvalues_mxfp4);
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 0] = q0.x;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 1] = q1.x;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 2] = q0.y;
            x_qs[i * MMQ_MMA_TILE_X_K_NVFP4 + kqs + 4 * sub + 3] = q1.y;
            x_df[i * MMQ_MMA_TILE_X_K_NVFP4 + ksc + sub] = ggml_cuda_ue4m3_to_fp32(src_d[sub]);
        }
    }
}

// ======================= vec_dot_q8_0_16_q8_1_mma (mmq.cuh:1495, TURING arm) =======================
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_nvfp4_w4a8_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K * 2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx * tile_A::I);

    tile_A A[ntx][8];
    float  dA[ntx][tile_C::ne / 2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;
            load_ldmatrix(((tile_A_8 *) A[n])[k01 / 8], x_qs + (i0 + n * tile_A::I) * MMQ_MMA_TILE_X_K_NVFP4 + k0, MMQ_MMA_TILE_X_K_NVFP4);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne / 2; ++l) {
            const int i = i0 + n * tile_C::I + tile_C::get_i(2 * l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;
                dA[n][l][k01 / 4] = x_df[i * MMQ_MMA_TILE_X_K_NVFP4 + k0 / 4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) { // QR3_K*VDR_Q3_K_Q8_1_MMQ == 4*2 == 8
            tile_B B[2];
            float dB[tile_C::ne / 2];

            // load_generic faster than load_ldmatrix here (llama comment).
            load_generic(B[0], y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0),           MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0 * MMQ_TILE_Y_K + (k01 + tile_B::J),   MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne / 2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01 / 4 + 0], B[0]);
                mma(C[1], A[n][k01 / 4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                        dB[l % 2] * (C[0].x[l] * dA[n][l / 2][k01 / 4 + 0] + C[1].x[l] * dA[n][l / 2][k01 / 4 + 1]);
                }
            }
        }
    }
}

// ======================= mmq_write_back_mma (mmq.cuh:3214) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_nvfp4_w4a8(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max, const float out_scale) {
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_y / 16;
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
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l] * out_scale;
            }
        }
    }
}

// ======================= mul_mat_q_process_tile (NVFP4 W4A8) =======================
template <int mmq_x, int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4_w4a8(
        const char * __restrict__ x, const char * __restrict__ x_sc, const int offset_x,
        const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = mmq_y / 16;
    constexpr int qk        = QK_NVFP4;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block        = 4 * QK8_1;                  // 128 values per block_q8_1_mmq
    constexpr int ITER_K          = MMQ_ITER_K;                 // 256
    constexpr int blocks_per_iter = ITER_K / qk;                // 4 NVFP4 blocks per iteration

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_nvfp4_w4a8<mmq_y, need_check, is_rp>(x, x_sc, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
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
        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_nvfp4_w4a8<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q_process_tile PIPELINED (BW24_PP_PIPE=1) =======================
// Same math as mul_mat_q_process_tile_nvfp4_w4a8 (identical dequant/vec_dot/write-back and FP add
// order); only the global->smem movement is restructured onto cp.async (schedule in the pipeline
// header comment above). smem: ids | ty0 | ty1 | tile_x | xr = 98816B at 128x128 (<= 99KB opt-in).
template <int mmq_x, int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4_w4a8_pipe(
        const char * __restrict__ x, const char * __restrict__ x_sc, const int offset_x,
        const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = mmq_y / 16;
    constexpr int qk        = QK_NVFP4;

    extern __shared__ int data_mul_mat_q[];
    constexpr int y_chunk_ints = GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);
    int  * tile_y0 = data_mul_mat_q + mmq_x;
    int  * tile_y1 = tile_y0 + y_chunk_ints;
    int  * tile_x  = tile_y1 + y_chunk_ints;
    char * xr      = (char *) (tile_x + mmq_y * MMQ_MMA_TILE_X_K_NVFP4);

    constexpr int ne_block        = 4 * QK8_1;                  // 128 values per block_q8_1_mmq
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;            // 4 NVFP4 blocks per iteration

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)
    // y chunk k: global base y + ncols_y*k*sz (same address math as the plain path's by0).
    // Per iteration kb0: chunks kb0*qk/ne_block and +1.

    if (kb0_start >= kb0_stop) { return; }

    // Prologue: X_0 raw (waited as "G2 of iter -1"), then Y chunk 0 (waited as "G3 of iter -1").
    if constexpr (is_rp) {
        pipe_stage_x_rp<mmq_y, need_check>(xr, x, x_sc, offset_x + kb0_start, tile_x_max_i, stride_row_x);
    } else {
        pipe_stage_x_gguf<mmq_y, need_check>(xr, x, offset_x + kb0_start, tile_x_max_i, stride_row_x);
    }
    pipe_commit();
    pipe_stage_y<mmq_x, nwarps>(tile_y0, y + ncols_y * (kb0_start * qk / ne_block) * sz);
    pipe_commit();

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int  kchunk   = kb0 * qk / ne_block;
        const bool has_next = kb0 + blocks_per_iter < kb0_stop;

        pipe_wait<1>();     // X_kb0 raw complete (all but {last y-chunk group} done)
        __syncthreads();    // (a) publish X raw; prev iter's reads of tile_x/ty1 are done

        dequant_tiles_nvfp4_w4a8<mmq_y, need_check, is_rp>(xr, tile_x, tile_x_max_i);

        pipe_stage_y<mmq_x, nwarps>(tile_y1, y + ncols_y * (kchunk + 1) * sz);
        pipe_commit();      // G1 = Y[2i+1]
        pipe_wait<1>();     // Y[2i] complete (G1 stays in flight)
        __syncthreads();    // (bc) publish dequant + Y[2i]; xr slot now free

        if (has_next) {     // G2 = X_{i+1} raw (empty group on last iter keeps wait counts uniform)
            if constexpr (is_rp) {
                pipe_stage_x_rp<mmq_y, need_check>(xr, x, x_sc, offset_x + kb0 + blocks_per_iter, tile_x_max_i, stride_row_x);
            } else {
                pipe_stage_x_gguf<mmq_y, need_check>(xr, x, offset_x + kb0 + blocks_per_iter, tile_x_max_i, stride_row_x);
            }
        }
        pipe_commit();

        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y0, sum, 0);
        __syncthreads();    // (d) ty0 free

        if (has_next) {     // G3 = Y[2i+2] (empty group on last iter)
            pipe_stage_y<mmq_x, nwarps>(tile_y0, y + ncols_y * (kchunk + 2) * sz);
        }
        pipe_commit();

        pipe_wait<2>();     // Y[2i+1] complete (G2/G3 stay in flight)
        __syncthreads();    // (e) publish Y[2i+1]

        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y1, sum, MMQ_TILE_NE_K);
    }
    pipe_wait<0>();         // drain (final G2/G3 are empty)

    mmq_write_back_nvfp4_w4a8<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q_process_tile PIPE2 (BW24_PP_PIPE=2, 2 CTA/SM) =======================
// Occupancy variant of the pipe above for the mmq_y=64 2-CTA/SM mode: SINGLE y slot (no ring) so
// smem fits two CTAs; the second resident CTA covers what the y-ring used to hide. Same dequant
// bytes/math/order and the same vec_dot(y[2i], k00=0) -> vec_dot(y[2i+1], k00=32) sequence as both
// other paths — bit-identical output per (token,row).
//
// Steady-state per k-iteration i (uniform positional waits; X_{i+1} always committed LAST so
// wait<1> at the y[2i+1] point leaves only it in flight; 4 syncthreads = baseline):
//   wait<0> -> X_i raw complete (it is the only pending group)    | sync (a)
//   G1: cp.async Y[2i] -> ty (flies under the dequant ALU work)
//   dequant xr -> tile_x (smem->smem)
//   wait<0> -> Y[2i] complete                                     | sync (b)  [xr free]
//   vec_dot(ty, k00=0)                                            | sync (c)  [ty free]
//   G2: cp.async Y[2i+1] -> ty
//   G3: cp.async X_{i+1} raw -> xr (empty group on last iter)
//   wait<1> -> Y[2i+1] complete (G3 stays in flight)              | sync (d)
//   vec_dot(ty, k00=32)
template <int mmq_x, int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4_w4a8_pipe2(
        const char * __restrict__ x, const char * __restrict__ x_sc, const int offset_x,
        const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = mmq_y / 16;
    constexpr int qk        = QK_NVFP4;

    extern __shared__ int data_mul_mat_q[];
    constexpr int y_chunk_ints = GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);
    int  * tile_y = data_mul_mat_q + mmq_x;
    int  * tile_x = tile_y + y_chunk_ints;
    char * xr     = (char *) (tile_x + mmq_y * MMQ_MMA_TILE_X_K_NVFP4);

    constexpr int ne_block        = 4 * QK8_1;                  // 128 values per block_q8_1_mmq
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;            // 4 NVFP4 blocks per iteration

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)

    if (kb0_start >= kb0_stop) { return; }

    // Prologue: X_0 raw only (y[0] is staged inside the first iteration, under the dequant).
    if constexpr (is_rp) {
        pipe_stage_x_rp<mmq_y, need_check>(xr, x, x_sc, offset_x + kb0_start, tile_x_max_i, stride_row_x);
    } else {
        pipe_stage_x_gguf<mmq_y, need_check>(xr, x, offset_x + kb0_start, tile_x_max_i, stride_row_x);
    }
    pipe_commit();

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int  kchunk   = kb0 * qk / ne_block;
        const bool has_next = kb0 + blocks_per_iter < kb0_stop;

        pipe_wait<0>();     // X_kb0 raw complete (only pending group)
        __syncthreads();    // (a) publish X raw; prev iter's reads of tile_x/ty are done

        pipe_stage_y<mmq_x, nwarps>(tile_y, y + ncols_y * (kchunk + 0) * sz);
        pipe_commit();      // G1 = Y[2i], in flight under the dequant

        dequant_tiles_nvfp4_w4a8<mmq_y, need_check, is_rp>(xr, tile_x, tile_x_max_i);

        pipe_wait<0>();     // Y[2i] complete
        __syncthreads();    // (b) publish dequant + Y[2i]; xr slot now free

        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
        __syncthreads();    // (c) ty free

        pipe_stage_y<mmq_x, nwarps>(tile_y, y + ncols_y * (kchunk + 1) * sz);
        pipe_commit();      // G2 = Y[2i+1]
        if (has_next) {     // G3 = X_{i+1} raw — committed LAST so it may stay in flight
            if constexpr (is_rp) {
                pipe_stage_x_rp<mmq_y, need_check>(xr, x, x_sc, offset_x + kb0 + blocks_per_iter, tile_x_max_i, stride_row_x);
            } else {
                pipe_stage_x_gguf<mmq_y, need_check>(xr, x, offset_x + kb0 + blocks_per_iter, tile_x_max_i, stride_row_x);
            }
        }
        pipe_commit();

        pipe_wait<1>();     // Y[2i+1] complete (G3 = X_{i+1} stays in flight)
        __syncthreads();    // (d) publish Y[2i+1]

        vec_dot_nvfp4_w4a8_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    }
    pipe_wait<0>();         // drain (final G3 is empty)

    mmq_write_back_nvfp4_w4a8<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q (conventional xy-tiling, NVFP4 W4A8) =======================
// pmode: 0 = plain tile loads, 1 = cp.async pipe (y ring + xr), 2 = 2-CTA slim pipe (single y slot).
template <int mmq_x, int mmq_y, int pmode, bool need_check, bool is_rp>
__launch_bounds__(MMQ_WARP_SIZE * (mmq_y / 16), 1)
static __global__ void mul_mat_q_nvfp4_w4a8(
        const char * __restrict__ x, const char * __restrict__ x_sc,
        const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00, const float out_scale) {
    constexpr int nwarps = mmq_y / 16;
    constexpr int warp_size = MMQ_WARP_SIZE;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) { break; }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int jt = blockIdx.y; // n-token tile
    const int it = blockIdx.x; // out-row tile

    const int col_diff = ncols_dst;
    const int offset_y   = (jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
    const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;

    const int tile_x_max_i = nrows_x  - it * mmq_y - 1;
    const int tile_y_max_j = col_diff - jt * mmq_x - 1;

    const int offset_x = it * mmq_y * stride_row_x;

    if constexpr (pmode == 2) {
        mul_mat_q_process_tile_nvfp4_w4a8_pipe2<mmq_x, mmq_y, need_check, is_rp>(
            x, x_sc, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
            stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
    } else if constexpr (pmode == 1) {
        mul_mat_q_process_tile_nvfp4_w4a8_pipe<mmq_x, mmq_y, need_check, is_rp>(
            x, x_sc, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
            stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
    } else {
        mul_mat_q_process_tile_nvfp4_w4a8<mmq_x, mmq_y, need_check, is_rp>(
            x, x_sc, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
            stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
    }
}

// ======================= activation quantizer (quantize.cu:276, D4 layout) =======================
// f32 -> block_q8_1_mmq with a symmetric FLOAT scale d per 32 values (NO sum term). NVFP4 is
// symmetric so D4 (not DS4) — this is the SAME activation quant class as bw24's default int8 GEMM.
static __global__ void quantize_mmq_q8_1_d4_kernel(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) { return; }

    const int64_t i1 = blockIdx.x;
    const int64_t i00 = i0;
    const int64_t i01 = i1;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib  = (i0 / (4 * QK8_1)) * ne1 + blockIdx.x; // block index (k-major, then column)
    const int64_t iqs = i0 % (4 * QK8_1);                      // quant index in block

    const float4 xi = i0 < ne00 ? x4[(i01 * s01 + i00) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between 8 threads (vals_per_scale/4 == 32/4 == 8).
#pragma unroll
    for (int offset = 32 / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs / 4] = q;

    if (iqs % 32 != 0) { return; }

    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;
    y[ib].d4[iqs / 32] = d;
}

// ======================= C-ABI host launcher =======================
extern "C" {

// Bytes needed for the block_q8_1_mmq activation scratch for (in_f, n_tokens).
size_t bw24_mmq_nvfp4_w4a8_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    return (size_t) nblocks * sizeof(block_q8_1_mmq);
}

static size_t mmq_nvfp4_w4a8_nbytes_shared(int pmode, int mmq_x, int mmq_y) {
    const size_t nbs_ids = (size_t) mmq_x * sizeof(int);
    const size_t nbs_x   = (size_t) mmq_y * MMQ_MMA_TILE_X_K_NVFP4 * sizeof(int);
    const size_t nbs_y   = (size_t) mmq_x * sizeof(block_q8_1_mmq);
    const size_t pad     = (size_t) (mmq_y / 16) * MMQ_WARP_SIZE * sizeof(int);
    if (pmode == 2) { // ids | ty x1 | tile_x | raw x stage — 49664B at 128x64 (2 CTA/SM)
        return nbs_ids + GGML_PAD(nbs_y, pad) + nbs_x + pipe_xr_bytes(mmq_y);
    }
    if (pmode == 1) { // ids | ty ring x2 | tile_x | raw x stage — 98816B at 128x128
        return nbs_ids + 2 * GGML_PAD(nbs_y, pad) + nbs_x + pipe_xr_bytes(mmq_y);
    }
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

// BW24_PP_PIPE: 1 (default) = cp.async pipeline, DEFAULT ON since 2026-07-09; bit-identical
// (0/6.3M mismatches at T=512), measured pp1845 27B +4.7% / 9B +5.6% on the rig, 1.135x kernel.
// 0 = plain tile loads. 2 (w4a8v2 experiment, default OFF) = 2-CTA/SM occupancy mode: mmq_y=64
// slim pipe (single y slot), attacks the ncu-measured warp starvation (2 warps/scheduler, 48%
// no-eligible, math_pipe_throttle-dominated) without duplicating dequant work. Bit-identical.
static int mmq_w4a8_pipe_mode() {
    static const int mode = [] {
        const char * v = std::getenv("BW24_PP_PIPE");
        if (v == nullptr) { return 1; }
        if (v[0] == '0') { return 0; }
        if (v[0] == '2') { return 2; }
        return 1;
    }();
    return mode;
}

// mode-2 row tile: 64 rows / 4 warps — the 2-CTA/SM geometry (48.5KB smem/CTA).
#define MMQ_Y_P2 64

// Run the NVFP4 W4A8 MMQ prefill GEMM. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
//   W_nvfp4_blocks : NVFP4 weight bytes. rp=0: GGUF block_nvfp4 36B blocks, in_f/64 per row.
//                    rp=1: A6 split-plane repack ([quant plane out_f x in_f/64 x 32B]
//                    [scale plane out_f x in_f/64 x 4B]) — the resident decode layout; the rp tile
//                    loader is a pure address remap of the GGUF loader (bit-identical output).
//   act_f32        : f32 activation [n_tokens, in_f].
//   y              : f32 output [n_tokens, out_f].
//   act_scratch    : pre-alloc'd quant buffer >= bw24_mmq_nvfp4_w4a8_act_bytes(in_f, n_tokens).
//   out_scale      : per-tensor NVFP4 macro-scale (folded into write-back). 1.0 for unscaled.
// Requires in_f % 64 == 0. Returns 0 on success, else (1000 + cudaError).
int bw24_mmq_nvfp4_w4a8(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                        int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                        float out_scale, int rp) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    // ---- 1) quantize activation f32 -> block_q8_1_mmq (D4) ----
    const int64_t ne10 = in_f;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    {
        const int64_t block_num_y = (ne10_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                                    (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks((unsigned) n_tokens, (unsigned) block_num_y, 1);
        quantize_mmq_q8_1_d4_kernel<<<num_blocks, block_size, 0, st>>>(
            act_f32, act_scratch, ne10, /*s01*/ in_f, ne10_padded, n_tokens);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }

    // ---- 2) launch mul_mat_q NVFP4 W4A8 (conventional xy-tiling) ----
    const int stride_row_x    = in_f / QK_NVFP4;          // block_nvfp4 per weight row
    const int blocks_per_ne00 = in_f / QK_NVFP4;
    const int stride_col_dst  = out_f;
    const int ncols_y         = n_tokens;

    const int pmode = mmq_w4a8_pipe_mode();
    const int mmq_y_run = pmode == 2 ? MMQ_Y_P2 : MMQ_Y;

    const int nty = (out_f    + mmq_y_run - 1) / mmq_y_run;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, mmq_y_run / 16, 1);
    const size_t smem = mmq_nvfp4_w4a8_nbytes_shared(pmode, MMQ_X, mmq_y_run);

    const bool need_check = (out_f % mmq_y_run) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_nvfp4_blocks;
    // rp: scale plane sits after the quant plane (out_f rows x in_f/64 groups x 32B). Unused for rp=0.
    const char * W_sc = W + (size_t) out_f * (in_f / QK_NVFP4) * 32;

    #define BW24_W4A8_LAUNCH(Y, PM, NC, RP) do {                                                         \
        cudaFuncSetAttribute(mul_mat_q_nvfp4_w4a8<MMQ_X, Y, PM, NC, RP>,                                 \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                         \
        mul_mat_q_nvfp4_w4a8<MMQ_X, Y, PM, NC, RP><<<grid, block, smem, st>>>(                           \
            W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00,    \
            out_scale);                                                                                  \
    } while (0)
    #define BW24_W4A8_LAUNCH4(Y, PM) do {                                                                \
        if (rp) {                                                                                        \
            if (need_check) { BW24_W4A8_LAUNCH(Y, PM, true,  true);  }                                   \
            else            { BW24_W4A8_LAUNCH(Y, PM, false, true);  }                                   \
        } else {                                                                                         \
            if (need_check) { BW24_W4A8_LAUNCH(Y, PM, true,  false); }                                   \
            else            { BW24_W4A8_LAUNCH(Y, PM, false, false); }                                   \
        }                                                                                                \
    } while (0)

    if      (pmode == 2) { BW24_W4A8_LAUNCH4(MMQ_Y_P2, 2); }
    else if (pmode == 1) { BW24_W4A8_LAUNCH4(MMQ_Y,    1); }
    else                 { BW24_W4A8_LAUNCH4(MMQ_Y,    0); }
    #undef BW24_W4A8_LAUNCH4
    #undef BW24_W4A8_LAUNCH
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

} // extern "C"

// ======================================================================================
// W4A8-FP8 (R-B route, research/prefill-mxf8f6f4-design.md): e4m3 weights x e4m3 acts on
// the 381-TF kind::f8f6f4 MMA — NVFP4 per-16 scales FOLD into the weight VALUES at tile
// load (per-sub 16-byte e4m3 LUT built with cvt.e4m3x2, then the SAME byte-perm gather as
// the int8 path). No x scale plane; epilogue = activation d4 only. Plain path only (pipe
// variants after the A/B proves the win). Seam: BW24_MMQ_F8F4=1 (mmq_ffi.rs dispatch).
// ======================================================================================
#define MMQ_MMA_TILE_X_K_F8F4 (2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 4 + 4)   // 76: 64 value-ints + 8 per-32 f32 scales + pad

// f32x2 -> packed e4m3x2 (round-to-nearest, saturate) — same op as the activation quantizer.
static __device__ __forceinline__ uint16_t bw24_cvt_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\n\tmov.b16 %0, t;\n}"
        : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

// e2m1 x2 value table (the kvalues_mxfp4 convention: values doubled to integers, compensated
// by ue4m3_to_fp32's /2 — both loaders must agree on this pairing; the raw-grid variant paired
// with the halved scale cost an exact x0.5 on every element, caught by synth_f8f4).
static __constant__ float bw24_e2m1_f32[16] =
    {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
     -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};

// plain-kind f8f6f4 MMA: D(f32 16x8) += A(e4m3 16x32) * B(e4m3 32x8). A = 4 regs (one
// tile_A_8 ldmatrix load), B = the two k16 sub-fragments the int8 path already loads.
static __device__ __forceinline__ void bw24_mma_f8f4(
        float * __restrict__ c, const int * __restrict__ a, const int b0, const int b1) {
    asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Loader: same addressing as load_tiles_nvfp4_w4a8. R-A' fold: per PAIR of 16-value
// sub-blocks, s32 = max(s_a, s_b) goes to the x_df scale plane (per-32, matching the ONE
// k32 MMA the vec_dot issues) and each sub's LUT folds only the RATIO s_sub/s32 in (0,1]
// into the e4m3 values — range stays 0.25..6 (no e4m3 subnormal underflow; the naive full
// fold s*v underflowed whole small-amax blocks below 2^-9 and failed f8f4-check at ~0.5 rel).
// Ratio==1 subs (8.5% measured) re-code exactly.
template <int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void load_tiles_nvfp4_f8f4(
        const char * __restrict__ x, const char * __restrict__ x_sc, int * __restrict__ x_tile,
        const int kb0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_y / 16;
    constexpr int warp_size = MMQ_WARP_SIZE;
    int * x_qs = x_tile;

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;      // 4
    constexpr int rows_per_warp = warp_size / threads_per_row;  // 8
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;
        if constexpr (need_check) { i = min(i, i_max); }

        const int ib = kb0 + i * stride + kbx;
        const uint32_t * __restrict__ src_qs;
        const uint8_t  * __restrict__ src_d;
        if constexpr (is_rp) {
            src_qs = reinterpret_cast<const uint32_t *>(x    + (size_t) ib * 32);
            src_d  = reinterpret_cast<const uint8_t  *>(x_sc + (size_t) ib * 4);
        } else {
            const block_nvfp4 * bxi = (const block_nvfp4 *) x + ib;
            src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
            src_d  = bxi->d;
        }
        const int kqs = 16 * kbx;
        float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const float sa = ggml_cuda_ue4m3_to_fp32(src_d[2 * pair + 0]);
            const float sb = ggml_cuda_ue4m3_to_fp32(src_d[2 * pair + 1]);
            const float s32 = fmaxf(fmaxf(sa, sb), 1e-30f);
            x_df[i * MMQ_MMA_TILE_X_K_F8F4 + 2 * kbx + pair] = s32;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int sub = 2 * pair + half;
                const float r = (half == 0 ? sa : sb) / s32;   // in (0, 1]
                int lut[4];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint16_t p0 = bw24_cvt_e4m3x2(bw24_e2m1_f32[4*w+0] * r, bw24_e2m1_f32[4*w+1] * r);
                    const uint16_t p1 = bw24_cvt_e4m3x2(bw24_e2m1_f32[4*w+2] * r, bw24_e2m1_f32[4*w+3] * r);
                    lut[w] = (int) ((uint32_t) p0 | ((uint32_t) p1 << 16));
                }
                const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], (const int8_t *) lut);
                const int2 q1 = get_int_from_table_16(src_qs[2 * sub + 1], (const int8_t *) lut);
                x_qs[i * MMQ_MMA_TILE_X_K_F8F4 + kqs + 4 * sub + 0] = q0.x;
                x_qs[i * MMQ_MMA_TILE_X_K_F8F4 + kqs + 4 * sub + 1] = q1.x;
                x_qs[i * MMQ_MMA_TILE_X_K_F8F4 + kqs + 4 * sub + 2] = q0.y;
                x_qs[i * MMQ_MMA_TILE_X_K_F8F4 + kqs + 4 * sub + 3] = q1.y;
            }
        }
    }
}

// vec_dot: ONE k32 f8f6f4 MMA per 8-int step (the int8 path issues two k16 imma there),
// f32 accumulators direct, epilogue = dB only (weight scales folded into the values).
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_nvfp4_f8f4_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx * tile_A_8::I);

    tile_A_8 A[ntx][4];
    float dA[16 / 4][2][4];   // [ntx<=4][row-half l/2][k32-step] per-32 weight scales
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01 / 8], x_qs + (i0 + n * tile_A_8::I) * MMQ_MMA_TILE_X_K_F8F4 + k0,
                          MMQ_MMA_TILE_X_K_F8F4);
        }
#pragma unroll
        for (int l = 0; l < tile_C::ne / 2; ++l) {
            const int i = i0 + n * tile_C::I + tile_C::get_i(2 * l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
                dA[n][l][k01 / 8] = x_df[i * MMQ_MMA_TILE_X_K_F8F4 + (k00 + k01) / 8];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne / 2];
            load_generic(B[0], y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0 * MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < tile_C::ne / 2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                bw24_mma_f8f4(C, A[n][k01 / 8].x, B[0].x[0], B[1].x[0]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dA[n][l / 2][k01 / 8] * dB[l % 2] * C[l];
                }
            }
        }
    }
}

// process_tile (plain): identical skeleton to mul_mat_q_process_tile_nvfp4_w4a8.
template <int mmq_x, int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4_f8f4(
        const char * __restrict__ x, const char * __restrict__ x_sc, const int offset_x,
        const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = mmq_y / 16;
    constexpr int qk        = QK_NVFP4;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block        = 4 * QK8_1;
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_nvfp4_f8f4<mmq_y, need_check, is_rp>(x, x_sc, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_nvfp4_f8f4_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
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
        vec_dot_nvfp4_f8f4_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_nvfp4_w4a8<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

template <int mmq_x, int mmq_y, bool need_check, bool is_rp>
static __global__ void mul_mat_q_nvfp4_f8f4(
        const char * __restrict__ x, const char * __restrict__ x_sc,
        const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00, const float out_scale) {
    constexpr int nwarps = mmq_y / 16;
    constexpr int warp_size = MMQ_WARP_SIZE;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) { break; }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int jt = blockIdx.y;
    const int it = blockIdx.x;

    const int offset_y   = (jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
    const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;
    const int tile_x_max_i = nrows_x   - it * mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;
    const int offset_x = it * mmq_y * stride_row_x;

    mul_mat_q_process_tile_nvfp4_f8f4<mmq_x, mmq_y, need_check, is_rp>(
        x, x_sc, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
        stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
}

extern "C" {
// activation quantizer lives in mmq_nvfp4_f8f4.cu (block_e4m3_mmq is footprint-identical to
// block_q8_1_mmq, so all smem/stride math here reuses the q8_1 constants).
void bw24_mmq_nvfp4_f8f4_quantize_act(const float * x, void * vy, int in_f, int n_tokens,
                                      int64_t s01, cudaStream_t st);

int bw24_mmq_nvfp4_f8f4(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                        int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                        float out_scale, int rp) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    bw24_mmq_nvfp4_f8f4_quantize_act(act_f32, act_scratch, in_f, n_tokens, in_f, st);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return 1000 + (int) e; } }

    const int stride_row_x    = in_f / QK_NVFP4;
    const int blocks_per_ne00 = in_f / QK_NVFP4;
    const int stride_col_dst  = out_f;
    const int ncols_y         = n_tokens;

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, MMQ_Y / 16, 1);
    const size_t smem = (size_t) MMQ_X * sizeof(int)
                      + (size_t) GGML_PAD(MMQ_X * MMQ_TILE_Y_K, (MMQ_Y / 16) * MMQ_WARP_SIZE) * sizeof(int)
                      + (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_F8F4 * sizeof(int);

    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_nvfp4_blocks;
    const char * W_sc = W + (size_t) out_f * (in_f / QK_NVFP4) * 32;

    #define BW24_F8F4_LAUNCH(NC, RP) do {                                                                \
        cudaFuncSetAttribute(mul_mat_q_nvfp4_f8f4<MMQ_X, MMQ_Y, NC, RP>,                                 \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                         \
        mul_mat_q_nvfp4_f8f4<MMQ_X, MMQ_Y, NC, RP><<<grid, block, smem, st>>>(                           \
            W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00,    \
            out_scale);                                                                                  \
    } while (0)
    if (rp) { if (need_check) { BW24_F8F4_LAUNCH(true, true);  } else { BW24_F8F4_LAUNCH(false, true);  } }
    else    { if (need_check) { BW24_F8F4_LAUNCH(true, false); } else { BW24_F8F4_LAUNCH(false, false); } }
    #undef BW24_F8F4_LAUNCH

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 2000 + (int) e; }
    return 0;
}
} // extern "C"
