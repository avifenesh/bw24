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
    constexpr int nwarps = MMQ_NWARPS;
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
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l] * out_scale;
            }
        }
    }
}

// ======================= mul_mat_q_process_tile (NVFP4 W4A8) =======================
template <int mmq_x, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4_w4a8(
        const char * __restrict__ x, const char * __restrict__ x_sc, const int offset_x,
        const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const float out_scale) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = MMQ_NWARPS;
    constexpr int qk        = QK_NVFP4;
    constexpr int mmq_y     = MMQ_Y;

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

// ======================= mul_mat_q (conventional xy-tiling, NVFP4 W4A8) =======================
template <int mmq_x, bool need_check, bool is_rp>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_nvfp4_w4a8(
        const char * __restrict__ x, const char * __restrict__ x_sc,
        const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00, const float out_scale) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int mmq_y = MMQ_Y;

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

    mul_mat_q_process_tile_nvfp4_w4a8<mmq_x, need_check, is_rp>(
        x, x_sc, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
        stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
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

static size_t mmq_nvfp4_w4a8_nbytes_shared() {
    const size_t nbs_ids = (size_t) MMQ_X * sizeof(int);
    const size_t nbs_x   = (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_NVFP4 * sizeof(int);
    const size_t nbs_y   = (size_t) MMQ_X * sizeof(block_q8_1_mmq);
    const size_t pad     = (size_t) MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

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

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    const size_t smem = mmq_nvfp4_w4a8_nbytes_shared();

    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_nvfp4_blocks;
    // rp: scale plane sits after the quant plane (out_f rows x in_f/64 groups x 32B). Unused for rp=0.
    const char * W_sc = W + (size_t) out_f * (in_f / QK_NVFP4) * 32;

    if (rp) {
        if (need_check) {
            cudaFuncSetAttribute(mul_mat_q_nvfp4_w4a8<MMQ_X, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            mul_mat_q_nvfp4_w4a8<MMQ_X, true, true><<<grid, block, smem, st>>>(
                W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
        } else {
            cudaFuncSetAttribute(mul_mat_q_nvfp4_w4a8<MMQ_X, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            mul_mat_q_nvfp4_w4a8<MMQ_X, false, true><<<grid, block, smem, st>>>(
                W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
        }
    } else {
        if (need_check) {
            cudaFuncSetAttribute(mul_mat_q_nvfp4_w4a8<MMQ_X, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            mul_mat_q_nvfp4_w4a8<MMQ_X, true, false><<<grid, block, smem, st>>>(
                W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
        } else {
            cudaFuncSetAttribute(mul_mat_q_nvfp4_w4a8<MMQ_X, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            mul_mat_q_nvfp4_w4a8<MMQ_X, false, false><<<grid, block, smem, st>>>(
                W, W_sc, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

} // extern "C"
