// mmq_q4_0.cu — Q4_0 int8-MMA MMQ prefill GEMM (vendored floor, ggml-decoupled, sm_75+ portable).
//
// gemma-4-12B lane: the QAT ggufs are q4_0 end to end, and the hand-rolled tiling GEMM
// `qmatvec_gemm_q4_0_rp` measures 77% of the 12B prime pass (~1.04s at ~40 TFLOPS) — the single
// biggest prefill lever. This file vendors llama's mul_mat_q<Q4_0> the same way mmq_q8_0.cu
// vendored the Q8_0 tile. Source: /home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/
//   - mmq.cuh      : load_tiles_q4_0 (TURING_MMA branch: packed nibbles -> int8 at tile load via
//                    __vsubss4((qs >> {0,4}) & 0x0F0F0F0F, 0x08080808) — the -8 offset folded into
//                    the quants so the D4 epilogue needs no min/sum term), then the SAME
//                    vec_dot_q8_0_q8_1_mma / write_back / process_tile as Q8_0 (GGML_TYPE_Q4_0 maps
//                    to MMQ_Q8_1_DS_LAYOUT_D4, mmq.cuh:64).
//   - quantize.cu  : quantize_mmq_q8_1<D4> activation (symmetric float scale per 32, no sum term).
//
// DECOUPLING: no ggml headers; all functions static/internal (same treatment as the sibling MMQ
// TUs, no link collisions).
//
// KEY DIFFS vs mmq_q8_0.cu (the direct template — both are D4/symmetric):
//   - Weight block is 18B (fp16 d + 16B of 32 packed nibbles), not 34B. QI4_0 = 4 ints of packed
//     qs per block; each loaded int expands to TWO x-tile ints (low nibbles then high nibbles),
//     so one warp pass still fills 64 qs ints = 256 values = 8 blocks per ITER_K.
//   - Nibble order: byte j of qs holds value j (low nibble) and value j+16 (high nibble), so the
//     low-int lands at kbx*(2*QI4_0)+kqsx and the high-int at +QI4_0 — natural v0..v31 order in
//     the x-tile, matching the activation's per-32 D4 blocking exactly.
//   - is_rp arm: bw24's BW24_Q4RP split-plane repack (qs plane 16B/block contiguous from base,
//     fp16 d plane dense at base + out_f*nblk*16). Pure address remap of the raw loader — same
//     dequant math, same FP op order, bit-identical output either way.
//
// EXACTNESS: (q-8) is exact in int8; s32 mma accumulate is exact; only the final f32
// (d_w * d_act * s32) reduction ORDER differs from qmatvec_gemm_q4_0's tiling reduction -> NOT
// bit-identical to the hand-rolled GEMM, gated as its own numeric config behind BW24_PP_Q4MMQ
// with the full exactness battery (same discipline as the Q8_0 / k-quant / W4A8 MMQ arms).
//
// C-ABI: bw24_mmq_q4_0 (+ bw24_mmq_q4_0_act_bytes). Compiled into libbw24_mmq.a, called via FFI.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

// ======================= ggml constants/macros (vendored) =======================
#define WARP_SIZE 32
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

#define QK4_0 32
#define QI4_0 4                  // QK4_0 / (4 * QR4_0), QR4_0 == 2
#define QI8_0 8
#define QK8_1 32
#define QI8_1 8
#define MATRIX_ROW_PADDING 512

// MMQ tile constants (mmq.cuh) — q4_0 shares the Q8_0 x-tile layout (int8 quants + float scales).
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K    256
// x-tile stride: 2*MMQ_TILE_NE_K int8-as-int quants + 2*MMQ_TILE_NE_K/QI8_0 float scales + 4 pad.
#define MMQ_MMA_TILE_X_K_Q8_0 (2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4)  // 76
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)                        // 36

// launch constants (same 128x128 / 8-warp tile as the sibling vendor kernels).
#define MMQ_WARP_SIZE 32
#define MMQ_NWARPS    8
#define MMQ_Y         128
#ifndef MMQ_X
#define MMQ_X         128
#endif

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// get_int_b2 (common.cuh): read an int from a >=2-byte-aligned buffer (raw q4_0 qs starts at +2
// inside an 18B block — only 2B alignment is guaranteed).
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] <<  0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

// ======================= weight / activation block structs =======================
// block_q4_0 (ggml-common.h): 18 bytes = fp16 block scale + 32 packed 4-bit quants.
typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "wrong q4_0 block size/padding");

// block_q8_1_mmq (mmq.cuh): D4 layout — 4x float scale (no sum term) + 128 int8 quants.
struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4 * QK8_1];           // 128 values
};
static_assert(sizeof(block_q8_1_mmq) == 4 * MMQ_TILE_Y_K, "block_q8_1_mmq != MMQ_TILE_Y_K ints");

// ======================= mma.cuh: tile<>, loads, int8 mma =======================
namespace ggml_cuda_mma {

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

    // int8 MMA (mma.cuh, Ampere+ path): D(s32) += A(s8) * B(s8).
    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
    }
} // namespace ggml_cuda_mma

using namespace ggml_cuda_mma;

// Turing+ granularity (mmq_get_granularity_device): mmq_x>=48 -> 16.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// ======================= load_tiles_q4_0 (mmq.cuh, TURING_MMA branch) =======================
// Packed nibbles -> int8 at tile load: low nibbles at kbx*(2*QI4_0)+kqsx, high nibbles at +QI4_0
// (natural v0..v31 order — byte j holds value j low / value j+16 high). The -8 zero-point is
// folded here via __vsubss4, so the D4 epilogue is plain C*dA*dB. x_df: per-32-block float scale.
// One call loads mmq_y rows x (2*MMQ_TILE_NE_K ints = 256 int8 = 8 q4_0 blocks).
//
// is_rp selects bw24's BW24_Q4RP split-plane layout: qs plane (16B/block, contiguous, 4B-aligned)
// at x, fp16 d plane (dense) at x_d. Raw ggml 18B blocks otherwise (x_d unused). Same dequant
// math and FP op order either way -> bit-identical output.
template <int mmq_y, bool need_check, bool is_rp>
static __device__ __forceinline__ void load_tiles_q4_0(
        const char * __restrict__ x, const char * __restrict__ x_d, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2 * MMQ_TILE_NE_K);

    const int txi  = threadIdx.x;
    const int kbx  = txi / QI4_0;    // 0..7 (8 q4_0 blocks per warp pass)
    const int kqsx = txi % QI4_0;    // 0..3 (4 packed-nibble ints per block)

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) { i = min(i, i_max); }

        int qs0;
        if constexpr (is_rp) {
            const size_t ib = (size_t) (kbx0 + kbx) + (size_t) i * stride;
            qs0 = ((const int *) (x + ib * 16))[kqsx];
        } else {
            const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride + kbx;
            qs0 = get_int_b2(bxi->qs, kqsx);
        }

        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + 0] =
            __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + QI4_0] =
            __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;       // 8
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;   // 4
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;              // 0..7

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }

        half d;
        if constexpr (is_rp) {
            const size_t ib = (size_t) (kbx0 + kbxd) + (size_t) i * stride;
            d = ((const half *) x_d)[ib];
        } else {
            const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride + kbxd;
            d = bxi->d;
        }
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(d);
    }
}

// ======================= vec_dot_q8_0_q8_1_mma D4 (mmq.cuh, TURING branch) =======================
// Identical to the Q8_0 file — the x-tile is already int8 with per-32 float scales, so Q4_0 rides
// the same int8 m16n8k32 mma + C*dA*dB epilogue (D4, no sum term).
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I; // Number of x minitiles per warp.

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

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}

// ======================= mmq_write_back_mma (mmq.cuh) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_q4_0(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
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
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

// ======================= mul_mat_q_process_tile (q4_0) =======================
template <int mmq_x, bool need_check, bool is_rp>
static __device__ __forceinline__ void mul_mat_q_process_tile_q4_0(
        const char * __restrict__ x, const char * __restrict__ x_d, const int offset_x,
        const int * __restrict__ y, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = MMQ_NWARPS;
    constexpr int qk        = QK4_0;                      // 32
    constexpr int mmq_y     = MMQ_Y;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block        = 4 * QK8_1;                  // 128 values per block_q8_1_mmq
    constexpr int ITER_K          = MMQ_ITER_K;                 // 256
    constexpr int blocks_per_iter = ITER_K / qk;                // 8 q4_0 blocks per iteration

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_q4_0<mmq_y, need_check, is_rp>(x, x_d, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
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
        vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_q4_0<mmq_x, mmq_y, need_check>(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

// ======================= mul_mat_q (conventional xy-tiling) =======================
// Grid: (nty = ceil(nrows_x/mmq_y), ntx = ceil(ncols_dst/mmq_x), 1). One tile per CTA.
template <int mmq_x, bool need_check, bool is_rp>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_q4_0(
        const char * __restrict__ x, const char * __restrict__ x_d, const int * __restrict__ y,
        float * __restrict__ dst, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int ncols_y, const int stride_col_dst, const int blocks_per_ne00) {
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

    mul_mat_q_process_tile_q4_0<mmq_x, need_check, is_rp>(
        x, x_d, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00);
}

// ======================= activation quantizer (quantize.cu, D4 layout) =======================
// f32 -> block_q8_1_mmq with a symmetric FLOAT scale d per 32 values (NO sum term). llama maps
// GGML_TYPE_Q4_0 to the same D4 layout as Q8_0 (the -8 zero-point is folded into the weight tile).
static __global__ void quantize_mmq_q8_1_d4_q4_0(
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

// ======================= host launcher =======================
static size_t mmq_q4_0_nbytes_shared() {
    const size_t nbs_ids = (size_t) MMQ_X * sizeof(int);
    const size_t nbs_x   = (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_Q8_0 * sizeof(int);
    const size_t nbs_y   = (size_t) MMQ_X * sizeof(block_q8_1_mmq);
    const size_t pad     = (size_t) MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

template <bool need_check, bool is_rp>
static int mmq_q4_0_launch(const char * W, const char * W_d, const int * y_q, float * y,
                           int in_f, int out_f, int n_tokens, cudaStream_t st) {
    const int stride_row_x    = in_f / QK4_0;   // block_q4_0 per weight row
    const int blocks_per_ne00 = in_f / QK4_0;
    const int stride_col_dst  = out_f;
    const int ncols_y         = n_tokens;

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    const size_t smem = mmq_q4_0_nbytes_shared();

    cudaFuncSetAttribute(mul_mat_q_q4_0<MMQ_X, need_check, is_rp>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    mul_mat_q_q4_0<MMQ_X, need_check, is_rp><<<grid, block, smem, st>>>(
        W, W_d, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

extern "C" {

// Bytes needed for the quantized activation buffer (block_q8_1_mmq stream): caller pre-allocs.
size_t bw24_mmq_q4_0_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    return (size_t) nblocks * sizeof(block_q8_1_mmq);
}

// Run the Q4_0 int8-MMA MMQ prefill GEMM. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
//   W_q4_0 : rp == 0 -> raw ggml block_q4_0 weight rows (18B blocks, in_f/32 per row).
//            rp != 0 -> BW24_Q4RP split-plane repack: qs plane (out_f * in_f/32 * 16B, block-major)
//                       at W, fp16 d plane (dense) at W + out_f*(in_f/32)*16.
//   act_f32       : f32 activation [n_tokens, in_f].
//   y             : f32 output [n_tokens, out_f].
//   act_scratch   : pre-alloc'd >= bw24_mmq_q4_0_act_bytes(in_f, n_tokens).
// Requires in_f % 32 == 0. Returns 0 on success, else (1000 + cudaError).
int bw24_mmq_q4_0(const void * W_q4_0, const float * act_f32, float * y,
                  int in_f, int out_f, int n_tokens, void * act_scratch, void * stream, int rp) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    // ---- 1) quantize activation f32 -> block_q8_1_mmq (D4) ----
    const int64_t ne10 = in_f;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    {
        const int64_t block_num_y = (ne10_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                                    (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks((unsigned) n_tokens, (unsigned) block_num_y, 1);
        quantize_mmq_q8_1_d4_q4_0<<<num_blocks, block_size, 0, st>>>(
            act_f32, act_scratch, ne10, /*s01*/ in_f, ne10_padded, n_tokens);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }

    // ---- 2) launch mul_mat_q q4_0 (conventional xy-tiling) ----
    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_q4_0;
    const char * W_d = W + (size_t) out_f * (size_t) (in_f / QK4_0) * 16;  // rp d plane

    if (rp) {
        return need_check
            ? mmq_q4_0_launch<true,  true>(W, W_d, y_q, y, in_f, out_f, n_tokens, st)
            : mmq_q4_0_launch<false, true>(W, W_d, y_q, y, in_f, out_f, n_tokens, st);
    }
    return need_check
        ? mmq_q4_0_launch<true,  false>(W, nullptr, y_q, y, in_f, out_f, n_tokens, st)
        : mmq_q4_0_launch<false, false>(W, nullptr, y_q, y, in_f, out_f, n_tokens, st);
}

} // extern "C"
