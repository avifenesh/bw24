// mmq_q45k.cu — Q4_K/Q5_K W4A8 int8-MMA MMQ prefill GEMM (vendored floor, ggml-decoupled).
//
// Same vendor-the-floor move as mmq_fp4.cu, for the two hand-rolled k-quant GEMMs that
// dominate the prefill busy-% (q4_K 32% + q5_K 28%). Source: /data/projects/llama.cpp/ggml/src/ggml-cuda/
//   - quantize.cu  : quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4> (activation f32 -> block_q8_1_mmq,
//                    (d, sum) half2 per 32 values — Q4_K/Q5_K carry a min-offset so the sum term is live)
//   - mmq.cuh      : load_tiles_q4_K / load_tiles_q5_K (TURING_MMA branch: dequant to int8 at tile-load,
//                    scales+mins folded to per-sub-block half2 (d*sc, -dmin*m)), unpack_scales_q45_K,
//                    vec_dot_q8_1_q8_1_mma (the SHARED int8 MMA inner loop for both k-quants),
//                    mmq_write_back_mma, mul_mat_q_process_tile, mul_mat_q (conventional xy-tiling)
//   - mma.cuh      : tile<>, load_ldmatrix, load_generic, mma.sync.m16n8k32.row.col.s32.s8.s8.s32
//
// DECOUPLING: no ggml headers, same treatment as the NVFP4 vendor file. Self-contained on purpose —
// duplicating ~120 lines of tile machinery keeps this translation unit independent of the proven
// NVFP4 kernel (all functions static/internal, no link collisions).
//
// KEY DIFFS vs the NVFP4 file (do not "unify" blindly):
//   - MMQ_ITER_K = 256 (not 512): one 256-value k-quant superblock per iteration, two 128-value
//     block_q8_1_mmq y-chunks per iteration.
//   - Activation is q8_1 int8 with DS4 (d, sum) scales, NOT block_fp4_mmq.
//   - MMA is plain int8 m16n8k32 with float (d*sc)*(d_act)*acc + (-dmin*m)*(sum_act) epilogue,
//     NOT the mxf4nvf4 block-scale op. -> No Blackwell-only asm: this file is sm_75+ portable and
//     goes to the sm_89 L40S branch unchanged.
//
// C-ABI launchers: bw24_mmq_q4_K / bw24_mmq_q5_K (+ shared bw24_mmq_q45k_act_bytes). Compiled to a
// static lib like the NVFP4 vendor file, called from Rust via FFI, dispatched behind BW24_MMQ=1.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

// ======================= ggml constants/macros (vendored) =======================
#define WARP_SIZE 32
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

// quant-format constants (ggml-common.h)
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_1 32
#define QI8_1 8                  // QK8_1 / (4 * QR8_1), QR8_1 == 1
#define QR4_K 2
#define QI4_K 32                 // QK_K / (4*QR4_K)
#define QR5_K 2
#define QI5_K 32                 // QK_K / (4*QR5_K)
#define MATRIX_ROW_PADDING 512

// MMQ tile constants (mmq.cuh) — k-quant path (NOT the FP4 512 iter).
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K 256
#define MMQ_MMA_TILE_X_K_Q8_1 (2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / 8 + 4)  // 76, QI8_0==8
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)                   // 36

// launch constants (same shape as the NVFP4 vendor kernel: 8 warps, 128x128 tile)
#define MMQ_WARP_SIZE 32
#define MMQ_NWARPS    8
#define MMQ_Y         128
// MMQ_X guarded for -D sweeps: 128 (57KB smem, 1 CTA/SM) vs 64 (47KB, 2 CTA/SM — the occupancy
// lever ncu pointed at: warps_active 16.7% at 1 CTA/SM).
#ifndef MMQ_X
#define MMQ_X         128
#endif

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume >= 4 byte alignment
}

// ======================= weight / activation block structs (ggml-common.h) =======================
// block_q4_K: 144 bytes. bw24 stores raw ggml bytes (see cu/qmatvec.cu deq_q4_k: scales at +4, qs at +16).
typedef struct {
    half2   dm;                     // (d, dmin) super-block scales
    uint8_t scales[K_SCALE_SIZE];   // 8x (6-bit scale, 6-bit min) packed
    uint8_t qs[QK_K / 2];           // 4-bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size/padding");

// block_q5_K: 176 bytes (qh at +16, qs at +48 — matches cu/qmatvec.cu deq_q5_k).
typedef struct {
    half2   dm;                     // (d, dmin) super-block scales
    uint8_t scales[K_SCALE_SIZE];   // 8x (6-bit scale, 6-bit min) packed
    uint8_t qh[QK_K / 8];           // quants, high bit
    uint8_t qs[QK_K / 2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size/padding");

// block_q8_1_mmq (mmq.cuh): 4x (d, partial-sum) scales + 128 int8 quants. DS4 layout.
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

    // int8 MMA (mma.cuh:946, Ampere+ path): D(s32) += A(s8) * B(s8).
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

// ======================= unpack_scales_q45_K (mmq.cuh:2083) =======================
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

// Shared scale/min fold for both k-quants (identical block in load_tiles_q4_K and load_tiles_q5_K):
// x_dm[row][sub] = (d*sc[sub], -dmin*m[sub]) as half2, sub = 0..7.
template <int mmq_y, bool need_check, typename block_t>
static __device__ __forceinline__ void load_scales_q45_K(
        const char * __restrict__ x, half2 * __restrict__ x_dm, const int kbx0, const int i_max,
        const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int rows_per_warp = warp_size / 2;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / 2) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }

        const block_t * bxi = (const block_t *) x + kbx0 + i * stride;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        const half2 dm = __hmul2(bxi->dm, __floats2half2_rn(1.0f, -1.0f));

#pragma unroll
        for (int l = 0; l < (int) sizeof(int); ++l) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int) * ksc + l] =
                __hmul2(dm, __floats2half2_rn(sc8[l], m8[l]));
        }
    }
}

// ======================= load_tiles_q4_K (mmq.cuh:2093, TURING_MMA branch) =======================
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q4_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
        const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);   // 32
    constexpr int nrows = warp_size / threads_per_row;          // 1
    const int txi = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i * stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    load_scales_q45_K<mmq_y, need_check, block_q4_K>(x, x_dm, kbx0, i_max, stride);
}

// ======================= load_tiles_q5_K (mmq.cuh:2240, TURING_MMA branch) =======================
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
        const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);   // 32
    constexpr int nrows = warp_size / threads_per_row;          // 1
    const int txi = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y;
        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i * stride;
        const int ky = QR5_K * txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
    }

    load_scales_q45_K<mmq_y, need_check, block_q5_K>(x, x_dm, kbx0, i_max, stride);
}

// ======================= vec_dot_q8_1_q8_1_mma (mmq.cuh:1330, Turing branch) =======================
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2 * MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K / QI8_1];
    float2 dmA[ntx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_1];

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;
                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

// ======================= mmq_write_back_mma (mmq.cuh:3214) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_q45k(
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

// ======================= mul_mat_q_process_tile (mmq.cuh:3447, k-quant) =======================
// QTYPE: 4 -> Q4_K, 5 -> Q5_K (compile-time load_tiles selection; vec_dot + write_back shared).
template <int mmq_x, bool need_check, int QTYPE>
static __device__ __forceinline__ void mul_mat_q_process_tile_q45k(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int nwarps    = MMQ_NWARPS;
    constexpr int qk        = QK_K;
    constexpr int mmq_y     = MMQ_Y;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block        = 4 * QK8_1;                  // 128 values per block_q8_1_mmq
    constexpr int ITER_K          = MMQ_ITER_K;                 // 256
    constexpr int blocks_per_iter = ITER_K / qk;                // 1 superblock per iteration

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        if constexpr (QTYPE == 4) {
            load_tiles_q4_K<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        } else {
            load_tiles_q5_K<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        }
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
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
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_q45k<mmq_x, mmq_y, need_check>(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

// ======================= mul_mat_q (conventional xy-tiling) =======================
// Grid: (nty = ceil(nrows_x/mmq_y), ntx = ceil(ncols_dst/mmq_x), 1). One tile per CTA.
// (A vendored stream-K decomposition existed behind BW24_MMQ_STREAMK — measured 1.11x per-GEMM
// but its k-split f32 reorder flipped the model argmax gate; removed 2026-07-08, record in
// rig5090.jsonl 2026-07-03. Conventional tiling is the only path.)
template <int mmq_x, bool need_check, int QTYPE>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_q45k(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int mmq_y = MMQ_Y;

    // ids identity (plain GEMM: dst row == column index).
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

    mul_mat_q_process_tile_q45k<mmq_x, need_check, QTYPE>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00);
}

// ======================= activation quantizer (quantize.cu:277, DS4 layout) =======================
// f32 -> block_q8_1_mmq with (d, sum) half2 per 32 values. The sum term feeds the k-quant min-offset
// (-dmin*m * sum_act) in the vec_dot epilogue — this is why Q4_K/Q5_K need DS4, not D4.
static __global__ void quantize_mmq_q8_1_ds4_kernel(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {
    constexpr int vals_per_scale = 32;

    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) { return; }

    const int64_t i1 = blockIdx.x;
    const int64_t i00 = i0;
    const int64_t i01 = i1;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib  = (i0 / (4 * QK8_1)) * ne1 + blockIdx.x; // block index (k-major, then column)
    const int64_t iqs = i0 % (4 * QK8_1);                      // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i01 * s01 + i00) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs / 4] = q;

    if (iqs % 32 != 0) { return; }

    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;
    y[ib].ds4[iqs / 32] = make_half2(d, sum);
}

// ======================= host launchers =======================

// Dynamic-smem byte count for the mul_mat_q kernel at mmq_x=MMQ_X.
static size_t mmq_q45k_nbytes_shared() {
    const size_t nbs_ids = (size_t) MMQ_X * sizeof(int);
    const size_t nbs_x   = (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 * sizeof(int);
    const size_t nbs_y   = (size_t) MMQ_X * sizeof(block_q8_1_mmq);
    const size_t pad     = (size_t) MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

// SM count of the current device (cached — the engine is single-device).
static int mmq_q45k_nsm() {
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) { nsm = 1; }
    }
    return nsm;
}


// Shared launcher body. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
// Requires in_f % QK_K == 0 (integral superblocks per weight row).
template <int QTYPE>
static int bw24_mmq_q45k_launch(const void * W_blocks, const float * act_f32, float * y,
                                int in_f, int out_f, int n_tokens, void * act_scratch,
                                void * stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    // ---- 1) quantize activation f32 -> block_q8_1_mmq DS4 (quantize_mmq_q8_1_cuda) ----
    const int64_t ne10 = in_f;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    {
        const int64_t block_num_y = (ne10_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                                    (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks((unsigned) n_tokens, (unsigned) block_num_y, 1);
        quantize_mmq_q8_1_ds4_kernel<<<num_blocks, block_size, 0, st>>>(
            act_f32, act_scratch, ne10, /*s01*/ in_f, ne10_padded, n_tokens);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }

    // ---- 2) launch mul_mat_q (conventional xy-tiling) ----
    const int stride_row_x    = in_f / QK_K;   // superblocks per weight row
    const int blocks_per_ne00 = in_f / QK_K;
    const int stride_col_dst  = out_f;
    const int ncols_y         = n_tokens;

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    const size_t smem = mmq_q45k_nbytes_shared();

    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_blocks;

    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    if (need_check) {
        cudaFuncSetAttribute(mul_mat_q_q45k<MMQ_X, true, QTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_q45k<MMQ_X, true, QTYPE><<<grid, block, smem, st>>>(
            W, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00);
    } else {
        cudaFuncSetAttribute(mul_mat_q_q45k<MMQ_X, false, QTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_q45k<MMQ_X, false, QTYPE><<<grid, block, smem, st>>>(
            W, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

extern "C" {

// Bytes needed for the quantized activation buffer (block_q8_1_mmq stream): caller pre-allocs.
// Shared by Q4_K and Q5_K (identical activation format).
size_t bw24_mmq_q45k_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    // +MMQ_X blocks: the mul_mat_q y-tile loader always reads a FULL mmq_x-column tile; for the
    // final k-block with n_tokens % MMQ_X != 0 that read runs past the last real column. Padding
    // the scratch keeps the overread mapped (values are garbage; write-back drops j > j_max).
    return (size_t) (nblocks + MMQ_X) * sizeof(block_q8_1_mmq);
}

// Run the Q4_K W4A8 MMQ prefill GEMM. Returns 0 on success, else (1000 + cudaError).
int bw24_mmq_q4_K(const void * W_q4k_blocks, const float * act_f32, float * y,
                  int in_f, int out_f, int n_tokens, void * act_scratch, void * stream) {
    return bw24_mmq_q45k_launch<4>(W_q4k_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch,
                                   stream);
}

// Run the Q5_K W4A8 MMQ prefill GEMM (176B superblocks). Same contract as bw24_mmq_q4_K.
int bw24_mmq_q5_K(const void * W_q5k_blocks, const float * act_f32, float * y,
                  int in_f, int out_f, int n_tokens, void * act_scratch, void * stream) {
    return bw24_mmq_q45k_launch<5>(W_q5k_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch,
                                   stream);
}

} // extern "C"
