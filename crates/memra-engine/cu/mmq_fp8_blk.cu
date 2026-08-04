// mmq_fp8_blk.cu — PER-BLOCK FP8 MMQ prefill GEMM (P1 option (b), lane/fp8-mmq, sm_120a).
//
// The arm that is BOTH exact and fast. Consumes the Qwen-official block-scaled FP8 checkpoint
// DIRECTLY:
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
// This kernel keeps EVERY block's own scale, exactly, at tensor-core-class throughput.
//
// THE KEY PROPERTY — THE WEIGHT SIDE IS NOT RE-QUANTIZED AT ALL:
// the checkpoint bytes ARE the A operand. e4m3 x e4m3 -> f32 is a native Blackwell MMA
// (mma.sync.aligned.kind::f8f6f4.m16n8k32, 381-TF class, the same op the W4A8-FP8 arm uses), so
// the tile loader is a pure global->smem COPY: no dequant, no LUT, no fold, zero weight-side
// precision loss. The per-128-block scale is applied in f32 in the epilogue.
//
// GEOMETRY (why the scale lookup is free):
//   mmq_y == 128 == the scale block edge, and row tiles start at it*128, so EVERY row in a CTA's
//   tile shares scale row (i>>7) == it — one uniform grid row per CTA, hoisted out of the tile
//   loop entirely. MMQ_ITER_K == 256 values, and each vec_dot covers exactly 128 k-values aligned
//   to a 128 boundary, so each vec_dot has ONE uniform scale scalar: a k-block boundary can never
//   fall inside an MMA. (The "a tile crosses at most 2 block columns" worry collapses to "each of
//   the two vec_dots per iteration reads one scalar".)
//
// ACTIVATIONS: NOT a new format. This file calls the EXISTING W4A8-FP8 activation quantizer
// (memra_mmq_nvfp4_f8f4_quantize_act, cu/mmq_nvfp4_f8f4.cu): block_e4m3_mmq = 4x f32 scale per 32
// + 128 e4m3 bytes, d = amax/448 — byte-footprint-identical to block_q8_1_mmq, so all y-tile smem
// and stride math is the vendored q8_1 math unchanged.
//
// ARITHMETIC CONTRACT (what the kernel-check host reference reproduces, in this order):
//   sum(i,j) = SUM over k-iterations kv0 (ascending, step 256)
//                SUM over halves h in {0,1} (values kv0+128h .. kv0+128h+127)
//                  SUM over k01 in {0,8,16,24} (ascending; 32 values each)
//                    (s_blk * dB) * C ,  C = SUM_{t=0..31} e4m3(W[i][g]) * e4m3(A[j][g])
//   with g = kv0 + 128h + 4*k01 + t, s_blk = blk_scale[it][min((kv0+128h)>>7, cols-1)],
//   dB = act_d4[j][chunk][k01/8], and dst = sum * out_scale.
//   The only order the kernel does NOT define is the 32-product reduction INSIDE one MMA
//   (hardware-internal). The kernel-check integer arm is exact-by-construction (all products
//   integers, |partial sums| < 2^24, scales powers of two) so that order cannot matter, giving a
//   true BIT-IDENTITY gate; a second random-e4m3 arm bounds the residual rounding.
//
// k TAIL: in_dim need not be a multiple of 256 (or of 128). k values >= in_dim are filled with
// 0x00 in smem — e4m3 0x00 is exactly 0.0, and the activation quantizer already zero-pads its
// side, so padded lanes contribute exact zeros. Requires in_dim % 16 == 0 (16B row alignment for
// the int4 tile copy; every real block-128 projection is a multiple of 128).
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

#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K    256                                       // k-values per tile iteration
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)   // 36 ints per y block
// x tile row stride, ints: 64 value-ints (256 e4m3 bytes) + 4 pad ints. 68*4 = 272 B == 17*16, so
// every ldmatrix row address stays 16B aligned; the +4 pad breaks the smem bank alignment the way
// MMQ_MMA_TILE_X_K_NVFP4 (84) does.
#define MMQ_MMA_TILE_X_K_FP8 (2 * MMQ_TILE_NE_K + 4)

#define MMQ_WARP_SIZE  32
#define FP8_BLK        128        // scale block edge (both axes) — the checkpoint's grid
#ifndef FP8_MMQ_Y
#define FP8_MMQ_Y      128        // MUST equal FP8_BLK: that is what makes scale row == blockIdx.x
#endif
#define FP8_MMQ_NWARPS (FP8_MMQ_Y / 16)
#ifndef FP8_MMQ_X
#define FP8_MMQ_X      128
#endif
static_assert(FP8_MMQ_Y == FP8_BLK, "scale-row hoist requires mmq_y == block edge");

// block_e4m3_mmq (cu/mmq_nvfp4_f8f4.cu) — the activation block this kernel consumes.
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

// plain-kind f8f6f4 MMA: D(f32 16x8) += A(e4m3 16x32) * B(e4m3 32x8). Same op as the W4A8-FP8 arm.
static __device__ __forceinline__ void memra_fp8_mma_f8f4(
        float * __restrict__ c, const int * __restrict__ a, const int b0, const int b1) {
    asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ======================= tile loader: a COPY, not a dequant =======================
// [mmq_y rows x MMQ_ITER_K k-values] of raw e4m3 -> smem, as int4 (16B) lines. k >= k_valid is
// zero-filled (e4m3 0x00 == exact 0.0). need_check clamps the SOURCE row to i_max exactly like the
// vendored loaders; the destination slot stays unclamped so no two threads alias.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_fp8_blk(
        const uint8_t * __restrict__ x, int * __restrict__ x_tile,
        const int kv0, const int i_max, const int stride_row, const int k_valid) {
    constexpr int nwarps        = mmq_y / 16;
    constexpr int nthreads      = nwarps * MMQ_WARP_SIZE;
    constexpr int lines_per_row = MMQ_ITER_K / 16;      // 16B lines per row (16)
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

// ======================= vec_dot: one k32 f8f6f4 MMA per 8-int step =======================
// s_blk is UNIFORM over this call (see the geometry note in the header): 128 k-values aligned to a
// 128 boundary, all rows of the CTA in one scale row. Epilogue = (s_blk * dB) * C.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_fp8_blk_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum,
        const int k00, const float s_blk) {
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
                          x_qs + (i0 + n * tile_A_8::I) * MMQ_MMA_TILE_X_K_FP8 + (k00 + k01),
                          MMQ_MMA_TILE_X_K_FP8);
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float  dB[tile_C::ne / 2];
            load_generic(B[0], y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0),           MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0 * MMQ_TILE_Y_K + (k01 + tile_B::J),   MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < tile_C::ne / 2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                memra_fp8_mma_f8f4(C, A[n][k01 / 8].x, B[0].x[0], B[1].x[0]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += (s_blk * dB[l % 2]) * C[l];
                }
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

    for (int kv0 = 0; kv0 < k_iter_end; kv0 += MMQ_ITER_K) {
        load_tiles_fp8_blk<mmq_y, need_check>(x, tile_x, kv0, tile_x_max_i, stride_row_x, k_valid);

        const int   c0  = kv0 >> 7;                                   // y chunk / scale column
        const int   c1  = c0 + 1;
        // Clamp only guards the fully-padded tail chunk (its weight bytes are all 0x00, so the
        // scale value is irrelevant — the clamp exists to keep the grid read in bounds).
        const float sc0 = s_row[min(c0, scale_cols - 1)];
        const float sc1 = s_row[min(c1, scale_cols - 1)];

        {
            const int * by0 = y + ncols_y * c0 * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_fp8_blk_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0, sc0);
        __syncthreads();
        {
            const int * by0 = y + ncols_y * c1 * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_fp8_blk_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K, sc1);
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
    const int it = blockIdx.x;   // out-row tile == scale grid ROW (mmq_y == FP8_BLK)

    const int offset_y     = (jt * mmq_x) * (sizeof(block_e4m3_mmq) / sizeof(int));
    const int offset_dst   = jt * mmq_x * stride_col_dst + it * mmq_y;
    const int tile_x_max_i = nrows_x   - it * mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;

    mul_mat_q_process_tile_fp8_blk<mmq_x, mmq_y, need_check>(
        x + (size_t) it * mmq_y * (size_t) stride_row_x,
        blk_scales + (size_t) it * (size_t) scale_cols,
        y + offset_y, ids_dst_shared, dst + offset_dst,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, k_valid, scale_cols,
        out_scale);
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
extern "C" {

// Activation quantizer + scratch sizing are SHARED with the W4A8-FP8 arm (cu/mmq_nvfp4_f8f4.cu) —
// this kernel deliberately introduces no activation format of its own.
size_t memra_mmq_nvfp4_f8f4_act_bytes(int in_f, int n_tokens);
void   memra_mmq_nvfp4_f8f4_quantize_act(const float * x, void * vy, int in_f, int n_tokens,
                                         int64_t s01, cudaStream_t st);

size_t memra_mmq_fp8_blk_act_bytes(int in_f, int n_tokens) {
    return memra_mmq_nvfp4_f8f4_act_bytes(in_f, n_tokens);
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

    memra_mmq_nvfp4_f8f4_quantize_act(act_f32, act_scratch, in_f, n_tokens, in_f, st);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return 1000 + (int) e; } }

    const int scale_cols = (in_f + FP8_BLK - 1) / FP8_BLK;
    const int nty = (out_f    + FP8_MMQ_Y - 1) / FP8_MMQ_Y;
    const int ntx = (n_tokens + FP8_MMQ_X - 1) / FP8_MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, FP8_MMQ_NWARPS, 1);

    const size_t nbs_ids = (size_t) FP8_MMQ_X * sizeof(int);
    const size_t nbs_y   = (size_t) FP8_MMQ_X * sizeof(block_e4m3_mmq);
    const size_t pad     = (size_t) FP8_MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    const size_t nbs_x   = (size_t) FP8_MMQ_Y * MMQ_MMA_TILE_X_K_FP8 * sizeof(int);
    const size_t smem    = nbs_ids + GGML_PAD(nbs_y, pad) + nbs_x;

    const bool need_check = (out_f % FP8_MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;

    #define MEMRA_FP8MMQ_LAUNCH(NC) do {                                                          \
        cudaFuncSetAttribute(mul_mat_q_fp8_blk<FP8_MMQ_X, FP8_MMQ_Y, NC>,                         \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                  \
        mul_mat_q_fp8_blk<FP8_MMQ_X, FP8_MMQ_Y, NC><<<grid, block, smem, st>>>(                   \
            (const uint8_t *) W_e4m3, blk_scales, y_q, y, out_f, n_tokens, in_f, n_tokens, out_f, \
            in_f, scale_cols, out_scale);                                                         \
    } while (0)
    if (need_check) { MEMRA_FP8MMQ_LAUNCH(true); } else { MEMRA_FP8MMQ_LAUNCH(false); }
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
