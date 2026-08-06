// mmq_fp4.cu — NVFP4 W4A4 block-scale MMQ prefill GEMM (vendored floor, ggml-decoupled, sm_120a).
//
// This is the 5150-pp512 kernel from llama.cpp brought into memra wholesale (the user's "copy the
// working fast kernel, tune the edges" mandate). Source: /data/projects/llama.cpp/ggml/src/ggml-cuda/
//   - quantize.cu  : quantize_mmq_nvfp4 (activation f32 -> block_fp4_mmq, 2-level FP8-e8m0/UE4M3 scale)
//   - mmq.cuh      : block_q8_1_mmq / block_fp4_mmq, load_tiles_nvfp4_nvfp4, vec_dot_fp4_fp4_mma,
//                    mmq_write_back_mma, mul_mat_q_process_tile, mul_mat_q (conventional xy-tiling)
//   - mma.cuh      : tile<>, load_ldmatrix, load_generic, mma_block_scaled_fp4 (mxf4nvf4 block-scale mma)
//   - common.cuh   : ggml_cuda_ue4m3_to_fp32 / fp32_to_ue4m3 / float_to_fp4_e2m1, kvalues_mxfp4
//
// DECOUPLING: no ggml headers. ggml_tensor/backend/pool/info stripped -> raw device pointers + the
// hardcoded sm_120 constants (warp_size=32, nwarps=8, mmq_y=128, BLACKWELL_MMA_AVAILABLE). We use the
// CONVENTIONAL xy-tiling launch (one tile/CTA, fixup=false) so there is NO stream-K and NO fixup buffer
// (the stream-K path only helps when ntiles << SMs; prefill GEMM has many tiles -> xy-tiling is fine).
//
// WEIGHT FORMAT (verified vs cu/cutlass_fp4_sm120.cu deinterleave): memra's stored NVFP4 weight bytes
// are EXACTLY llama's block_nvfp4 = per row, in_f/64 blocks of 36 bytes = [4 UE4M3 scale bytes | 32
// e2m1 qs bytes], qs packed so element w of sub-block s lives in qs[s*8 + (w&7)] lo/hi at w<8/w>=8.
// That is the SAME packing quantize_mmq_nvfp4 emits for activations -> load_tiles_nvfp4_nvfp4 reads
// the raw weight bytes directly (pure u32 copy, no repack).
//
// C-ABI launcher: memra_mmq_nvfp4(W_nvfp4, act_f32, y, in_f, out_f, n_tokens, stream). Internally
// quantizes act_f32 -> block_fp4_mmq, then launches mul_mat_q NVFP4. Compiled to a static lib (same as
// cutlass_fp4_sm120.cu), called from Rust via FFI.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cstdint>
#include <cfloat>
#include <cmath>

// ======================= ggml constants/macros (vendored, sm_120) =======================
#define BLACKWELL_MMA_AVAILABLE
#define TURING_MMA_AVAILABLE
#define WARP_SIZE 32
#define NO_DEVICE_CODE __trap()
#define GGML_UNUSED(x) (void)(x)
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

// quant-format constants (ggml-common.h)
#define QK_K 256
#define QK8_1 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16           // 16-element sub-block (one UE4M3 micro-scale each)
#define QI8_1 (QK8_1 / (4 * 1))   // QR8_1 == 1 -> QI8_1 == 8
#define MATRIX_ROW_PADDING 512

// MMQ tile constants (mmq.cuh)
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K_FP4 512
#define MMQ_MMA_TILE_X_K_FP4 (2 * MMQ_TILE_NE_K + 8 + 4)            // 76
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)        // 36

// sm_120 launch constants (resolved from mmq_get_* device helpers)
#define MMQ_WARP_SIZE 32
#define MMQ_NWARPS    8        // 256 / 32
#define MMQ_Y         128      // get_mmq_y_device()
#define MMQ_X         128      // prefill batch tile (n-tokens tile)

// FP4 e2m1 reconstruction LUT (ggml-common.h kvalues_mxfp4) — used by the activation quantizer's
// per-sub-block scale search.
__constant__ int8_t kvalues_mxfp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// ======================= FP4 / UE4M3 scalar helpers (common.cuh) =======================
static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // NaN -> 0.0f to match CPU impl
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
}

__device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) { best_err = err; best_i = i; }
    }
    return static_cast<uint8_t>(best_i | sign_bit);
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume >= 4 byte alignment
}

// ======================= weight / activation block structs =======================
// llama block_nvfp4 (ggml-common.h): 36 bytes = 4 UE4M3 scales (per 16) + 32 packed e2m1 (64 vals).
typedef struct {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4 / 2];           // packed 4-bit e2m1 (32 bytes)
} block_nvfp4;

// llama block_q8_1_mmq / block_fp4_mmq (mmq.cuh) — the activation tile layout the MMA consumes.
struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4 * QK8_1];               // 128 values
};
struct block_fp4_mmq {
    uint32_t d4[4];
    int8_t   qs[4 * 32];                // 256 e2m1 values packed 2/byte
};

// ======================= mma.cuh: tile<>, loads, block-scaled FP4 mma =======================
namespace ggml_cuda_mma {
    enum data_layout {
        DATA_LAYOUT_I_MAJOR = 0,
        DATA_LAYOUT_J_MAJOR = 10,
    };

    template <int I_, int J_, typename T, data_layout ds_ = DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;
        static constexpr int         ne = I * J / 32;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
        }
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    // NVFP4 block-scale MMA: mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X (UE4M3 scales).
    // rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md
    //   16.06 cyc/warp-MMA, 619.2 TFLOP/s = 3.99x the int8-k16 rate and THE FASTEST form on sm_120.
    //   The scale_vec::2X ue8m0 variant measures identically (619.1), so the granularity choice is
    //   free. ptxas REJECTS m16n8k128 mxf4nvf4 -- nothing deeper exists. OPTIMAL, no swap available.
    static __device__ __forceinline__ void mma_block_scaled_fp4_nvfp4(
            tile<16, 8, float> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B,
            uint32_t a_scale, uint32_t b_scale) {
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]),
              "r"(a_scale), "r"(b_scale));
    }
} // namespace ggml_cuda_mma

using namespace ggml_cuda_mma;

// sm_120 granularity (mmq_get_granularity_device): mmq_x>=48 -> 16.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// ======================= load_tiles_nvfp4_nvfp4 (mmq.cuh:945) =======================
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_nvfp4_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
        const int stride) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int warp_size = MMQ_WARP_SIZE;
    constexpr int iter_k = MMQ_ITER_K_FP4;
    constexpr int threads_per_row = iter_k / QK_NVFP4; // = 8, each thread processes 1 block
    constexpr int rows_per_warp = warp_size / threads_per_row;

    uint32_t * x_u32 = (uint32_t *) x_tile;

    const int txi = threadIdx.x;
    const int kbx = txi % threads_per_row;
    const int row_in_warp = txi / threads_per_row;

    const block_nvfp4 * bxi_base = (const block_nvfp4 *) x + kbx0 + kbx;
    uint32_t * x_u32_scale = x_u32 + 64 + kbx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;
        if constexpr (need_check) { i = min(i, i_max); }

        const block_nvfp4 * bxi = bxi_base + i * stride;
        const int row_base = i * MMQ_MMA_TILE_X_K_FP4;
        const int q_base = row_base + 8 * kbx;

        const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            x_u32[q_base + 2 * sub + 0] = src_qs[2 * sub + 0];
            x_u32[q_base + 2 * sub + 1] = src_qs[2 * sub + 1];
        }
        x_u32_scale[row_base] = get_int_b4(bxi->d, 0);
    }
}

// ======================= vec_dot_fp4_fp4_mma (mmq.cuh:991, NVFP4) =======================
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_nvfp4_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int stride        = MMQ_MMA_TILE_X_K_FP4;
    constexpr int granularity   = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * stride + k0, stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled_fp4_nvfp4(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

// ======================= mmq_write_back_mma (mmq.cuh:3214) =======================
template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_nvfp4(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max,
        const float out_scale) {
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
                // TWO folded scales, both applied here so neither costs a separate launch or a
                // full y round-trip:
                //   out_scale   — per-tensor NVFP4 macro-scale (was a scale_f32 launch, 4.2ms of
                //                 pp512). 1.0 for non-scaled tensors.
                //   y_scale[j]  — per-TOKEN activation row scale from the quantizer's level-1 amax.
                //                 The quantizer divided token j's row through by it before the
                //                 sub-block search, so the dot products come back scaled down by
                //                 the same factor and multiplying restores them. nullptr when the
                //                 quantizer ran without row scaling (the v1 oracle path).
                const float acc = sum[(j0 / tile_C::J + n) * tile_C::ne + l] * out_scale;
                dst[ids_dst[j] * stride + i] = y_scale ? acc * y_scale[j] : acc;
            }
        }
    }
}

// ======================= mul_mat_q_process_tile (mmq.cuh:3447, NVFP4) =======================
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const float * __restrict__ y_scale,
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

    // FP4 tile stores 8 blocks (QK_K=256 values per block_fp4_mmq).
    constexpr int ne_block = QK_K;
    constexpr int ITER_K          = MMQ_ITER_K_FP4;
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int); // == MMQ_TILE_Y_K (36)

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_nvfp4_nvfp4<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_nvfp4_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
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
        vec_dot_nvfp4_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_nvfp4<mmq_x, mmq_y, need_check>(
        sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q (conventional xy-tiling, NVFP4) =======================
// Grid: (nty = ceil(nrows_x/mmq_y), ntx = ceil(ncols_dst/mmq_x), 1). One tile per CTA, fixup=false.
// (2D plain GEMM: 1 channel, 1 sample -> all the stride_channel/sample/expert plumbing drops out.)
template <int mmq_x, bool need_check>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_nvfp4(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const float * __restrict__ y_scale,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const int blocks_per_ne00, const float out_scale) {
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
    // 64-bit offset_dst (audit Q7, 2026-08-05): wraps at n_tokens*out_f >= 2^31 — see mmq_q8_0.cu.
    const int64_t offset_dst = (int64_t) jt * mmq_x * stride_col_dst + (int64_t) it * mmq_y;

    const int tile_x_max_i = nrows_x  - it * mmq_y - 1;
    const int tile_y_max_j = col_diff - jt * mmq_x - 1;

    const int offset_x = it * mmq_y * stride_row_x;

    // The per-token scale array is indexed by GLOBAL token, but write_back sees tile-local j, so
    // advance the base pointer by this tile's first token.
    const float * y_scale_tile = y_scale ? y_scale + jt * mmq_x : nullptr;

    mul_mat_q_process_tile_nvfp4<mmq_x, need_check>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, y_scale_tile, stride_row_x,
        ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
}

// ======================= activation quantizer (quantize.cu quantize_mmq_nvfp4) =======================
//
// PER-TOKEN AMAX SCALING (ported from llama.cpp 1a064ab09, 2026-08-03).
//
// The original quantizer scaled each 16-element sub-block by its own UE4M3 micro-scale and nothing
// else. UE4M3 has a 4-bit exponent, so a sub-block whose amax sits far from the row's dynamic range
// gets a micro-scale that clamps: the e2m1 grid it then quantizes onto is the wrong decade, and the
// error is a systematic bias rather than rounding noise. Upstream's fix is a SECOND, coarser level
// of scaling: reduce the amax over the whole token row, divide the row through by it before the
// sub-block search, and hand the row factor to the GEMM epilogue to undo. Every sub-block then
// searches a normalized range where UE4M3 has headroom on both sides.
//
// `amax / (6 * 448)`: 6 is the largest e2m1 magnitude and 448 the largest UE4M3 value, so the
// product is the largest value the two-level product can represent. Dividing the row amax by it
// maps the row's peak onto the top of the representable range with no clamping.

// 32-byte load struct: maps to 256-bit PTX loads on Blackwell, else two 128-bit loads.
struct __builtin_align__(32) float8 {
    float x; float y; float z; float w;
    float p; float q; float r; float s;
};

// Full-warp max reduction (common.cuh warp_reduce_max<32>).
static __device__ __forceinline__ float mmq_warp_reduce_max(float x) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE));
    }
    return x;
}

#if CUDART_VERSION >= 12080
// Squared reconstruction error of one candidate sub-block scale, computed through the e2m1 HARDWARE
// converter (`cvt.rn.satfinite.e2m1x2.f32` behind __nv_fp4x4_e2m1) rather than the scalar
// nearest-of-8 LUT search. The hardware's rounding is what the packed bytes will actually carry, so
// the search now scores the code it will really emit: the scalar LUT picks the nearest grid point by
// absolute distance, while the hardware rounds to nearest-even in the e2m1 encoding, and the two
// disagree exactly at the ties that a scale search spends its time on.
static __device__ __forceinline__ float nvfp4_native_scale_error(
        const float vals[QK_NVFP4_SUB], const float inv_col_scale, const float inv_scale, const float scale) {
    const float scale_dequant = 2.0f * scale;
    float err = 0.0f;

#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k += 4) {
        const float v0 = vals[k + 0] * inv_col_scale;
        const float v1 = vals[k + 1] * inv_col_scale;
        const float v2 = vals[k + 2] * inv_col_scale;
        const float v3 = vals[k + 3] * inv_col_scale;

        const __nv_fp4x4_e2m1 q(make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);

        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
        const float2 dq_lo = __half22float2(static_cast<__half2>(hraw2_lo));
        const float2 dq_hi = __half22float2(static_cast<__half2>(hraw2_hi));

        const float err0 = fabsf(v0) - fabsf(dq_lo.x) * scale_dequant;
        const float err1 = fabsf(v1) - fabsf(dq_lo.y) * scale_dequant;
        const float err2 = fabsf(v2) - fabsf(dq_hi.x) * scale_dequant;
        const float err3 = fabsf(v3) - fabsf(dq_hi.y) * scale_dequant;

        err = fmaf(err0, err0, err);
        err = fmaf(err1, err1, err);
        err = fmaf(err2, err2, err);
        err = fmaf(err3, err3, err);
    }

    return err;
}
#endif // CUDART_VERSION >= 12080

#define MMQ_QUANT_BLOCK_SIZE 128

// One CTA per token row (was: one thread per sub-block over a 2D grid). The row amax reduction needs
// every element of the row in one CTA, which fixes the grid shape: blockIdx.x = token, and the CTA
// then strides its threads over the row's sub-blocks.
__launch_bounds__(MMQ_QUANT_BLOCK_SIZE, 1)
static __global__ void quantize_mmq_nvfp4_kernel(
        const float * __restrict__ x, void * __restrict__ vy, float * __restrict__ scale,
        const uint8_t * __restrict__ chan_skip,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const bool use_aligned_float8_in) {
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;

    const int64_t i2  = blockIdx.y % ne2;
    const int64_t i3  = blockIdx.y / ne2;
    const int64_t i01 = blockIdx.x;
    const float * __restrict__ x_row = x + (i3 * s03 + i2 * s02 + i01 * s01);

    // Residual channels are zeroed here and added back exactly in the epilogue correction. Zeroing
    // before the amax reduction is the point: it removes the outliers from the scale decision, so
    // every remaining sub-block gets a finer micro-scale.
    // (The aligned-float8 fast path can't do per-element masking, so it steps aside when active.)
    const bool residual = chan_skip != nullptr;
    const bool use_aligned_float8 = use_aligned_float8_in && !residual;

    // ---- level 1: row amax ----
    float amax = 0.0f;
    if (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const float8 v = reinterpret_cast<const float8 *>(x_row + i0)[0];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
            amax = fmaxf(amax, fabsf(v.p));
            amax = fmaxf(amax, fabsf(v.q));
            amax = fmaxf(amax, fabsf(v.r));
            amax = fmaxf(amax, fabsf(v.s));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            if (residual && chan_skip[i0]) { continue; }
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
    }

    amax = mmq_warp_reduce_max(amax);

    __shared__ float warp_amax[MMQ_QUANT_BLOCK_SIZE / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) { warp_amax[warp] = amax; }
    __syncthreads();

    if (warp == 0) {
        amax = threadIdx.x < (MMQ_QUANT_BLOCK_SIZE / WARP_SIZE) ? warp_amax[lane] : 0.0f;
        amax = mmq_warp_reduce_max(amax);
        if (lane == 0) {
            // 6 = max e2m1 magnitude, 448 = max UE4M3 value: the row peak lands at the top of the
            // two-level representable range, so no sub-block scale has to clamp.
            warp_amax[0] = amax / (6.0f * 448.0f);
            scale[blockIdx.y * ne1 + blockIdx.x] = warp_amax[0];
        }
    }
    __syncthreads();

    const float row_scale     = warp_amax[0];
    const float inv_col_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;

    // ---- level 2: per-sub-block UE4M3 micro-scale, searched in the normalized range ----
    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_K;
        const int     sub     = (i0_base % QK_K) / QK_NVFP4_SUB;

        float vals[QK_NVFP4_SUB];
        if (use_aligned_float8) {
            const float * x_base = x_row + i0_base;
            const float8 zero = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const float8 v0 = i0_base +  7 < ne00 ? reinterpret_cast<const float8 *>(x_base)[0]     : zero;
            const float8 v1 = i0_base + 15 < ne00 ? reinterpret_cast<const float8 *>(x_base + 8)[0] : zero;
            vals[0]  = v0.x; vals[1]  = v0.y; vals[2]  = v0.z; vals[3]  = v0.w;
            vals[4]  = v0.p; vals[5]  = v0.q; vals[6]  = v0.r; vals[7]  = v0.s;
            vals[8]  = v1.x; vals[9]  = v1.y; vals[10] = v1.z; vals[11] = v1.w;
            vals[12] = v1.p; vals[13] = v1.q; vals[14] = v1.r; vals[15] = v1.s;
        } else {
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const int64_t i00 = i0_base + k;
                const bool    skip = residual && i00 < ne00 && chan_skip[i00];
                vals[k] = (i00 < ne00 && !skip) ? x_row[i00] : 0.0f;
            }
        }

        float amax_sub = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
        }

        static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
        const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

        uint8_t fp8_code       = (uint8_t) first_fp8_code;
        float   subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
        float   inv_scale_err  = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#if CUDART_VERSION >= 12080
        float best_err = nvfp4_native_scale_error(vals, inv_col_scale, inv_scale_err, subblock_scale);
#else
        float best_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float   v        = vals[k] * inv_col_scale;
            const uint8_t q        = ggml_cuda_float_to_fp4_e2m1(v, inv_scale_err);
            const float   err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * subblock_scale;
            best_err = fmaf(err_diff, err_diff, best_err);
        }
#endif

#pragma unroll
        for (int i = 1; i < 5; ++i) {
            const int test_code = first_fp8_code + test_offsets[i];
            if (test_code < 0 || test_code > 0x7e) { continue; }
            const float test_scale     = ggml_cuda_ue4m3_to_fp32((uint8_t) test_code);
            const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
#if CUDART_VERSION >= 12080
            const float cur_err = nvfp4_native_scale_error(vals, inv_col_scale, test_inv_scale, test_scale);
#else
            float cur_err = 0.0f;
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const float   v        = vals[k] * inv_col_scale;
                const uint8_t q        = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
                const float   err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
                cur_err = fmaf(err_diff, err_diff, cur_err);
            }
#endif
            if (cur_err < best_err) {
                best_err       = cur_err;
                fp8_code       = (uint8_t) test_code;
                subblock_scale = test_scale;
            }
        }

        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        uint32_t q0 = 0;
        uint32_t q1 = 0;
#if CUDART_VERSION >= 12080
        // Pack through the hardware pair converter. The (0,8,1,9)/(2,10,3,11)/... interleave is the
        // byte order load_tiles_nvfp4_nvfp4 reads: element w of the sub-block sits in nibble
        // (w & 7) lo for w < 8 and hi for w >= 8.
        const float s = inv_col_scale * inv_scale;
        __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
        __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
        __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
        __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

        const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
        const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
        const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
        const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

        q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q0_hi_c.x)) << 16) | (uint32_t(uint8_t(q0_hi_c.y)) << 24);
        q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q1_hi_c.x)) << 16) | (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
            q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals[k +  0] * inv_col_scale, inv_scale) << (8 * k);
            q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals[k +  8] * inv_col_scale, inv_scale) << (8 * k + 4);
            q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals[k +  4] * inv_col_scale, inv_scale) << (8 * k);
            q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale) << (8 * k + 4);
        }
#endif

        block_fp4_mmq * yb = y + (blockIdx.y * (blocks_per_col * ne1) + k_block * ne1 + blockIdx.x);
        uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
        yqs[2 * sub + 0] = q0;
        yqs[2 * sub + 1] = q1;
        reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
    }
}

// The pre-port quantizer: per-sub-block UE4M3 micro-scale only, no row scale. Kept as the numeric
// oracle so the kernel-check arm can measure what the two-level scaling actually bought, and as the
// rollback seam if the per-token path ever has to be backed out.
static __global__ void quantize_mmq_nvfp4_kernel_v1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) { return; }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = i1;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) { return; }

    const int64_t ib = blockIdx.z * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x;
    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    block_fp4_mmq * yb = y + ib;

    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const float v = x[base_idx + i00];
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);

    float best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;

#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) { continue; }
        const uint8_t code = (uint8_t) test_code;
        const float test_scale = ggml_cuda_ue4m3_to_fp32(code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals_raw[k];
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }
        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = (uint8_t) test_code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0;
    uint32_t q1 = 0;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  0], inv_scale) << (8 * k);
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  4], inv_scale) << (8 * k);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }

    uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
}

// ======================= residual high-precision channels (ARCQuant-style) =======================
//
// Two-level scaling fixed the sub-blocks whose micro-scale was CLAMPING, but it cannot fix a
// sub-block whose values span more dynamic range than the e2m1 grid has points. Transformer
// activations have a small number of persistent outlier CHANNELS — the same feature dimensions,
// across every token — that run one to two decades above their neighbours. Normalizing the row by
// its amax makes those outliers the thing that sets the scale, so the other 15 values in their
// sub-block get pushed toward the bottom of the grid and lose resolution.
//
// The fix is to take the outliers out of the quantized path entirely: pick the k channels with the
// largest magnitude (ranked per-channel across the whole batch, because the outlier set is a
// property of the weight matrix's input space, not of one token), zero them before quantization,
// and add their exact f32 contribution back as a rank-k correction. Zeroing pays twice: the
// correction is exact for the channels that mattered most, AND the row amax drops, so every
// remaining sub-block gets a finer scale.
//
// k is small (4/8/16) — the correction is a thin rank-k update, not a second GEMM.

// Cap on residual channels. Raised 16 -> 64 once the perf window showed the correction is FREE
// (interleaved x5, q9 pp1845: W4A4 k=0 7404.9 vs k=16 7417.0 tok/s — inside the 0.8% spread), and
// the exactness sweep then landed on k=32, so the cap has to clear it.
#define MMQ_MAX_RESIDUAL_K 64
// The correction kernel is instantiated at these compile-time channel counts and a runtime k is
// rounded UP to the next bucket, with the padding channels marked -1 (zero weight, zero activation).
// Compile-time K is what keeps the register weight array out of local memory and keeps y touched
// exactly once; see the kernel comment for the measured cost of the runtime-k form.
#define MMQ_RESIDUAL_BUCKETS 4

// Per-channel amax across the whole token batch. Threads own channels, so reads across a token row
// are coalesced.
static __global__ void nvfp4_channel_amax_kernel(
        const float * __restrict__ act, float * __restrict__ chan_amax,
        const int n_tokens, const int in_f, const int64_t s11) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= in_f) { return; }
    float a = 0.0f;
    for (int j = 0; j < n_tokens; ++j) {
        a = fmaxf(a, fabsf(act[(int64_t) j * s11 + c]));
    }
    chan_amax[c] = a;
}

// Top-k channel selection, one CTA, k masking passes. k <= 64 and in_f is a few thousand, so the
// O(k * in_f / nthreads) scan is cheaper to write and to run than a sort.
//
// k_pad >= k is the compile-time bucket the correction kernel will run at. Entries in [k, k_pad) are
// left at -1 so the correction's padding lanes contribute nothing.
static __global__ void nvfp4_topk_channels_kernel(
        const float * __restrict__ chan_amax, uint8_t * __restrict__ chan_skip,
        int * __restrict__ topk_idx, const int in_f, const int k, const int k_pad) {
    constexpr int NT = 256;
    __shared__ float sv[NT];
    __shared__ int   si[NT];

    for (int c = threadIdx.x; c < in_f; c += NT) { chan_skip[c] = 0; }
    for (int s = threadIdx.x; s < k_pad; s += NT) { topk_idx[s] = -1; }
    __syncthreads();

    for (int pass = 0; pass < k; ++pass) {
        float bv = -1.0f;
        int   bi = -1;
        for (int c = threadIdx.x; c < in_f; c += NT) {
            if (chan_skip[c]) { continue; }
            const float v = chan_amax[c];
            if (v > bv) { bv = v; bi = c; }
        }
        sv[threadIdx.x] = bv;
        si[threadIdx.x] = bi;
        __syncthreads();
#pragma unroll
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                if (sv[threadIdx.x + off] > sv[threadIdx.x]) {
                    sv[threadIdx.x] = sv[threadIdx.x + off];
                    si[threadIdx.x] = si[threadIdx.x + off];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int win = si[0];
            topk_idx[pass] = win;
            if (win >= 0) { chan_skip[win] = 1; }
        }
        __syncthreads();
    }
}

// Rank-k correction: y[j][i] += out_scale * sum_s act[j][c_s] * W_dequant[i][c_s].
//
// NOT scaled by the per-token row factor: the correction consumes the ORIGINAL f32 activations,
// which were never divided by it.
//
// TEMPLATED on the channel count, and the reason is measured, not stylistic. A runtime-k form has to
// keep the per-row dequantized weights in a dynamically indexed array (nvcc puts that in local
// memory) and, once k passes the register budget, has to visit y once per channel tile — a
// read-modify-write of the whole output per pass. Priced on q9 pp1845 interleaved x5: the runtime-k
// tiled form ran 1.088x vs W4A8 at k=16 and 0.840x at k=32, against 1.631x for the untiled k=16
// form. With a compile-time K the weight array unrolls into registers and y is touched EXACTLY ONCE
// regardless of k, so the correction stays a thin rank-k update instead of extra output traffic.
//
// Runtime k <= K: topk_idx is pre-filled with -1, and a -1 channel contributes a zero weight and a
// zero activation, so the padding lanes are arithmetic no-ops. Each thread owns one output row and
// reads its K weight values once for the whole token loop.
//
// SUMMATION ORDER IS LOAD-BEARING, and this is a receipt, not a preference. The kernel that gated
// IDENTICAL on all five measurable cells at k=32 summed in groups of 16 channels, each group rounded
// into y separately. Accumulating all K in one f32 chain instead is more accurate in isolation but
// changes the last bits, and two q9 cells (p2-code-medium, p3-agentic-long) went DIVERGENT on that
// change alone — greedy decode sits on knife-edge argmax margins, so "different rounding" is
// indistinguishable from "wrong". The GROUP structure below reproduces the gated arithmetic exactly
// (group sums rounded to f32, then summed in group order) while still writing y ONCE, which is where
// the cost was. Do not fuse the groups to "clean this up" without re-running the gate.
template <int K>
static __global__ void nvfp4_residual_correct_kernel(
        const char * __restrict__ W, const float * __restrict__ act, float * __restrict__ y,
        const int * __restrict__ topk_idx, const int in_f, const int out_f,
        const int n_tokens, const int64_t s11, const float out_scale) {
    constexpr int NT = 256;
    // Tokens staged per CTA. Sized so a_sh is 32 KiB at every K (8192/K floats * K channels), which
    // fits the 48 KiB static shared-memory limit with room to spare.
    //
    // This is the weight-traffic knob, and it is the cost that remains after the grid fix. Every
    // token-chunk CTA re-dequantizes the SAME out_f*K outlier weights, and those reads are
    // inherently scattered: consecutive threads hold consecutive output rows, which sit row_bytes
    // apart (2304 B for in_f=4096), so each one pulls its own sector. At CHUNK=64 that was 29 passes
    // over pp1845 — order 121 MB of sector traffic against 60 MB for y, i.e. the redundant weight
    // reads outweighed the output they were correcting. Staging 8192/K tokens instead cuts the pass
    // count by the same factor (29 -> 8 at K=32) without touching per-element arithmetic: each CTA
    // still owns a disjoint (row, token) set and still sums groups of 16 in the same order.
    constexpr int CHUNK = 8192 / K;

    const int i = blockIdx.x * NT + threadIdx.x;
    const int row_bytes = (in_f / QK_NVFP4) * (int) sizeof(block_nvfp4);

    // GRID IS 2D: x over output rows, y over token chunks. A row-only grid launches out_f/NT CTAs —
    // 16 for out_f=4096 — on a 170-SM GPU, with each thread then walking every token serially. nsys
    // priced that at 756us average against a 40us bandwidth bound for the y traffic (60.5 MB at
    // pp1845), i.e. ~19x off and occupancy-bound, which made the correction cost MORE than the
    // 650us GEMM it corrects. Splitting the token axis across CTAs takes the same work to 464 CTAs.
    // Each CTA owns a disjoint set of (row, token) pairs, so no element's arithmetic or rounding
    // order changes — the exactness result is preserved by construction, not by luck.
    const int j_base = blockIdx.y * CHUNK;
    if (j_base >= n_tokens) { return; }

    __shared__ float a_sh[K * CHUNK];
    __shared__ int   c_sh[K];
    for (int s = threadIdx.x; s < K; s += NT) { c_sh[s] = topk_idx[s]; }
    __syncthreads();

    float wdeq[K];
#pragma unroll
    for (int s = 0; s < K; ++s) {
        wdeq[s] = 0.0f;
    }
    if (i < out_f) {
#pragma unroll
        for (int s = 0; s < K; ++s) {
            const int c = c_sh[s];
            if (c < 0) { continue; }
            const int blk = c / QK_NVFP4;
            const int sub = (c % QK_NVFP4) / QK_NVFP4_SUB;
            const int w   = c % QK_NVFP4_SUB;
            const block_nvfp4 * b =
                reinterpret_cast<const block_nvfp4 *>(W + (int64_t) i * row_bytes) + blk;
            const uint8_t byte = b->qs[sub * 8 + (w & 7)];
            const uint8_t nib  = (w < 8) ? (byte & 0x0f) : (byte >> 4);
            // GGUF NVFP4 dequant is EXACTLY kvalues_mxfp4[nib] * ue4m3_to_fp32(d) — the doubled
            // codebook (0,1,2,3,4,6,8,12) and the /2 already inside ggml_cuda_ue4m3_to_fp32
            // cancel, so no further factor belongs here. kvalues_mxfp4 carries its own sign in
            // codes 8..15. (An extra 0.5f here halved the correction; the residual kernel-check
            // arm caught it as rel == 0.52, exactly half the outlier contribution unrecovered.)
            wdeq[s] = (float) kvalues_mxfp4[nib] * ggml_cuda_ue4m3_to_fp32(b->d[sub]);
        }
    }

    {
        const int j0 = j_base;
        const int nj = min(CHUNK, n_tokens - j0);
        for (int t = threadIdx.x; t < K * nj; t += NT) {
            const int s  = t / nj;
            const int jj = t % nj;
            const int c  = c_sh[s];
            a_sh[s * CHUNK + jj] = c >= 0 ? act[(int64_t) (j0 + jj) * s11 + c] : 0.0f;
        }
        __syncthreads();
        if (i < out_f) {
            for (int jj = 0; jj < nj; ++jj) {
                const int64_t off = (int64_t) (j0 + jj) * out_f + i;
                // Groups of GROUP channels, accumulated INTO the y value in a register. The old tiled
                // kernel did `y[off] += out_scale * acc` once per group, so each group sum rounded
                // against the running y — including the large GEMM term. Summing the groups on their
                // own first and adding y at the end pairs different magnitudes and lands on different
                // last bits, which is exactly what turned two q9 cells DIVERGENT. Hoisting y into a
                // register keeps the addition sequence, the operands, and the rounding identical while
                // costing one load and one store instead of K/GROUP read-modify-writes. Each y element
                // is owned by exactly one thread for the whole kernel, so the hoist is safe.
                constexpr int GROUP = 16;
                float yv = y[off];
#pragma unroll
                for (int s0 = 0; s0 < K; s0 += GROUP) {
                    float acc = 0.0f;
#pragma unroll
                    for (int s = s0; s < s0 + GROUP && s < K; ++s) {
                        acc = fmaf(wdeq[s], a_sh[s * CHUNK + jj], acc);
                    }
                    yv += out_scale * acc;
                }
                y[off] = yv;
            }
        }
    }
}

// ======================= C-ABI host launcher =======================
extern "C" {

// Scratch layout, one allocation so the caller's scratch-slot cache (MMQ_ACT_SLOT) is untouched:
//   [0]                    block_fp4_mmq quantized activation stream
//   [+scale_off]           float  scale[n_tokens]     per-token row factor
//   [+chan_amax_off]       float  chan_amax[in_f]     per-channel amax (residual only)
//   [+chan_skip_off]       u8     chan_skip[in_f]     residual channel mask
//   [+topk_off]            int    topk_idx[MAX_K]     selected channel ids
static size_t mmq_nvfp4_qbytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    // s12 = ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) ints, *sizeof(int) bytes.
    // The full stream (1 channel/sample) = ne11 * ne10_padded/QK_K blocks of block_fp4_mmq.
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / QK_K);
    return (size_t) nblocks * sizeof(block_fp4_mmq);
}

static size_t mmq_nvfp4_scale_off(int in_f, int n_tokens) {
    // The scale array is read as float, so it starts 4-byte aligned (block_fp4_mmq is 272 bytes,
    // already a multiple of 4, but pad explicitly rather than rely on that).
    return GGML_PAD(mmq_nvfp4_qbytes(in_f, n_tokens), 16);
}
static size_t mmq_nvfp4_chan_amax_off(int in_f, int n_tokens) {
    return GGML_PAD(mmq_nvfp4_scale_off(in_f, n_tokens) + (size_t) n_tokens * sizeof(float), 16);
}
static size_t mmq_nvfp4_chan_skip_off(int in_f, int n_tokens) {
    return GGML_PAD(mmq_nvfp4_chan_amax_off(in_f, n_tokens) + (size_t) in_f * sizeof(float), 16);
}
static size_t mmq_nvfp4_topk_off(int in_f, int n_tokens) {
    return GGML_PAD(mmq_nvfp4_chan_skip_off(in_f, n_tokens) + (size_t) in_f, 16);
}

// Bytes needed for the activation scratch. Always sized for the residual arrays: they are a few KB
// against a multi-MB quant stream, and a single size keeps the caller's cached slot valid whether or
// not residual channels are enabled for a given call.
size_t memra_mmq_nvfp4_act_bytes(int in_f, int n_tokens) {
    return mmq_nvfp4_topk_off(in_f, n_tokens) + MMQ_MAX_RESIDUAL_K * sizeof(int);
}

static float * mmq_nvfp4_scale_ptr(void * s, int in_f, int n_tokens) {
    return (float *) ((char *) s + mmq_nvfp4_scale_off(in_f, n_tokens));
}

// Dynamic-smem byte count for the mul_mat_q kernel at mmq_x=MMQ_X (must opt-in via cudaFuncSetAttribute).
static size_t mmq_nvfp4_nbytes_shared() {
    const size_t nbs_ids = (size_t) MMQ_X * sizeof(int);
    const size_t nbs_x   = (size_t) MMQ_Y * MMQ_MMA_TILE_X_K_FP4 * sizeof(int);
    const size_t nbs_y   = (size_t) MMQ_X * sizeof(block_q8_1_mmq);
    const size_t pad     = (size_t) MMQ_NWARPS * MMQ_WARP_SIZE * sizeof(int);
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, pad);
}

// Run the NVFP4 W4A4 MMQ prefill GEMM. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
//   W_nvfp4_blocks : raw memra NVFP4 weight rows (block_nvfp4 = 36B blocks, in_f/64 per row).
//   act_f32        : f32 activation [n_tokens, in_f].
//   y              : f32 output [n_tokens, out_f].
//   act_scratch    : pre-allocated quant buffer, >= memra_mmq_nvfp4_act_bytes(in_f, n_tokens).
//   per_token_scale: 1 = two-level scaling (per-token row amax + per-sub-block UE4M3, the tuned
//                    path), 0 = the v1 sub-block-only quantizer, kept as the numeric oracle.
//   residual_k     : 0 = off. k>0 keeps the k largest-magnitude activation CHANNELS out of the
//                    quantized path and adds their exact f32 contribution back as a rank-k
//                    correction (requires per_token_scale=1). Clamped to MMQ_MAX_RESIDUAL_K.
// Returns 0 on success, else (1000 + cudaError).
int memra_mmq_nvfp4_ex2(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                   int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                   float out_scale, int per_token_scale, int residual_k) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    // ---- 1) quantize activation f32 -> block_fp4_mmq (quantize_mmq_fp4_cuda, NVFP4 branch) ----
    const int64_t ne10 = in_f;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t ne11 = n_tokens;
    const int64_t s11 = in_f; // row stride of act (contiguous [n_tokens, in_f])
    float * act_scale = per_token_scale ? mmq_nvfp4_scale_ptr(act_scratch, in_f, n_tokens) : nullptr;

    // The residual path rides on top of two-level scaling: it changes which values reach the
    // quantizer, not how they are scaled.
    const int rk = (per_token_scale && residual_k > 0)
                 ? (residual_k < MMQ_MAX_RESIDUAL_K ? residual_k : MMQ_MAX_RESIDUAL_K)
                 : 0;
    // Round up to the compile-time bucket the correction kernel is instantiated at. The padding
    // channels are -1 and contribute nothing, so a k of 20 costs a k=32 launch but is numerically
    // identical to k=20.
    const int rk_pad = rk <= 0 ? 0 : (rk <= 8 ? 8 : (rk <= 16 ? 16 : (rk <= 32 ? 32 : 64)));
    float   * chan_amax = (float *)   ((char *) act_scratch + mmq_nvfp4_chan_amax_off(in_f, n_tokens));
    uint8_t * chan_skip = (uint8_t *) ((char *) act_scratch + mmq_nvfp4_chan_skip_off(in_f, n_tokens));
    int     * topk_idx  = (int *)     ((char *) act_scratch + mmq_nvfp4_topk_off(in_f, n_tokens));

    if (rk > 0) {
        // Rank channels by amax over the WHOLE batch: the outlier set is a property of the weight
        // matrix's input space, so a per-token choice would make the correction inconsistent
        // between tokens of the same GEMM.
        {
            const int nt = 256;
            nvfp4_channel_amax_kernel<<<(in_f + nt - 1) / nt, nt, 0, st>>>(
                act_f32, chan_amax, n_tokens, in_f, s11);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { return 1000 + (int) e; }
        }
        {
            nvfp4_topk_channels_kernel<<<1, 256, 0, st>>>(
                chan_amax, chan_skip, topk_idx, in_f, rk, rk_pad);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { return 1000 + (int) e; }
        }
    }

    if (per_token_scale) {
        // One CTA per token: the level-1 amax reduction needs the whole row in one block.
        const dim3 block_size(MMQ_QUANT_BLOCK_SIZE, 1, 1);
        const dim3 num_blocks((unsigned) ne11, 1, 1);
        // 32-byte loads need the row base AND the stride 32-byte aligned. Activations come from
        // cudaMalloc (256-byte aligned), so only the stride has to be checked.
        const bool aligned8 = (s11 % 8 == 0) && (ne10 % 8 == 0);
        quantize_mmq_nvfp4_kernel<<<num_blocks, block_size, 0, st>>>(
            act_f32, act_scratch, act_scale, rk > 0 ? chan_skip : nullptr, ne10, s11, /*s02*/0,
            /*s03*/0, ne10_padded, ne11, /*ne2*/1, aligned8);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    } else {
        constexpr int nvfp4_block_size = 128;
        const int64_t block_num_y = (ne10_padded + (int64_t) QK_NVFP4_SUB * nvfp4_block_size - 1) /
                                     ((int64_t) QK_NVFP4_SUB * nvfp4_block_size);
        const dim3 block_size(nvfp4_block_size, 1, 1);
        const dim3 num_blocks((unsigned) ne11, (unsigned) block_num_y, 1);
        quantize_mmq_nvfp4_kernel_v1<<<num_blocks, block_size, 0, st>>>(
            act_f32, act_scratch, ne10, s11, /*s02*/0, /*s03*/0, ne10_padded, ne11, /*ne2*/1);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }

    // ---- 2) launch mul_mat_q NVFP4 (conventional xy-tiling) ----
    // mmq_args mapping (mmq.cu): ncols_x=in_f, nrows_x=out_f, ncols_dst=n_tokens,
    //   stride_row_x = blocks per weight row = in_f/QK_NVFP4, ncols_y = n_tokens,
    //   stride_col_dst = out_f (dst row stride), blocks_per_ne00 = in_f/QK_NVFP4.
    const int stride_row_x   = in_f / QK_NVFP4;          // block_nvfp4 per weight row
    const int blocks_per_ne00 = in_f / QK_NVFP4;
    const int stride_col_dst = out_f;
    const int ncols_y        = n_tokens;

    const int nty = (out_f    + MMQ_Y - 1) / MMQ_Y;
    const int ntx = (n_tokens + MMQ_X - 1) / MMQ_X;
    const dim3 grid((unsigned) nty, (unsigned) ntx, 1);
    const dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    const size_t smem = mmq_nvfp4_nbytes_shared();

    const bool need_check = (out_f % MMQ_Y) != 0;
    const int * y_q = (const int *) act_scratch;
    const char * W  = (const char *) W_nvfp4_blocks;

    if (need_check) {
        cudaFuncSetAttribute(mul_mat_q_nvfp4<MMQ_X, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_nvfp4<MMQ_X, true><<<grid, block, smem, st>>>(
            W, y_q, y, act_scale, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst,
            blocks_per_ne00, out_scale);
    } else {
        cudaFuncSetAttribute(mul_mat_q_nvfp4<MMQ_X, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_nvfp4<MMQ_X, false><<<grid, block, smem, st>>>(
            W, y_q, y, act_scale, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst,
            blocks_per_ne00, out_scale);
    }
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }

    // ---- 3) rank-k residual correction (must follow the GEMM: it accumulates into y) ----
    if (rk > 0) {
        constexpr int nt = 256;
        // 2D: x over output rows, y over token chunks. The chunk width is the kernel's CHUNK, which
        // is K-dependent (8192/K), so it is derived from the same expression here rather than
        // duplicated as a literal — a mismatch would silently skip or double-correct tokens.
        #define MMQ_RESIDUAL_LAUNCH(KK)                                                        \
            do {                                                                               \
                const dim3 nb((unsigned) ((out_f + nt - 1) / nt),                              \
                              (unsigned) ((n_tokens + (8192 / (KK)) - 1) / (8192 / (KK))), 1); \
                nvfp4_residual_correct_kernel<KK><<<nb, nt, 0, st>>>(                          \
                    W, act_f32, y, topk_idx, in_f, out_f, n_tokens, s11, out_scale);           \
            } while (0)
        switch (rk_pad) {
            case 8:  MMQ_RESIDUAL_LAUNCH(8);  break;
            case 16: MMQ_RESIDUAL_LAUNCH(16); break;
            case 32: MMQ_RESIDUAL_LAUNCH(32); break;
            default: MMQ_RESIDUAL_LAUNCH(64); break;
        }
        #undef MMQ_RESIDUAL_LAUNCH
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { return 1000 + (int) e; }
    }
    return 0;
}

int memra_mmq_nvfp4_ex(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                   int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                   float out_scale, int per_token_scale) {
    return memra_mmq_nvfp4_ex2(W_nvfp4_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch,
                               stream, out_scale, per_token_scale, /*residual_k=*/0);
}

// Default entry point: two-level scaling on, residual channels off unless MEMRA_MMQ_RESIDUAL_K asks
// (the Rust side reads the env var and passes k through ex2). per_token_scale=0 is reachable only
// through the _ex entry points, which is what the kernel-check accuracy arms use.
int memra_mmq_nvfp4(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                   int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                   float out_scale) {
    return memra_mmq_nvfp4_ex2(W_nvfp4_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch,
                               stream, out_scale, /*per_token_scale=*/1, /*residual_k=*/0);
}

} // extern "C"
