// mmq_fp4.cu — NVFP4 W4A4 block-scale MMQ prefill GEMM (vendored floor, ggml-decoupled, sm_120a).
//
// This is the 5150-pp512 kernel from llama.cpp brought into bw24 wholesale (the user's "copy the
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
// WEIGHT FORMAT (verified vs cu/cutlass_fp4_sm120.cu deinterleave): bw24's stored NVFP4 weight bytes
// are EXACTLY llama's block_nvfp4 = per row, in_f/64 blocks of 36 bytes = [4 UE4M3 scale bytes | 32
// e2m1 qs bytes], qs packed so element w of sub-block s lives in qs[s*8 + (w&7)] lo/hi at w<8/w>=8.
// That is the SAME packing quantize_mmq_nvfp4 emits for activations -> load_tiles_nvfp4_nvfp4 reads
// the raw weight bytes directly (pure u32 copy, no repack).
//
// C-ABI launcher: bw24_mmq_nvfp4(W_nvfp4, act_f32, y, in_f, out_f, n_tokens, stream). Internally
// quantizes act_f32 -> block_fp4_mmq, then launches mul_mat_q NVFP4. Compiled to a static lib (same as
// cutlass_fp4_sm120.cu), called from Rust via FFI.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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
                // FOLDED per-tensor NVFP4 macro-scale (was a separate scale_f32 launch + full
                // y round-trip per matmul, 4.2ms of pp512). scale==1.0 for non-scaled tensors.
                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l] * out_scale;
            }
        }
    }
}

// ======================= mul_mat_q_process_tile (mmq.cuh:3447, NVFP4) =======================
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void mul_mat_q_process_tile_nvfp4(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
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

    mmq_write_back_nvfp4<mmq_x, mmq_y, need_check>(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j, out_scale);
}

// ======================= mul_mat_q (conventional xy-tiling, NVFP4) =======================
// Grid: (nty = ceil(nrows_x/mmq_y), ntx = ceil(ncols_dst/mmq_x), 1). One tile per CTA, fixup=false.
// (2D plain GEMM: 1 channel, 1 sample -> all the stride_channel/sample/expert plumbing drops out.)
template <int mmq_x, bool need_check>
__launch_bounds__(MMQ_WARP_SIZE * MMQ_NWARPS, 1)
static __global__ void mul_mat_q_nvfp4(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
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
    const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;

    const int tile_x_max_i = nrows_x  - it * mmq_y - 1;
    const int tile_y_max_j = col_diff - jt * mmq_x - 1;

    const int offset_x = it * mmq_y * stride_row_x;

    mul_mat_q_process_tile_nvfp4<mmq_x, need_check>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, stride_row_x, ncols_y,
        stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00, out_scale);
}

// ======================= activation quantizer (quantize.cu:78 quantize_mmq_nvfp4) =======================
static __global__ void quantize_mmq_nvfp4_kernel(
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

// ======================= C-ABI host launcher =======================
extern "C" {

// Bytes needed for the quantized activation buffer (block_fp4_mmq stream): caller pre-allocs.
size_t bw24_mmq_nvfp4_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    // s12 = ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) ints, *sizeof(int) bytes.
    // The full stream (1 channel/sample) = ne11 * ne10_padded/QK_K blocks of block_fp4_mmq.
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / QK_K);
    return (size_t) nblocks * sizeof(block_fp4_mmq);
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
//   W_nvfp4_blocks : raw bw24 NVFP4 weight rows (block_nvfp4 = 36B blocks, in_f/64 per row).
//   act_f32        : f32 activation [n_tokens, in_f].
//   y              : f32 output [n_tokens, out_f].
//   act_scratch    : pre-allocated quant buffer, >= bw24_mmq_nvfp4_act_bytes(in_f, n_tokens).
// Returns 0 on success, else (1000 + cudaError).
int bw24_mmq_nvfp4(const void * W_nvfp4_blocks, const float * act_f32, float * y,
                   int in_f, int out_f, int n_tokens, void * act_scratch, void * stream,
                   float out_scale) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

    // ---- 1) quantize activation f32 -> block_fp4_mmq (quantize_mmq_fp4_cuda, NVFP4 branch) ----
    const int64_t ne10 = in_f;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t ne11 = n_tokens;
    const int64_t s11 = in_f; // row stride of act (contiguous [n_tokens, in_f])
    {
        constexpr int nvfp4_block_size = 128;
        const int64_t block_num_y = (ne10_padded + (int64_t) QK_NVFP4_SUB * nvfp4_block_size - 1) /
                                     ((int64_t) QK_NVFP4_SUB * nvfp4_block_size);
        const dim3 block_size(nvfp4_block_size, 1, 1);
        const dim3 num_blocks((unsigned) ne11, (unsigned) block_num_y, 1);
        quantize_mmq_nvfp4_kernel<<<num_blocks, block_size, 0, st>>>(
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
            W, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
    } else {
        cudaFuncSetAttribute(mul_mat_q_nvfp4<MMQ_X, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mul_mat_q_nvfp4<MMQ_X, false><<<grid, block, smem, st>>>(
            W, y_q, y, out_f, n_tokens, stride_row_x, ncols_y, stride_col_dst, blocks_per_ne00, out_scale);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { return 1000 + (int) e; }
    return 0;
}

} // extern "C"
