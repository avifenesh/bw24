// W4A8-FP8 prefill MMQ tile — R-B route (research/prefill-mxf8f6f4-design.md).
// Twin of mmq_nvfp4_w4a8.cu with three substitutions:
//   1. Weight tile: NVFP4 per-16 scales FOLD into values at load — byte = cvt_e4m3(kvalue[nib] * s16)
//      (weights stay 4-bit resident in VRAM; 8-bit e4m3 containers exist only in smem).
//   2. Activations: e4m3 bytes + f32 amax/448 scale per 32 (d4 layout kept from q8_1_mmq).
//   3. MMA: mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 (plain kind, no
//      scale regs, f32 accumulator) — 381 TF class vs the int8 path's ~219 (probe/mixed_f8f4_probe).
// Epilogue: sum_f32 * d_act only (weight scale folded into values).
// Seam: BW24_MMQ_F8F4=1 (default OFF until the full battery is green in-config).
//
// STATUS: piece 1 of the arc — activation quantize kernel + MMA primitive + ABI size helpers.
// The tile mainloop (load_tiles fold + vec_dot + writeback + pipeline) lands next.
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))
#define QK8_1 32
#define MATRIX_ROW_PADDING 512

// Activation block: 4x f32 scale (per 32) + 128 e4m3 bytes — byte-compatible footprint with
// block_q8_1_mmq so the y-tile smem math of the w4a8 kernel carries over unchanged.
struct block_e4m3_mmq {
    float   d4[4];
    uint8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_e4m3_mmq) == 144, "y-tile stride contract");

// f32x2 -> packed e4m3x2 (Blackwell cvt; round-to-nearest, saturate to +-448).
static __device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\n\tmov.b16 %0, t;\n}"
        : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

// Plain-kind f8f6f4 MMA, e4m3 x e4m3, f32 accumulate (CUTLASS SM120_16x8x32_TN form).
// A = 16x32 weights (e4m3 containers), B = 32x8 activations, D/C = f32 16x8.
static __device__ __forceinline__ void mma_f8f4_16x8x32(
        float * __restrict__ d, const uint32_t * __restrict__ a, const uint32_t * __restrict__ b) {
    asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Activation quantize: twin of quantize_mmq_q8_1_d4_kernel with the int8 grid swapped for e4m3.
// Same launch geometry, same k-major block order, same d4 scale slots.
static __global__ void quantize_mmq_e4m3_d4_kernel(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) { return; }
    const int64_t i1 = blockIdx.x;

    const float4 * x4 = (const float4 *) x;
    block_e4m3_mmq * y = (block_e4m3_mmq *) vy;

    const int64_t ib  = (i0 / (4 * QK8_1)) * ne1 + i1;
    const int64_t iqs = i0 % (4 * QK8_1);

    const float4 xi = i0 < ne00 ? x4[(i1 * s01 + i0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32 / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    // e4m3 top-of-grid is 448; d maps the block amax onto it (mirror of 127/amax for int8).
    const float d_inv = amax == 0.0f ? 0.0f : 448.0f / amax;
    const uint16_t q01 = cvt_e4m3x2(xi.x * d_inv, xi.y * d_inv);
    const uint16_t q23 = cvt_e4m3x2(xi.z * d_inv, xi.w * d_inv);

    uint32_t * yqs4 = (uint32_t *) y[ib].qs;
    yqs4[iqs / 4] = (uint32_t) q01 | ((uint32_t) q23 << 16);

    if (iqs % 32 != 0) { return; }
    y[ib].d4[iqs / 32] = amax == 0.0f ? 0.0f : amax / 448.0f;
}

extern "C" {

size_t bw24_mmq_nvfp4_f8f4_act_bytes(int in_f, int n_tokens) {
    const int64_t ne10_padded = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int64_t nblocks = (int64_t) n_tokens * (ne10_padded / (4 * QK8_1));
    // +MMQ_X blocks: the mul_mat_q y-tile loader always reads a FULL mmq_x-column tile; for the
    // final k-block with n_tokens % MMQ_X != 0 that read runs past the last real column. Padding
    // the scratch keeps the overread mapped (values are garbage; write-back drops j > j_max).
    return (size_t) (nblocks + 128) * sizeof(block_e4m3_mmq);   // 128 = max mmq_x tile width
}

void bw24_mmq_nvfp4_f8f4_quantize_act(
        const float * x, void * vy, int in_f, int n_tokens, int64_t s01, cudaStream_t st) {
    const int64_t ne0 = GGML_PAD((int64_t) in_f, MATRIX_ROW_PADDING);
    const int block_size = 128;
    const dim3 num_blocks((unsigned) n_tokens, (unsigned) ((ne0 / 4 + block_size - 1) / block_size), 1);
    quantize_mmq_e4m3_d4_kernel<<<num_blocks, block_size, 0, st>>>(x, vy, in_f, s01, ne0, n_tokens);
}

} // extern "C"
