// ARM B' (lane fp8-gemm-arm, 2026-08-03): DEVICE-SIDE dequant of block-128 FP8 (E4M3)
// weights straight into GGUF Q8_0 blocks, at model load.
//
//   in:  f8_weights [out_dim x in_dim] uint8 e4m3 codes, row-major (checkpoint order)
//        blk_scales [ceil(out_dim/128) x ceil(in_dim/128)] f32, row-major
//                   (the Fp8BlockScales / F8BlockGrid layout contract — scales[(o>>7)*cols
//                    + (e>>7)] scales element W[o][e])
//   out: Q8_0 blocks, 34 B each ({ half d; int8 qs[32] }), out_dim * (in_dim/32) of them,
//        row-major — byte-for-byte the layout the host re-encode
//        (memra_gguf::nvfp4_repack::f32_to_q8_0 over f8_deq_f32) produces.
//
// WHY: ARM A folds the 128x128 grid into ONE per-tensor scale (lossy where block dynamic
// range varies). This arm keeps every block's own scale and lands on Q8_0's FINER per-32
// grid, so precision is class-equal-or-better than the CPU path while costing one extra
// load-time device pass instead of a full host dequant + host re-encode. After this pass
// there is NO new GEMM code: the slab rides the existing Q8_0 MMQ/MMVQ path unchanged.
//
// BIT-PARITY CONTRACT (the kernel-check arm asserts byte-equality against the host path):
//   * e4m3 decode mirrors nvfp4_repack::fp8_e4m3_to_f32 EXACTLY, including its modelopt
//     convention that the NaN code (magnitude 0x7F) decodes to 0.0. The hardware
//     __nv_cvt_fp8_to_halfraw intrinsic returns NaN there, so this file decodes with the
//     same closed-form bit math the host uses (exact in f32, no table lookup, no
//     constant-memory broadcast serialization). A 256-entry constant LUT is kept as the
//     portable fallback (-DMEMRA_FP8_BLK_LUT).
//   * d = amax/127 in f32, stored as round-to-nearest-even f16 (__float2half_rn ==
//     nvfp4_repack::f32_to_f16_bits).
//   * q = rintf(x * id) — round-to-nearest-EVEN, matching Rust `round_ties_even()`.
//     NOT roundf() (ties-away-from-zero; disagrees on exact .5 products).
//   * id = 1/d from the f32 d (NOT the f16-rounded d), same as the host.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define QK8_0 32
#define Q8_0_BYTES 34 // sizeof(half) + 32 * sizeof(int8_t)

#ifndef MEMRA_FP8_BLK_LUT
#define MEMRA_FP8_BLK_USE_BITMATH 1
#endif

// Portable fallback path only (see header note). Seeded host-side, uploaded once per call.
__constant__ float c_e4m3_lut[256];

// e4m3 code -> f32, bit-identical to nvfp4_repack::fp8_e4m3_to_f32.
static __device__ __host__ __forceinline__ float memra_e4m3_to_f32(uint8_t x) {
    const uint32_t mag = (uint32_t)x & 0x7Fu;
    if (mag == 0x7Fu) {
        return 0.0f; // NaN code -> 0.0 (modelopt convention; the host does the same)
    }
    const int exp = (int)((mag >> 3) & 0xFu);
    const float man = (float)(mag & 0x7u);
    // Both arms are exact in f32: man*0.125 scales a small integer by a power of two,
    // and ldexpf is an exponent adjust.
    const float raw = (exp == 0) ? (man * 0.125f) * 0.015625f            // (man/8) * 2^-6
                                 : ldexpf(1.0f + man * 0.125f, exp - 7); // (1+man/8)*2^(e-7)
    return (x & 0x80u) ? -raw : raw;
}

static __device__ __forceinline__ float e4m3_decode(uint8_t code) {
#ifdef MEMRA_FP8_BLK_USE_BITMATH
    return memra_e4m3_to_f32(code);
#else
    return c_e4m3_lut[code];
#endif
}

// ONE WARP = one 128-element row segment (4 Q8_0 blocks); 4 warps (4 segments) per CTA. Each
// thread owns FOUR consecutive elements and loads them as a single `uchar4`.
//
// WHY THIS SHAPE (lane/fp8-blk128-decode, 2026-08-05 — the prefill-regression fix). The original
// mapping below (`_scalar`, kept only as the portable/reference form) gave every thread ONE byte:
// per 128 bytes moved it issued 128 LDG.8 + 128 STS.8 and 128*5 = 640 warp shuffles. On the 27B
// block-128 checkpoint that made `try_e4m3_blk_prefill`'s per-call dequant the single largest
// prefill cost — nsys, pp512, 27B: 66.5 ms per prefill pass across 208 projections, moving
// 6.88 GB of e4m3 + 7.31 GB of Q8_0 = 14.19 GB at ~213 GB/s effective, ~8x off this card's DRAM
// roofline. Nothing there is bandwidth: it is one memory instruction per BYTE with no ILP, so the
// warp never has more than one outstanding load per lane. e2e cost: pp512 1541.1 -> 1329.2 tok/s.
//
// This form issues, per 128 bytes: 32 LDG.32 + 64 STS.16 + 32*3 = 96 shuffles. 4x fewer loads,
// 4 independent decodes in flight per lane, and the per-32 amax collapses to a 4-way serial max
// plus a 3-step butterfly INSIDE a group of 8 lanes.
//
// ARITHMETIC IS UNCHANGED, ELEMENT FOR ELEMENT: x = decode(code) * bscale; amax = max|x| over the
// same 32 elements (max is exact and associative, so the reduce order cannot move a bit);
// d = amax/127, id = 1/d from the f32 d, q = rintf(x*id). Only the thread->element map moved, so
// the output slab stays BYTE-IDENTICAL and the [fp8-blk-gpu] host-reference gate is the proof.
//
// GROUP MASK, not 0xffffffff: a group of 8 lanes is exactly one Q8_0 block (8 * 4 elements), so
// the ragged-in_dim early return below is group-uniform and whole groups leave the warp. A
// surviving lane's shuffle partners (laneMask 4/2/1 never crosses a group boundary) are exactly
// its own group, so the mask must name those 8 lanes and only those.
__global__ __launch_bounds__(128) void fp8_blk_dequant_q8_0_kernel(
    const uint8_t *__restrict__ f8_weights,
    const float *__restrict__ blk_scales,
    uint8_t *__restrict__ out_q8,
    const long long nseg, // out_dim * scale_cols
    const int in_dim,
    const int scale_cols,
    const int blocks_per_row) {
    const long long sid = (long long)blockIdx.x * 4LL + (long long)(threadIdx.x >> 5);
    if (sid >= nseg) {
        return;
    }
    const int row = (int)(sid / (long long)scale_cols);
    const int seg = (int)(sid % (long long)scale_cols);

    const int lane = threadIdx.x & 31;
    const int grp = lane >> 3; // which of this segment's 4 Q8_0 blocks
    const int sub = lane & 7;  // 4 elements each -> 32 per group

    const int qb = seg * 4 + grp;
    if (qb >= blocks_per_row) {
        return; // group-uniform: 8 lanes at a time, never splits a group
    }

    const float bscale = blk_scales[(size_t)(row >> 7) * (size_t)scale_cols + (size_t)seg];

    // 4-byte aligned: in_dim % 32 == 0 makes every row start 32-byte aligned from a cudaMalloc'd
    // base, and `col` is a multiple of 4.
    const int col = seg * 128 + grp * QK8_0 + sub * 4;
    const uchar4 c = *(const uchar4 *)(f8_weights + (size_t)row * (size_t)in_dim + (size_t)col);
    const float x0 = e4m3_decode(c.x) * bscale;
    const float x1 = e4m3_decode(c.y) * bscale;
    const float x2 = e4m3_decode(c.z) * bscale;
    const float x3 = e4m3_decode(c.w) * bscale;

    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    const unsigned gmask = 0xFFu << (grp * 8);
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, off, 32));
    }

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    // rintf == round-to-nearest-even == Rust round_ties_even (the host re-encode's rounding).
    const uint32_t q0 = (uint32_t)(uint8_t)(int8_t)(int)rintf(x0 * id);
    const uint32_t q1 = (uint32_t)(uint8_t)(int8_t)(int)rintf(x1 * id);
    const uint32_t q2 = (uint32_t)(uint8_t)(int8_t)(int)rintf(x2 * id);
    const uint32_t q3 = (uint32_t)(uint8_t)(int8_t)(int)rintf(x3 * id);

    uint8_t *dst = out_q8 + ((size_t)row * (size_t)blocks_per_row + (size_t)qb) * Q8_0_BYTES;
    if (sub == 0) {
        // ONE aligned u16 store, NOT two byte stores — see the nvcc 13.0.88 miscompile note on the
        // scalar kernel below. `dst` walks in 34-byte strides from a cudaMalloc'd base, so it is
        // always even, and `dst + 2 + 4*sub` is therefore even too.
        *(uint16_t *)dst = __half_as_ushort(__float2half_rn(d));
    }
    uint16_t *qd = (uint16_t *)(dst + 2 + sub * 4);
    qd[0] = (uint16_t)(q0 | (q1 << 8));
    qd[1] = (uint16_t)(q2 | (q3 << 8));
}

// SCALAR reference form: one thread per element, one CTA per 128-element segment. Superseded by the
// vector kernel above (which is byte-identical and ~4x cheaper in memory instructions) and kept
// ONLY as the -DMEMRA_FP8_BLK_LUT portable/debug companion — a form whose per-element mapping is
// trivially readable next to the host reference. Not reachable in a default build.
__global__ __launch_bounds__(128) void fp8_blk_dequant_q8_0_scalar_kernel(
    const uint8_t *__restrict__ f8_weights,
    const float *__restrict__ blk_scales,
    uint8_t *__restrict__ out_q8,
    const int out_dim,
    const int in_dim,
    const int scale_cols,
    const int blocks_per_row) {
    const long long bid = (long long)blockIdx.x;
    const int row = (int)(bid / (long long)scale_cols);
    const int seg = (int)(bid % (long long)scale_cols);
    if (row >= out_dim) {
        return;
    }

    const float bscale = blk_scales[(size_t)(row >> 7) * (size_t)scale_cols + (size_t)seg];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Q8_0 block index inside this row. A ragged in_dim (multiple of 32 but not of 128)
    // leaves trailing warps of the last segment with no block to write — the predicate is
    // warp-uniform, so the early exit never splits a warp and the shuffle reduce below
    // keeps its full 32-lane mask.
    const int qb = seg * 4 + warp;
    if (qb >= blocks_per_row) {
        return;
    }

    const int col = seg * 128 + warp * QK8_0 + lane;
    const float x = e4m3_decode(f8_weights[(size_t)row * (size_t)in_dim + (size_t)col]) * bscale;

    // Per-32 amax: warp shuffle butterfly over the 32 lanes of this warp.
    float amax = fabsf(x);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
    }

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    // rintf == round-to-nearest-even == Rust round_ties_even (the host re-encode's rounding).
    const int qi = (int)rintf(x * id);

    uint8_t *dst = out_q8 + ((size_t)row * (size_t)blocks_per_row + (size_t)qb) * Q8_0_BYTES;
    if (lane == 0) {
        // ONE aligned u16 store, NOT two byte stores. nvcc 13.0.88 (CUDA 13.0, sm_120a)
        // miscompiles the byte-split form: dst[0] lands as 0x00 (isolated 40-line repro,
        // 8/8 blocks; nvcc 13.1 compiles the same source correctly). Found by the
        // kernel-check [fp8-blk-gpu] arm failing on the vast 2x5090 box, 2026-08-04 —
        // research/fp8ship-20260804/official/BLK-GPU-MISCOMPILE.md. The alignment is
        // guaranteed: dst walks in 34-byte strides from a cudaMalloc'd base, so dst is
        // always even.
        *(uint16_t *)dst = __half_as_ushort(__float2half_rn(d));
    }
    dst[2 + lane] = (uint8_t)(int8_t)qi;
}

extern "C" {

// Byte size of the Q8_0 slab this pass writes for an [out_dim x in_dim] weight. 0 = bad dims.
size_t memra_fp8_blk_q8_0_bytes(int out_dim, int in_dim) {
    if (out_dim <= 0 || in_dim <= 0 || (in_dim % QK8_0) != 0) {
        return 0;
    }
    return (size_t)out_dim * (size_t)(in_dim / QK8_0) * (size_t)Q8_0_BYTES;
}

// Error bands: 1 = bad dims (in_dim must be a multiple of 32), otherwise a cudaError_t.
int memra_fp8_blk_dequant_q8_0(
    const void *f8_weights,
    const float *blk_scales,
    void *out_q8,
    int out_dim,
    int in_dim,
    void *stream) {
    if (out_dim <= 0 || in_dim <= 0 || (in_dim % QK8_0) != 0) {
        return 1;
    }
    cudaStream_t st = (cudaStream_t)stream;
#ifndef MEMRA_FP8_BLK_USE_BITMATH
    {
        float lut[256];
        for (int i = 0; i < 256; ++i) {
            lut[i] = memra_e4m3_to_f32((uint8_t)i);
        }
        cudaError_t lrc = cudaMemcpyToSymbolAsync(
            c_e4m3_lut, lut, sizeof(lut), 0, cudaMemcpyHostToDevice, st);
        if (lrc != cudaSuccess) {
            return (int)lrc;
        }
        cudaStreamSynchronize(st); // `lut` is a stack buffer — the async copy must land first
    }
#endif
    const int scale_cols = (in_dim + 127) / 128;
    const int blocks_per_row = in_dim / QK8_0;
    const long long nseg = (long long)out_dim * (long long)scale_cols;

#ifdef MEMRA_FP8_BLK_USE_BITMATH
    // Vector form: one WARP per 128-element segment, 4 segments (4 warps) per CTA.
    const long long nctas = (nseg + 3LL) / 4LL;
    fp8_blk_dequant_q8_0_kernel<<<(unsigned)nctas, 128, 0, st>>>(
        (const uint8_t *)f8_weights, blk_scales, (uint8_t *)out_q8,
        nseg, in_dim, scale_cols, blocks_per_row);
#else
    // Portable/debug LUT build keeps the scalar reference mapping (one CTA per segment).
    fp8_blk_dequant_q8_0_scalar_kernel<<<(unsigned)nseg, 128, 0, st>>>(
        (const uint8_t *)f8_weights, blk_scales, (uint8_t *)out_q8,
        out_dim, in_dim, scale_cols, blocks_per_row);
#endif
    return (int)cudaGetLastError();
}

} // extern "C"
