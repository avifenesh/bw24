// mma_tile.cuh — PORT of llama.cpp ggml/src/ggml-cuda/mma.cuh, concretized for
// bw24 (sm_120a / consumer Blackwell, head_dim=256, bf16 inputs, f32 accumulate).
//
// SOURCE (verbatim port, no ggml templates):
//   tile lane maps : mma.cuh non-AMD DATA_LAYOUT_I_MAJOR specializations
//                    float    tile<I,J,float>        : lines 239-271
//                    bf16/h2  tile<I,J,nv_bfloat162> : lines 479-503 (== half2 400-428)
//   load_ldmatrix  : mma.cuh lines 829-859 (tile<16,8>, x4, per-lane addr 834)
//   load_ldmatrix_trans : mma.cuh lines 884-918 (x4.trans, reg reorder 893)
//   mma m16n8k16 bf16 : mma.cuh lines 1181-1194 (.f32.bf16.bf16.f32)
//
// CRITICAL FIX (review C1): every lane supplies its OWN row address.
//   per-lane offset = (lane % t.I)*stride + (lane / t.I)*(t.J/2)
// This is folded INSIDE every loader; callers pass the TILE BASE pointer only.
//
// Physical-element note (mma.cuh:12): for the bf16 tile, T = nv_bfloat162 (one u32
// = 2 packed bf16). J is counted in 32-bit physical elements, NOT logical bf16.
// So tile<16,8,bf16> spans 16 rows x (8 physical u32 = 16 logical bf16) and has
// ne = 16*8/32 = 4 u32 regs/lane. tile<8,8,bf16> -> ne = 2 u32 regs/lane.
// The f32 accumulator tile<16,8,float> has ne = 16*8/32 = 4 f32 regs/lane.

#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace bw {

// ============================================================================
//  TILE STRUCTS (concretized — only the shapes bw24 needs)
//  Layout convention (mma.cuh:7-13): A is row-major MxK, B is col-major KxN,
//  C is col-major MxN; all stored as a row-major IxJ tile of physical u32.
// ============================================================================

// f32 accumulator / C-D operand of m16n8k16.  ne = 4 f32 / lane.
// (mma.cuh tile<16,8,float> get_i/get_j, lines 245/262)
struct CTile_m16n8_f32 {
    float x[4];
    // logical row in [0,16), logical col in [0,8)
    static __device__ __forceinline__ int get_i(int l) { return ((l / 2) * 8) + (threadIdx.x / 4); }
    static __device__ __forceinline__ int get_j(int l) { return ((threadIdx.x % 4) * 2) + (l % 2); }
};

// bf16 A operand of m16n8k16 : logical 16 rows x 16 contraction(k).  ne = 4 u32 / lane.
// (mma.cuh tile<16,8,nv_bfloat162> get_i/get_j, lines 485/498 — J phys = 8)
struct ATile_m16k16_bf16 {
    nv_bfloat162 x[4];   // 4 u32, each = 2 packed bf16
    // get_i: logical row in [0,16) for the lth packed element
    static __device__ __forceinline__ int get_i(int l) { return ((l % 2) * 8) + (threadIdx.x / 4); }
    // get_j: logical *u32* col in [0,8) -> logical bf16 k = 2*get_j (+0/+1 within the pair)
    static __device__ __forceinline__ int get_j(int l) { return ((l / 2) * 4) + (threadIdx.x % 4); }
};

// bf16 B operand of m16n8k16 : logical 8 cols(n) x 16 contraction(k).  ne = 2 u32 / lane.
// (mma.cuh tile<8,8,nv_bfloat162> get_i/get_j, lines 481/493 — J phys = 8)
struct BTile_n8k16_bf16 {
    nv_bfloat162 x[2];   // 2 u32
    // get_i: logical n (the matrix is read col-major: I direction == column n) in [0,8)
    static __device__ __forceinline__ int get_i(int l) { return threadIdx.x / 4; }
    // get_j: logical *u32* k-pair index -> logical bf16 k = 2*get_j (+0/+1)
    static __device__ __forceinline__ int get_j(int l) { return (l * 4) + (threadIdx.x % 4); }
};

// ============================================================================
//  ldmatrix LOADERS  (the C1 fix: per-LANE address baked in)
//  `xs0` points at the (0,0) element of the tile in shared memory.
//  `stride` is the row stride of that smem region in elements of T-base type.
//  For bf16 tiles, xs0/stride are in **logical bf16** elements (we cast to int*
//  exactly as mma.cuh does: an int == one u32 == one bf16 pair).
// ============================================================================

// A-operand load: ldmatrix.x4 (4 u32/lane).  PORT of mma.cuh:829-859 (addr 834).
// xs0: const __nv_bfloat16* at tile origin; stride: bf16 elems per row.
static __device__ __forceinline__ void load_ldmatrix_A(
        ATile_m16k16_bf16 & t, const __nv_bfloat16 * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    // t.I = 16, t.J(phys u32) = 8 -> t.J/2 = 4. mma.cuh:834 exactly.
    const int * xs = (const int *) xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// B-operand load: ldmatrix.x2 (2 u32/lane).  PORT of the tile<8,8> path.
// Per mma.cuh:790 the tile<8,8> per-lane addr is (lane%8)*stride + ((lane/8)*(8/2))%8.
// For an 8x8 b16 tile that loads one m8n8 fragment; here we load the n8k16 B operand
// as a single x4 over the 16-wide k by treating it as two stacked m8n8 — but the
// proven minimal form is x2 with the 8x8 addressing repeated, matching how
// fattn drives the n8 B operand. We use x4 to fill both k-halves in one shot
// (lane%16 spans the 16 k-rows ldmatrix needs for k=16).  See note below.
static __device__ __forceinline__ void load_ldmatrix_B(
        BTile_n8k16_bf16 & t, const __nv_bfloat16 * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    // B is col-major KxN read as I=8(n) rows of J(phys)=8.  Treat the 16-row k
    // address like the x4 loader's first two register slots: per-lane row = lane%8
    // over the n8 rows, k-pair selected by lane/8.  mma.cuh:790 form, t.I=8.
    const int * xs = (const int *) xs0 + (threadIdx.x % 8) * stride + ((threadIdx.x / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

// V^T load for PV: ldmatrix.x4.trans with the register REORDER (xi[0],xi[2],xi[1],xi[3]).
// PORT of mma.cuh:884-918 (addr 891, reorder 893).  Fills an A-shaped 16x16 tile
// transposed from a [k=16 rows][d cols] smem region — used to feed V as the
// m16n8k16 B operand (V supplied as [n=D_chunk, k=keys] = V^T).
static __device__ __forceinline__ void load_ldmatrix_A_trans(
        ATile_m16k16_bf16 & t, const __nv_bfloat16 * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    // identical per-lane addr to the non-trans x4 loader (mma.cuh:891 == :834)
    const int * xs = (const int *) xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    // NOTE the output operand order: %1 and %2 are SWAPPED vs the source register
    // index — this is the xi[0],xi[2],xi[1],xi[3] reorder from mma.cuh:893.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(xs));
}

// ============================================================================
//  MMA  m16n8k16  .f32.bf16.bf16.f32   (PORT of mma.cuh:1181-1194)
//  D[16x8 f32] += A[16x16 bf16] @ B[8x16 bf16]^T  (.row.col)
// ============================================================================
static __device__ __forceinline__ void mma_m16n8k16_bf16(
        CTile_m16n8_f32 & D, const ATile_m16k16_bf16 & A, const BTile_n8k16_bf16 & B) {
    const int * Axi = (const int *) A.x;   // 4 regs
    const int * Bxi = (const int *) B.x;   // 2 regs
    float     * Dxi = (float     *) D.x;   // 4 f32
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
}

} // namespace bw
