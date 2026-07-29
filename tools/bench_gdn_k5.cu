// GDN K5 (gdn_chunk_output) mma dev harness (task 9, follows the proven K4 arc/playbook).
// K5 = 62.9us x 24/prime = 1.5ms (nsys 2026-07-26); two GEMM phases (o_inter = q.St gated
// by b_j, then += P.Y with P's upper triangle zero by the K2 contract). Chunk-parallel
// (grid NC x H x C/32) — no serial state, output-path only (less numerically sensitive
// than K4's recurrent state; same gated chunked config).
//
// Build (box): nvcc -O3 -arch=sm_90a -o /tmp/gdnk5 tools/bench_gdn_k5.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
using std::uint32_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#define GDN_D 128

extern "C" __global__ void gdn_chunk_output_f32(
        const float* __restrict__ q, const float* __restrict__ gcum,
        const float* __restrict__ P, const float* __restrict__ Y,
        const float* __restrict__ Ssnap, float* __restrict__ o,
        int H, int T, int C, float scale) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int j0 = blockIdx.z * 32;
    if (j0 >= Cc) return;                      // uniform per block (tail chunk)
    __shared__ float ts[32][GDN_D];            // phase 1: St sub-tile; phase 2: Y sub-tile
    __shared__ float qs[32][GDN_D];            // the block's q rows (zero-padded tail)
    const int tid = threadIdx.x;
    const int cg = tid % 32, rg = tid / 32;    // 4x4 register tile: cols c0=4cg, rows r0=4rg
    const int c0 = cg * 4, r0 = rg * 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;
    for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
        int r = idx / GDN_D, d = idx % GDN_D;
        qs[r][d] = (r < jn) ? q[((size_t)(t0 + j0 + r) * H + h) * GDN_D + d] : 0.0f;
    }
    float acc[4][4];
    #pragma unroll
    for (int rr = 0; rr < 4; rr++)
        #pragma unroll
        for (int cc = 0; cc < 4; cc++) acc[rr][cc] = 0.0f;
    // phase 1: inter-chunk term q_j . S_c[:,col] (4 rows x 4 cols per thread; one float4
    // ts read + 4 qs broadcasts feed 16 FMAs — the m-outer form was smem-issue-bound)
    const float* st = Ssnap + ((size_t)c * H + h) * GDN_D * GDN_D;
    for (int it0 = 0; it0 < GDN_D; it0 += 32) {
        __syncthreads();
        for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
            int r = idx / GDN_D, d = idx % GDN_D;
            ts[r][d] = st[(size_t)(it0 + r) * GDN_D + d];
        }
        __syncthreads();
        #pragma unroll 4
        for (int ii = 0; ii < 32; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float qv = qs[r0 + rr][it0 + ii];
                acc[rr][0] += qv * tv.x; acc[rr][1] += qv * tv.y;
                acc[rr][2] += qv * tv.z; acc[rr][3] += qv * tv.w;
            }
        }
    }
    // gate the inter-chunk term by b_j before the intra-chunk add
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int jj = r0 + rr;
        if (jj < jn) {
            const float b = expf(gcum[(size_t)(t0 + j0 + jj) * H + h]);
            #pragma unroll
            for (int cc = 0; cc < 4; cc++) acc[rr][cc] *= b;
        }
    }
    // phase 2: intra-chunk term P @ Y (rectangular: P upper triangle is ZERO by the K2
    // contract, so no per-row bounds in the inner loop)
    for (int it0 = 0; it0 < jend; it0 += 32) {
        const int itn = min(32, jend - it0);
        __syncthreads();
        for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
            int r = idx / GDN_D, d = idx % GDN_D;
            ts[r][d] = (r < itn) ? Y[(((size_t)c * H + h) * C + it0 + r) * GDN_D + d] : 0.0f;
        }
        __syncthreads();
        const float* P0 = P + (((size_t)c * H + h) * C + j0 + r0) * C + it0;
        for (int ii = 0; ii < itn; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float pv = (r0 + rr < jn) ? P0[(size_t)rr * C + ii] : 0.0f;
                acc[rr][0] += pv * tv.x; acc[rr][1] += pv * tv.y;
                acc[rr][2] += pv * tv.z; acc[rr][3] += pv * tv.w;
            }
        }
    }
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int j = j0 + r0 + rr;
        if (j < jend) {
            const float4 ov = make_float4(scale * acc[rr][0], scale * acc[rr][1],
                                          scale * acc[rr][2], scale * acc[rr][3]);
            *reinterpret_cast<float4*>(&o[((size_t)(t0 + j) * H + h) * GDN_D + c0]) = ov;
        }
    }
}


// ---- mma helpers (K4-proven set) ----
struct CTile { float x[4]; };
struct ATile { nv_bfloat162 x[4]; };
struct BTile { nv_bfloat162 x[2]; };
static __device__ __forceinline__ void ld_A(ATile& t, const __nv_bfloat16* xs0, int stride_pairs, int lane){
    int* xi = (int*)t.x;
    const uint32_t* xs = (const uint32_t*)xs0 + (lane % 16)*stride_pairs + (lane / 16)*4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[1]),"=r"(xi[2]),"=r"(xi[3]) : "r"(addr));
}
static __device__ __forceinline__ void ld_A_trans(ATile& t, const __nv_bfloat16* xs0, int stride_pairs, int lane){
    int* xi = (int*)t.x;
    const uint32_t* xs = (const uint32_t*)xs0 + (lane % 16)*stride_pairs + (lane / 16)*4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[2]),"=r"(xi[1]),"=r"(xi[3]) : "r"(addr));
}
static __device__ __forceinline__ void mma_bf16(CTile& D, const ATile& A, const BTile& B){
    const int* Ax=(const int*)A.x; const int* Bx=(const int*)B.x; float* Dx=D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(Dx[0]),"+f"(Dx[1]),"+f"(Dx[2]),"+f"(Dx[3])
        : "r"(Ax[0]),"r"(Ax[1]),"r"(Ax[2]),"r"(Ax[3]),"r"(Bx[0]),"r"(Bx[1]));
}

// ---- v1: mma form. Output tile 32j x 128col per CTA, 8 warps x 4 CTiles (K4 step-B shape):
// warp w: mh = w/4 (j 16-half), nq = w%4 (col 32-quarter). Phase 1 A=qs[j][i] (bf16 stage),
// B=St[i][col] via ld_A_trans per 32-i sub-tile. b_j gate = per-fragment-row scale between
// phases. Phase 2 A=P[j][i'] (bf16 stage, upper-zero holds), B=Y[i'][col] via ld_A_trans.
extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_output_mma(const float* __restrict__ q, const float* __restrict__ gcum,
                     const float* __restrict__ P, const float* __restrict__ Y,
                     const float* __restrict__ Ssnap, float* __restrict__ o,
                     int H, int T, int C, float scale) {
    constexpr int D = GDN_D;
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int j0 = blockIdx.z * 32;
    if (j0 >= Cc) return;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / 4, nq = warp % 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;

    __shared__ __nv_bfloat16 qs[32 * D];            // A: q rows (zero-padded)
    __shared__ __nv_bfloat16 ts[32 * D];            // B source: St / Y sub-tiles ([k][n])
    __shared__ __nv_bfloat16 ps[32 * 40];           // phase-2 A: P tile [j][i'], stride 40

    for (int idx = tid; idx < 32 * D; idx += 256) {
        int r = idx / D, d = idx % D;
        float v = (r < jn) ? q[((size_t)(t0 + j0 + r) * H + h) * D + d] : 0.0f;
        qs[r * D + d] = __float2bfloat16(v);
    }

    CTile acc[4];
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) { acc[t4].x[0]=acc[t4].x[1]=acc[t4].x[2]=acc[t4].x[3]=0.0f; }

    // phase 1: q[32j x 128i] . St[128i x 128col] over 32-i sub-tiles
    const float* st = Ssnap + ((size_t)c * H + h) * D * D;
    for (int it0 = 0; it0 < D; it0 += 32) {
        __syncthreads();
        for (int idx = tid; idx < 32 * D; idx += 256) {
            int r = idx / D, d = idx % D;
            ts[r * D + d] = __float2bfloat16(st[(size_t)(it0 + r) * D + d]);
        }
        __syncthreads();
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, qs + (mh * 16) * D + it0 + k16 * 16, D / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;   // ts rows i (k-dim), cols col (n-dim): [k][n] -> transpose-load
                ld_A_trans(Bt, ts + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(acc[p2 * 2 + 0], A, Blo);
                mma_bf16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
    }
    // b_j gate: fragment rows are j = mh*16 + fr (+8 for l>=2)
    {
        float b_lo = 0.0f, b_hi = 0.0f;
        int j_lo = mh * 16 + fr, j_hi = j_lo + 8;
        if (j_lo < jn) b_lo = expf(gcum[(size_t)(t0 + j0 + j_lo) * H + h]);
        if (j_hi < jn) b_hi = expf(gcum[(size_t)(t0 + j0 + j_hi) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            acc[t4].x[0] *= b_lo; acc[t4].x[1] *= b_lo;
            acc[t4].x[2] *= b_hi; acc[t4].x[3] *= b_hi;
        }
    }
    // phase 2: P[32j x itn] . Y[itn x 128col] over 32-i' sub-tiles (P upper triangle zero)
    for (int it0 = 0; it0 < jend; it0 += 32) {
        const int itn = min(32, jend - it0);
        __syncthreads();
        for (int idx = tid; idx < 32 * D; idx += 256) {
            int r = idx / D, d = idx % D;
            float v = (r < itn) ? Y[(((size_t)c * H + h) * C + it0 + r) * D + d] : 0.0f;
            ts[r * D + d] = __float2bfloat16(v);
        }
        for (int idx = tid; idx < 32 * 32; idx += 256) {
            int r = idx / 32, i = idx % 32;
            float v = (r < jn && i < itn && j0 + r < Cc)
                ? P[(((size_t)c * H + h) * C + j0 + r) * C + it0 + i] : 0.0f;
            ps[r * 40 + i] = __float2bfloat16(v);
        }
        __syncthreads();
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, ps + (mh * 16) * 40 + k16 * 16, 20, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, ts + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(acc[p2 * 2 + 0], A, Blo);
                mma_bf16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
    }
    // epilogue: o[t0+j][col] = scale * acc
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int j = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int col = nq * 32 + t4 * 8 + fc + (l & 1);
            if (j < jn)
                o[((size_t)(t0 + j0 + j) * H + h) * D + col] = scale * acc[t4].x[l];
        }
    }
}


// ---- v2: coupled form — St and Y arrive as BF16 (written by K4-mma directly; identical
// numerics to v1 which rounds them anyway) through a cp.async ring. P stays f32->bf16.
__device__ __forceinline__ void cp_async16_k5(void* dst, const void* src, int src_size) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit_k5() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait_k5() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_output_mma2(const float* __restrict__ q, const float* __restrict__ gcum,
                      const float* __restrict__ P, const __nv_bfloat16* __restrict__ Yb,
                      const __nv_bfloat16* __restrict__ Stb, float* __restrict__ o,
                      int H, int T, int C, float scale) {
    constexpr int D = GDN_D;
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int j0 = blockIdx.z * 32;
    if (j0 >= Cc) return;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / 4, nq = warp % 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;

    __shared__ __nv_bfloat16 qs[32 * D];
    __shared__ __nv_bfloat16 ts[2][32 * D];    // double-buffered B sub-tiles (bf16, 8KB each)
    __shared__ __nv_bfloat16 ps[32 * 40];

    // stage sub-tile: St rows it0..+31 (phase 1) or Y rows (phase 2); 32 rows x 128 bf16 = 16B x 16/row
    const __nv_bfloat16* stb = Stb + ((size_t)c * H + h) * D * D;
    #define K5_STAGE_ST(it0_, buf_) do {                                                  \
        for (int idx = tid; idx < 32 * (D / 8); idx += 256) {                             \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_k5(&ts[buf_][r * D + seg * 8],                                     \
                          stb + (size_t)((it0_) + r) * D + seg * 8, 16);                  \
        }                                                                                 \
        cp_commit_k5();                                                                   \
    } while (0)
    #define K5_STAGE_Y(it0_, itn_, buf_) do {                                             \
        for (int idx = tid; idx < 32 * (D / 8); idx += 256) {                             \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_k5(&ts[buf_][r * D + seg * 8],                                     \
                          Yb + (((size_t)c * H + h) * C + (it0_) + r) * D + seg * 8,      \
                          (r < (itn_)) ? 16 : 0);                                         \
        }                                                                                 \
        cp_commit_k5();                                                                   \
    } while (0)

    K5_STAGE_ST(0, 0);
    for (int idx = tid; idx < 32 * D; idx += 256) {
        int r = idx / D, d = idx % D;
        float v = (r < jn) ? q[((size_t)(t0 + j0 + r) * H + h) * D + d] : 0.0f;
        qs[r * D + d] = __float2bfloat16(v);
    }
    // P staged up front too (phase 2 A operand; independent of the ring)
    for (int idx = tid; idx < 32 * 32; idx += 256) {
        int r = idx / 32, i = idx % 32;
        float v = (r < jn && i < min(32, jend) && i <= j0 + r)
            ? P[(((size_t)c * H + h) * C + j0 + r) * C + i] : 0.0f;
        ps[r * 40 + i] = __float2bfloat16(v);
    }

    CTile acc[4];
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) { acc[t4].x[0]=acc[t4].x[1]=acc[t4].x[2]=acc[t4].x[3]=0.0f; }

    for (int it = 0; it < 4; it++) {           // phase 1: 4 x 32-i sub-tiles
        cp_wait_k5<0>();
        __syncthreads();
        int cur = it & 1;
        if (it < 3) K5_STAGE_ST((it + 1) * 32, cur ^ 1);
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, qs + (mh * 16) * D + it * 32 + k16 * 16, D / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, ts[cur] + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(acc[p2 * 2 + 0], A, Blo);
                mma_bf16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
        if (it == 3) K5_STAGE_Y(0, min(32, jend), 0);   // prefetch phase-2's first tile
        __syncthreads();
    }
    {
        float b_lo = 0.0f, b_hi = 0.0f;
        int j_lo = mh * 16 + fr, j_hi = j_lo + 8;
        if (j_lo < jn) b_lo = expf(gcum[(size_t)(t0 + j0 + j_lo) * H + h]);
        if (j_hi < jn) b_hi = expf(gcum[(size_t)(t0 + j0 + j_hi) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            acc[t4].x[0] *= b_lo; acc[t4].x[1] *= b_lo;
            acc[t4].x[2] *= b_hi; acc[t4].x[3] *= b_hi;
        }
    }
    const int nit2 = (jend + 31) / 32;
    for (int it = 0; it < nit2; it++) {        // phase 2 (j0=0,C=32 -> usually 1 sub-tile)
        cp_wait_k5<0>();
        __syncthreads();
        int cur = it & 1;
        if (it + 1 < nit2) K5_STAGE_Y((it + 1) * 32, min(32, jend - (it + 1) * 32), cur ^ 1);
        // refresh ps for sub-tiles beyond the first (P cols shift by it*32)
        if (it > 0) {
            for (int idx = tid; idx < 32 * 32; idx += 256) {
                int r = idx / 32, i = idx % 32;
                int gi = it * 32 + i;
                float v = (r < jn && gi < jend && gi <= j0 + r)
                    ? P[(((size_t)c * H + h) * C + j0 + r) * C + gi] : 0.0f;
                ps[r * 40 + i] = __float2bfloat16(v);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, ps + (mh * 16) * 40 + k16 * 16, 20, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, ts[cur] + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(acc[p2 * 2 + 0], A, Blo);
                mma_bf16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int j = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int col = nq * 32 + t4 * 8 + fc + (l & 1);
            if (j < jn)
                o[((size_t)(t0 + j0 + j) * H + h) * D + col] = scale * acc[t4].x[l];
        }
    }
}

int main(int argc, char** argv) {
    const int H = argc > 1 ? atoi(argv[1]) : 32;
    const int T = argc > 2 ? atoi(argv[2]) : 512, C = 32, D = GDN_D;
    const int NC = (T + C - 1) / C;
    printf("GDN K5 harness: H=%d T=%d C=%d NC=%d\n", H, T, C, NC);
    srand(9);
    auto rf = [](float s) { return ((rand() % 2001) - 1000) * 1e-3f * s; };
    std::vector<float> hq((size_t)T * H * D), hgc((size_t)T * H);
    std::vector<float> hP((size_t)NC * H * C * C, 0.0f), hY((size_t)NC * H * C * D);
    std::vector<float> hS((size_t)NC * H * D * D);
    for (auto& v : hq) v = rf(1.0f);
    for (auto& v : hY) v = rf(1.0f);
    for (auto& v : hS) v = rf(1.0f);
    for (int h = 0; h < H; h++) { float a = 0; for (int t = 0; t < T; t++) { a += -0.02f - (rand()%100)*2e-4f; hgc[(size_t)t*H+h] = a; } }
    // P lower-triangular-inclusive per (c,h): P[j][i] nonzero for i<=j
    for (int c = 0; c < NC; c++) for (int h = 0; h < H; h++)
        for (int j = 0; j < C; j++) for (int i = 0; i <= j; i++)
            hP[(((size_t)c*H+h)*C+j)*C+i] = rf(0.8f);
    float scale = 1.0f / sqrtf((float)D);

    float *dq,*dgc,*dP,*dY,*dS,*dO;
    CK(cudaMalloc(&dq,hq.size()*4)); CK(cudaMalloc(&dgc,hgc.size()*4));
    CK(cudaMalloc(&dP,hP.size()*4)); CK(cudaMalloc(&dY,hY.size()*4));
    CK(cudaMalloc(&dS,hS.size()*4)); CK(cudaMalloc(&dO,(size_t)T*H*D*4));
    CK(cudaMemcpy(dq,hq.data(),hq.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgc,hgc.data(),hgc.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP,hP.data(),hP.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dY,hY.data(),hY.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS,hS.data(),hS.size()*4,cudaMemcpyHostToDevice));

    // CPU ref
    std::vector<float> rO((size_t)T*H*D);
    for (int c = 0; c < NC; c++) for (int h = 0; h < H; h++) {
        int t0 = c*C, Cc = std::min(C, T-t0);
        for (int j = 0; j < Cc; j++) {
            double bj = exp((double)hgc[(size_t)(t0+j)*H+h]);
            for (int col = 0; col < D; col++) {
                double a = 0;
                for (int i = 0; i < D; i++)
                    a += (double)hq[((size_t)(t0+j)*H+h)*D+i] * hS[(((size_t)c*H+h)*D+i)*D+col];
                a *= bj;
                for (int i = 0; i <= j && i < Cc; i++)
                    a += (double)hP[(((size_t)c*H+h)*C+j)*C+i] * hY[(((size_t)c*H+h)*C+i)*D+col];
                rO[((size_t)(t0+j)*H+h)*D+col] = (float)(a * scale);
            }
        }
    }

    dim3 g(NC, H, (C+31)/32), blk(256);
    auto check = [&](const char* tag, double band) {
        std::vector<float> gO((size_t)T*H*D);
        CK(cudaMemcpy(gO.data(), dO, gO.size()*4, cudaMemcpyDeviceToHost));
        double m = 0, sc = 0;
        for (size_t i = 0; i < gO.size(); i++) { m = fmax(m, fabs((double)gO[i]-rO[i])); sc = fmax(sc, fabs((double)rO[i])); }
        printf("%s vs CPU: rel %.3e %s\n", tag, m/fmax(sc,1e-3), (m/fmax(sc,1e-3) < band) ? "OK" : "FAIL");
    };
    gdn_chunk_output_f32<<<g, blk>>>(dq, dgc, dP, dY, dS, dO, H, T, C, scale);
    CK(cudaDeviceSynchronize());
    check("v0", 1e-4);
    gdn_chunk_output_mma<<<g, blk>>>(dq, dgc, dP, dY, dS, dO, H, T, C, scale);
    CK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e) { printf("v1 launch: %s\n", cudaGetErrorString(e)); return 1; }
    check("v1", 3e-2);
    // v2 inputs: bf16 mirrors of Y and St (what K4-mma would write directly)
    std::vector<__nv_bfloat16> hYb(hY.size()), hSb(hS.size());
    for (size_t i = 0; i < hY.size(); i++) hYb[i] = __float2bfloat16(hY[i]);
    for (size_t i = 0; i < hS.size(); i++) hSb[i] = __float2bfloat16(hS[i]);
    __nv_bfloat16 *dYb, *dSb;
    CK(cudaMalloc(&dYb, hYb.size()*2)); CK(cudaMalloc(&dSb, hSb.size()*2));
    CK(cudaMemcpy(dYb, hYb.data(), hYb.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSb, hSb.data(), hSb.size()*2, cudaMemcpyHostToDevice));
    gdn_chunk_output_mma2<<<g, blk>>>(dq, dgc, dP, dYb, dSb, dO, H, T, C, scale);
    CK(cudaDeviceSynchronize());
    e = cudaGetLastError();
    if (e) { printf("v2 launch: %s\n", cudaGetErrorString(e)); return 1; }
    check("v2", 3e-2);

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int v = 0; v < 3; v++) {
        #define K5L() do { \
            if (v==0)      gdn_chunk_output_f32<<<g, blk>>>(dq, dgc, dP, dY, dS, dO, H, T, C, scale); \
            else if (v==1) gdn_chunk_output_mma<<<g, blk>>>(dq, dgc, dP, dY, dS, dO, H, T, C, scale); \
            else           gdn_chunk_output_mma2<<<g, blk>>>(dq, dgc, dP, dYb, dSb, dO, H, T, C, scale); \
        } while (0)
        for (int i = 0; i < 5; i++) K5L();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 50; i++) K5L();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("v%d: %.1f us/launch\n", v, ms*1000.0f/50);
        #undef K5L
    }
    return 0;
}
