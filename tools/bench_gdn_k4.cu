// GDN K4 (gdn_chunk_state) mma-rewrite dev harness (task 9 arc-1, ARCHITECTURE-H100.md).
// ncu 2026-07-26: K4 at 59.6% mem SOL, 130us/launch, 2.8ms/prime (the largest GDN slice).
// The step-A dot chains (Y = U - W.M) and step-B rank updates (M += (gk k) y^T) are GEMM-
// shaped; this harness develops the bf16-mma form against a CPU reference of the EXACT
// f32 semantics (v0 = the shipped kernel verbatim), standalone before any engine change.
// NUMERIC NOTE: an mma form rounds the multiply INPUTS to bf16 (state storage stays f32)
// — a numeric change WITHIN the already-gated chunked prefill config (MEMRA_GDN_CHUNKED);
// the MEMRA_GDN_DIFF oracle + argmax battery arbitrate any engine adoption.
//
// Build (box): nvcc -O3 -arch=sm_90a -o /tmp/gdnk4 tools/bench_gdn_k4.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#define GDN_D 128
#define GDN_NSPLIT 4

extern "C" __global__ void gdn_chunk_state_f32(
        const float* __restrict__ k, const float* __restrict__ gcum,
        const float* __restrict__ beta,
        const float* __restrict__ U, const float* __restrict__ W,
        float* __restrict__ Y, float* __restrict__ Ssnap,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        int H, int T, int C) {
    constexpr int COLS = GDN_D / GDN_NSPLIT;   // 32
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    __shared__ float Ms[COLS][GDN_D + 4];      // +4 pad: float4-aligned, bank-spread rows
    __shared__ float wt[32][GDN_D];            // W sub-tile; step B reuses it for k
    __shared__ float ys[32][COLS + 1];         // step-A Y slice (step B reads smem, not L2)
    __shared__ float gk[128];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
        int cl2 = idx / GDN_D, i = idx % GDN_D;
        Ms[cl2][i] = state_in[((size_t)h * GDN_D + col0 + cl2) * GDN_D + i];
    }
    __syncthreads();
    const int NC = (T + C - 1) / C;
    const int cl = tid % COLS, jr = tid / COLS;   // 8 row-groups (A) / 8 i-groups (B) per col
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        if (tid < Cc) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                    * beta[(size_t)(t0 + tid) * H + h];
        }
        // snapshot the chunk-START state for K5's inter-chunk output term (col-fast writes,
        // TRANSPOSED to St[i][col] so K5 reads coalesce). Moves the o_inter dot OFF the
        // sequential path into the fully chunk-parallel output kernel.
        float* sc_out = Ssnap + ((size_t)c * H + h) * GDN_D * GDN_D;
        for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
            int i = idx / COLS, cl2 = idx % COLS;
            sc_out[(size_t)i * GDN_D + col0 + cl2] = Ms[cl2][i];
        }
        float acc[GDN_D / 8];   // step-B accumulators (16 i's/thread), built across sub-tiles
        #pragma unroll
        for (int r = 0; r < GDN_D / 8; r++) acc[r] = 0.0f;
        // Per 32-row sub-tile: step A (Y = U - W S_c, 4 rows/thread, float4 smem dots,
        // U loads HOISTED above the dot chains) then step B (rank update from the smem
        // Y slice + re-staged k rows). The naive global-broadcast form was L2-bound.
        for (int jt = 0; jt < Cc; jt += 32) {
            const int jn = min(32, Cc - jt);
            __syncthreads();
            for (int idx = tid; idx < 32 * (GDN_D / 4); idx += 256) {
                int r = idx / (GDN_D / 4), d4 = idx % (GDN_D / 4);
                *reinterpret_cast<float4*>(&wt[r][d4 * 4]) = (r < jn)
                    ? *reinterpret_cast<const float4*>(
                        &W[(((size_t)c * H + h) * C + jt + r) * GDN_D + d4 * 4])
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
            {
                const size_t yb = (((size_t)c * H + h) * C + jt) * GDN_D + col0 + cl;
                const float u0 = (jr      < jn) ? U[yb + (size_t)jr * GDN_D] : 0.0f;
                const float u1 = (jr + 8  < jn) ? U[yb + (size_t)(jr + 8) * GDN_D] : 0.0f;
                const float u2 = (jr + 16 < jn) ? U[yb + (size_t)(jr + 16) * GDN_D] : 0.0f;
                const float u3 = (jr + 24 < jn) ? U[yb + (size_t)(jr + 24) * GDN_D] : 0.0f;
                float pw0 = 0.0f, pw1 = 0.0f, pw2 = 0.0f, pw3 = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < GDN_D; i += 4) {
                    const float4 m = *reinterpret_cast<const float4*>(&Ms[cl][i]);
                    const float4 w0 = *reinterpret_cast<const float4*>(&wt[jr][i]);
                    const float4 w1 = *reinterpret_cast<const float4*>(&wt[jr + 8][i]);
                    const float4 w2 = *reinterpret_cast<const float4*>(&wt[jr + 16][i]);
                    const float4 w3 = *reinterpret_cast<const float4*>(&wt[jr + 24][i]);
                    pw0 += w0.x * m.x + w0.y * m.y + w0.z * m.z + w0.w * m.w;
                    pw1 += w1.x * m.x + w1.y * m.y + w1.z * m.z + w1.w * m.w;
                    pw2 += w2.x * m.x + w2.y * m.y + w2.z * m.z + w2.w * m.w;
                    pw3 += w3.x * m.x + w3.y * m.y + w3.z * m.z + w3.w * m.w;
                }
                const float y0 = u0 - pw0, y1 = u1 - pw1, y2 = u2 - pw2, y3 = u3 - pw3;
                if (jr      < jn) { Y[yb + (size_t)jr * GDN_D] = y0;        ys[jr][cl] = y0; }
                if (jr + 8  < jn) { Y[yb + (size_t)(jr + 8) * GDN_D] = y1;  ys[jr + 8][cl] = y1; }
                if (jr + 16 < jn) { Y[yb + (size_t)(jr + 16) * GDN_D] = y2; ys[jr + 16][cl] = y2; }
                if (jr + 24 < jn) { Y[yb + (size_t)(jr + 24) * GDN_D] = y3; ys[jr + 24][cl] = y3; }
            }
            __syncthreads();
            for (int idx = tid; idx < 32 * (GDN_D / 4); idx += 256) {
                int r = idx / (GDN_D / 4), d4 = idx % (GDN_D / 4);
                *reinterpret_cast<float4*>(&wt[r][d4 * 4]) = (r < jn)
                    ? *reinterpret_cast<const float4*>(
                        &k[((size_t)(t0 + jt + r) * H + h) * GDN_D + d4 * 4])
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
            for (int jj = 0; jj < jn; jj++) {
                float yv = ys[jj][cl] * gk[jt + jj];
                #pragma unroll
                for (int r = 0; r < GDN_D / 8; r++)
                    acc[r] += wt[jj][jr * (GDN_D / 8) + r] * yv;
            }
        }
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        #pragma unroll
        for (int r = 0; r < GDN_D / 8; r++) {
            int i = jr * (GDN_D / 8) + r;
            Ms[cl][i] = bC * Ms[cl][i] + acc[r];
        }
        __syncthreads();   // Ms/gk stable before the next chunk rewrites them
    }
    for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
        int cl2 = idx / GDN_D, i = idx % GDN_D;
        state_out[((size_t)h * GDN_D + col0 + cl2) * GDN_D + i] = Ms[cl2][i];
    }
}



// ---------------- v1: bf16-mma form (M resident in accumulator fragments) ----------------
// Per CTA (h, 32-col block), 256 threads = 8 warps, C=32 (one j sub-tile per chunk):
//   Macc: M[32col x 128i] lives in mma CTile accumulators (16 f32/thread) across ALL chunks.
//   step A: Y[32j x 32col] = U - W[32j x 128i] . M[128i x 32col]   (8 warp-tiles, k-loop 8)
//   step B: M[32col x 128i] += (ys=Y*gk)^T[32col x 32j] . k[32j x 128i]  (4 tiles/warp, k-loop 2)
//   bC fold = register scale of Macc between A and B. Operands round to bf16 per chunk
//   (Mb mirror, Wb, kb, ys); state carry itself stays f32 in registers.
#include <cuda_bf16.h>
#include <cstdint>
using std::uint32_t;
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

#define MB_PAD 40   // Mb/ys row stride in bf16 (80B: 16B-aligned rows for ldmatrix)

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_state_mma(const float* __restrict__ k, const float* __restrict__ gcum,
                    const float* __restrict__ beta,
                    const float* __restrict__ U, const float* __restrict__ W,
                    float* __restrict__ Y, float* __restrict__ Ssnap,
                    const float* __restrict__ state_in, float* __restrict__ state_out,
                    int H, int T, int C) {
    constexpr int D = GDN_D;
    constexpr int COLS = 32;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;

    __shared__ __nv_bfloat16 Wb[32 * D];          // step-A A operand (row-major [j][i])
    __shared__ __nv_bfloat16 kb[32 * D];          // step-B B operand ([j][i])
    __shared__ __nv_bfloat16 Mb[32 * (D + 8)];    // step-A B operand mirror, NATURAL [col][i]
    constexpr int MB_STR = D + 8;                  // 136 bf16 = 272B rows (16B-aligned)
    __shared__ __nv_bfloat16 ys[32 * MB_PAD];     // step-B A operand ([j][col], gk-folded)
    __shared__ float gk[32];

    // fragment lane coords (m16n8 D-tile): row r_frag(l) = lane/4 + (l<2?0:8), col (lane%4)*2 + (l&1)
    const int fr = lane / 4, fc = (lane % 4) * 2;

    // step-B warp tiling: Macc[t4] covers cols mh*16 + frag-row, i = nq*32 + t4*8 + frag-col
    const int mh = warp / 4, nq = warp % 4;
    CTile Macc[4];
    // init from state_in: M[col][i] with col = col0 + mh*16 + row(l), i = nq*32 + t4*8 + colf(l)
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            Macc[t4].x[l] = state_in[((size_t)h * D + col0 + col) * D + i];
        }
    }

    const int NC = (T + C - 1) / C;
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        // ---- chunk top: Ssnap (f32, from fragments) + Mb mirror (bf16) + gk + Wb stage ----
        float* sc_out = Ssnap + ((size_t)c * H + h) * D * D;
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
                int i = nq * 32 + t4 * 8 + fc + (l & 1);
#ifndef K4_NOSNAP
                sc_out[(size_t)i * D + col0 + col] = Macc[t4].x[l];
#endif
                Mb[col * MB_STR + i] = __float2bfloat16(Macc[t4].x[l]);
            }
        }
        if (tid < Cc) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                    * beta[(size_t)(t0 + tid) * H + h];
        } else if (tid < 32) {
            gk[tid] = 0.0f;
        }
#ifndef K4_NOSTAGE
        for (int idx = tid; idx < 32 * D; idx += 256) {
            int r = idx / D, i = idx % D;
            float w = (r < Cc) ? W[(((size_t)c * H + h) * C + r) * D + i] : 0.0f;
            Wb[r * D + i] = __float2bfloat16(w);
        }
#endif
        __syncthreads();

        // ---- step A: S[32j x 32col] = W . M; Y = U - S ----
        // warp tiling: mj = warp/4 (j-16-half), colg = (warp%4)/2 (col-16-group), half = warp%2.
        {
            const int mj = warp / 4, colg = (warp % 4) / 2, half = warp % 2;
            CTile Sc; Sc.x[0] = Sc.x[1] = Sc.x[2] = Sc.x[3] = 0.0f;
            #pragma unroll
            for (int k16 = 0; k16 < D / 16; k16++) {
                ATile A;
                ld_A(A, Wb + (mj * 16) * D + k16 * 16, D / 2, lane);
                ATile Bt;   // Mb tile: rows = col (n-dim), cols = i (k-dim) — [n][k] as mma wants
                ld_A(Bt, Mb + (colg * 16) * MB_STR + k16 * 16, MB_STR / 2, lane);
                BTile B;
                if (half == 0) { B.x[0] = Bt.x[0]; B.x[1] = Bt.x[2]; }
                else           { B.x[0] = Bt.x[1]; B.x[1] = Bt.x[3]; }
                mma_bf16(Sc, A, B);
            }
            // epilogue: y = U - S; write Y global (f32) + ys smem (bf16, *gk[j])
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int j = mj * 16 + fr + ((l < 2) ? 0 : 8);
                int col = colg * 16 + half * 8 + fc + (l & 1);
                if (j < Cc) {
                    float u = U[(((size_t)c * H + h) * C + j) * D + col0 + col];
                    float y = u - Sc.x[l];
                    Y[(((size_t)c * H + h) * C + j) * D + col0 + col] = y;
                    ys[j * MB_PAD + col] = __float2bfloat16(y * gk[j]);
                } else {
                    ys[j * MB_PAD + col] = __float2bfloat16(0.0f);
                }
            }
        }
        // stage kb while ys settles (no dependency between them)
#ifndef K4_NOSTAGE
        for (int idx = tid; idx < 32 * D; idx += 256) {
            int r = idx / D, i = idx % D;
            float kv = (r < Cc) ? k[((size_t)(t0 + r) * H + h) * D + i] : 0.0f;
            kb[r * D + i] = __float2bfloat16(kv);
        }
#endif
        __syncthreads();

        // ---- step B: Macc = bC*Macc + ys^T . kb ----
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) Macc[t4].x[l] *= bC;
        }
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;   // A[col][j] = ys[j][col]^T
            ld_A_trans(A, ys + (k16 * 16) * MB_PAD + mh * 16, MB_PAD / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {   // i-16-pairs within this warp's 32-i quarter
                ATile Bt;   // kb tile rows j (k-dim), cols i (n-dim): [k][n] -> transpose-load
                ld_A_trans(Bt, kb + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(Macc[p2 * 2 + 0], A, Blo);
                mma_bf16(Macc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();   // Wb/kb/Mb/ys stable before next chunk restages
    }
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            state_out[((size_t)h * D + col0 + col) * D + i] = Macc[t4].x[l];
        }
    }
}


// ---------------- v2: v1 + bf16 W/k inputs + cp.async double-buffered staging ----------------
// Probe verdict on v1: synchronous global->bf16 staging = 72us of 133 (54%); Ssnap 15us.
// v2 takes W and k PRE-CONVERTED to bf16 (engine side: K3 casts W on store for free; k gets
// a bf16 mirror pass) and pipelines the 8KB tiles through a 2-deep cp.async ring.
__device__ __forceinline__ void cp_async16_g(void* dst, const void* src) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_state_mma2(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                     const float* __restrict__ beta,
                     const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                     float* __restrict__ Y, float* __restrict__ Ssnap,
                     const float* __restrict__ state_in, float* __restrict__ state_out,
                     int H, int T, int C) {
    constexpr int D = GDN_D;
    constexpr int COLS = 32;
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;

    __shared__ __nv_bfloat16 Wb[2][32 * D];
    __shared__ __nv_bfloat16 kb[2][32 * D];
    __shared__ __nv_bfloat16 Mb[32 * (D + 8)];
    constexpr int MB_STR = D + 8;
    __shared__ __nv_bfloat16 ys[32 * MB_PAD];
    __shared__ float gk[32];

    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / 4, nq = warp % 4;
    CTile Macc[4];
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            Macc[t4].x[l] = state_in[((size_t)h * D + col0 + col) * D + i];
        }

    const int NC = (T + C - 1) / C;
    // stage(chunk, buf): W tile 32xD bf16 (8KB) + k tile (8KB) = 8 x 16B per thread
    #define V2_STAGE(c_, buf_) do {                                                       \
        int t0_ = (c_) * C;                                                               \
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {                               \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_g(&Wb[buf_][r * D + seg * 8],                                      \
                         Wb16 + (((size_t)(c_) * H + h) * C + r) * D + seg * 8);          \
            cp_async16_g(&kb[buf_][r * D + seg * 8],                                      \
                         kb16 + ((size_t)(t0_ + r) * H + h) * D + seg * 8);               \
        }                                                                                 \
        cp_commit();                                                                      \
    } while (0)

    V2_STAGE(0, 0);
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        const int cur = c & 1;
        // chunk top: Ssnap + Mb mirror + gk (independent of the in-flight stage)
        float* sc_out = Ssnap + ((size_t)c * H + h) * D * D;
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
                int i = nq * 32 + t4 * 8 + fc + (l & 1);
                sc_out[(size_t)i * D + col0 + col] = Macc[t4].x[l];
                Mb[col * MB_STR + i] = __float2bfloat16(Macc[t4].x[l]);
            }
        if (tid < Cc) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                    * beta[(size_t)(t0 + tid) * H + h];
        } else if (tid < 32) {
            gk[tid] = 0.0f;
        }
        cp_wait<0>();
        __syncthreads();
        if (c + 1 < NC) V2_STAGE(c + 1, cur ^ 1);

        // step A
        {
            const int mj = warp / 4, colg = (warp % 4) / 2, half = warp % 2;
            CTile Sc; Sc.x[0] = Sc.x[1] = Sc.x[2] = Sc.x[3] = 0.0f;
            #pragma unroll
            for (int k16 = 0; k16 < D / 16; k16++) {
                ATile A;
                ld_A(A, Wb[cur] + (mj * 16) * D + k16 * 16, D / 2, lane);
                ATile Bt;
                ld_A(Bt, Mb + (colg * 16) * MB_STR + k16 * 16, MB_STR / 2, lane);
                BTile B;
                if (half == 0) { B.x[0] = Bt.x[0]; B.x[1] = Bt.x[2]; }
                else           { B.x[0] = Bt.x[1]; B.x[1] = Bt.x[3]; }
                mma_bf16(Sc, A, B);
            }
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int j = mj * 16 + fr + ((l < 2) ? 0 : 8);
                int col = colg * 16 + half * 8 + fc + (l & 1);
                if (j < Cc) {
                    float u = U[(((size_t)c * H + h) * C + j) * D + col0 + col];
                    float y = u - Sc.x[l];
                    Y[(((size_t)c * H + h) * C + j) * D + col0 + col] = y;
                    ys[j * MB_PAD + col] = __float2bfloat16(y * gk[j]);
                } else {
                    ys[j * MB_PAD + col] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // step B
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++)
            #pragma unroll
            for (int l = 0; l < 4; l++) Macc[t4].x[l] *= bC;
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A_trans(A, ys + (k16 * 16) * MB_PAD + mh * 16, MB_PAD / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, kb[cur] + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(Macc[p2 * 2 + 0], A, Blo);
                mma_bf16(Macc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            state_out[((size_t)h * D + col0 + col) * D + i] = Macc[t4].x[l];
        }
}

// ---- CPU reference (exact f32 semantics of the kernel above) ----
static void cpu_ref(const float* k, const float* gcum, const float* beta,
                    const float* U, const float* W,
                    float* Y, float* Ssnap, const float* state_in, float* state_out,
                    int H, int T, int C) {
    const int D = GDN_D, NC = (T + C - 1) / C;
    for (int h = 0; h < H; h++) {
        // M[col][i]
        std::vector<float> M((size_t)D * D);
        for (int col = 0; col < D; col++)
            for (int i = 0; i < D; i++)
                M[(size_t)col * D + i] = state_in[((size_t)h * D + col) * D + i];
        for (int c = 0; c < NC; c++) {
            const int t0 = c * C, Cc = (T - t0 < C) ? T - t0 : C;
            const float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            // snapshot (transposed)
            for (int i = 0; i < D; i++)
                for (int col = 0; col < D; col++)
                    Ssnap[(((size_t)c * H + h) * D + i) * D + col] = M[(size_t)col * D + i];
            // step A: Y[j,col] = U[j,col] - sum_i W[j,i] M[col][i]
            std::vector<float> Yc((size_t)Cc * D);
            for (int j = 0; j < Cc; j++)
                for (int col = 0; col < D; col++) {
                    double acc = 0;
                    for (int i = 0; i < D; i++)
                        acc += (double)W[(((size_t)c * H + h) * C + j) * D + i] * M[(size_t)col * D + i];
                    float y = U[(((size_t)c * H + h) * C + j) * D + col] - (float)acc;
                    Yc[(size_t)j * D + col] = y;
                    Y[(((size_t)c * H + h) * C + j) * D + col] = y;
                }
            // step B: M[col][i] = bC M + sum_j gk_j k_j[i] y_j[col]
            const float bC = expf(gC);
            std::vector<float> Mn((size_t)D * D);
            for (int col = 0; col < D; col++)
                for (int i = 0; i < D; i++) {
                    double acc = bC * (double)M[(size_t)col * D + i];
                    for (int j = 0; j < Cc; j++) {
                        float gk = expf(gC - gcum[(size_t)(t0 + j) * H + h]) * beta[(size_t)(t0 + j) * H + h];
                        acc += (double)gk * k[((size_t)(t0 + j) * H + h) * D + i] * Yc[(size_t)j * D + col];
                    }
                    Mn[(size_t)col * D + i] = (float)acc;
                }
            M.swap(Mn);
        }
        for (int col = 0; col < D; col++)
            for (int i = 0; i < D; i++)
                state_out[((size_t)h * D + col) * D + i] = M[(size_t)col * D + i];
    }
}


int main(int argc, char** argv) {
    const int H = argc > 1 ? atoi(argv[1]) : 32;
    const int T = argc > 2 ? atoi(argv[2]) : 512, C = 32, D = GDN_D;
    const int NC = (T + C - 1) / C;
    printf("GDN K4 harness: H=%d T=%d C=%d D=%d NC=%d\n", H, T, C, D, NC);
    srand(7);
    auto rf = [](float s) { return ((rand() % 2001) - 1000) * 1e-3f * s; };

    std::vector<float> hk((size_t)T * H * D), hgc((size_t)T * H), hb((size_t)T * H);
    std::vector<float> hU((size_t)NC * H * C * D), hW((size_t)NC * H * C * D);
    std::vector<float> hSi((size_t)H * D * D);
    for (auto& v : hk) v = rf(1.0f);
    for (auto& v : hU) v = rf(1.0f);
    for (auto& v : hW) v = rf(0.3f);
    for (auto& v : hSi) v = rf(0.5f);
    for (auto& v : hb) v = 0.5f + ((rand() % 1000) * 5e-4f);
    // gcum: per (h): cumulative sum of small negatives along t (log-gate law)
    for (int h = 0; h < H; h++) {
        float acc = 0.0f;
        for (int t = 0; t < T; t++) {
            acc += -0.02f - (rand() % 100) * 2e-4f;
            hgc[(size_t)t * H + h] = acc;
        }
    }

    float *dk, *dgc, *db, *dU, *dW, *dY, *dS, *dSi, *dSo;
    CK(cudaMalloc(&dk, hk.size() * 4)); CK(cudaMalloc(&dgc, hgc.size() * 4));
    CK(cudaMalloc(&db, hb.size() * 4)); CK(cudaMalloc(&dU, hU.size() * 4));
    CK(cudaMalloc(&dW, hW.size() * 4)); CK(cudaMalloc(&dY, hU.size() * 4));
    CK(cudaMalloc(&dS, (size_t)NC * H * D * D * 4));
    CK(cudaMalloc(&dSi, hSi.size() * 4)); CK(cudaMalloc(&dSo, hSi.size() * 4));
    CK(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgc, hgc.data(), hgc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dU, hU.data(), hU.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, hW.data(), hW.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSi, hSi.data(), hSi.size() * 4, cudaMemcpyHostToDevice));

    dim3 grid(H, GDN_NSPLIT), blk(256);
    gdn_chunk_state_f32<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
    CK(cudaDeviceSynchronize());

    // CPU reference + compare (Y, state_out, Ssnap probes)
    std::vector<float> rY(hU.size()), rS((size_t)NC * H * D * D), rSo(hSi.size());
    cpu_ref(hk.data(), hgc.data(), hb.data(), hU.data(), hW.data(),
            rY.data(), rS.data(), hSi.data(), rSo.data(), H, T, C);
    std::vector<float> gY(hU.size()), gSo(hSi.size());
    CK(cudaMemcpy(gY.data(), dY, gY.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gSo.data(), dSo, gSo.size() * 4, cudaMemcpyDeviceToHost));
    double mY = 0, mS = 0, sY = 0, sS = 0;
    for (size_t i = 0; i < gY.size(); i++) { mY = fmax(mY, fabs((double)gY[i] - rY[i])); sY = fmax(sY, fabs((double)rY[i])); }
    for (size_t i = 0; i < gSo.size(); i++) { mS = fmax(mS, fabs((double)gSo[i] - rSo[i])); sS = fmax(sS, fabs((double)rSo[i])); }
    printf("v0 vs CPU: Y maxdiff %.3e (scale %.2f)  state maxdiff %.3e (scale %.2f)  %s\n",
           mY, sY, mS, sS, (mY / fmax(sY, 1e-3) < 1e-4 && mS / fmax(sS, 1e-3) < 1e-4) ? "OK" : "FAIL");

    // bf16 mirrors of W and k for v2 (engine side: K3 casts on store; k gets a mirror pass)
    std::vector<__nv_bfloat16> hWb(hW.size()), hkb(hk.size());
    for (size_t i = 0; i < hW.size(); i++) hWb[i] = __float2bfloat16(hW[i]);
    for (size_t i = 0; i < hk.size(); i++) hkb[i] = __float2bfloat16(hk[i]);
    __nv_bfloat16 *dWb16, *dkb16;
    CK(cudaMalloc(&dWb16, hWb.size() * 2));
    CK(cudaMalloc(&dkb16, hkb.size() * 2));
    CK(cudaMemcpy(dWb16, hWb.data(), hWb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkb16, hkb.data(), hkb.size() * 2, cudaMemcpyHostToDevice));

    // ---- v1 (mma): correctness vs CPU ref (bf16 band) + timing ----
    {
        gdn_chunk_state_mma<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
        CK(cudaDeviceSynchronize());
        cudaError_t e = cudaGetLastError();
        if (e) { printf("v1 launch: %s\n", cudaGetErrorString(e)); return 1; }
        std::vector<float> gY1(hU.size()), gSo1(hSi.size());
        CK(cudaMemcpy(gY1.data(), dY, gY1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gSo1.data(), dSo, gSo1.size() * 4, cudaMemcpyDeviceToHost));
        double mY1 = 0, mS1 = 0;
        for (size_t i = 0; i < gY1.size(); i++) mY1 = fmax(mY1, fabs((double)gY1[i] - rY[i]));
        for (size_t i = 0; i < gSo1.size(); i++) mS1 = fmax(mS1, fabs((double)gSo1[i] - rSo[i]));
        double relY = mY1 / fmax(sY, 1e-3), relS = mS1 / fmax(sS, 1e-3);
        printf("v1 vs CPU: Y rel %.3e  state rel %.3e  %s (bf16 band 3e-2)\n",
               relY, relS, (relY < 3e-2 && relS < 3e-2) ? "OK" : "FAIL");
    }
    {
        gdn_chunk_state_mma2<<<grid, blk>>>(dkb16, dgc, db, dU, dWb16, dY, dS, dSi, dSo, H, T, C);
        CK(cudaDeviceSynchronize());
        cudaError_t e = cudaGetLastError();
        if (e) { printf("v2 launch: %s\n", cudaGetErrorString(e)); return 1; }
        std::vector<float> gY2(hU.size()), gSo2(hSi.size());
        CK(cudaMemcpy(gY2.data(), dY, gY2.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gSo2.data(), dSo, gSo2.size() * 4, cudaMemcpyDeviceToHost));
        double mY2 = 0, mS2 = 0;
        for (size_t i = 0; i < gY2.size(); i++) mY2 = fmax(mY2, fabs((double)gY2[i] - rY[i]));
        for (size_t i = 0; i < gSo2.size(); i++) mS2 = fmax(mS2, fabs((double)gSo2[i] - rSo[i]));
        double relY2 = mY2 / fmax(sY, 1e-3), relS2 = mS2 / fmax(sS, 1e-3);
        printf("v2 vs CPU: Y rel %.3e  state rel %.3e  %s (bf16 band 3e-2)\n",
               relY2, relS2, (relY2 < 3e-2 && relS2 < 3e-2) ? "OK" : "FAIL");
    }
    // timing
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; i++) gdn_chunk_state_f32<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) gdn_chunk_state_f32<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    printf("v0: %.1f us/launch (engine measured ~130us at the real H)\n", ms * 1000.0f / 50);
    for (int i = 0; i < 5; i++) gdn_chunk_state_mma<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) gdn_chunk_state_mma<<<grid, blk>>>(dk, dgc, db, dU, dW, dY, dS, dSi, dSo, H, T, C);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("v1: %.1f us/launch\n", ms * 1000.0f / 50);
    for (int i = 0; i < 5; i++) gdn_chunk_state_mma2<<<grid, blk>>>(dkb16, dgc, db, dU, dWb16, dY, dS, dSi, dSo, H, T, C);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) gdn_chunk_state_mma2<<<grid, blk>>>(dkb16, dgc, db, dU, dWb16, dY, dS, dSi, dSo, H, T, C);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("v2: %.1f us/launch (v0 %.1fx)\n", ms * 1000.0f / 50, 0.0f);
    return 0;
}
