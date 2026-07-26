// wgmma prefill-GEMM dev harness (task 8, ARCHITECTURE-H100.md): develop the sm_90a
// warpgroup int8 GEMM against synthetic Q8_0 data with a CPU reference, standalone —
// no engine coupling until it beats the int8-mma class (MMQ 688us/launch = 60% of prime).
//
// v0 kernel: one warpgroup (128 threads) per 64-row tile, wgmma.mma_async.sync.m64n64k32.s8
// over K in 32-elem steps, per-32-block f32 scale fold on the s32 accumulators (the
// Q8_0 block-quant law — same math as qmatvec_gemm: s32 accumulate exact, fold per block).
// No cp.async pipeline yet (correctness first; staging is v1).
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/wgmma tools/bench_q8_gemm_wgmma.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

// ---- wgmma primitives (sm_90a) ----
// smem descriptor: matrix start addr (bits 0-13, >>4), leading byte offset (16-29, >>4),
// stride byte offset (32-45, >>4), swizzle mode (62-63).
__device__ __forceinline__ unsigned long long make_desc(const void* smem_ptr,
                                                        unsigned lead_bytes,
                                                        unsigned stride_bytes,
                                                        unsigned swizzle) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    d |= (unsigned long long)(swizzle & 0x3) << 62;
    return d;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
template<int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
}

// m64n64k32 s8*s8->s32: 32 accumulator regs per thread (64*64/128 threads = 32).
__device__ __forceinline__ void wgmma_m64n64k32_s8(int acc[32], unsigned long long da,
                                                   unsigned long long db, int scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p;\n"
        "}\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]),
          "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]),
          "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
          "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
          "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]),
          "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]),
          "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
          "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// ---- v0 kernel: C[m64 x n64] tile per warpgroup; K stepped 32 at a time.
// A = weights int8 [out_f, in_f] row-major (from the rp qplane), per-block scales wd.
// B = activation int8 [n_tok, in_f] row-major, per-block scales ad (q8_1-style d).
// For Q8_0 block-quant: acc_f32 += (s32 wgmma of the k-slice) * wscale[blk] * ascale[tok][blk]
// — requires resetting the s32 accumulator each 32-K step (scale_d=0 first mma of slice).
// Layout: A slice [64 rows x 32k] into smem as 8x(8x16?) core matrices — v0 uses the
// canonical "K-major 64x32" arrangement wgmma expects with SW none (interleaved manual).
extern "C" __global__ void __launch_bounds__(128, 1)
q8_gemm_wgmma_v0(const signed char* __restrict__ A, const half* __restrict__ Ascale,
                 const signed char* __restrict__ B, const float* __restrict__ Bscale,
                 float* __restrict__ C, int in_f, int out_f, int n_tok) {
    // tile origin
    int row0 = blockIdx.x * 64;
    int col0 = blockIdx.y * 64;
    if (row0 >= out_f || col0 >= n_tok) return;
    int tid = threadIdx.x;           // 0..127 (one warpgroup)
    int nblk = in_f / 32;

    // smem: A-slice 64x32 int8 (2KB) + B-slice 64x32 int8 (2KB), wgmma-canonical layout.
    __shared__ __align__(128) signed char sA[64 * 32];
    __shared__ __align__(128) signed char sB[64 * 32];

    float facc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) facc[i] = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        // load A slice: 64 rows x 32 int8 — each thread copies 16 bytes (64*32/128 = 16B/thread).
        // wgmma smem A layout (no swizzle, K-major): row r, k -> sA[r*32 + k] with the
        // descriptor lead/stride encoding the 8x8 core-matrix tiling; v0 uses the plain
        // "canonical row-major with LBO=16, SBO=512" arrangement (K%32==0 tiles).
        // CORE-MATRIX order (canonical no-swizzle): 8-row x 16B cores; tile 64x32 =
        // 8 M-cores x 2 K-cores; core(m,k) base = ((m*2+k)*8)*16, row r at +r*16.
        {
            int r = tid / 2;              // source row 0..63
            int seg = tid % 2;            // k-core 0..1
            const signed char* src = A + (size_t)(row0 + r) * in_f + blk * 32 + seg * 16;
            int m_core = r / 8, r_in = r % 8;
#ifndef CORE_K_OUTER
            signed char* dst = sA + (((m_core * 2 + seg) * 8) + r_in) * 16;
#else
            signed char* dst = sA + (((seg * 8 + m_core) * 8) + r_in) * 16;
#endif
            *(int4*)dst = (row0 + r < out_f) ? *(const int4*)src : make_int4(0,0,0,0);
        }
        {
            // canonical K-major B (PTX core matrices): core = 8 N-rows x 16 K-bytes;
            // core(n_core, k_core) at n_core*SBO(256) + k_core*LBO(128), row n_in at
            // +n_in*16 — token-major source is k-contiguous per token: clean int4 copies.
            int c = tid / 2;              // token 0..63
            int seg = tid % 2;            // k-core 0..1
            const signed char* src = B + (size_t)(col0 + c) * in_f + blk * 32 + seg * 16;
            int n_core = c / 8, c_in = c % 8;
            signed char* dst = sB + n_core * 256 + seg * 128 + c_in * 16;
            *(int4*)dst = (col0 + c < n_tok) ? *(const int4*)src : make_int4(0,0,0,0);
        }
        __syncthreads();
        // smem was written by the GENERIC proxy (st.shared); wgmma reads it in the
        // ASYNC proxy — without this fence the reads are undefined (PTX mem model).
        asm volatile("fence.proxy.async.shared::cta;");

        int acc[32];
        wgmma_fence();
#ifndef DESC_LBO
#define DESC_LBO 128
#endif
#ifndef DESC_SBO
#define DESC_SBO 256
#endif
#ifndef DESC_LBO_B
#define DESC_LBO_B DESC_LBO
#endif
#ifndef DESC_SBO_B
#define DESC_SBO_B DESC_SBO
#endif
        unsigned long long da = make_desc(sA, DESC_LBO, DESC_SBO, 0);
        unsigned long long db = make_desc(sB, DESC_LBO_B, DESC_SBO_B, 0);
        wgmma_m64n64k32_s8(acc, da, db, /*scale_d=*/0);   // fresh s32 accumulate per k-slice
        wgmma_commit();
        wgmma_wait<0>();

        // fold: per-thread accumulator fragment mapping for m64n64 s32 (wgmma canonical):
        // thread t owns rows (t/32)*16 + (t%32)/4 + {0,8}, cols (t%4)*2 + n8*8 + {0,1}
        float wsc[2];
        {
            int warp = tid / 32;
            int r_base = warp * 16 + (tid % 32) / 4;
            wsc[0] = __half2float(Ascale[(size_t)(row0 + r_base) * nblk + blk]);
            wsc[1] = __half2float(Ascale[(size_t)(row0 + r_base + 8) * nblk + blk]);
        }
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4;               // 8 column-groups
            int reg = i % 4;              // 2 rows x 2 cols
            int col = (tid % 4) * 2 + n8 * 8 + (reg % 2);
            float bs = Bscale[(size_t)(col0 + col) * nblk + blk];
            facc[i] += (float)acc[i] * wsc[reg / 2] * bs;
        }
        __syncthreads();
    }

    // store C [n_tok, out_f] token-major (engine convention y[m, out])
    {
        int warp = tid / 32;
        int r_base = row0 + warp * 16 + (tid % 32) / 4;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int n8 = i / 4;
            int reg = i % 4;
            int col = col0 + (tid % 4) * 2 + n8 * 8 + (reg % 2);
            int row = r_base + (reg / 2) * 8;
            if (row < out_f && col < n_tok) C[(size_t)col * out_f + row] = facc[i];
        }
    }
}

// ---- host: synthetic Q8_0-style data + CPU reference + timing ----
int main() {
    const int in_f = 4096, out_f = 4096, n_tok = 512;
    int nblk = in_f / 32;
    printf("q8 wgmma v0: %dx%d x %d tok\n", out_f, in_f, n_tok);

    // synthetic int8 weights + half block scales; int8 activations + float block scales
    signed char* hA = (signed char*)malloc((size_t)out_f * in_f);
    half* hAs = (half*)malloc((size_t)out_f * nblk * 2);
    signed char* hB = (signed char*)malloc((size_t)n_tok * in_f);
    float* hBs = (float*)malloc((size_t)n_tok * nblk * 4);
    srand(7);
    for (size_t i = 0; i < (size_t)out_f * in_f; i++) hA[i] = (signed char)(rand() % 17 - 8);
    for (size_t i = 0; i < (size_t)out_f * nblk; i++) hAs[i] = __float2half(0.01f + (rand() % 100) * 1e-4f);
    for (size_t i = 0; i < (size_t)n_tok * in_f; i++) hB[i] = (signed char)(rand() % 17 - 8);
    for (size_t i = 0; i < (size_t)n_tok * nblk; i++) hBs[i] = 0.02f + (rand() % 100) * 1e-4f;

    signed char *dA, *dB; half* dAs; float *dBs, *dC;
    CK(cudaMalloc(&dA, (size_t)out_f * in_f));
    CK(cudaMalloc(&dAs, (size_t)out_f * nblk * 2));
    CK(cudaMalloc(&dB, (size_t)n_tok * in_f));
    CK(cudaMalloc(&dBs, (size_t)n_tok * nblk * 4));
    CK(cudaMalloc(&dC, (size_t)n_tok * out_f * 4));
    CK(cudaMemcpy(dA, hA, (size_t)out_f * in_f, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAs, hAs, (size_t)out_f * nblk * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, (size_t)n_tok * in_f, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBs, hBs, (size_t)n_tok * nblk * 4, cudaMemcpyHostToDevice));

    dim3 grid(out_f / 64, n_tok / 64), block(128);
    q8_gemm_wgmma_v0<<<grid, block>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok);
    CK(cudaDeviceSynchronize());

    // CPU reference on a sample of outputs
    float* hC = (float*)malloc((size_t)n_tok * out_f * 4);
    CK(cudaMemcpy(hC, dC, (size_t)n_tok * out_f * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (int probe = 0; probe < 64; probe++) {
        int r = (probe * 131) % out_f, c = (probe * 197) % n_tok;
        double acc = 0;
        for (int b = 0; b < nblk; b++) {
            long s = 0;
            for (int k = 0; k < 32; k++)
                s += (long)hA[(size_t)r * in_f + b * 32 + k] * hB[(size_t)c * in_f + b * 32 + k];
            acc += (double)s * __half2float(hAs[(size_t)r * nblk + b]) * hBs[(size_t)c * nblk + b];
        }
        double got = hC[(size_t)c * out_f + r];
        double rel = fabs(got - acc) / fmax(fabs(acc), 1e-3);
        if (rel > maxrel) maxrel = rel;
    }
    printf("correctness: max rel err %.3e %s\n", maxrel, maxrel < 1e-3 ? "OK" : "FAIL");

    // timing
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 10; i++) q8_gemm_wgmma_v0<<<grid, block>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) q8_gemm_wgmma_v0<<<grid, block>>>(dA, dAs, dB, dBs, dC, in_f, out_f, n_tok);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    double us = ms * 1000.0 / 50;
    double tops = 2.0 * out_f * in_f * n_tok / (us * 1e6);
    printf("v0: %.1f us  (%.1f TOP/s int8; MMQ class ref ~688us for this shape family)\n", us, tops);
    return 0;
}
