// s8 wgmma canonical-layout probe (round 36, the prefill-crossing arc): crack the
// K-major canonical staging + descriptor for wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8
// — int8 tensor-core GEMMs with exact i32 accumulation (Q8_0 block-scale epilogues
// become exact math; the prior W8A8 refusal covered cuBLASLt epilogues only).
//
// Layout candidates (element (r, kk) of k32-step st, 1B elems):
//   F1: st*2048 + (r/8)*256 + (kk/16)*128 + (r%8)*16 + (kk%16)   desc (128, 256)
//   F2: st*2048 + (r/8)*256 + (kk/8)*64  + (r%8)*8  + (kk%8)    desc (64, 256)?
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DS8_F=1 -o /tmp/s8 tools/probe_s8.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#ifndef S8_F
#define S8_F 1
#endif
#ifndef S8_LEAD
#define S8_LEAD 128
#endif
#ifndef S8_STRIDE
#define S8_STRIDE 256
#endif

__device__ __forceinline__ unsigned long long make_desc(const void* p, unsigned lead, unsigned stride) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ size_t s8_off(int st, int r, int kk) {
#if S8_F == 2
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 64 + (r % 8) * 8 + (kk % 8);
#else
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 16) * 128 + (r % 8) * 16 + (kk % 16);
#endif
}
__device__ __forceinline__ void wgmma_s8(int acc[32], unsigned long long da,
                                         unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p;\n}"
        : "+r"(acc[0]),"+r"(acc[1]),"+r"(acc[2]),"+r"(acc[3]),"+r"(acc[4]),"+r"(acc[5]),"+r"(acc[6]),"+r"(acc[7]),
          "+r"(acc[8]),"+r"(acc[9]),"+r"(acc[10]),"+r"(acc[11]),"+r"(acc[12]),"+r"(acc[13]),"+r"(acc[14]),"+r"(acc[15]),
          "+r"(acc[16]),"+r"(acc[17]),"+r"(acc[18]),"+r"(acc[19]),"+r"(acc[20]),"+r"(acc[21]),"+r"(acc[22]),"+r"(acc[23]),
          "+r"(acc[24]),"+r"(acc[25]),"+r"(acc[26]),"+r"(acc[27]),"+r"(acc[28]),"+r"(acc[29]),"+r"(acc[30]),"+r"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void wg_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;"); }

// C(64x64) = A(64x64) . B(64x64)^T over k=64 (2 k32-steps)
extern "C" __global__ void s8_probe(const signed char* __restrict__ A,
                                    const signed char* __restrict__ B,
                                    int* __restrict__ Cout) {
    __shared__ __align__(1024) signed char sA[64 * 64];
    __shared__ __align__(1024) signed char sB[64 * 64];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 64; idx += 128) {
        int r = idx / 64, kv = idx % 64;
        sA[s8_off(kv / 32, r, kv % 32)] = A[r * 64 + kv];
        sB[s8_off(kv / 32, r, kv % 32)] = B[r * 64 + kv];
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    int acc[32];
    wg_fence();
    for (int st = 0; st < 2; st++) {
        unsigned long long da = make_desc((char*)sA + st * 2048, S8_LEAD, S8_STRIDE);
        unsigned long long db = make_desc((char*)sB + st * 2048, S8_LEAD, S8_STRIDE);
        wgmma_s8(acc, da, db, st == 0 ? 0 : 1);
    }
    wg_commit();
    wg_wait();
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        int n8 = i / 4;
        Cout[(r0 + 0) * 64 + c0 + n8 * 8 + 0] = acc[i + 0];
        Cout[(r0 + 0) * 64 + c0 + n8 * 8 + 1] = acc[i + 1];
        Cout[(r0 + 8) * 64 + c0 + n8 * 8 + 0] = acc[i + 2];
        Cout[(r0 + 8) * 64 + c0 + n8 * 8 + 1] = acc[i + 3];
    }
}

// ---- rescale-cost probe: the V1-exact question (round 36). 64x64 tile, k=4096:
// (a) pure i32 chain: 128 wgmmas, scale_d=1 (the w8a8-class upper bound)
// (b) per-block-exact: scale_d=0 each step, i32 fragment readback, f32 FMA accumulate
//     with per-element scale product (Q8_0 x q8_1 exact math)
// Verdict drives V1(exact) vs V2(gated w8a8-class config) for the prefill GEMM arc.
extern "C" __global__ void __launch_bounds__(128, 1)
s8_chain(const signed char* __restrict__ A, const signed char* __restrict__ B,
         int* __restrict__ Cout, int K, int iters) {
    __shared__ __align__(1024) signed char sA[64 * 128];   // 4 k32-steps resident
    __shared__ __align__(1024) signed char sB[64 * 128];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int r = idx / 128, kv = idx % 128;
        sA[s8_off(kv / 32, r, kv % 32)] = A[r * 128 + kv];
        sB[s8_off(kv / 32, r, kv % 32)] = B[r * 128 + kv];
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    int acc[32];
    for (int it = 0; it < iters; it++) {
        wg_fence();
        for (int kk = 0; kk < K / 32; kk++) {
            int st = kk & 3;                       // cycle the resident tiles
            unsigned long long da = make_desc((char*)sA + st * 2048, S8_LEAD, S8_STRIDE);
            unsigned long long db = make_desc((char*)sB + st * 2048, S8_LEAD, S8_STRIDE);
            wgmma_s8(acc, da, db, kk == 0 ? 0 : 1);
        }
        wg_commit();
        wg_wait();
    }
    const int warp = tid / 32, lane = tid % 32;
    Cout[warp * 32 + lane] = acc[0] + acc[31];
}

extern "C" __global__ void __launch_bounds__(128, 1)
s8_rescale(const signed char* __restrict__ A, const signed char* __restrict__ B,
           const float* __restrict__ scA, const float* __restrict__ scB,
           float* __restrict__ Cout, int K, int iters) {
    __shared__ __align__(1024) signed char sA[64 * 128];
    __shared__ __align__(1024) signed char sB[64 * 128];
    __shared__ float fsA[64 * 4], fsB[64 * 4];     // per-row per-k32-block scales (resident window)
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int r = idx / 128, kv = idx % 128;
        sA[s8_off(kv / 32, r, kv % 32)] = A[r * 128 + kv];
        sB[s8_off(kv / 32, r, kv % 32)] = B[r * 128 + kv];
    }
    for (int idx = tid; idx < 64 * 4; idx += 128) { fsA[idx] = scA[idx]; fsB[idx] = scB[idx]; }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float fac[32];
    #pragma unroll
    for (int q = 0; q < 32; q++) fac[q] = 0.0f;
    int acc[32];
    for (int it = 0; it < iters; it++) {
        for (int kk = 0; kk < K / 32; kk++) {
            int st = kk & 3;
            unsigned long long da = make_desc((char*)sA + st * 2048, S8_LEAD, S8_STRIDE);
            unsigned long long db = make_desc((char*)sB + st * 2048, S8_LEAD, S8_STRIDE);
            wg_fence();
            wgmma_s8(acc, da, db, 0);              // fresh i32 block-dot each step
            wg_commit();
            wg_wait();
            // exact epilogue: fac[m,n] += i32 * sA[m,block] * sB[n,block]
            float a0 = fsA[(r0 + 0) * 4 + st], a1 = fsA[(r0 + 8) * 4 + st];
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cc = c0 + n8 * 8;
                float b0 = fsB[(cc + 0) * 4 + st], b1 = fsB[(cc + 1) * 4 + st];
                fac[q + 0] += (float)acc[q + 0] * a0 * b0;
                fac[q + 1] += (float)acc[q + 1] * a0 * b1;
                fac[q + 2] += (float)acc[q + 2] * a1 * b0;
                fac[q + 3] += (float)acc[q + 3] * a1 * b1;
            }
        }
    }
    Cout[warp * 32 + lane] = fac[0] + fac[31];
}


// pipelined per-block-exact: ping-pong acc banks, wait<1> overlaps rescale with the
// NEXT block's wgmma in flight (the drain in s8_rescale was the cost, not the FMAs)
template<int N> __device__ __forceinline__ void wg_wait_n() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
}
extern "C" __global__ void __launch_bounds__(128, 1)
s8_rescale_pipe(const signed char* __restrict__ A, const signed char* __restrict__ B,
                const float* __restrict__ scA, const float* __restrict__ scB,
                float* __restrict__ Cout, int K, int iters) {
    __shared__ __align__(1024) signed char sA[64 * 128];
    __shared__ __align__(1024) signed char sB[64 * 128];
    __shared__ float fsA[64 * 4], fsB[64 * 4];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int r = idx / 128, kv = idx % 128;
        sA[s8_off(kv / 32, r, kv % 32)] = A[r * 128 + kv];
        sB[s8_off(kv / 32, r, kv % 32)] = B[r * 128 + kv];
    }
    for (int idx = tid; idx < 64 * 4; idx += 128) { fsA[idx] = scA[idx]; fsB[idx] = scB[idx]; }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float fac[32];
    #pragma unroll
    for (int q = 0; q < 32; q++) fac[q] = 0.0f;
    int accA[32], accB[32];
    for (int it = 0; it < iters; it++) {
        for (int kk = 0; kk < K / 32; kk++) {
            int st = kk & 3;
            unsigned long long da = make_desc((char*)sA + st * 2048, S8_LEAD, S8_STRIDE);
            unsigned long long db = make_desc((char*)sB + st * 2048, S8_LEAD, S8_STRIDE);
            int* bank = (kk & 1) ? accB : accA;
            wg_fence();
            wgmma_s8(bank, da, db, 0);
            wg_commit();
            if (kk > 0) {
                wg_wait_n<1>();                    // previous bank done; current in flight
                int* prev = (kk & 1) ? accA : accB;
                int pst = (kk - 1) & 3;
                float a0 = fsA[(r0 + 0) * 4 + pst], a1 = fsA[(r0 + 8) * 4 + pst];
                #pragma unroll
                for (int q = 0; q < 32; q += 4) {
                    int n8 = q / 4;
                    int cc = c0 + n8 * 8;
                    float b0 = fsB[(cc + 0) * 4 + pst], b1 = fsB[(cc + 1) * 4 + pst];
                    fac[q + 0] += (float)prev[q + 0] * a0 * b0;
                    fac[q + 1] += (float)prev[q + 1] * a0 * b1;
                    fac[q + 2] += (float)prev[q + 2] * a1 * b0;
                    fac[q + 3] += (float)prev[q + 3] * a1 * b1;
                }
            }
        }
        wg_wait_n<0>();
        {   // drain the last bank
            int last = (K / 32 - 1);
            int* prev = (last & 1) ? accB : accA;
            int pst = last & 3;
            float a0 = fsA[(r0 + 0) * 4 + pst], a1 = fsA[(r0 + 8) * 4 + pst];
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cc = c0 + n8 * 8;
                float b0 = fsB[(cc + 0) * 4 + pst], b1 = fsB[(cc + 1) * 4 + pst];
                fac[q + 0] += (float)prev[q + 0] * a0 * b0;
                fac[q + 1] += (float)prev[q + 1] * a0 * b1;
                fac[q + 2] += (float)prev[q + 2] * a1 * b0;
                fac[q + 3] += (float)prev[q + 3] * a1 * b1;
            }
        }
    }
    Cout[warp * 32 + lane] = fac[0] + fac[31];
}

int main() {
    srand(23);
    signed char *hA = (signed char*)malloc(64 * 64), *hB = (signed char*)malloc(64 * 64);
    for (int i = 0; i < 64 * 64; i++) { hA[i] = (signed char)(rand() % 255 - 127); hB[i] = (signed char)(rand() % 255 - 127); }
    int* ref = (int*)malloc(64 * 64 * 4);
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++) {
            int s = 0;
            for (int k = 0; k < 64; k++) s += (int)hA[r * 64 + k] * (int)hB[c * 64 + k];
            ref[r * 64 + c] = s;
        }
    signed char *dA, *dB; int* dC;
    CK(cudaMalloc(&dA, 64 * 64)); CK(cudaMalloc(&dB, 64 * 64)); CK(cudaMalloc(&dC, 64 * 64 * 4));
    CK(cudaMemcpy(dA, hA, 64 * 64, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, 64 * 64, cudaMemcpyHostToDevice));
    s8_probe<<<1, 128>>>(dA, dB, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int* hC = (int*)malloc(64 * 64 * 4);
    CK(cudaMemcpy(hC, dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 64 * 64 && bad < 4; i++)
        if (hC[i] != ref[i]) { if (bad < 2) printf("  (%d,%d): got %d want %d\n", i / 64, i % 64, hC[i], ref[i]); bad++; }
    int total_bad = 0;
    for (int i = 0; i < 64 * 64; i++) if (hC[i] != ref[i]) total_bad++;
    printf("s8 m64n64k32 F=%d lead=%d stride=%d: %d/4096 bad %s\n",
           S8_F, S8_LEAD, S8_STRIDE, total_bad, total_bad == 0 ? "MATCH (EXACT)" : "MISMATCH");
    if (total_bad) return 1;

    // ---- rescale-cost probe (single CTA, k=4096, 100 iters) ----
    {
        signed char *dA2, *dB2; float *dsc, *dCo; int* dCi;
        CK(cudaMalloc(&dA2, 64 * 128)); CK(cudaMalloc(&dB2, 64 * 128));
        CK(cudaMalloc(&dsc, 64 * 4 * 4)); CK(cudaMalloc(&dCo, 128 * 4)); CK(cudaMalloc(&dCi, 128 * 4));
        CK(cudaMemset(dA2, 1, 64 * 128)); CK(cudaMemset(dB2, 1, 64 * 128));
        CK(cudaMemset(dsc, 0, 64 * 4 * 4));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const int K = 4096, IT = 100;
        s8_chain<<<1, 128>>>(dA2, dB2, dCi, K, 3);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        s8_chain<<<1, 128>>>(dA2, dB2, dCi, K, IT);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms1; CK(cudaEventElapsedTime(&ms1, a, b));
        s8_rescale<<<1, 128>>>(dA2, dB2, dsc, dsc, dCo, K, 3);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        s8_rescale<<<1, 128>>>(dA2, dB2, dsc, dsc, dCo, K, IT);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms2; CK(cudaEventElapsedTime(&ms2, a, b));
        printf("rescale-cost: i32-chain %.1fus/iter vs per-block-exact %.1fus/iter = %.2fx overhead\n",
               ms1 * 1000.0f / IT, ms2 * 1000.0f / IT, ms2 / ms1);
        s8_rescale_pipe<<<1, 128>>>(dA2, dB2, dsc, dsc, dCo, K, 3);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        s8_rescale_pipe<<<1, 128>>>(dA2, dB2, dsc, dsc, dCo, K, IT);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms3; CK(cudaEventElapsedTime(&ms3, a, b));
        printf("pipelined-exact: %.1fus/iter = %.2fx vs chain (V1 lives if ~1.2x)\n",
               ms3 * 1000.0f / IT, ms3 / ms1);
    }
    return 0;
}
