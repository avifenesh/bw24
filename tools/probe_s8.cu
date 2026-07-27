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
    return total_bad != 0;
}
