// tf32 wgmma canonical-layout probe (solve Route A prerequisite, ledger dd40a605):
// crack the K-major canonical staging + descriptor pairing for
// wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 — the f32-class GEMM kind
// needed to run the K3 triangular-inverse product without bf16 error compounding.
//
// Layout candidates (element (r, kk) of k-step st, 4B elems, k8 steps):
//   F1: st*2048 + (r/8)*256 + (kk/4)*128 + (r%8)*16 + (kk%4)*4    desc(lead,stride)=(128,256)
//   F2: st*4096 + (r/8)*512 + (kk/4)*256 + (r%8)*32 + (kk%4)*4    desc=(256,512)
//   F3: st*2048 + (r/8)*256 + (kk/4)*128 + (r%8)*16 + (kk%4)*4    desc=(256,128) swap
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DTF_F=1 -o /tmp/tf32 tools/probe_tf32.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#ifndef TF_F
#define TF_F 1
#endif
#ifndef TF_LEAD
#if TF_F == 2
#define TF_LEAD 256
#define TF_STRIDE 512
#elif TF_F == 3
#define TF_LEAD 256
#define TF_STRIDE 128
#else
#define TF_LEAD 128
#define TF_STRIDE 256
#endif
#endif

__device__ __forceinline__ unsigned long long make_desc(const void* p, unsigned lead, unsigned stride) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ size_t tf_off(int st, int r, int kk) {
#if TF_F == 2
    return (size_t)st * 4096 + (r / 8) * 512 + (kk / 4) * 256 + (r % 8) * 32 + (kk % 4) * 4;
#else
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 4) * 128 + (r % 8) * 16 + (kk % 4) * 4;
#endif
}

__device__ __forceinline__ void wgmma_tf32(float acc[32], unsigned long long da,
                                           unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void wg_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;"); }

// C(64x64) = A(64x32) . B(64x32)^T over k=32 (4 k8-steps); A/B staged canonical
extern "C" __global__ void tf32_probe(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ Cout) {
    __shared__ __align__(1024) float sA[64 * 32];
    __shared__ __align__(1024) float sB[64 * 32];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 32; idx += 128) {
        int r = idx / 32, kv = idx % 32;
        float av = A[r * 32 + kv], bv = B[r * 32 + kv];
        unsigned ua, ub;
        asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(ua) : "f"(av));
        asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(ub) : "f"(bv));
        *(unsigned*)((char*)sA + tf_off(kv / 8, r, kv % 8)) = ua;
        *(unsigned*)((char*)sB + tf_off(kv / 8, r, kv % 8)) = ub;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    float acc[32];
    wg_fence();
    for (int st = 0; st < 4; st++) {
#if TF_F == 2
        unsigned long long da = make_desc((char*)sA + st * 4096, TF_LEAD, TF_STRIDE);
        unsigned long long db = make_desc((char*)sB + st * 4096, TF_LEAD, TF_STRIDE);
#else
        unsigned long long da = make_desc((char*)sA + st * 2048, TF_LEAD, TF_STRIDE);
        unsigned long long db = make_desc((char*)sB + st * 2048, TF_LEAD, TF_STRIDE);
#endif
        wgmma_tf32(acc, da, db, st == 0 ? 0 : 1);
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

static float tf32r(float x) {           // host tf32 round-to-nearest-away emulation
    unsigned u; memcpy(&u, &x, 4);
    unsigned r = (u + 0x1000u) & 0xFFFFE000u;
    float y; memcpy(&y, &r, 4);
    return y;
}
#include <cstring>
int main() {
    srand(11);
    float *hA = (float*)malloc(64 * 32 * 4), *hB = (float*)malloc(64 * 32 * 4);
    for (int i = 0; i < 64 * 32; i++) {
        hA[i] = ((rand() % 2001) - 1000) * 1e-3f;
        hB[i] = ((rand() % 2001) - 1000) * 1e-3f;
    }
    float* ref = (float*)malloc(64 * 64 * 4);
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++) {
            double s = 0;
            for (int k = 0; k < 32; k++) s += (double)tf32r(hA[r * 32 + k]) * tf32r(hB[c * 32 + k]);
            ref[r * 64 + c] = (float)s;
        }
    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, 64 * 32 * 4)); CK(cudaMalloc(&dB, 64 * 32 * 4)); CK(cudaMalloc(&dC, 64 * 64 * 4));
    CK(cudaMemcpy(dA, hA, 64 * 32 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, 64 * 32 * 4, cudaMemcpyHostToDevice));
    tf32_probe<<<1, 128>>>(dA, dB, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    float* hC = (float*)malloc(64 * 64 * 4);
    CK(cudaMemcpy(hC, dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
    float mr = 0; int bad = 0;
    for (int i = 0; i < 64 * 64; i++) {
        float rl = fabsf(hC[i] - ref[i]) / fmaxf(fabsf(ref[i]), 1e-4f);
        if (rl > mr) mr = rl;
        if (rl > 1e-3f) bad++;
    }
    printf("tf32 m64n64k8 F=%d lead=%d stride=%d: max_rel %.3e bad %d/4096 %s\n",
           TF_F, TF_LEAD, TF_STRIDE, mr, bad, bad == 0 ? "MATCH" : "MISMATCH");
    return bad != 0;
}
