// ACF per-TU runner for the PRODUCTION fa3 prefill kernel (carry-over 8, 2026-07-31).
// Round-38 found harness-searched ACFs do NOT transfer across TUs (byte-identical SASS
// on fa3_prefill.cu) — this runner IS the production TU's objective: it links directly
// against fa3_prefill.cu compiled in the same nvcc invocation, so an ACF applied via
// -Xptxas --apply-controls lands on the real memra_fa3_prefill kernel being timed.
//
// Build (box, 13.3 for --apply-controls):
//   ~/cuda-13.3.1/bin/nvcc -std=c++17 -O3 -arch=sm_90a --expt-relaxed-constexpr \
//     [-Xptxas=--apply-controls=CAND.acf] \
//     -I/usr/local/cuda-13.1/targets/x86_64-linux/include -L/usr/local/cuda-13.1/lib64 \
//     -lcuda -o /tmp/acfr tools/acf_fa3_runner.cu crates/memra-engine/cu/fa3_prefill.cu
// Run: /tmp/acfr [T]   -> "fa3_prefill T=2048: NNNus/call" + a correctness fingerprint
// (output checksum — the ACF search gates candidates on the fingerprint matching the
// no-ACF baseline: scheduling controls must not change results).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" int memra_fa3_prefill(const void*, const void*, const void*, float*,
                                int, int, int, int, float, void*);

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

int main(int argc, char** argv) {
    // qwen35 attention shape (the promoted single-seq config): H=16, HKV=4, D=256.
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    const int H = 16, HKV = 4, D = 256;
    const float scale = 1.0f / sqrtf((float)D);
    size_t qn = (size_t)T * H * D, kn = (size_t)T * HKV * D;
    __nv_bfloat16 *dQ, *dK, *dV; float* dO;
    CK(cudaMalloc(&dQ, qn * 2)); CK(cudaMalloc(&dK, kn * 2));
    CK(cudaMalloc(&dV, kn * 2)); CK(cudaMalloc(&dO, qn * 4));
    {
        __nv_bfloat16* h = (__nv_bfloat16*)malloc(qn * 2);
        srand(7);
        for (size_t i = 0; i < qn; i++) h[i] = __float2bfloat16((rand() % 200 - 100) * 0.01f);
        CK(cudaMemcpy(dQ, h, qn * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < kn; i++) h[i] = __float2bfloat16((rand() % 200 - 100) * 0.01f);
        CK(cudaMemcpy(dK, h, kn * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < kn; i++) h[i] = __float2bfloat16((rand() % 200 - 100) * 0.011f);
        CK(cudaMemcpy(dV, h, kn * 2, cudaMemcpyHostToDevice));
        free(h);
    }
    int rc = memra_fa3_prefill(dQ, dK, dV, dO, T, H, HKV, D, scale, nullptr);
    if (rc) { printf("fa3_prefill rc=%d\n", rc); return 1; }
    CK(cudaDeviceSynchronize());
    // correctness fingerprint: double-sum of |o| (schedule-invariant iff results identical)
    {
        float* h = (float*)malloc(qn * 4);
        CK(cudaMemcpy(h, dO, qn * 4, cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (size_t i = 0; i < qn; i++) s += fabs((double)h[i]);
        printf("fingerprint %.6e\n", s);
        free(h);
    }
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 10; i++) memra_fa3_prefill(dQ, dK, dV, dO, T, H, HKV, D, scale, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) memra_fa3_prefill(dQ, dK, dV, dO, T, H, HKV, D, scale, nullptr);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    printf("fa3_prefill T=%d: %.0fus/call\n", T, ms * 20.0);
    return 0;
}
