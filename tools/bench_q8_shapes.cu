// Per-shape m=1 microbench for the Q8_0 rp decode matvec (H100 wall table).
// Benches qmatvec_q8_0_mmvq_rp standalone on the Qwen3.5-9B trunk shapes with
// synthetic data: reports achieved GB/s (weight bytes only) vs device peak, per shape.
// Build (on box):
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DBW24_PORTABLE_CUDA=1 -DBW24_HOPPER_MMA=1 \
//     -o /tmp/bench_q8_shapes tools/bench_q8_shapes.cu
// The production kernels are #included (bench_mapped_qmatvec.cu pattern).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../crates/bw24-engine/cu/qmatvec.cu"

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

int main() {
    // (in_f, out_f, label) — the 9B trunk decode shapes (per-layer) + lm_head.
    struct Shape { int in_f, out_f; const char* label; };
    Shape shapes[] = {
        {4096, 12288, "wqkv (lin)"},
        {4096,  4096, "wqkv_gate"},
        {4096,  4096, "ssm_out"},
        {4096, 11008, "ffn_gate/up"},
        {11008, 4096, "ffn_down"},
        {4096,    32, "ssm_beta/alpha"},
        {4096,  2048, "attn qkv (full)"},
        {4096, 248320, "lm_head"},
    };
    int dev = 0;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    double peak_gbs = 3350.0;   // H100 SXM HBM3 spec (memoryClockRate gone in CUDA 13)
    printf("device %s, %d SMs, theoretical %.0f GB/s\n", p.name, p.multiProcessorCount, peak_gbs);

    for (auto& s : shapes) {
        int nblk = s.in_f / 32;
        size_t wbytes = (size_t)s.out_f * nblk * 34;   // split-plane total (32B q + 2B d)
        unsigned char* W; CK(cudaMalloc(&W, wbytes));
        CK(cudaMemset(W, 1, wbytes));
        signed char* aq; CK(cudaMalloc(&aq, s.in_f));
        CK(cudaMemset(aq, 1, s.in_f));
        float* ad; CK(cudaMalloc(&ad, nblk * 4));
        CK(cudaMemset(ad, 0, nblk * 4));
        float* y; CK(cudaMalloc(&y, s.out_f * 4));

        dim3 block(32, BW24_MMVQ_ROWS, 1);
        dim3 grid((s.out_f + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS, 1, 1);
        // warm
        for (int i = 0; i < 20; i++)
            qmatvec_q8_0_mmvq_rp<<<grid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const int REPS = 200;
        CK(cudaEventRecord(a));
        for (int i = 0; i < REPS; i++)
            qmatvec_q8_0_mmvq_rp<<<grid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double us = ms * 1000.0 / REPS;
        double gbs = wbytes / (us * 1e3);
        int blocks = grid.x;
        double waves = blocks / (double)(p.multiProcessorCount * 4); // ~4 CTAs/SM at 128thr
        printf("%-18s in=%6d out=%7d  %8.1f us  %7.1f GB/s (%4.1f%% peak)  blocks=%6d waves~%.2f\n",
               s.label, s.in_f, s.out_f, us, gbs, 100.0 * gbs / peak_gbs, blocks, waves);
        cudaFree(W); cudaFree(aq); cudaFree(ad); cudaFree(y);
    }
    return 0;
}
