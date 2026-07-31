// Per-shape m=1 microbench for the Q8_0 rp decode matvec (H100 wall table).
// Benches qmatvec_q8_0_mmvq_rp standalone on the Qwen3.5-9B trunk shapes with
// synthetic data: reports achieved GB/s (weight bytes only) vs device peak, per shape.
// Build (on box):
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DMEMRA_PORTABLE_CUDA=1 -DMEMRA_HOPPER_MMA=1 \
//     -o /tmp/bench_q8_shapes tools/bench_q8_shapes.cu
// The production kernels are #included (bench_mapped_qmatvec.cu pattern).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../crates/memra-engine/cu/qmatvec.cu"

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// ldg twin of qmatvec_q8_0_mmvq_rp: identical program, plain cached loads instead of
// __ldcs streaming hints — the wide-shape 80%-of-peak A/B (does the streaming hint help
// or hurt on H100's 50MB L2 when the weight re-walks every token?).
extern "C" __global__ void bench_q8_rp_ldg(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldg((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldg((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// Persistent-CTA twin: grid = exact machine fill; each CTA strides row-groups
// (rg += gridDim.x). Per-row program identical (one warp per row, same walk) ->
// bit-identical; kills the partial-wave tail on wide shapes (5.8 waves at 3072 blocks).
extern "C" __global__ void bench_q8_rp_pers(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes; (void)m;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const signed char* arow = aq;
    const float* adrow = ad;
    int ngroups = (out_f + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    for (int rg = blockIdx.x; rg < ngroups; rg += gridDim.x) {
        int o = rg * MEMRA_MMVQ_ROWS + threadIdx.y;
        if (o >= out_f) continue;
        const unsigned char* wq; const unsigned short* wd;
        q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) {
            int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            float dw = half_to_float(wd[blk]);
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc += dw * adrow[blk] * (float)sumi;
        }
        float v = warp_reduce_sum(acc);
        if (lane == 0) y[o] = v;
    }
}

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

        dim3 block(32, MEMRA_MMVQ_ROWS, 1);
        dim3 grid((s.out_f + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS, 1, 1);
        // warm
        for (int i = 0; i < 20; i++)
            qmatvec_q8_0_mmvq_rp<<<grid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const int REPS = 200;
        dim3 pgrid(min((int)grid.x, p.multiProcessorCount * 4), 1, 1);
        double us_v[3];
        for (int v = 0; v < 3; v++) {
            CK(cudaEventRecord(a));
            for (int i = 0; i < REPS; i++) {
                if (v == 0)      qmatvec_q8_0_mmvq_rp<<<grid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
                else if (v == 1) bench_q8_rp_ldg<<<grid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
                else             bench_q8_rp_pers<<<pgrid, block>>>(W, aq, ad, y, s.in_f, s.out_f, 1, 0);
            }
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b));
            us_v[v] = ms * 1000.0 / REPS;
        }
        double us = us_v[0];
        double gbs = wbytes / (us * 1e3);
        int blocks = grid.x;
        double waves = blocks / (double)(p.multiProcessorCount * 4); // ~4 CTAs/SM at 128thr
        printf("%-18s out=%7d  ldcs %7.1f us (%5.1f%% pk)  ldg %+5.1f%%  pers %+5.1f%%  waves~%.2f\n",
               s.label, s.out_f, us, 100.0 * gbs / peak_gbs,
               100.0 * (us_v[0] - us_v[1]) / us_v[0],
               100.0 * (us_v[0] - us_v[2]) / us_v[0], waves);
        cudaFree(W); cudaFree(aq); cudaFree(ad); cudaFree(y);
    }
    return 0;
}
