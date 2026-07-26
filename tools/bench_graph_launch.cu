// Graph-launch economics probe (task #15 decision gate, ARCHITECTURE-H100.md):
// does one cuGraphLaunch of a K-kernel segment beat K eager submissions, and
// at what K does it pay? The interleaved A/B refuted K=2 (segment net -0.7%);
// this measures the crossover directly so the S-prep/S-attn slab refactor
// (multi-hour) is priced with data instead of projection.
//
// Two regimes measured, both with kernels sized like the real prime glue
// (~2-6us: rms-norm / add / repack at bucket=512 shapes):
//   1. HOST-BOUND: submission wall time with NO sync until the end at huge
//      depth — what the prime loop pays if the GPU ever gets ahead of host.
//   2. E2E: wall time for a fixed kernel count, stream-serialized, eager vs
//      graphs of size K in {2, 4, 8, 16} — the number that matches pp512.
//
// Build (box): nvcc -O3 -arch=sm_90a -o /tmp/graphlaunch tools/bench_graph_launch.cu
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

// glue-kernel stand-in: streaming add over m x n floats (bucket-shaped)
extern "C" __global__ void glue_add(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ y, int nelem) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < nelem) {
        float4 va = *(const float4*)(a + i);
        float4 vb = *(const float4*)(b + i);
        *(float4*)(y + i) = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    } else {
        for (; i < nelem; i++) y[i] = a[i] + b[i];
    }
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // 512 x 4096 f32 = the h-slab shape; glue_add on it runs ~4-5us (measured band
    // of the real add/norm glue kernels at bucket 512)
    int nelem = 512 * 4096;
    if (argc > 1) nelem = atoi(argv[1]);
    const int TOTAL = 960;          // divisible by 2,4,8,16; ~= glue launches/prime
    const int REPS = 50;

    float *a, *b, *y;
    CK(cudaMalloc(&a, (size_t)nelem * 4));
    CK(cudaMalloc(&b, (size_t)nelem * 4));
    CK(cudaMalloc(&y, (size_t)nelem * 4));
    CK(cudaMemset(a, 0, (size_t)nelem * 4));
    CK(cudaMemset(b, 0, (size_t)nelem * 4));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int blocks = (nelem / 4 + 255) / 256;

    // single-kernel GPU duration (context for the reader)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int i = 0; i < 10; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < 200; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("glue kernel GPU duration: %.2fus (nelem=%d)\n", ms * 1000.0 / 200.0, nelem);
    }

    double eager_sub = 1.0;
    // 1. HOST submission cost: enqueue TOTAL launches, measure wall BEFORE sync
    {
        // warm
        for (int i = 0; i < TOTAL; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
        CK(cudaStreamSynchronize(s));
        double best = 1e9;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_s();
            for (int i = 0; i < TOTAL; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
            double t1 = now_s();
            CK(cudaStreamSynchronize(s));
            double us = (t1 - t0) * 1e6 / TOTAL;
            if (us < best) best = us;
        }
        printf("eager submission host cost: %.3fus/kernel (best of %d)\n", best, REPS);
        eager_sub = best;
    }

    // 2. graphs of size K: host submission cost + E2E wall
    for (int K : {2, 4, 8, 16}) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < K; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        int launches = TOTAL / K;
        // warm
        for (int i = 0; i < launches; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        double best_sub = 1e9, best_e2e = 1e9;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_s();
            for (int i = 0; i < launches; i++) CK(cudaGraphLaunch(ge, s));
            double t1 = now_s();
            CK(cudaStreamSynchronize(s));
            double t2 = now_s();
            double sub = (t1 - t0) * 1e6 / launches;
            double e2e = (t2 - t0) * 1e6;
            if (sub < best_sub) best_sub = sub;
            if (e2e < best_e2e) best_e2e = e2e;
        }
        // eager E2E reference at the same TOTAL
        double eager_e2e = 1e9;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_s();
            for (int i = 0; i < TOTAL; i++) glue_add<<<blocks, 256, 0, s>>>(a, b, y, nelem);
            CK(cudaStreamSynchronize(s));
            double us = (now_s() - t0) * 1e6;
            if (us < eager_e2e) eager_e2e = us;
        }
        printf("K=%2d | graph launch host %.3fus (= %.2f kernel submissions) | "
               "E2E %6.0fus vs eager %6.0fus (%+.1f%%)\n",
               K, best_sub, best_sub / eager_sub, best_e2e, eager_e2e,
               100.0 * (eager_e2e / best_e2e - 1.0));
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
    }
    return 0;
}
