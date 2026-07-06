// PDL (Programmatic Dependent Launch) win-probe for sm_120 laptop.
// Chain of N dependent tiny kernels (the bw24 MoE decode launch pattern: 2-8us kernels,
// each reading the previous one's output). Three arms:
//   A: plain stream-ordered launches (cudaLaunchKernel, the bw24 status quo)
//   B: PDL launches (cudaLaunchKernelEx + PROGRAMMATIC_STREAM_SERIALIZATION; kernel does
//      griddepcontrol wait before its first read and launch_dependents after last write)
//   C: PDL with the sync placed late (overlap prolog with predecessor tail)
// Reports ns/kernel for each arm. If B/C don't beat A by >= 5% here, PDL is closed for this rig.
#include <cuda_runtime.h>
#include <cstdio>

#define N_CHAIN 2000
#define ELEMS (2*1024*1024)

__global__ void step_plain(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 1.0001f + 0.5f;
}

__global__ void step_pdl(const float* in, float* out, int n) {
    // prolog work that does NOT depend on `in`: index math (cheap here, but the mechanism
    // is what we time — real kernels have real prologs).
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 900 || __CUDA_ARCH__ >= 1200
    cudaGridDependencySynchronize();
#endif
    if (i < n) out[i] = in[i] * 1.0001f + 0.5f;
#if __CUDA_ARCH__ >= 900 || __CUDA_ARCH__ >= 1200
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__global__ void step_pdl_late(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float bias = 0.5f;              // pretend-prolog
#if __CUDA_ARCH__ >= 900 || __CUDA_ARCH__ >= 1200
    cudaTriggerProgrammaticLaunchCompletion();  // release next launch EARLY (before our reads!)
    cudaGridDependencySynchronize();            // then wait for OUR inputs
#endif
    if (i < n) out[i] = in[i] * 1.0001f + bias;
}

static float run(void (*k)(const float*, float*, int), float* a, float* b, bool pdl) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    dim3 grid((ELEMS + 255) / 256), block(256);
    // warmup
    for (int i = 0; i < 50; i++) {
        const float* in = (i & 1) ? b : a; float* out = (i & 1) ? a : b;
        if (pdl) {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = grid; cfg.blockDim = block; cfg.attrs = &attr; cfg.numAttrs = 1;
            void* args[] = {(void*)&in, (void*)&out, (void*)&(int&)*(new int(ELEMS))};
            int n = ELEMS; void* a2[] = {&in, &out, &n};
            cudaLaunchKernelExC(&cfg, (const void*)k, a2);
        } else {
            k<<<grid, block>>>(in, out, ELEMS);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < N_CHAIN; i++) {
        const float* in = (i & 1) ? b : a; float* out = (i & 1) ? a : b;
        if (pdl) {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = grid; cfg.blockDim = block; cfg.attrs = &attr; cfg.numAttrs = 1;
            int n = ELEMS; void* a2[] = {&in, &out, &n};
            cudaLaunchKernelExC(&cfg, (const void*)k, a2);
        } else {
            k<<<grid, block>>>(in, out, ELEMS);
        }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return ms * 1e6f / N_CHAIN;  // ns per kernel
}

int main() {
    float *a, *b;
    cudaMalloc(&a, ELEMS * 4); cudaMalloc(&b, ELEMS * 4);
    cudaMemset(a, 0, ELEMS * 4);
    for (int rep = 0; rep < 3; rep++) {
        float pa = run(step_plain, a, b, false);
        float pb = run(step_pdl, a, b, true);
        float pc = run(step_pdl_late, a, b, true);
        printf("rep %d: plain %.0f ns/k | pdl-sync-first %.0f (%.1f%%) | pdl-launch-first %.0f (%.1f%%)\n",
               rep, pa, pb, (pa-pb)/pa*100, pc, (pa-pc)/pa*100);
    }
    return 0;
}
