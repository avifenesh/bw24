// W8A8 per-row decision probe (task 9 arc-4 gate, ARCHITECTURE-H100.md): is the vLLM
// numeric config (int8 W per-row scale x int8 act per-token scale, s32 GEMM + f32
// dequant epilogue) worth the cutlass integration on H100?
//
// Measures at the real m=512 shapes:
//   1. cublasGemmEx int8 s32-accumulate GEMM rate (the IMMA ceiling Lt can reach)
//   2. the dequant epilogue cost (y_f32 = s32 * row_scale x col_scale, float4 kernel)
//   3. net vs the SHIPPED fp16-mirror times (measured 2026-07-26)
// Decision band: net >= 1.4x fp16 -> arc justified; else refuted (epilogue tax eats it).
//
// Build (box): nvcc -O3 -arch=sm_90a -lcublas -o /tmp/lti8 tools/bench_lt_i8.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_) { printf("cuBLAS %d @%d\n", (int)s_, __LINE__); exit(1);} } while (0)

// dequant epilogue: y[m,n] f32 = s32[m,n] * arow[m] * wcol[n]  (token-major y, col=out row)
extern "C" __global__ void dequant_rc(const int* __restrict__ s, const float* __restrict__ arow,
                                      const float* __restrict__ wcol, float* __restrict__ y,
                                      int m, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total = m * n;
    if (base + 3 < total) {
        int4 v = *(const int4*)(s + base);
        int row = base / n, col0 = base % n;   // 4 elems same row when n%4==0
        float a = arow[row];
        float4 o = make_float4(v.x * a * wcol[col0], v.y * a * wcol[col0 + 1],
                               v.z * a * wcol[col0 + 2], v.w * a * wcol[col0 + 3]);
        *(float4*)(y + base) = o;
    } else {
        for (int i = base; i < total; i++)
            y[i] = s[i] * arow[i / n] * wcol[i % n];
    }
}

struct ShapeRef { int in_f, out_f; float f16_us; const char* tag; };

int main() {
    const int m = getenv("BENCH_M") ? atoi(getenv("BENCH_M")) : 512;
    // f16_us = shipped fp16-mirror per-launch medians (nsys/probe 2026-07-26)
    ShapeRef shapes[] = {
        {4096, 12288, 78.4f, "wqkv (lin)"},
        {4096,  8192, 52.0f, "mid"},
        {4096,  4096, 28.1f, "square"},
        {11008, 4096, 67.2f, "ffn_down"},
        {4096, 11008, 69.7f, "ffn_gate/up"},
        {4096,  1024, 11.6f, "small"},
    };
    cublasHandle_t h; CB(cublasCreate(&h));
    srand(7);
    for (auto& sh : shapes) {
        int in_f = sh.in_f, out_f = sh.out_f;
        signed char *dW, *dX; int* dS; float *dY, *dAs, *dWs;
        CK(cudaMalloc(&dW, (size_t)out_f * in_f));
        CK(cudaMalloc(&dX, (size_t)m * in_f));
        CK(cudaMalloc(&dS, (size_t)m * out_f * 4));
        CK(cudaMalloc(&dY, (size_t)m * out_f * 4));
        CK(cudaMalloc(&dAs, m * 4));
        CK(cudaMalloc(&dWs, out_f * 4));
        CK(cudaMemset(dW, 3, (size_t)out_f * in_f));
        CK(cudaMemset(dX, 2, (size_t)m * in_f));
        CK(cudaMemset(dAs, 0, m * 4));
        CK(cudaMemset(dWs, 0, out_f * 4));

        int alpha = 1, beta = 0;
        auto gemm = [&]() {
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out_f, m, in_f,
                            &alpha, dW, CUDA_R_8I, in_f, dX, CUDA_R_8I, in_f,
                            &beta, dS, CUDA_R_32I, out_f,
                            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        auto epi = [&]() {
            int total = m * out_f;
            int blocks = (total / 4 + 255) / 256;
            dequant_rc<<<blocks, 256>>>(dS, dAs, dWs, dY, m, out_f);
        };
        for (int i = 0; i < 10; i++) { gemm(); epi(); }
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; i++) gemm();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double g_us = ms * 10.0;
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; i++) epi();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        double e_us = ms * 10.0;
        double tops = 2.0 * out_f * in_f * m / (g_us * 1e6);
        // MEASURED f16 reference at THIS m (2026-07-31: the hardcoded table is m=512-only;
        // the m=2048 crossing decision needs a live fp16 GemmEx run, same reps protocol).
        __half *fW, *fX; float* fY;
        CK(cudaMalloc(&fW, (size_t)out_f * in_f * 2));
        CK(cudaMalloc(&fX, (size_t)m * in_f * 2));
        CK(cudaMalloc(&fY, (size_t)m * out_f * 4));
        CK(cudaMemset(fW, 0x3c, (size_t)out_f * in_f * 2));
        CK(cudaMemset(fX, 0x3c, (size_t)m * in_f * 2));
        float falpha = 1.0f, fbeta = 0.0f;
        auto fgemm = [&]() {
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out_f, m, in_f,
                            &falpha, fW, CUDA_R_16F, in_f, fX, CUDA_R_16F, in_f,
                            &fbeta, fY, CUDA_R_32F, out_f,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        for (int i = 0; i < 10; i++) fgemm();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; i++) fgemm();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        double f_us = ms * 10.0;
        printf("%-12s in=%5d out=%6d | i8 gemm %6.1fus (%4.0f TOP)  epi %5.1fus  net %6.1fus  vs f16(m=%d) %6.1fus  net-speedup %.2fx  [table512 %5.1fus]\n",
               sh.tag, in_f, out_f, g_us, tops, e_us, g_us + e_us, m, f_us,
               f_us / (g_us + e_us), sh.f16_us);
        cudaFree(fW); cudaFree(fX); cudaFree(fY);
        cudaFree(dW); cudaFree(dX); cudaFree(dS); cudaFree(dY); cudaFree(dAs); cudaFree(dWs);
    }
    return 0;
}
