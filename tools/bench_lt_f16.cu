// fp16-dequant prefill GEMM probe (task 8 next swing, ARCHITECTURE-H100.md): if Q8_0
// weights get a RESIDENT fp16 mirror at load, prefill projections become plain fp16
// tensor-core GEMMs (f32 accumulate, ZERO per-block folds — the thing wgmma-int8 can't
// have). This measures cublasGemmEx fp16 at the real m=512 shapes vs the MMQ medians.
// Numerics: new config, opt-in seam + argmax/tolerance gate (MEMRA_PP_FP8 precedent).
//
// Build (box): nvcc -O3 -arch=sm_90a -lcublas -o /tmp/ltf16 tools/bench_lt_f16.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_) { printf("cuBLAS %d @%d\n", (int)s_, __LINE__); exit(1);} } while (0)

struct ShapeRef { int in_f, out_f, mmq_us; const char* tag; };

int main() {
    const int m = 512;
    ShapeRef shapes[] = {
        {4096, 12288, 253, "wqkv (lin)"},
        {4096,  8192, 168, "mid"},
        {4096,  4096,  90, "square"},
        {11008, 4096, 247, "ffn_down"},
        {4096, 11008, 236, "ffn_gate/up"},
        {4096,  1024,  82, "small"},
    };
    cublasHandle_t h; CB(cublasCreate(&h));
    srand(7);
    for (auto& sh : shapes) {
        int in_f = sh.in_f, out_f = sh.out_f;
        // W fp16 [out,in] row-major; acts fp16 [m,in] row-major; y f32 [m,out] row-major.
        half *dW, *dX; float* dY;
        CK(cudaMalloc(&dW, (size_t)out_f * in_f * 2));
        CK(cudaMalloc(&dX, (size_t)m * in_f * 2));
        CK(cudaMalloc(&dY, (size_t)m * out_f * 4));
        {
            half* tmp = (half*)malloc((size_t)out_f * in_f * 2);
            for (size_t i = 0; i < (size_t)out_f * in_f; i++) tmp[i] = __float2half((rand() % 255 - 127) * 0.01f);
            CK(cudaMemcpy(dW, tmp, (size_t)out_f * in_f * 2, cudaMemcpyHostToDevice));
            free(tmp);
            half* tx = (half*)malloc((size_t)m * in_f * 2);
            for (size_t i = 0; i < (size_t)m * in_f; i++) tx[i] = __float2half((rand() % 255 - 127) * 0.005f);
            CK(cudaMemcpy(dX, tx, (size_t)m * in_f * 2, cudaMemcpyHostToDevice));
            free(tx);
        }
        // col-major view: y_cm[out,m] = W_cm[in,out]^T * X_cm[in,m]
        float alpha = 1.0f, beta = 0.0f;
        auto run = [&]() {
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out_f, m, in_f,
                            &alpha, dW, CUDA_R_16F, in_f, dX, CUDA_R_16F, in_f,
                            &beta, dY, CUDA_R_32F, out_f,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        for (int i = 0; i < 10; i++) run();
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; i++) run();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double us = ms * 1000.0 / 100;
        double tf = 2.0 * out_f * in_f * m / (us * 1e6);
        printf("%-12s in=%5d out=%6d | f16 %6.1fus (%5.0f TF)  MMQ %4dus  f16/MMQ %.2fx\n",
               sh.tag, in_f, out_f, us, tf, sh.mmq_us, us / sh.mmq_us);
        cudaFree(dW); cudaFree(dX); cudaFree(dY);
    }
    return 0;
}
