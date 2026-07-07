// fp8_lt_scale_probe.cu — probe 2 for the FP8-act prefill card (sm_120):
//  (a) does cuBLASLt accept CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F for the activation
//      (per-token f32 scale vector) with a scalar A scale on sm_120? If yes the epilogue is free.
//  (b) cost of the f32 -> fp8-e4m3 per-token-scale activation quantize kernel at 27B shapes.
//  (c) end-to-end numeric spot check: dequant(fp8(act)) @ W vs f32 reference.
//
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp8_lt_scale_probe.cu -o t -lcublasLt

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    printf("cublasLt ERR %s:%d status=%d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

// per-token (row) f32 -> fp8 e4m3 quantize: one block per token row, two passes in smem
// (amax reduce, then scale+convert). k up to 17408. scale[t] = amax/448 (e4m3 max normal).
__global__ void quantize_fp8_act(const float* __restrict__ x, __nv_fp8_e4m3* __restrict__ q,
                                 float* __restrict__ s, int k) {
    const int t = blockIdx.x;
    const float* row = x + (size_t)t * k;
    __shared__ float red[256];
    float amax = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) amax = fmaxf(amax, fabsf(row[i]));
    red[threadIdx.x] = amax; __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + w]);
        __syncthreads();
    }
    const float sc = red[0] > 0.f ? red[0] / 448.f : 1.f;
    const float inv = 1.f / sc;
    if (threadIdx.x == 0) s[t] = sc;
    __nv_fp8_e4m3* qrow = q + (size_t)t * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        qrow[i] = __nv_fp8_e4m3(row[i] * inv);
}

__global__ void fill_f32(float* p, size_t n, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    p[i] = (((h >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 2.0f;
}
__global__ void fill_fp8(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    p[i] = __nv_fp8_e4m3((((h >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 0.5f);
}

int main() {
    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    const int n = 12288, k = 5120;   // q_gate shape (largest attn GEMM)
    const int ms[] = {512, 2048, 4096, 6257};
    const int MMAX = 6257;

    float* dX; CK(cudaMalloc(&dX, (size_t)MMAX * k * 4));
    __nv_fp8_e4m3* dA; CK(cudaMalloc(&dA, (size_t)MMAX * k));
    float* dS; CK(cudaMalloc(&dS, MMAX * 4));
    __nv_fp8_e4m3* dW; CK(cudaMalloc(&dW, (size_t)n * k));
    float* dY; CK(cudaMalloc(&dY, (size_t)MMAX * n * 4));
    float* dWs; CK(cudaMalloc(&dWs, 4));
    float onef = 1.f;
    CK(cudaMemcpy(dWs, &onef, 4, cudaMemcpyHostToDevice));
    fill_f32<<<(unsigned)(((size_t)MMAX * k + 255) / 256), 256>>>(dX, (size_t)MMAX * k, 3);
    fill_fp8<<<(unsigned)(((size_t)n * k + 255) / 256), 256>>>(dW, (size_t)n * k, 7);
    CK(cudaDeviceSynchronize());

    // ---- (b) quantize kernel cost ----
    printf("== activation quantize f32->fp8 per-token (k=%d) ==\n", k);
    for (int m : ms) {
        quantize_fp8_act<<<m, 256, 0, stream>>>(dX, dA, dS, k);
        CK(cudaStreamSynchronize(stream));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, stream));
        for (int i = 0; i < 50; i++) quantize_fp8_act<<<m, 256, 0, stream>>>(dX, dA, dS, k);
        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
        float msec; CK(cudaEventElapsedTime(&msec, e0, e1));
        printf("  m=%5d  %.4f ms  (%.1f GB/s eff)\n", m, msec / 50,
               (double)m * k * 5 / (msec / 50 * 1e-3) / 1e9);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }

    // ---- (a) OUTER_VEC B scale mode ----
    printf("\n== cuBLASLt FP8 GEMM w/ OUTER_VEC per-token B scale (n=%d k=%d) ==\n", n, k);
    for (int m : ms) {
        cublasLtMatmulDesc_t op;
        LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
        // A = W (fp8, scalar scale), B = act (fp8, per-token OUTER_VEC scale)
        const void* aps = dWs; LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aps, sizeof(aps)));
        const void* bps = dS;  LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bps, sizeof(bps)));
        int32_t amode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        int32_t bmode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        cublasStatus_t s1 = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &amode, sizeof(amode));
        cublasStatus_t s2 = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &bmode, sizeof(bmode));
        if (s1 != CUBLAS_STATUS_SUCCESS || s2 != CUBLAS_STATUS_SUCCESS) {
            printf("  scale-mode set REJECTED (%d/%d)\n", (int)s1, (int)s2); return 1;
        }
        cublasLtMatrixLayout_t la, lb, ld;
        LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, n, k));
        LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, m, k));
        LK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, n, m, n));
        size_t ws_sz = 64ull << 20; static void* ws = nullptr;
        if (!ws) CK(cudaMalloc(&ws, ws_sz));
        cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
        LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz)));
        cublasLtMatmulHeuristicResult_t heur[4]; int nh = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 4, heur, &nh);
        if (hs != CUBLAS_STATUS_SUCCESS || nh == 0) {
            printf("  m=%5d  NO ALGO for OUTER_VEC (status=%d nh=%d)\n", m, (int)hs, nh);
        } else {
            float alpha = 1.f, beta = 0.f;
            quantize_fp8_act<<<m, 256, 0, stream>>>(dX, dA, dS, k);
            cublasStatus_t rs = cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta, dY, ld, dY, ld,
                                               &heur[0].algo, ws, ws_sz, stream);
            if (rs != CUBLAS_STATUS_SUCCESS) { printf("  m=%5d  matmul RUN failed %d\n", m, (int)rs); }
            else {
                CK(cudaStreamSynchronize(stream));
                // numeric spot check y[0,0..2] vs host f32 ref through the SAME fp8 act codes
                if (m == 512) {
                    float* hx = (float*)malloc((size_t)k * 4);
                    __nv_fp8_e4m3* ha = (__nv_fp8_e4m3*)malloc(k);
                    __nv_fp8_e4m3* hw = (__nv_fp8_e4m3*)malloc(k);
                    float hs0, hy0;
                    CK(cudaMemcpy(hx, dX, (size_t)k * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ha, dA, k, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hw, dW, k, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&hs0, dS, 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&hy0, dY, 4, cudaMemcpyDeviceToHost));
                    double accq = 0, accf = 0;
                    for (int i = 0; i < k; i++) {
                        accq += (double)float(ha[i]) * float(hw[i]);
                        accf += (double)hx[i] * float(hw[i]);
                    }
                    printf("  spot: lt=%.4f  fp8ref=%.4f (scaled %.4f)  f32ref=%.4f  relerr(fp8 vs f32)=%.4f\n",
                           hy0, accq * hs0, accq * hs0, accf, fabs(accq * hs0 - accf) / (fabs(accf) + 1e-6));
                    free(hx); free(ha); free(hw);
                }
                // timing: quantize + GEMM chained (the real per-matmul cost)
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                CK(cudaEventRecord(e0, stream));
                for (int i = 0; i < 20; i++) {
                    quantize_fp8_act<<<m, 256, 0, stream>>>(dX, dA, dS, k);
                    cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta, dY, ld, dY, ld,
                                   &heur[0].algo, ws, ws_sz, stream);
                }
                CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
                float msec; CK(cudaEventElapsedTime(&msec, e0, e1));
                double per = msec / 20;
                double tf = 2.0 * m * (double)n * k / (per * 1e-3) / 1e12;
                printf("  m=%5d  quant+GEMM %.3f ms  = %.1f TFLOP/s effective\n", m, per, tf);
                CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            }
        }
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
    }
    printf("done\n");
    return 0;
}
