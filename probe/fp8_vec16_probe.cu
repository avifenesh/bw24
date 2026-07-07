// fp8_vec16_probe.cu — is CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 (block-scaled FP8, the
// mxf8f6f4-style path) available for the A (weight) operand on sm_120 via cuBLASLt 13.2?
// If yes: NVFP4 weights could ride FP8 GEMM with the per-16 ue4m3 scales applied IN the MMA
// (weight side bit-exact, e2m1 codes are exactly representable in e4m3). If no: fold scales
// into the e4m3 codes at convert (small weight-side rounding on 5-bit-mantissa products).
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp8_vec16_probe.cu -o t -lcublasLt

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %d\n", (int)e_); exit(1); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    printf("lt ERR line %d status=%d\n", __LINE__, (int)s_); exit(1); } } while (0)

int main() {
    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    const int n = 17408, k = 5120, m = 4096;
    void *dW, *dA, *dY, *dSa, *dSb;
    CK(cudaMalloc(&dW, (size_t)n * k));
    CK(cudaMalloc(&dA, (size_t)m * k));
    CK(cudaMalloc(&dY, (size_t)m * n * 4));
    CK(cudaMalloc(&dSa, (size_t)n * k / 16));   // ue4m3 per-16 scales for A
    CK(cudaMalloc(&dSb, 4));
    float one = 1.f; CK(cudaMemcpy(dSb, &one, 4, cudaMemcpyHostToDevice));

    // combos: (A mode, B mode, D type)
    struct Combo { int32_t am, bm; cudaDataType_t dt; const char* tag; };
    Combo combos[] = {
        {CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUDA_R_32F, "A=VEC16_UE4M3 B=scalar D=f32"},
        {CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUDA_R_32F, "A=VEC16 B=VEC16 D=f32"},
        {CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUDA_R_32F, "A=VEC32_UE8M0 B=VEC32_UE8M0 D=f32 (mxfp8)"},
        {CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUDA_R_16BF, "mxfp8 D=bf16"},
    };
    for (auto& c : combos) {
        cublasLtMatmulDesc_t op;
        LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
        const void* ap = dSa; const void* bp = dSb;
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ap, sizeof(ap)));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bp, sizeof(bp)));
        cublasStatus_t s1 = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &c.am, sizeof(c.am));
        cublasStatus_t s2 = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &c.bm, sizeof(c.bm));
        if (s1 != CUBLAS_STATUS_SUCCESS || s2 != CUBLAS_STATUS_SUCCESS) {
            printf("%-42s  desc-set REJECTED (%d/%d)\n", c.tag, (int)s1, (int)s2);
            cublasLtMatmulDescDestroy(op); continue;
        }
        cublasLtMatrixLayout_t la, lb, ld;
        LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, n, k));
        LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, m, k));
        LK(cublasLtMatrixLayoutCreate(&ld, c.dt, n, m, n));
        cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws_sz = 64ull << 20;
        LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz)));
        cublasLtMatmulHeuristicResult_t heur[4]; int nh = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 4, heur, &nh);
        printf("%-42s  heuristic status=%d nh=%d%s\n", c.tag, (int)hs, nh,
               (hs == CUBLAS_STATUS_SUCCESS && nh > 0) ? "  << SUPPORTED" : "");
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
    }
    return 0;
}
