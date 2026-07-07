// fp8_lt_prefill.cu — MICRO-PROBE: cuBLASLt FP8-E4M3 GEMM at the 27B prefill shapes (sm_120).
//
// QUESTION (FP8-ACT prefill card): can a cuBLASLt FP8xFP8 GEMM beat the current prefill GEMM
// class at the NVIDIA-27B shapes on the g7e RTX PRO 6000 (188 SM sm_120)?
// Current class measured via nsys pp6257 (2026-07-07, commit aa3c4ff5f):
//   qmatvec_gemm_q8_0 (attn/linear-attn layers, FP8-origin weights re-encoded Q8_0):
//     n=5120 k=6144: 62 TF | n=10240 k=5120: 72 TF | n=12288 k=5120: 72 TF |
//     n=1024 k=5120: 47 TF | n=6144 k=5120: 68 TF   (m=4096 chunk)
//   mul_mat_q_nvfp4_w4a8 (MLP gate/up/down, NVFP4 weights): ~241 TF (m=4096)
// GATE: FP8 arm must be >= 1.3x the current class at real shapes, else closed-negative.
//
// LAYOUT: y[m,n] = act[m,k] @ W[n,k]^T. Both operands row-major k-major = the cuBLASLt FP8
// TN requirement (opA=T on W[k x n co-major], opB=N on act). D = FP32. Per-token act scale is
// factored OUT of the GEMM (row-rescale epilogue outside; scale is per-row so it commutes) —
// probe times the GEMM alone plus, separately, a fused quantize+scale estimate.
//
// Build: nvcc -O3 -arch=sm_120a fp8_lt_prefill.cu -o fp8_lt_prefill -lcublasLt
// Run:   ./fp8_lt_prefill

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    printf("cublasLt ERR %s:%d status=%d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

__global__ void fill_fp8(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    // cheap hash -> [-1, 1]
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    float v = ((h >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    p[i] = __nv_fp8_e4m3(v * 0.5f);
}

// reference f32 GEMM row (for a spot numeric sanity check)
__global__ void spot_ref(const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* w, float* out,
                         int k, int n_col) {
    // one thread: y[0, n_col] = sum_k a[0,k] * w[n_col,k]
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.f;
    for (int i = 0; i < k; i++) acc += float(a[i]) * float(w[(size_t)n_col * k + i]);
    out[0] = acc;
}

struct Shape { int n, k; const char* tag; float cur_tf; };

int main() {
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s  SMs=%d\n", prop.name, prop.multiProcessorCount);

    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    // 27B prefill shapes (n=out_f, k=in_f) + measured current-class TFLOPS at m=4096.
    Shape shapes[] = {
        {5120, 6144, "o_proj      (q8_0 62TF)", 62.f},
        {10240, 5120, "lin_qkv     (q8_0 72TF)", 72.f},
        {12288, 5120, "q_gate      (q8_0 72TF)", 72.f},
        {1024, 5120, "kv_proj     (q8_0 47TF)", 47.f},
        {6144, 5120, "lin_ba      (q8_0 68TF)", 68.f},
        {17408, 5120, "ffn_gate/up (mmq 241TF)", 241.f},
        {5120, 17408, "ffn_down    (mmq 241TF)", 241.f},
    };
    int ms[] = {512, 2048, 4096, 6257};

    size_t ws_sz = 64ull << 20;
    void* ws; CK(cudaMalloc(&ws, ws_sz));

    printf("\n%-26s %6s %10s %10s %8s\n", "shape", "m", "ms/iter", "TFLOP/s", "vs cur");
    for (auto& s : shapes) {
        // weight [n, k] row-major fp8, activation [m_max, k] row-major fp8, out f32
        size_t wn = (size_t)s.n * s.k;
        __nv_fp8_e4m3 *dW, *dA; float* dY;
        CK(cudaMalloc(&dW, wn));
        CK(cudaMalloc(&dA, (size_t)6257 * s.k));
        CK(cudaMalloc(&dY, (size_t)6257 * s.n * 4));
        fill_fp8<<<(unsigned)((wn + 255) / 256), 256>>>(dW, wn, 7);
        fill_fp8<<<(unsigned)(((size_t)6257 * s.k + 255) / 256), 256>>>(dA, (size_t)6257 * s.k, 13);
        CK(cudaDeviceSynchronize());

        for (int m : ms) {
            // cuBLAS col-major view: D[n, m] = A^T(W as [k x n] col-major, opT -> [n x k]) *
            // B(act as [k x m] col-major, opN). lda=k (W row-major k-contig), ldb=k, ldd=n.
            cublasLtMatmulDesc_t op;
            LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
            LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
            LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
            cublasLtMatrixLayout_t la, lb, ld;
            LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, s.k, s.n, s.k)); // W: k x n col-major, ld=k
            LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, s.k, m, s.k));   // act: k x m col-major, ld=k
            LK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, s.n, m, s.n));       // out: n x m col-major, ld=n

            cublasLtMatmulPreference_t pref;
            LK(cublasLtMatmulPreferenceCreate(&pref));
            LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &ws_sz, sizeof(ws_sz)));
            cublasLtMatmulHeuristicResult_t heur[4]; int nh = 0;
            cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 4, heur, &nh);
            if (hs != CUBLAS_STATUS_SUCCESS || nh == 0) {
                printf("%-26s %6d   NO ALGO (status=%d nh=%d)\n", s.tag, m, (int)hs, nh);
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
                cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
                continue;
            }
            float alpha = 1.f, beta = 0.f;
            // warmup + spot check
            LK(cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta, dY, ld, dY, ld,
                              &heur[0].algo, ws, ws_sz, stream));
            CK(cudaStreamSynchronize(stream));
            if (m == 512) { // one-time numeric spot check vs f32 ref: y[0,0]
                float *dref, href, hy;
                CK(cudaMalloc(&dref, 4));
                spot_ref<<<1, 32>>>(dA, dW, dref, s.k, 0);
                CK(cudaMemcpy(&href, dref, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&hy, dY, 4, cudaMemcpyDeviceToHost)); // col-major [n,m]: (0,0)
                if (fabsf(href - hy) > 1e-2f * (fabsf(href) + 1.f))
                    printf("  !! spot mismatch %s: ref=%f lt=%f\n", s.tag, href, hy);
                cudaFree(dref);
            }
            int iters = 20;
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, stream));
            for (int i = 0; i < iters; i++)
                LK(cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta, dY, ld, dY, ld,
                                  &heur[0].algo, ws, ws_sz, stream));
            CK(cudaEventRecord(e1, stream));
            CK(cudaEventSynchronize(e1));
            float msec; CK(cudaEventElapsedTime(&msec, e0, e1));
            double per = msec / iters;
            double tf = 2.0 * m * (double)s.n * s.k / (per * 1e-3) / 1e12;
            printf("%-26s %6d %10.3f %10.1f %7.2fx\n", s.tag, m, per, tf, tf / s.cur_tf);
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
        }
        CK(cudaFree(dW)); CK(cudaFree(dA)); CK(cudaFree(dY));
        printf("\n");
    }
    cudaFree(ws);
    cublasLtDestroy(lt);
    printf("done\n");
    return 0;
}
