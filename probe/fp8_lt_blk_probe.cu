// fp8_lt_blk_probe.cu — P1 of the FP8-ST track (research/8bit-decision-20260803/DECISION.md §7 B3):
// does cuBLASLt on sm_120 accept BLOCK-SCALED FP8 GEMM — specifically the Qwen-official-FP8
// weight granularity CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F (weight side) and the
// DeepSeek-recipe activation granularity VEC128_32F (per-token per-128-k) — or do we fold the
// block scales in a pre-pass / extend MEMRA_MMQ_F8F4 instead?
//
// Prior art on this rig class (g7e probe 2026-07-08, research/tune-data/g7e-rtx6000.jsonl:39):
// per-token OUTER_VEC_32F B-scales came back NOT_SUPPORTED on sm120 (AlgoGetHeuristic status=7
// nh=0 at every m). Block-scale modes were NOT probed there — that is this probe's single job.
//
// Shape: q_gate 12288x5120 (the largest attn GEMM; same shape family the July probe measured at
// 668-779 TF with scalar scales). GEMM convention mirrors cu/fp8_prefill.cu: A = W e4m3
// [k x n] col-major with OP_T (row-major [out,in] weight viewed col-major), B = act e4m3
// [k x m], D = [n x m].
//
// Scale-mode matrix probed (A-mode x B-mode, D dtype variants):
//   1. SCALAR      x SCALAR       (control — the shipped MEMRA_PP_FP8 config, must be SUPPORTED)
//   2. BLK128x128  x SCALAR       (weight block grid, one act scalar)
//   3. BLK128x128  x VEC128       (the DeepSeek/Qwen serving recipe)
//   4. BLK128x128  x OUTER_VEC    (per-token act — expected NOT_SUPPORTED per the July finding)
//   5. VEC128      x VEC128       (1-D blocks both sides)
// Each combo is probed for D=R_32F and (if that fails) D=R_16BF — some Lt block-scale kernels
// only ship narrow-output epilogues.
//
// Default run is HEURISTIC-ONLY (host-side query; no GEMM launched — safe next to a busy GPU).
// `--run` additionally executes the supported combos with a NON-UNIFORM scale grid and
// numerically arbitrates the 2-D grid's linear order against BOTH candidate layouts
// (k-major: s[kb + nb*kblk] vs n-major: s[nb + kb*nblk]) — whichever matches the f64 host
// reference is the order Fp8BlockScales must upload for the GEMM consumer (loader keeps
// checkpoint order regardless; the reorder, if any, happens at GEMM plan build).
//
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp8_lt_blk_probe.cu -o fp8_lt_blk_probe -lcublasLt
// Run:   flock -n /tmp/gpu5090.lock ./fp8_lt_blk_probe [--run]

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    printf("cublasLt ERR %s:%d status=%d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static const char* mode_name(int32_t m) {
    switch (m) {
        case CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:      return "SCALAR";
        case CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F:   return "OUTER_VEC";
        case CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F:      return "VEC128";
        case CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F:  return "BLK128x128";
        default: return "?";
    }
}
static const char* dt_name(cudaDataType_t t) {
    return t == CUDA_R_32F ? "R_32F" : t == CUDA_R_16BF ? "R_16BF" : "?";
}

// deterministic fills (same hash family as the July probes)
__global__ void fill_fp8(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    p[i] = __nv_fp8_e4m3((((h >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 0.5f);
}

int main(int argc, char** argv) {
    const bool do_run = argc > 1 && !strcmp(argv[1], "--run");
    int dev = 0; cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s (sm_%d%d)  cublasLt %zu  mode: %s\n",
           prop.name, prop.major, prop.minor, cublasLtGetVersion(),
           do_run ? "heuristic+run" : "heuristic-only");

    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    const int n = 12288, k = 5120;             // q_gate [out=12288, in=5120]
    const int kblk = (k + 127) / 128;          // 40
    const int nblk = (n + 127) / 128;          // 96
    const int ms[] = {512, 2048, 4096};
    const int MMAX = 4096;

    __nv_fp8_e4m3 *dW, *dA;
    CK(cudaMalloc(&dW, (size_t)n * k));
    CK(cudaMalloc(&dA, (size_t)MMAX * k));
    void* dY; CK(cudaMalloc(&dY, (size_t)MMAX * n * 4)); // f32-sized; bf16 fits inside
    fill_fp8<<<(unsigned)(((size_t)n * k + 255) / 256), 256>>>(dW, (size_t)n * k, 7);
    fill_fp8<<<(unsigned)(((size_t)MMAX * k + 255) / 256), 256>>>(dA, (size_t)MMAX * k, 3);

    // ---- host scale grids (non-uniform so a --run spot check can arbitrate layout order) ----
    // weight 2-D grid, k-major candidate order: sA[kb + nb*kblk] = 0.5 + small drift
    float* hSa = (float*)malloc((size_t)kblk * nblk * 4);
    for (int nb = 0; nb < nblk; nb++)
        for (int kb = 0; kb < kblk; kb++)
            hSa[kb + (size_t)nb * kblk] = 0.5f + 0.001f * kb + 0.01f * nb;
    // act VEC128 grid [kblk x m] (col-major per token): uniform 1.0 keeps refs simple
    float* hSb = (float*)malloc((size_t)kblk * MMAX * 4);
    for (size_t i = 0; i < (size_t)kblk * MMAX; i++) hSb[i] = 1.0f;
    float one = 1.0f;

    float *dSa, *dSb, *dS1a, *dS1b;
    CK(cudaMalloc(&dSa, (size_t)kblk * nblk * 4));
    CK(cudaMalloc(&dSb, (size_t)kblk * MMAX * 4));
    CK(cudaMalloc(&dS1a, 4)); CK(cudaMalloc(&dS1b, 4));
    CK(cudaMemcpy(dSa, hSa, (size_t)kblk * nblk * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSb, hSb, (size_t)kblk * MMAX * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS1a, &one, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS1b, &one, 4, cudaMemcpyHostToDevice));
    // per-token OUTER_VEC B grid (combo 4): m floats
    float* dSo; CK(cudaMalloc(&dSo, (size_t)MMAX * 4));
    CK(cudaMemcpy(dSo, hSb, (size_t)MMAX * 4, cudaMemcpyHostToDevice)); // 1.0s
    CK(cudaDeviceSynchronize());

    size_t ws_sz = 64ull << 20; void* ws; CK(cudaMalloc(&ws, ws_sz));

    struct Combo { const char* tag; int32_t am, bm; const void *ap, *bp; };
    const Combo combos[] = {
        {"1 ctrl  ", CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                     CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,     dS1a, dS1b},
        {"2 blkW  ", CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
                     CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,     dSa,  dS1b},
        {"3 dseek ", CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
                     CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,     dSa,  dSb},
        {"4 outer ", CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
                     CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,  dSa,  dSo},
        {"5 vecs  ", CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
                     CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,     dSb,  dSb}, // A reuses a [kblk x n]-sized grid? see note
    };
    // NOTE combo 5: a true A-side VEC128 grid is [kblk x n] = 40*12288 floats; dSb (40*4096) is
    // too small to RUN but heuristic-status only needs the descriptor, and --run skips combo 5.
    const cudaDataType_t dts[] = {CUDA_R_32F, CUDA_R_16BF};

    printf("\n== heuristic matrix: q_gate n=%d k=%d (A=W e4m3 [kxn] OP_T, B=act e4m3 [kxm]) ==\n", n, k);
    printf("%-9s %-11s %-11s %-7s %-6s status nh\n", "combo", "A_mode", "B_mode", "D", "m");

    for (const Combo& c : combos) {
        for (cudaDataType_t dt : dts) {
            for (int m : ms) {
                cublasLtMatmulDesc_t op;
                LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
                cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
                LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
                LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
                LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &c.ap, sizeof(c.ap)));
                LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &c.bp, sizeof(c.bp)));
                cublasStatus_t sa = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &c.am, sizeof(c.am));
                cublasStatus_t sb = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &c.bm, sizeof(c.bm));
                if (sa != CUBLAS_STATUS_SUCCESS || sb != CUBLAS_STATUS_SUCCESS) {
                    printf("%-9s %-11s %-11s %-7s %-6d DESC_REJECT (%d/%d)\n", c.tag,
                           mode_name(c.am), mode_name(c.bm), dt_name(dt), m, (int)sa, (int)sb);
                    cublasLtMatmulDescDestroy(op);
                    continue;
                }
                cublasLtMatrixLayout_t la, lb, ld;
                LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, n, k));
                LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, m, k));
                LK(cublasLtMatrixLayoutCreate(&ld, dt, n, m, n));
                cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
                LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz)));
                cublasLtMatmulHeuristicResult_t heur[4]; int nh = 0;
                cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 4, heur, &nh);
                printf("%-9s %-11s %-11s %-7s %-6d %-6d %d %s\n", c.tag,
                       mode_name(c.am), mode_name(c.bm), dt_name(dt), m, (int)hs, nh,
                       (hs == CUBLAS_STATUS_SUCCESS && nh > 0) ? "SUPPORTED" : "NOT_SUPPORTED");

                // ---- optional run: numerics (grid-order arbitration) + timing ----
                if (do_run && hs == CUBLAS_STATUS_SUCCESS && nh > 0 && strncmp(c.tag, "5", 1)) {
                    float alpha = 1.f, beta = 0.f;
                    cublasStatus_t rs = cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta,
                                                       dY, ld, dY, ld, &heur[0].algo, ws, ws_sz, stream);
                    if (rs != CUBLAS_STATUS_SUCCESS) {
                        printf("          RUN failed status=%d\n", (int)rs);
                    } else {
                        CK(cudaStreamSynchronize(stream));
                        if (m == 512 && c.am == CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) {
                            // host reference for D[0..3, 0] under BOTH grid-order candidates
                            __nv_fp8_e4m3 *hw = (__nv_fp8_e4m3*)malloc((size_t)4 * k),
                                          *ha = (__nv_fp8_e4m3*)malloc(k);
                            // A col-major [k x n], ld=k: column j = row j of the weight
                            CK(cudaMemcpy(hw, dW, (size_t)4 * k, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(ha, dA, k, cudaMemcpyDeviceToHost));
                            for (int j = 0; j < 4; j++) {
                                double acc_kmaj = 0, acc_nmaj = 0;
                                for (int i = 0; i < k; i++) {
                                    double w = (double)(float)hw[(size_t)j * k + i];
                                    double a = (double)(float)ha[i];
                                    int kb = i / 128, nb = j / 128;
                                    acc_kmaj += w * a * hSa[kb + (size_t)nb * kblk];
                                    acc_nmaj += w * a * hSa[nb + (size_t)kb * nblk];
                                }
                                float y;
                                if (dt == CUDA_R_32F) {
                                    CK(cudaMemcpy(&y, (float*)dY + j, 4, cudaMemcpyDeviceToHost));
                                } else {
                                    unsigned short b16;
                                    CK(cudaMemcpy(&b16, (unsigned short*)dY + j, 2, cudaMemcpyDeviceToHost));
                                    unsigned int u = ((unsigned int)b16) << 16;
                                    memcpy(&y, &u, 4);
                                }
                                printf("          spot j=%d: lt=%.5f  kmaj=%.5f (rel %.2e)  nmaj=%.5f (rel %.2e)\n",
                                       j, y, acc_kmaj, fabs(y - acc_kmaj) / (fabs(acc_kmaj) + 1e-9),
                                       acc_nmaj, fabs(y - acc_nmaj) / (fabs(acc_nmaj) + 1e-9));
                            }
                            free(hw); free(ha);
                        }
                        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                        CK(cudaEventRecord(e0, stream));
                        for (int i = 0; i < 20; i++)
                            cublasLtMatmul(lt, op, &alpha, dW, la, dA, lb, &beta, dY, ld, dY, ld,
                                           &heur[0].algo, ws, ws_sz, stream);
                        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
                        float msec; CK(cudaEventElapsedTime(&msec, e0, e1));
                        printf("          %.3f ms  = %.1f TFLOP/s\n", msec / 20,
                               2.0 * m * (double)n * k / (msec / 20 * 1e-3) / 1e12);
                        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
                    }
                }
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
                cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
            }
        }
    }
    printf("done\n");
    return 0;
}
