// fp8_prefill.cu — BW24_PP_FP8 prefill GEMM: cuBLASLt FP8-E4M3 TN + per-batch activation quantize.
//
// Provenance: probe/fp8_lt_prefill.cu + probe/fp8_lt_scale_probe.cu (2026-07-08 probe verdict GO:
// 620-795 TF at the 27B prefill shapes vs 47-72 TF for the qmatvec_gemm_q8_0 class those F8-origin
// weights ride today = 8.7-14.2x). This TU is the ENGINE version of that verified call:
//   y[m,n] = act[m,k] @ W[n,k]^T,  W = raw checkpoint e4m3 bytes (EXACT weight side, no re-quant),
//   act    = f32 -> e4m3 with ONE per-batch scalar scale (amax/448; e4m3 max normal = 448).
// SCALE LAW (probed, sm_120 cuBLASLt 13.2): per-token OUTER_VEC_32F B-scale NOT supported ->
// per-batch scalar only; scalar A/B f32 scale pointers ARE supported and verified exact
// (y == raw*sa*sb). The activation dequant scale and the per-tensor weight_scale are FOLDED into
// one device scalar fed to CUBLASLT_MATMUL_DESC_B_SCALE_POINTER (the act operand is B).
//
// Host C-ABI (called from fp8_ffi.rs): one call = amax reduce + scale finalize + quantize +
// cublasLtMatmul, all on the caller's stream. Matmul descriptors + heuristic algo are cached per
// (m,n,k) under a mutex; the lock also covers the matmul enqueue (the cached desc is mutated per
// call to refresh the scale pointer — cublasLtMatmul consumes the desc synchronously on the host,
// so post-return mutation is safe). bw24 runs ONE GPU worker, so contention is nil.

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

// ---- device kernels -------------------------------------------------------------------------

// Per-batch amax over the whole activation [m*k]. Grid-stride block-reduce, then one atomicMax
// per block on the float-as-int monotonic trick (all values >= 0 after fabsf; non-negative IEEE
// floats order identically as ints). s[0] must be memset to 0 before launch.
extern "C" __global__ void bw24_fp8_amax_kernel(const float* __restrict__ x, size_t n, float* s) {
    float local = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        local = fmaxf(local, fabsf(x[i]));
    __shared__ float sm[256];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((int*)s, __float_as_int(sm[0]));
}

// Finalize the two scalars from amax: s[1] = quant multiplier (448/amax, applied to x before the
// e4m3 convert), s[2] = folded dequant scale ((amax/448) * weight_scale, the GEMM B_SCALE).
// amax == 0 (all-zero activation) -> quantize to all-zero codes and scale 0 -> y exactly 0.
extern "C" __global__ void bw24_fp8_scale_kernel(float* s, float w_scale) {
    float amax = s[0];
    if (amax > 0.f && isfinite(amax)) {
        s[1] = 448.f / amax;
        s[2] = (amax / 448.f) * w_scale;
    } else {
        s[1] = 0.f;
        s[2] = 0.f;
    }
}

// f32 -> e4m3 elementwise with the device-resident multiplier s[1] (no host sync anywhere in the
// chain). __nv_fp8_e4m3 ctor = cvt rn.satfinite (round-nearest-even, clamp to +-448).
extern "C" __global__ void bw24_fp8_quant_kernel(const float* __restrict__ x,
                                                 __nv_fp8_e4m3* __restrict__ q,
                                                 size_t n, const float* s) {
    const float mul = s[1];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        q[i] = __nv_fp8_e4m3(x[i] * mul);
}

// ---- host: cached cuBLASLt plans ------------------------------------------------------------

namespace {
struct Fp8Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulAlgo_t algo;
};
std::mutex g_mu;
cublasLtHandle_t g_lt = nullptr;
std::map<std::tuple<int, int, int>, Fp8Plan>* g_plans = nullptr;  // leaked on purpose (process-lifetime)
}  // namespace

// Run one FP8 prefill GEMM. Layout mirrors the probe exactly:
//   cuBLAS col-major view D[n,m] = A^T(W as [k x n] col-major, opT) * B(act [k x m] col-major, opN),
//   lda = ldb = k, ldd = n  ->  y is [m,n] row-major token-major (the engine's GEMM contract).
// Returns 0 on success; 1xxxx = cudaError from the quant chain; 2xxxx = heuristic failure
// (status, or nh==0); 3xxxx = cublasLtMatmul status; plain cublasStatus for handle/desc creation.
extern "C" int bw24_fp8_pp_gemm(
    const void* w_e4m3,     // device [n, k] row-major raw checkpoint e4m3 bytes
    const float* x_f32,     // device [m, k] row-major f32 activation
    void* xq_e4m3,          // device scratch [m, k] e4m3 (fully overwritten)
    float* scales,          // device scratch float[4]: [0]=amax [1]=quant-mul [2]=B_SCALE
    float* y_f32,           // device out [m, n] row-major f32 (fully overwritten)
    int m, int n, int k,
    float w_scale,          // per-tensor checkpoint weight_scale (host scalar, folded on device)
    void* ws, size_t ws_bytes,
    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;

    // 1) activation quantize chain (amax -> scales -> e4m3 codes), all on `stream`.
    size_t nelem = (size_t)m * (size_t)k;
    cudaError_t ce = cudaMemsetAsync(scales, 0, sizeof(float), stream);
    if (ce != cudaSuccess) return 10000 + (int)ce;
    const int threads = 256;
    size_t want = (nelem + threads - 1) / threads;
    int blocks = (int)(want < 1024 ? want : 1024);
    if (blocks < 1) blocks = 1;
    bw24_fp8_amax_kernel<<<blocks, threads, 0, stream>>>(x_f32, nelem, scales);
    bw24_fp8_scale_kernel<<<1, 1, 0, stream>>>(scales, w_scale);
    bw24_fp8_quant_kernel<<<blocks, threads, 0, stream>>>(
        x_f32, (__nv_fp8_e4m3*)xq_e4m3, nelem, scales);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) return 10000 + (int)ce;

    // 2) cuBLASLt FP8 TN matmul with the (m,n,k)-cached plan. The lock covers plan build AND the
    //    per-call B_SCALE_POINTER refresh + enqueue (the desc is shared mutable state).
    std::lock_guard<std::mutex> guard(g_mu);
    if (!g_lt) {
        cublasStatus_t s = cublasLtCreate(&g_lt);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    }
    if (!g_plans) g_plans = new std::map<std::tuple<int, int, int>, Fp8Plan>();
    auto key = std::make_tuple(m, n, k);
    auto it = g_plans->find(key);
    if (it == g_plans->end()) {
        Fp8Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        const void* bsp = scales + 2;  // refreshed every call below; set before the heuristic
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bsp, sizeof(bsp));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, k, n, k);  // W: k x n col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, k, m, k);  // act: k x m col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, n, m, n);      // out: n x m col-major, ld=n
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_plans->emplace(key, p).first;
    }
    Fp8Plan& plan = it->second;
    // Refresh the scale pointer: the resident scales buffer is stable per Engine, but a second
    // Engine in the same process (tests) would otherwise reuse the first Engine's pointer.
    const void* bsp = scales + 2;
    cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bsp, sizeof(bsp));
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(g_lt, plan.op, &alpha, w_e4m3, plan.la, xq_e4m3, plan.lb,
                                      &beta, y_f32, plan.ld, y_f32, plan.ld, &plan.algo,
                                      ws, ws_bytes, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}
