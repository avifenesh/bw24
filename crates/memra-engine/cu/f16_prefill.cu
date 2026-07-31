// f16_prefill.cu — MEMRA_PP_F16 prefill GEMM: cuBLASLt FP16 TN on a resident fp16 dequant
// mirror of Q8_0 weights.
//
// Provenance: tools/bench_lt_f16.cu probe (2026-07-26, H100 box): 611-687 TF at the 9B m=512
// prefill shapes vs the vendored MMQ class those Q8_0 weights ride (253/168/90/247/236/82us
// per-shape medians) = 3.2-3.7x per launch. The exact-int8 wgmma arc (v0-v4 + ceiling probe,
// ARCHITECTURE-H100.md) established WHY a fold-free fp16 GEMM is the right swing: Q8_0's
// per-32-block scale fold serializes Hopper's warpgroup MMA pipe (ptxas C7514); fp16 with f32
// accumulate has zero mid-loop accumulator reads and streams at full tensor-core rate.
//
// NUMERIC CONFIG (new, explicit, gated — GDN-chunked/MEMRA_PP_FP8 precedent): weight int8 -> fp16
// dequant is EXACT for the int8 part (|q|<=127 needs 7 mantissa bits; fp16 has 11) — the only
// rounding is d(half)*q products vs the s32-exact-then-f32-fold law, plus activation f32->fp16
// (rel ~2^-11 per element, NO per-32 rescale — finer-grained than q8_1's int8 in that sense).
// Decode (m=1..8) NEVER reaches this path — the decode==verify law is untouched.
//
// Host C-ABI (called from f16_ffi.rs), mirrors fp8_prefill.cu exactly:
//   memra_f16_pp_gemm : f32->fp16 activation convert + cublasLtMatmul TN, all on one stream,
//                      (m,n,k)-cached descriptors + heuristic algo under a mutex.
//   memra_q8_0_dequant_f16 : GGUF Q8_0 34B blocks -> row-major fp16 mirror (load-time, per tensor).

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

// ---- device kernels -------------------------------------------------------------------------

// f32 -> fp16 elementwise (activations; fp16 max 65504 >> activation range, no scale needed).
// float4/half4-vectorized grid-stride (elementwise -> bit-identical; H100 sweep 2026-07-26).
extern "C" __global__ void memra_f16_cvt_kernel(const float* __restrict__ x,
                                               __half* __restrict__ o, size_t n) {
    size_t n4 = n / 4;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = *(const float4*)(x + i * 4);
        __half2 lo = __floats2half2_rn(v.x, v.y);
        __half2 hi = __floats2half2_rn(v.z, v.w);
        *(__half2*)(o + i * 4) = lo;
        *(__half2*)(o + i * 4 + 2) = hi;
    }
    // tail (n % 4) by the first threads
    size_t t0 = n4 * 4;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid < n - t0) o[t0 + tid] = __float2half(x[t0 + tid]);
}

// GGUF Q8_0 (34B blocks: half d + 32 int8) -> row-major fp16. One thread per 32-block.
extern "C" __global__ void memra_q8f16_dequant_kernel(const unsigned char* __restrict__ src,
                                                     __half* __restrict__ dst,
                                                     size_t nblk_total, int nblk_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nblk_total) return;
    const unsigned char* b = src + i * 34;
    float d = __half2float(*(const __half*)b);
    const signed char* q = (const signed char*)(b + 2);
    __half* o = dst + i * 32;   // block i = row (i/nblk_row), k-block (i%nblk_row): dst is dense
    #pragma unroll
    for (int k = 0; k < 32; k++) o[k] = __float2half(d * (float)q[k]);
}

extern "C" int memra_q8_0_dequant_f16(const void* w_q8, void* w_f16, long out_f, long nblk_row,
                                     void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nblk_total = (size_t)out_f * (size_t)nblk_row;
    int threads = 256;
    size_t blocks = (nblk_total + threads - 1) / threads;
    memra_q8f16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q8, (__half*)w_f16, nblk_total, (int)nblk_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// GGUF Q4_0 (18B blocks: half d + 16 nibble bytes) -> row-major fp16 (campaign A, 2026-07-31:
// the gemma QAT trunk's f16 mirror — int4 magnitudes are exact in fp16, rounding only at d*q;
// same accuracy class as the promoted Q8_0 mirror). One thread per 32-block.
extern "C" __global__ void memra_q4f16_dequant_kernel(const unsigned char* __restrict__ src,
                                                     __half* __restrict__ dst,
                                                     size_t nblk_total, int nblk_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nblk_total) return;
    const unsigned char* b = src + i * 18;
    float d = __half2float(*(const __half*)b);
    const unsigned char* qs = b + 2;
    __half* o = dst + i * 32;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        const int lo = (qs[k] & 0x0F) - 8;
        const int hi = (qs[k] >> 4) - 8;
        o[k]      = __float2half(d * (float)lo);
        o[k + 16] = __float2half(d * (float)hi);
    }
}

extern "C" int memra_q4_0_dequant_f16(const void* w_q4, void* w_f16, long out_f, long nblk_row,
                                     void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nblk_total = (size_t)out_f * (size_t)nblk_row;
    int threads = 256;
    size_t blocks = (nblk_total + threads - 1) / threads;
    memra_q4f16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q4, (__half*)w_f16, nblk_total, (int)nblk_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// ---- host: cached cuBLASLt plans (fp8_prefill.cu pattern) ------------------------------------

namespace {
struct F16Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulAlgo_t algo;
};
std::mutex g_mu16;
cublasLtHandle_t g_lt16 = nullptr;
std::map<std::tuple<int, int, int>, F16Plan>* g_plans16 = nullptr;  // leaked (process-lifetime)
}  // namespace

// Standalone f32->fp16 activation convert (grouped-dispatch entry: hybrid layers run 2-4
// GEMMs on ONE activation — convert once, feed memra_f16_pp_gemm_pre per weight).
extern "C" int memra_f16_cvt(const float* x_f32, void* xh_f16, size_t nelem, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = 256;
    size_t want = (nelem + threads - 1) / threads;
    int blocks = (int)(want < 1024 ? want : 1024);
    if (blocks < 1) blocks = 1;
    memra_f16_cvt_kernel<<<blocks, threads, 0, stream>>>(x_f32, (__half*)xh_f16, nelem);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// One FP16 prefill GEMM: y[m,n] row-major = x[m,k] @ W[n,k]^T, f32 accumulate/output.
// Col-major view: D[n,m] = A^T(W as k x n, opT) * B(xh as k x m, opN), lda=ldb=k, ldd=n.
// Returns 0 ok; 1xxxx = cudaError from the convert; 2xxxx = no heuristic; 3xxxx = matmul status.
extern "C" int memra_f16_pp_gemm_pre(
    const void* w_f16,      // device [n, k] row-major fp16 mirror
    const void* xh_f16,     // device [m, k] fp16 activation (pre-converted)
    float* y_f32,           // device out [m, n] row-major f32 (fully overwritten)
    int m, int n, int k,
    void* ws, size_t ws_bytes,
    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> guard(g_mu16);
    if (!g_lt16) {
        cublasStatus_t s = cublasLtCreate(&g_lt16);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    }
    if (!g_plans16) g_plans16 = new std::map<std::tuple<int, int, int>, F16Plan>();
    auto key = std::make_tuple(m, n, k);
    auto it = g_plans16->find(key);
    if (it == g_plans16->end()) {
        F16Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, k, n, k);  // W: k x n col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, k, m, k);  // act: k x m col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, n, m, n);  // out: n x m col-major, ld=n
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(g_lt16, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_plans16->emplace(key, p).first;
    }
    F16Plan& plan = it->second;
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(g_lt16, plan.op, &alpha, w_f16, plan.la, xh_f16, plan.lb,
                                      &beta, y_f32, plan.ld, y_f32, plan.ld, &plan.algo,
                                      ws, ws_bytes, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}

// Combined convert + GEMM (the single-consumer path and the kernel-check raw entry).
extern "C" int memra_f16_pp_gemm(
    const void* w_f16, const float* x_f32, void* xh_f16, float* y_f32,
    int m, int n, int k, void* ws, size_t ws_bytes, void* stream_v) {
    int rc = memra_f16_cvt(x_f32, xh_f16, (size_t)m * (size_t)k, stream_v);
    if (rc != 0) return rc;
    return memra_f16_pp_gemm_pre(w_f16, xh_f16, y_f32, m, n, k, ws, ws_bytes, stream_v);
}
