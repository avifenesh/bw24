// K3 solve Route A harness (ledger 876cdcb7): forward substitution vs tf32-wgmma
// triangular inverse-product. Semantics (gdn_chunk_solve_kernel<32>):
//   U_j = v_j − Σ_{i<j} A[j,i]·U_i          (RHS = v rows, 32 x 128)
//   W_j = e^{gcum_j}·k_j − Σ_{i<j} A[j,i]·W_i
// Route A: T = (I+L)^{-1} = (I−L)(I+L²)(I+L⁴)(I+L⁸)(I+L¹⁶) for strictly-lower L=A
// (nilpotent at 32), then U = T·Rv, W = T·Rw — 8 product GEMMs (32x32) + 2 apps
// (32x128), all m64n64k8.f32.tf32 (canonical pairing proven in probe_tf32.cu).
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/gs tools/bench_gdn_solve.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define D 128

__device__ __forceinline__ unsigned long long make_desc(const void* p, unsigned lead, unsigned stride) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride >> 4) & 0x3FFF) << 32;
    return d;
}
// tf32 canonical: element (r, kk of k8-step st) at 4B (probe_tf32.cu F1)
__device__ __forceinline__ size_t tf_off(int st, int r, int kk) {
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 4) * 128 + (r % 8) * 16 + (kk % 4) * 4;
}
__device__ __forceinline__ void wgmma_tf32(float acc[32], unsigned long long da,
                                           unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void wg_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;"); }
__device__ __forceinline__ unsigned tf32r_dev(float x) {
    unsigned u; asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return u;
}

// ---- baseline: the engine's f32 forward substitution (verbatim structure) ----
extern "C" __global__ void solve32_f32(const float* __restrict__ v, const float* __restrict__ k,
                                       const float* __restrict__ A, const float* __restrict__ gcum,
                                       float* __restrict__ U, float* __restrict__ W,
                                       int H, int T) {
    constexpr int CT = 32;
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * CT;
    const int Cc = min(CT, T - t0);
    const int tid = threadIdx.x;
    const int col = tid & (D - 1);
    const bool is_w = tid >= D;
    float* R = is_w ? W : U;
    __shared__ float As[CT][CT];
    for (int idx = tid; idx < Cc * CT; idx += 256) {
        int j = idx / CT, i = idx % CT;
        if (i < j) As[j][i] = A[(((size_t)c * H + h) * CT + j) * CT + i];
    }
    __syncthreads();
    const size_t rbase = ((size_t)c * H + h) * (size_t)CT * D;
    float hist[CT];
    #pragma unroll
    for (int j = 0; j < CT; j++) {
        if (j >= Cc) break;
        float acc;
        if (is_w) acc = expf(gcum[(size_t)(t0 + j) * H + h]) * k[((size_t)(t0 + j) * H + h) * D + col];
        else      acc = v[((size_t)(t0 + j) * H + h) * D + col];
        #pragma unroll
        for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
        hist[j] = acc;
        R[rbase + (size_t)j * D + col] = acc;
    }
}

// ---- Route A: tf32 inverse-product. CTA (chunk, head), 128 threads. ----
// T = (I−L)(I+L²)(I+L⁴)(I+L⁸)(I+L¹⁶); stages keep the running product M in
// fragments, restaged to tf32 smem between GEMMs; powers P = L^(2^i) likewise.
extern "C" __global__ void __launch_bounds__(128, 1)
solve32_tf32(const float* __restrict__ v, const float* __restrict__ k,
             const float* __restrict__ A, const float* __restrict__ gcum,
             float* __restrict__ U, float* __restrict__ W,
             int H, int T) {
    constexpr int CT = 32;
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * CT;
    const int Cc = min(CT, T - t0);
    if (Cc <= 0) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;

    // canonical tf32 tiles (m64 x k32 = 4 k8-steps = 8KB each)
    __shared__ __align__(1024) unsigned sP[64 * 32];    // current power L^(2^i)
    __shared__ __align__(1024) unsigned sM[64 * 32];    // running product
    __shared__ __align__(1024) unsigned sR[2][64 * 32]; // RHS halves as B (n=64 cols, k=j2) x2 for D=128... B tiles below
    __shared__ float gct[32];

    // stage L (A strictly lower, pad zero) into sP AND sM-init source; gct
    if (tid < 32) gct[tid] = (tid < Cc) ? gcum[(size_t)(t0 + tid) * H + h] : 0.0f;
    for (int idx = tid; idx < 64 * 32; idx += 128) {
        int r = idx / 32, i = idx % 32;
        float lv = (r < Cc && i < r) ? A[(((size_t)c * H + h) * CT + r) * CT + i] : 0.0f;
        unsigned t32 = tf32r_dev(lv);
        *(unsigned*)((char*)sP + tf_off(i / 8, r, i % 8)) = t32;
        // M0 = I − L
        float mv = (r < 64 && i < 32) ? ((r == i ? 1.0f : 0.0f) - lv) : 0.0f;
        *(unsigned*)((char*)sM + tf_off(i / 8, r, i % 8)) = tf32r_dev(mv);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    // stages i = 1..4: Pnew = P·P ; Mnew = M + M·Pnew  (M, P consumed as A and B)
    // B operand for X·Y: B(n, k) = Y^T?? — we need C[m,n] = Σ_k X[m,k]·Y[k,n]:
    // canonical B is staged by (r=n, kk=k) = Y[k][n] → stage powers TRANSPOSED as B.
    // L and its powers: staged sP as A-layout (r=row m, kk=col k). For B we need
    // the SAME matrix by (r=n=col, kk=k=row) — i.e. the transpose staging. Keep TWO
    // copies? Simpler: keep sP staged as B-layout (r=col, kk=row) throughout:
    //   - P·P: A must be P in A-layout — so keep BOTH: sPA (A-layout) + sP (B-layout).
    __shared__ __align__(1024) unsigned sPA[64 * 32];
    // re-stage L into both layouts (redo: sP above was A-layout; fix roles)
    for (int idx = tid; idx < 64 * 32; idx += 128) {
        int r = idx / 32, i = idx % 32;
        float lv = (r < Cc && i < r && i < Cc) ? A[(((size_t)c * H + h) * CT + r) * CT + i] : 0.0f;
        *(unsigned*)((char*)sPA + tf_off(i / 8, r, i % 8)) = tf32r_dev(lv);          // A-layout: (m=r, k=i)
    }
    for (int idx = tid; idx < 64 * 32; idx += 128) {
        int n = idx / 32, kk = idx % 32;   // B-layout: value = L[kk][n]
        float lv = (kk < Cc && n < kk && n < Cc) ? A[(((size_t)c * H + h) * CT + kk) * CT + n] : 0.0f;
        *(unsigned*)((char*)sP + tf_off(kk / 8, n, kk % 8)) = tf32r_dev(lv);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");

    float mAcc[32], pAcc[32];
    __shared__ __align__(1024) unsigned sG[64 * 32];    // G = M.P scratch (A-layout)
    // stage i: P <- P^2 and M <- M + (M.P_old).P_old  == M.(I + P_old^2) = M.(I+L^{2^i})
    for (int stage = 0; stage < 4; stage++) {
        // round A: pAcc = PA.PB (the square), gAcc = MA.PB (first half of the M update)
        float gAcc[32];
        wg_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long dpa = make_desc((char*)sPA + st * 2048, 128, 256);
            unsigned long long dma = make_desc((char*)sM + st * 2048, 128, 256);
            unsigned long long dpb = make_desc((char*)sP + st * 2048, 128, 256);
            wgmma_tf32(pAcc, dpa, dpb, st == 0 ? 0 : 1);
            wgmma_tf32(gAcc, dma, dpb, st == 0 ? 0 : 1);
        }
        wg_commit();
        wg_wait();
        // restage G (A-layout) — sP/sPA/sM untouched (still needed)
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cc = fc + n8 * 8;
            int r0 = warp * 16 + fr;
            if (cc < 32) {
                #pragma unroll
                for (int e2 = 0; e2 < 4; e2++) {
                    int r = r0 + (e2 >= 2 ? 8 : 0);
                    int cl = cc + (e2 & 1);
                    *(unsigned*)((char*)sG + tf_off(cl / 8, r, cl % 8)) = tf32r_dev(gAcc[q + e2]);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        // round B: mAcc = M(old, from smem) + GA.PB
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cc = fc + n8 * 8;
            int r0 = warp * 16 + fr;
            #pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int r = r0 + (e2 >= 2 ? 8 : 0);
                int cl = cc + (e2 & 1);
                float mo = 0.0f;
                if (cl < 32) {
                    unsigned mu = *(unsigned*)((char*)sM + tf_off(cl / 8, r, cl % 8));
                    memcpy(&mo, &mu, 4);
                }
                mAcc[q + e2] = mo;
            }
        }
        wg_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long dga = make_desc((char*)sG + st * 2048, 128, 256);
            unsigned long long dpb = make_desc((char*)sP + st * 2048, 128, 256);
            wgmma_tf32(mAcc, dga, dpb, 1);
        }
        wg_commit();
        wg_wait();
        __syncthreads();                             // all reads of sM/sP/sPA done
        // restage: sM <- mAcc; sPA/sP <- pAcc (both layouts)
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cc = fc + n8 * 8;
            int r0 = warp * 16 + fr;
            if (cc < 32) {
                #pragma unroll
                for (int e2 = 0; e2 < 4; e2++) {
                    int r = r0 + (e2 >= 2 ? 8 : 0);
                    int cl = cc + (e2 & 1);
                    unsigned pu = tf32r_dev(pAcc[q + e2]);
                    unsigned mu = tf32r_dev(mAcc[q + e2]);
                    if (r < 32) {
                        *(unsigned*)((char*)sPA + tf_off(cl / 8, r, cl % 8)) = pu;   // A-layout
                        *(unsigned*)((char*)sP + tf_off(r / 8, cl, r % 8)) = pu;     // B-layout (transposed roles)
                    }
                    *(unsigned*)((char*)sM + tf_off(cl / 8, r, cl % 8)) = mu;
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
    }
    // applications: U = M·Rv, W = M·Rw — Rv/Rw staged as B (n = D cols in 2 atoms, k = j2)
    // Rv[j2][d] = v[t0+j2][d]; Rw[j2][d] = exp(gct[j2])·k[t0+j2][d]
    for (int half = 0; half < 2; half++) {
        for (int rhs = 0; rhs < 2; rhs++) {
            for (int idx = tid; idx < 64 * 32; idx += 128) {
                int n = idx / 32, kk = idx % 32;   // value = R[kk][half*64+n]
                float rv = 0.0f;
                if (kk < Cc) {
                    if (rhs == 0) rv = v[((size_t)(t0 + kk) * H + h) * D + half * 64 + n];
                    else          rv = expf(gct[kk]) * k[((size_t)(t0 + kk) * H + h) * D + half * 64 + n];
                }
                *(unsigned*)((char*)sR[rhs] + tf_off(kk / 8, n, kk % 8)) = tf32r_dev(rv);
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        float uAcc[32], wAcc[32];
        wg_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long dma = make_desc((char*)sM + st * 2048, 128, 256);
            unsigned long long dv = make_desc((char*)sR[0] + st * 2048, 128, 256);
            unsigned long long dw = make_desc((char*)sR[1] + st * 2048, 128, 256);
            wgmma_tf32(uAcc, dma, dv, st == 0 ? 0 : 1);
            wgmma_tf32(wAcc, dma, dw, st == 0 ? 0 : 1);
        }
        wg_commit();
        wg_wait();
        const size_t rbase = ((size_t)c * H + h) * (size_t)CT * D;
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cc = fc + n8 * 8;
            int r0 = warp * 16 + fr;
            #pragma unroll
            for (int pr = 0; pr < 2; pr++) {
                int j = r0 + pr * 8;
                if (j < Cc) {
                    *(float2*)(U + rbase + (size_t)j * D + half * 64 + cc) =
                        make_float2(uAcc[q + pr * 2 + 0], uAcc[q + pr * 2 + 1]);
                    *(float2*)(W + rbase + (size_t)j * D + half * 64 + cc) =
                        make_float2(wAcc[q + pr * 2 + 0], wAcc[q + pr * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    const int H = 32, T = (argc > 1 ? atoi(argv[1]) : 512), C = 32;
    const int NC = (T + C - 1) / C;
    printf("K3 solve harness: H=%d T=%d NC=%d\n", H, T, NC);
    srand(7);
    auto rf = [](float s) { return ((rand() % 2001) - 1000) * 1e-3f * s; };
    size_t nv = (size_t)T * H * D, na = (size_t)NC * H * C * C, ng = (size_t)T * H;
    size_t nr = (size_t)NC * H * C * D;
    float *hv = (float*)malloc(nv * 4), *hk = (float*)malloc(nv * 4);
    float *hA = (float*)malloc(na * 4), *hg = (float*)malloc(ng * 4);
    for (size_t i = 0; i < nv; i++) { hv[i] = rf(1.0f); hk[i] = rf(1.0f); }
    for (size_t i = 0; i < na; i++) hA[i] = rf(0.3f);
    for (int h = 0; h < H; h++) { float a = 0; for (int t = 0; t < T; t++) { a += -0.02f - (rand() % 100) * 2e-4f; hg[(size_t)t * H + h] = a; } }
    // CPU ref
    float *rU = (float*)malloc(nr * 4), *rW = (float*)malloc(nr * 4);
    for (int c = 0; c < NC; c++)
        for (int h = 0; h < H; h++) {
            const int t0 = c * C, Cc = (T - t0 < C) ? T - t0 : C;
            for (int col = 0; col < D; col++) {
                float hu[32], hw[32];
                for (int j = 0; j < Cc; j++) {
                    float au = hv[((size_t)(t0 + j) * H + h) * D + col];
                    float aw = expf(hg[(size_t)(t0 + j) * H + h]) * hk[((size_t)(t0 + j) * H + h) * D + col];
                    for (int i = 0; i < j; i++) {
                        float a = hA[(((size_t)c * H + h) * C + j) * C + i];
                        au -= a * hu[i]; aw -= a * hw[i];
                    }
                    hu[j] = au; hw[j] = aw;
                    rU[(((size_t)c * H + h) * C + j) * D + col] = au;
                    rW[(((size_t)c * H + h) * C + j) * D + col] = aw;
                }
            }
        }
    float *dv, *dk, *dA, *dg, *dU, *dW;
    CK(cudaMalloc(&dv, nv * 4)); CK(cudaMalloc(&dk, nv * 4)); CK(cudaMalloc(&dA, na * 4));
    CK(cudaMalloc(&dg, ng * 4)); CK(cudaMalloc(&dU, nr * 4)); CK(cudaMalloc(&dW, nr * 4));
    CK(cudaMemcpy(dv, hv, nv * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk, nv * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA, hA, na * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, hg, ng * 4, cudaMemcpyHostToDevice));
    dim3 g1(NC, H);
    float *hU = (float*)malloc(nr * 4), *hW = (float*)malloc(nr * 4);
    // f32 baseline correctness + timing
    solve32_f32<<<g1, 256>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hU, dU, nr * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hW, dW, nr * 4, cudaMemcpyDeviceToHost));
    double m0 = 0, s0 = 0;
    for (size_t i = 0; i < nr; i++) {
        m0 = fmax(m0, fabs((double)hU[i] - rU[i])); m0 = fmax(m0, fabs((double)hW[i] - rW[i]));
        s0 = fmax(s0, fabs((double)rU[i])); s0 = fmax(s0, fabs((double)rW[i]));
    }
    printf("f32 baseline: rel %.3e (scale %.2f)\n", m0 / fmax(s0, 1e-3), s0);
    // tf32 route
    CK(cudaMemset(dU, 0, nr * 4)); CK(cudaMemset(dW, 0, nr * 4));
    solve32_tf32<<<g1, 128>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hU, dU, nr * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hW, dW, nr * 4, cudaMemcpyDeviceToHost));
    double m1 = 0;
    for (size_t i = 0; i < nr; i++) {
        m1 = fmax(m1, fabs((double)hU[i] - rU[i])); m1 = fmax(m1, fabs((double)hW[i] - rW[i]));
    }
    double rel1 = m1 / fmax(s0, 1e-3);
    printf("tf32 route: rel %.3e %s (tf32 band 5e-3)\n", rel1, rel1 < 5e-3 ? "IN-BAND" : "OUT-OF-BAND");
    // timing
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; i++) solve32_f32<<<g1, 256>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) solve32_f32<<<g1, 256>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    printf("f32 timing: %.1fus/call\n", ms * 20.0f);
    for (int i = 0; i < 5; i++) solve32_tf32<<<g1, 128>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
    for (int i = 0; i < 50; i++) solve32_tf32<<<g1, 128>>>(dv, dk, dA, dg, dU, dW, H, T);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("tf32 timing: %.1fus/call\n", ms * 20.0f);
    return rel1 < 5e-3 ? 0 : 1;
}
