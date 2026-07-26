// FA3 v10 engine shim (task #20, ARCHITECTURE-H100.md round 29): the harness-proven
// TMA-swizzled wgmma FA (tools/bench_fa3.cu v10, 883us vs the shipped 993us at T=2048,
// correct at all T vs the online-semantics reference). NEW NUMERIC CONFIG (explicit,
// opt-in BW24_FA3=1 — the GDN-mma precedent): online-softmax + bf16 P at running max,
// same class as the shipped kernel but not bit-paired; run-gen argmax/stream batteries
// arbitrate. Fresh causal prefill, head_dim 256 only this increment.
//
// Host entry encodes the CUtensorMaps per call (~us) for the caller's PRE-MIRRORED
// bf16 Q/K buffers and launches the kernel on the caller's stream. Static-lib kind
// (build.rs), links the driver API for cuTensorMapEncodeTiled.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>

// wgmma/TMA are sm_90a-only; other arches build a fail-closed stub (the dispatch arm
// is opt-in + hd256, and gates fail loudly on rc=3).
#if defined(BW24_FA3_STUB)
extern "C" int bw24_fa3_prefill(const void*, const void*, const void*, float*,
                                int, int, int, int, float, void*) { return 3; }
#else

// ---- wgmma/mbarrier helpers (harness-proven set) ----
__device__ __forceinline__ unsigned long long make_desc(const void* smem_ptr,
                                                        unsigned lead_bytes,
                                                        unsigned stride_bytes) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
template<int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
}
__device__ __forceinline__ void bar_sync(int id, int cnt) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt));
}
__device__ __forceinline__ void wgmma_m64n64k16_bf16(float acc[32], unsigned long long da,
                                                     unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n64k16_bf16_tb(float acc[32], unsigned long long da,
                                                        unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 1;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

__device__ __forceinline__ void v10_mbar_init(void* mbar, unsigned count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void v10_expect_tx(void* mbar, unsigned bytes) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void v10_arrive(void* mbar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(a));
}
__device__ __forceinline__ void v10_wait(void* mbar, unsigned phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(a), "r"(phase));
    }
}
__device__ __forceinline__ void v10_tma3d(const CUtensorMap* map, void* dst,
                                          int c0, int c1, int c2, void* mbar) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    unsigned b = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
}
__device__ __forceinline__ unsigned long long v10_desc_swz(const void* p) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((1024u >> 4) & 0x3FFF) << 32;
    d |= (unsigned long long)1 << 62;
    return d;
}

extern "C" __global__ void __launch_bounds__(256, 1)
bw24_fa3_v10_kernel(const __grid_constant__ CUtensorMap tQ, const __grid_constant__ CUtensorMap tK,
        const __grid_constant__ CUtensorMap tV, float* __restrict__ O,
        int T, int H, int HKV, int D, float scale) {
    extern __shared__ char smem[];
    const int wg = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    char* bQ = smem + wg * 32768;
    char* bK[2] = { smem + 65536, smem + 65536 + 32768 };
    char* bV[2] = { smem + 131072, smem + 131072 + 32768 };
    char* bP = smem + 196608 + wg * 8192;
    unsigned long long* mFull = (unsigned long long*)(smem + 212992);
    unsigned long long* mEmpty = mFull + 2;
    const int q0 = (blockIdx.x * 2 + wg) * 64;
    const int head = blockIdx.y;
    const int kvh = head / (H / HKV);

    if (threadIdx.x == 0) {
        v10_mbar_init(&mFull[0], 1); v10_mbar_init(&mFull[1], 1);
        v10_mbar_init(&mEmpty[0], 256); v10_mbar_init(&mEmpty[1], 256);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const int blk_q_end = (blockIdx.x * 2 + 1) * 64;
    const int kv_end_all = blk_q_end + 64 <= T ? blk_q_end + 64 : T;
    const int n_tiles = (kv_end_all + 63) / 64;
    const int kv_end_own = q0 + 64 <= T ? q0 + 64 : T;
    const int n_own = q0 < T ? (kv_end_own + 63) / 64 : 0;

    // prologue: thread 0 issues Q (both WGs) + K(0) + V(0) on FULL[0] — all TMA
    if (threadIdx.x == 0) {
        v10_expect_tx(&mFull[0], 2 * 32768 + 2 * 32768);
        for (int a = 0; a < 4; a++) {
            v10_tma3d(&tQ, smem + a * 8192, a * 64, head, (blockIdx.x * 2) * 64, &mFull[0]);
            v10_tma3d(&tQ, smem + 32768 + a * 8192, a * 64, head, (blockIdx.x * 2 + 1) * 64, &mFull[0]);
            v10_tma3d(&tK, bK[0] + a * 8192, a * 64, kvh, 0, &mFull[0]);
            v10_tma3d(&tV, bV[0] + a * 8192, a * 64, kvh, 0, &mFull[0]);
        }
    }
    __syncthreads();

    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    float m[2] = {-1e30f, -1e30f};
    float l[2] = {0.0f, 0.0f};
    float oacc[4][32];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int i = 0; i < 32; i++) oacc[nb][i] = 0.0f;

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * 64;
        const int cur = t & 1;
        const bool active = t < n_own;
        v10_wait(&mFull[cur], (t >> 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;");
        // issue K(t+1) (stage freed two tiles ago; EMPTY guards reuse)
        if (threadIdx.x == 0 && t + 1 < n_tiles) {
            if (t + 1 >= 2) v10_wait(&mEmpty[cur ^ 1], ((t - 1) >> 1) & 1);
            v10_expect_tx(&mFull[cur ^ 1], 2 * 32768);
            for (int a = 0; a < 4; a++) {
                v10_tma3d(&tK, bK[cur ^ 1] + a * 8192, a * 64, kvh, (t + 1) * 64, &mFull[cur ^ 1]);
                v10_tma3d(&tV, bV[cur ^ 1] + a * 8192, a * 64, kvh, (t + 1) * 64, &mFull[cur ^ 1]);
            }
        }
        if (active) {
            float acc[32];
            wgmma_fence();
            for (int st = 0; st < D / 16; st++) {
                unsigned long long da = v10_desc_swz(bQ + (st / 4) * 8192 + (st % 4) * 32);
                unsigned long long db = v10_desc_swz(bK[cur] + (st / 4) * 8192 + (st % 4) * 32);
                wgmma_m64n64k16_bf16(acc, da, db, st == 0 ? 0 : 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            float mn[2] = {m[0], m[1]};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = q0 + r0 + ((i % 4) / 2) * 8;
                int cc = k0 + c0 + (i / 4) * 8 + (i % 2);
                acc[i] = (cc <= rr && cc < T) ? acc[i] * scale : -1e30f;
                int half = (i % 4) / 2;
                if (acc[i] > mn[half]) mn[half] = acc[i];
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
                mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
            }
            float alpha[2] = {expf(m[0] - mn[0]), expf(m[1] - mn[1])};
            if (m[0] == -1e30f) alpha[0] = 0.0f;
            if (m[1] == -1e30f) alpha[1] = 0.0f;
            m[0] = mn[0]; m[1] = mn[1];
            float ladd[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int half = (i % 4) / 2;
                float pv = expf(acc[i] - m[half]);
                acc[i] = pv;
                ladd[half] += pv;
            }
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ladd[0] += __shfl_xor_sync(0xffffffffu, ladd[0], o);
                ladd[1] += __shfl_xor_sync(0xffffffffu, ladd[1], o);
            }
            l[0] = l[0] * alpha[0] + ladd[0];
            l[1] = l[1] * alpha[1] + ladd[1];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++)
                #pragma unroll
                for (int i = 0; i < 32; i++) oacc[nb][i] *= alpha[(i % 4) / 2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = r0 + ((i % 4) / 2) * 8;
                int cc = c0 + (i / 4) * 8 + (i % 2);
                int st = cc / 16, kk = cc % 16;
                size_t off = (size_t)st * 2048 + (rr / 8) * 256 + (kk / 8) * 128 + (rr % 8) * 16 + (kk % 8) * 2;
                *(__nv_bfloat16*)(bP + off) = __float2bfloat16(acc[i]);
            }
            bar_sync(6 + wg, 128);
            asm volatile("fence.proxy.async.shared::cta;");
            wgmma_fence();
            for (int st = 0; st < 4; st++) {
                unsigned long long da = make_desc(bP + st * 2048, 128, 256);
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    // V rows TMA-swizzled; trans_b MN-major B: atom = nb*8192, k-slice = st*2048
                    unsigned long long db = v10_desc_swz(bV[cur] + nb * 8192 + st * 2048);
                    wgmma_m64n64k16_bf16_tb(oacc[nb], da, db, 1);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();
        v10_arrive(&mEmpty[cur]);
    }
    if (q0 < T) {
        float il[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f, l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
        #pragma unroll
        for (int nb = 0; nb < 4; nb++)
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                int n8 = i / 4;
                int cc = nb * 64 + c0 + n8 * 8;
                int ra = q0 + r0, rb = q0 + r0 + 8;
                if (ra < T) {
                    O[((size_t)ra * H + head) * D + cc + 0] = oacc[nb][i + 0] * il[0];
                    O[((size_t)ra * H + head) * D + cc + 1] = oacc[nb][i + 1] * il[0];
                }
                if (rb < T) {
                    O[((size_t)rb * H + head) * D + cc + 0] = oacc[nb][i + 2] * il[1];
                    O[((size_t)rb * H + head) * D + cc + 1] = oacc[nb][i + 3] * il[1];
                }
            }
    }
}


// ---- host entry: encode maps + launch. q16/k16/v16 = caller's bf16 mirrors
// ([T,H,D] / [T,HKV,D] rows). Returns 0 ok; 1 map-encode fail; 2 launch fail.
extern "C" int bw24_fa3_prefill(const void* q16, const void* k16, const void* v16,
                                float* o, int T, int H, int HKV, int D, float scale,
                                void* stream) {
    if (D != 256) return 3;
    CUtensorMap tQ, tK;
    cuuint64_t gdq[3] = { (cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T };
    cuuint64_t gsq[2] = { (cuuint64_t)D * 2, (cuuint64_t)H * D * 2 };
    cuuint32_t bx[3] = { 64, 1, 64 };
    cuuint32_t es[3] = { 1, 1, 1 };
    if (cuTensorMapEncodeTiled(&tQ, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)q16,
            gdq, gsq, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) return 1;
    cuuint64_t gdk[3] = { (cuuint64_t)D, (cuuint64_t)HKV, (cuuint64_t)T };
    cuuint64_t gsk[2] = { (cuuint64_t)D * 2, (cuuint64_t)HKV * D * 2 };
    if (cuTensorMapEncodeTiled(&tK, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)k16,
            gdk, gsk, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) return 1;
    CUtensorMap tV;
    if (cuTensorMapEncodeTiled(&tV, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)v16,
            gdk, gsk, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) return 1;
    int shmem = 212992 + 64;
    static int attr_set = 0;
    if (!attr_set) {
        if (cudaFuncSetAttribute(bw24_fa3_v10_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)) return 2;
        attr_set = 1;
    }
    dim3 grid((T + 127) / 128, H);
    bw24_fa3_v10_kernel<<<grid, 256, shmem, (cudaStream_t)stream>>>(
        tQ, tK, tV, o, T, H, HKV, D, scale);
    return cudaPeekAtLastError() ? 2 : 0;
}

#endif  // BW24_FA3_STUB
