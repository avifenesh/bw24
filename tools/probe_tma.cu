// TMA smoke (FA3 v10 step 1, ARCHITECTURE-H100.md design): host cuTensorMapEncodeTiled +
// cp.async.bulk.tensor.2d with mbarrier complete_tx — the byte-exact foundation for the
// swizzled-descriptor campaign. No swizzle here: box lands row-major in smem, host compares.
//
// Build (box): nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o /tmp/tma tools/probe_tma.cu -lcuda
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CD(x) do { CUresult r_ = (x); if (r_) { const char* s_; cuGetErrorString(r_, &s_); printf("CU %s @%d\n", s_, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(void* mbar, unsigned bytes) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void wgmma_fence_probe() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit_probe() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void wgmma_wait_probe() { asm volatile("wgmma.wait_group.sync.aligned 0;"); }

__device__ __forceinline__ void mbar_wait(void* mbar, unsigned phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(mbar);
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(a), "r"(phase));
    }
}

extern "C" __global__ void tma_smoke(const __grid_constant__ CUtensorMap tmap,
                                     __nv_bfloat16* out, int cy, int cx) {
    __shared__ __align__(128) __nv_bfloat16 tile[64 * 64];
    __shared__ __align__(8) unsigned long long mbar;
    if (threadIdx.x == 0) {
        mbar_init(&mbar, 1);
        asm volatile("fence.proxy.async.shared::cta;");
        mbar_expect_tx(&mbar, 64 * 64 * 2);
        unsigned t = (unsigned)__cvta_generic_to_shared(tile);
        unsigned b = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(t), "l"(&tmap), "r"(cx), "r"(cy), "r"(b) : "memory");
    }
    __syncthreads();
    mbar_wait(&mbar, 0);
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x)
        out[i] = tile[i];
}

// ---- swizzle pairing probe: SWIZZLE_128B tensor map + swizzle-mode wgmma descriptor.
// K-major bf16 tile 64x64-elem boxes; desc knobs -DSWZ_LBO -DSWZ_SBO (bytes), mode bits
// fixed to 1 (128B). One wgmma k-step reads 32B columns inside the swizzled atom — the
// sweep finds the LBO/SBO pair that makes S == QK^T (the canonical-layout crack, redux).
#ifndef SWZ_LBO
#define SWZ_LBO 1
#endif
#ifndef SWZ_SBO
#define SWZ_SBO 64
#endif
__device__ __forceinline__ unsigned long long make_desc_swz(const void* smem_ptr,
                                                            unsigned lead, unsigned stride) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride >> 4) & 0x3FFF) << 32;
    d |= (unsigned long long)1 << 62;   // swizzle mode 1 = 128B
    return d;
}
__device__ __forceinline__ void wgmma64_swz(float acc[32], unsigned long long da,
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

extern "C" __global__ void tma_qk_swz(const __grid_constant__ CUtensorMap tq,
                                      const __grid_constant__ CUtensorMap tk,
                                      float* S) {
    // Q,K tiles: 64 rows x 256 cols as 4 boxes of 64x64 elems each (8KB swizzled atoms)
    __shared__ __align__(1024) __nv_bfloat16 sQ[64 * 256];
    __shared__ __align__(1024) __nv_bfloat16 sK[64 * 256];
    __shared__ __align__(8) unsigned long long mbar;
    if (threadIdx.x == 0) {
        mbar_init(&mbar, 1);
        asm volatile("fence.proxy.async.shared::cta;");
        mbar_expect_tx(&mbar, 2 * 64 * 256 * 2);
        unsigned b = (unsigned)__cvta_generic_to_shared(&mbar);
        for (int q = 0; q < 4; q++) {
            unsigned tq_s = (unsigned)__cvta_generic_to_shared(sQ + q * 64 * 64);
            unsigned tk_s = (unsigned)__cvta_generic_to_shared(sK + q * 64 * 64);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1, {%2, %3}], [%4];" :: "r"(tq_s), "l"(&tq), "r"(q * 64), "r"(0), "r"(b) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1, {%2, %3}], [%4];" :: "r"(tk_s), "l"(&tk), "r"(q * 64), "r"(0), "r"(b) : "memory");
        }
    }
    __syncthreads();
    mbar_wait(&mbar, 0);
    asm volatile("fence.proxy.async.shared::cta;");

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    wgmma_fence_probe();
    // k-steps: 16 total; each swizzled 64x64 atom holds 4 k16 slices. Descriptor start
    // per (atom q, slice j): atom base + j*32 bytes?? — the sweep's LBO/SBO arbitrate.
    for (int q = 0; q < 4; q++) {
        for (int j = 0; j < 4; j++) {
            unsigned long long da = make_desc_swz((char*)(sQ + q * 64 * 64) + j * 32, SWZ_LBO, SWZ_SBO);
            unsigned long long db = make_desc_swz((char*)(sK + q * 64 * 64) + j * 32, SWZ_LBO, SWZ_SBO);
            wgmma64_swz(acc, da, db, (q == 0 && j == 0) ? 0 : 1);
        }
    }
    wgmma_commit_probe();
    wgmma_wait_probe();
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        int n8 = i / 4;
        S[(r0 + 0) * 64 + c0 + n8 * 8 + 0] = acc[i + 0];
        S[(r0 + 0) * 64 + c0 + n8 * 8 + 1] = acc[i + 1];
        S[(r0 + 8) * 64 + c0 + n8 * 8 + 0] = acc[i + 2];
        S[(r0 + 8) * 64 + c0 + n8 * 8 + 1] = acc[i + 3];
    }
}

// ---- PV trans_b probe: O = P(64x64, canonical A) x V(64kv x 64d slice, TMA-swizzled
// rows as MN-major B with the wgmma trans_b bit). If a desc pairing MATCHes, v10's
// scalar V^T staging is replaced by TMA (its last staging cost). Knobs: -DTB_SBO.
#ifndef TB_SBO
#define TB_SBO 1024
#endif
__device__ __forceinline__ unsigned long long make_desc(const void* smem_ptr,
                                                        unsigned lead_bytes, unsigned stride_bytes) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ unsigned long long make_desc_tb(
const void* p, unsigned sbo) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((sbo >> 4) & 0x3FFF) << 32;
    d |= (unsigned long long)1 << 62;
    return d;
}
__device__ __forceinline__ void wgmma64_tb(float acc[32], unsigned long long da,
                                           unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 1;\n}"   // trans_b = 1
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

extern "C" __global__ void tma_pv_tb(const __grid_constant__ CUtensorMap tv,
                                     const __nv_bfloat16* __restrict__ P,
                                     float* __restrict__ O) {
    // V: one 64(kv) x 64(d) swizzled atom via TMA; P: 64x64 canonical (staged scalar).
    __shared__ __align__(1024) __nv_bfloat16 sV[64 * 64];
    __shared__ __align__(128) __nv_bfloat16 sP[64 * 64];
    __shared__ __align__(8) unsigned long long mbar;
    const int tid = threadIdx.x;
    if (tid == 0) {
        mbar_init(&mbar, 1);
        asm volatile("fence.proxy.async.shared::cta;");
        mbar_expect_tx(&mbar, 64 * 64 * 2);
        unsigned d = (unsigned)__cvta_generic_to_shared(sV);
        unsigned b = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3}], [%4];" :: "r"(d), "l"(&tv), "r"(0), "r"(0), "r"(b) : "memory");
    }
    // canonical P staging (k-steps over kv: 4 x 64x16 tiles)
    char* bP = (char*)sP;
    for (int idx = tid; idx < 64 * 64; idx += 128) {
        int r = idx / 64, c = idx % 64;
        int st = c / 16, kk = c % 16;
        size_t off = (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
        *(__nv_bfloat16*)(bP + off) = P[r * 64 + c];
    }
    __syncthreads();
    mbar_wait(&mbar, 0);
    asm volatile("fence.proxy.async.shared::cta;");
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    wgmma_fence_probe();
    // 4 k16-steps over kv; A = P canonical (desc lead128/stride256); B = V atom with
    // trans_b: kv is B's k dim, rows of the swizzled atom — slice via base + j*?? sweep:
    // try k-slice = row-group offset j*SBO/4?? use j * 32 first (as K-major) and j * 2048.
    for (int j = 0; j < 4; j++) {
        unsigned long long da = make_desc((char*)sP + j * 2048, 128, 256);
#ifndef TB_JSTRIDE
#define TB_JSTRIDE 32
#endif
        unsigned long long db = make_desc_tb((char*)sV + j * TB_JSTRIDE, TB_SBO);
        wgmma64_tb(acc, da, db, j == 0 ? 0 : 1);
    }
    wgmma_commit_probe();
    wgmma_wait_probe();
    const int warp = tid / 32, lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int c0 = (lane % 4) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        int n8 = i / 4;
        O[(r0 + 0) * 64 + c0 + n8 * 8 + 0] = acc[i + 0];
        O[(r0 + 0) * 64 + c0 + n8 * 8 + 1] = acc[i + 1];
        O[(r0 + 8) * 64 + c0 + n8 * 8 + 0] = acc[i + 2];
        O[(r0 + 8) * 64 + c0 + n8 * 8 + 1] = acc[i + 3];
    }
}

int main() {
    const int ROWS = 256, COLS = 256;
    __nv_bfloat16* h = (__nv_bfloat16*)malloc(ROWS * COLS * 2);
    for (int i = 0; i < ROWS * COLS; i++) h[i] = __float2bfloat16((float)(i % 1017) * 0.25f);
    __nv_bfloat16 *dg, *dout;
    CK(cudaMalloc(&dg, ROWS * COLS * 2));
    CK(cudaMalloc(&dout, 64 * 64 * 2));
    CK(cudaMemcpy(dg, h, ROWS * COLS * 2, cudaMemcpyHostToDevice));

    CUtensorMap tmap;
    cuuint64_t gdim[2] = { COLS, ROWS };              // inner (elements), outer (rows)
    cuuint64_t gstride[1] = { COLS * 2 };             // bytes between rows
    cuuint32_t box[2] = { 64, 64 };
    cuuint32_t estride[2] = { 1, 1 };
    CD(cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                              (void*)dg, gdim, gstride, box, estride,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                              CU_TENSOR_MAP_L2_PROMOTION_NONE,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    // load the box at (cx=64, cy=128): rows 128..191, cols 64..127
    tma_smoke<<<1, 128>>>(tmap, dout, 128, 64);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    __nv_bfloat16* ho = (__nv_bfloat16*)malloc(64 * 64 * 2);
    CK(cudaMemcpy(ho, dout, 64 * 64 * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < 64 && bad < 5; r++)
        for (int c = 0; c < 64; c++) {
            unsigned short got, want;
            memcpy(&got, &ho[r * 64 + c], 2);
            memcpy(&want, &h[(128 + r) * COLS + 64 + c], 2);
            if (got != want) {
                if (bad < 3) printf("  mismatch (%d,%d): got %04x want %04x\n", r, c, got, want);
                bad++;
            }
        }
    printf("TMA smoke: %s (box byte-compare, %d bad)\n", bad == 0 ? "MATCH" : "MISMATCH", bad);
    if (bad) return 1;

    // ---- swizzle QK^T probe: Q,K 64x256 bf16; S vs CPU ref ----
    {
        const int D = 256;
        __nv_bfloat16 *hq = (__nv_bfloat16*)malloc(64 * D * 2), *hk = (__nv_bfloat16*)malloc(64 * D * 2);
        srand(13);
        for (int i = 0; i < 64 * D; i++) {
            hq[i] = __float2bfloat16((rand() % 255 - 127) * 0.011f);
            hk[i] = __float2bfloat16((rand() % 255 - 127) * 0.009f);
        }
        float* ref = (float*)malloc(64 * 64 * 4);
        for (int r = 0; r < 64; r++)
            for (int c = 0; c < 64; c++) {
                float sv = 0;
                for (int d = 0; d < D; d++)
                    sv += __bfloat162float(hq[r * D + d]) * __bfloat162float(hk[c * D + d]);
                ref[r * 64 + c] = sv;
            }
        __nv_bfloat16 *dq, *dk; float* dS;
        CK(cudaMalloc(&dq, 64 * D * 2)); CK(cudaMalloc(&dk, 64 * D * 2)); CK(cudaMalloc(&dS, 64 * 64 * 4));
        CK(cudaMemcpy(dq, hq, 64 * D * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, hk, 64 * D * 2, cudaMemcpyHostToDevice));
        CUtensorMap tq, tk;
        cuuint64_t gd[2] = { (cuuint64_t)D, 64 };
        cuuint64_t gs[1] = { (cuuint64_t)D * 2 };
        cuuint32_t bx[2] = { 64, 64 };
        cuuint32_t es[2] = { 1, 1 };
        CD(cuTensorMapEncodeTiled(&tq, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dq, gd, gs, bx, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        CD(cuTensorMapEncodeTiled(&tk, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dk, gd, gs, bx, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        tma_qk_swz<<<1, 128>>>(tq, tk, dS);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        float* hS = (float*)malloc(64 * 64 * 4);
        CK(cudaMemcpy(hS, dS, 64 * 64 * 4, cudaMemcpyDeviceToHost));
        float mr = 0; int b2 = 0;
        for (int i = 0; i < 64 * 64; i++) {
            float rl = fabsf(hS[i] - ref[i]) / fmaxf(fabsf(ref[i]), 1e-3f);
            if (rl > mr) mr = rl;
            if (rl > 2e-2f) b2++;
        }
        printf("SWZ QK^T: max_rel %.3e bad %d %s (LBO=%d SBO=%d)\n", mr, b2,
               b2 == 0 ? "MATCH" : "MISMATCH", SWZ_LBO, SWZ_SBO);
        if (b2) return 1;

        // ---- PV trans_b: O = P(64x64) x V(64x64) ----
        __nv_bfloat16 *hp = (__nv_bfloat16*)malloc(64 * 64 * 2), *hv2 = (__nv_bfloat16*)malloc(64 * 64 * 2);
        for (int i = 0; i < 64 * 64; i++) {
            hp[i] = __float2bfloat16((rand() % 255) * 0.003f);
            hv2[i] = __float2bfloat16((rand() % 255 - 127) * 0.01f);
        }
        float* ref2 = (float*)malloc(64 * 64 * 4);
        for (int r = 0; r < 64; r++)
            for (int d2 = 0; d2 < 64; d2++) {
                float o = 0;
                for (int c = 0; c < 64; c++)
                    o += __bfloat162float(hp[r * 64 + c]) * __bfloat162float(hv2[c * 64 + d2]);
                ref2[r * 64 + d2] = o;
            }
        __nv_bfloat16 *dp2, *dv2; float* dO2;
        CK(cudaMalloc(&dp2, 64 * 64 * 2)); CK(cudaMalloc(&dv2, 64 * 64 * 2)); CK(cudaMalloc(&dO2, 64 * 64 * 4));
        CK(cudaMemcpy(dp2, hp, 64 * 64 * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv2, hv2, 64 * 64 * 2, cudaMemcpyHostToDevice));
        CUtensorMap tv;
        cuuint64_t gdv[2] = { 64, 64 };
        cuuint64_t gsv[1] = { 64 * 2 };
        cuuint32_t bxv[2] = { 64, 64 };
        cuuint32_t esv[2] = { 1, 1 };
        CD(cuTensorMapEncodeTiled(&tv, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dv2, gdv, gsv, bxv, esv,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        tma_pv_tb<<<1, 128>>>(tv, dp2, dO2);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        float* hO2 = (float*)malloc(64 * 64 * 4);
        CK(cudaMemcpy(hO2, dO2, 64 * 64 * 4, cudaMemcpyDeviceToHost));
        float mr2 = 0; int b3 = 0;
        for (int i = 0; i < 64 * 64; i++) {
            float rl = fabsf(hO2[i] - ref2[i]) / fmaxf(fabsf(ref2[i]), 1e-3f);
            if (rl > mr2) mr2 = rl;
            if (rl > 2e-2f) b3++;
        }
        printf("PV trans_b: max_rel %.3e bad %d %s (SBO=%d JS=%d)\n", mr2, b3,
               b3 == 0 ? "MATCH" : "MISMATCH", TB_SBO, TB_JSTRIDE);
        return b3 != 0;
    }
}
