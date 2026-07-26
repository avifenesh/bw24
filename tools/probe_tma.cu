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
    return bad != 0;
}
