// Phase-0 smoke kernels. Proves: (1) a custom warp mma.sync kernel runs via cudarc-loaded fatbin,
// (2) the block-scale FP4 instruction the engine relies on is present in a real compiled module.
#include <cuda_runtime.h>

// SAXPY-ish elementwise — trivial correctness oracle for the launch path.
extern "C" __global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// FP16 warp tensor-core mma.sync m16n8k16 — proves tensor-core path from a cudarc-launched kernel.
// One warp computes a 16x8 tile = A(16x16) * B(16x8). Inputs preloaded as raw fragments (smoke only).
extern "C" __global__ void mma_fp16_smoke(const unsigned* a_frag, const unsigned* b_frag, float* out) {
    unsigned a[4] = {a_frag[0], a_frag[1], a_frag[2], a_frag[3]};
    unsigned b[2] = {b_frag[0], b_frag[1]};
    float c[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    int lane = threadIdx.x & 31;
    out[lane * 4 + 0] = c[0]; out[lane * 4 + 1] = c[1];
    out[lane * 4 + 2] = c[2]; out[lane * 4 + 3] = c[3];
}

// FP4 block-scale mma — the engine's headline weapon. Presence in this fatbin proves it
// compiles+links in our real build pipeline (not just a /tmp probe).
extern "C" __global__ void mma_fp4_blockscale_smoke(const unsigned* a_frag, const unsigned* b_frag,
                                                    const unsigned* scales, float* out) {
    unsigned a[4] = {a_frag[0], a_frag[1], a_frag[2], a_frag[3]};
    unsigned b[2] = {b_frag[0], b_frag[1]};
    unsigned sa = scales[0], sb = scales[1];
    float c[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
    int lane = threadIdx.x & 31;
    out[lane * 4 + 0] = c[0]; out[lane * 4 + 1] = c[1];
    out[lane * 4 + 2] = c[2]; out[lane * 4 + 3] = c[3];
}
