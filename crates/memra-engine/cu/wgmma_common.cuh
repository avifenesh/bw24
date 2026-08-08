// Shared Hopper wgmma toolkit (task #22 extraction): canonical descriptors, core-matrix
// offsets, m64n64 wgmma wrappers, cp.async 16B — the probe-verified pairings from
// ARCHITECTURE-H100.md (bf16 canonical: canon staging + desc(128,256); tf32: the bf16
// formula with 4-element kk groups, ledger 876cdcb7). Consumers: hybrid.cu (K2/K45
// wgmma family). The asm wrappers are sm_90a-only and sit behind MEMRA_K45_REAL; the
// pure-arithmetic helpers (offsets, descriptors) compile everywhere.
#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define MEMRA_K45_REAL 1
#endif

__device__ __forceinline__ unsigned long long k45_desc(const void* smem_ptr,
                                                       unsigned lead_bytes, unsigned stride_bytes) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ size_t k45_canon(int st, int r, int kk) {
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
}
#ifdef MEMRA_K45_REAL
// rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md
//   NOT-APPLICABLE on sm_120a: wgmma is an sm_90a (Hopper) instruction and does not exist in
//   the sm_120a ISA -- unmeasurable on this rig, and no instruction is emitted here in the
//   shipped build. Gated by MEMRA_K45_REAL, which is an internal arch-derived macro
//   (__CUDA_ARCH__ == 900), not a user-settable flag.
__device__ __forceinline__ void k45_wgmma(float acc[32], unsigned long long da,
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
__device__ __forceinline__ void k45_fence()  { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void k45_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void k45_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;"); }
__device__ __forceinline__ void k45_cp16(void* dst, const void* src, int sz) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(sz));
}
#endif

// tf32 canonical offset (probe_tf32.cu F1): element (r, kk of k8-step st), 4B elems
__device__ __forceinline__ size_t k45_tf_off(int st, int r, int kk) {
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 4) * 128 + (r % 8) * 16 + (kk % 4) * 4;
}
#ifdef MEMRA_K45_REAL
__device__ __forceinline__ void k45_wgmma_tf32(float acc[32], unsigned long long da,
                                               unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        // rate-audited 2026-08-06: wgmma is sm_90a-only, NOT-APPLICABLE on sm_120a (see
        // research/sm120-empirical-capabilities.md). For reference the sm_120a mma.sync tf32
        // form (m16n8k8, the only tf32 shape the ISA offers) is the slowest form on this rig:
        // 32.03 cyc/warp-MMA for just 1024 MACs = 38.9 TFLOP/s.
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
__device__ __forceinline__ unsigned k45_tf32r(float x) {
    unsigned u; asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return u;
}
#endif
