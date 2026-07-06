// Probe: SLAB-REPACK design arc (2026-07-06) — measure a decode-friendly SPLIT-PLANE repack of
// IQ3_S / IQ4_XS expert weights vs the current GGUF block layout, at the EXACT 35B expert decode
// shapes and launch geometries, BEFORE any engine integration.
//
// CONTEXT (research/tune-data/g7e-rtx6000.jsonl 2026-07-06 18:00 row): expert decode kernels sit
// at 43-51% of the 1537 GB/s box wall (gate_up IQ3_S 790 GB/s, down IQ4_XS 660 GB/s) and every
// cheap lever measured flat (variant sweeps x2, sg/j8sg smem-grid, wider loads, k-split/gs4).
// The remaining structural lever: expert_dot_iq3s_g reads FIVE scattered streams per 32-elem
// group across a 110B block (d@0, qs@2+g*8, qh@66+g, signs@74+g*4, scale nibble@106+g/2);
// expert_dot_iq4xs_g reads d/sh/sl scattered + 16 byte-wise qs loads. A6 split-plane (NVFP4,
// nvfp4_repack.rs) proved the class: repack at slab-fill into walk-order records -> 1-2 aligned
// LDG.128s per group, bit-identical outputs (pure address remap; dequant VALUES and dp4a order
// unchanged).
//
// REPACKED LAYOUTS measured here:
//   IQ3_S  (gate_up, in_f=2048): per-group 16B record [qs 8B | signs 4B | qh 1B | sc_nib 1B |
//          d fp16 2B] — d/scale DUPLICATED per group so ONE int4 load serves the whole dot.
//          Row stride 16B*nsb (1024B) vs GGUF 880B (+16.4% bytes). Consecutive lanes (groups)
//          read consecutive 16B records -> fully coalesced warp stream.
//   IQ4_XS (down, in_f=512): qs plane (16B/group, the GGUF qs bytes verbatim) + meta plane
//          (4B/group: [d fp16 | ls byte | pad]) — 2 aligned loads/group. Row cost 320B vs 272B
//          (+17.6%).
//
// GEOMETRY mirrors the engine defaults exactly (lib.rs moe_*_dev_q8 "auto"):
//   gate_up: moe_gate_up_silu8_dev_q8   grid(n_ff=512, n_used=8) block(32)   [qmatvec.cu:4328]
//   down:    moe_down8_fma_dev_q8_w8h2  grid(out_f/2=1024)      block(32,8)  [qmatvec.cu:4575]
// Base kernels below are those kernels VERBATIM (dot bodies lifted from expert_dot_iq3s_g /
// expert_dot_iq4xs_g qmatvec.cu:3651/3678); rp twins change ONLY the weight addressing.
//
// MEASUREMENT LAW: 40 independent layer-sets cycled per launch (the box L2 is 128MB — a single
// 7MB layer looped would measure L2 bandwidth, not the real decode DRAM regime where 40 MoE
// layers stream through). Gate: bit-identity of act/dst arrays (memcmp) + interleaved A/B/A
// timing, N=5 reps, median. Decision per brief: rp kernel win < 15% => record NEGATIVE and close
// the direction; >= 15% => integrate behind BW24_MOE_RP.
//
// Build:
//   nvcc -O3 -arch=compute_120a -code=sm_120a probe/slab_repack_expert.cu -o probe/slab_repack_expert
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// IQ3_S grid + IQ4_NL codebook — spliced VERBATIM from qmatvec.cu by the build script.
__device__ unsigned int iq3s_grid_const[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};
__device__ signed char kvalues_iq4nl_d[16] =
    {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

__device__ __forceinline__ unsigned int iq3s_grid_d(int idx) { return iq3s_grid_const[idx]; }

// ---- BASE dot bodies (qmatvec.cu:3651 / 3678 VERBATIM) ----
__device__ __forceinline__ float expert_dot_iq3s_g(const unsigned char* wrow, int g,
                                                   const signed char* aqb, float d8) {
    int sblk = g >> 3, ib32 = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    float d = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs    = b + 2  + ib32 * 8;
    unsigned char qh           = b[66 + ib32];
    const unsigned char* signs = b + 74 + ib32 * 4;
    const unsigned char* scales= b + 106;
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
        unsigned char sb = signs[l0 / 2];
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        int grid_l = __vsub4(gl ^ signs0, signs0);
        int grid_h = __vsub4(gh ^ signs1, signs1);
        sumi = dp4a(grid_l, aq4[l0 + 0], sumi);
        sumi = dp4a(grid_h, aq4[l0 + 1], sumi);
    }
    return db * (float)sumi * d8;
}
__device__ __forceinline__ float expert_dot_iq4xs_g(const unsigned char* wrow, int g,
                                                    const signed char* aqb, float d8) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    float d_sb = half_to_float(*(const unsigned short*)b);
    unsigned short sh = *(const unsigned short*)(b + 2);
    const unsigned char* sl = b + 4;
    const unsigned char* qs = b + 8 + ib * 16;
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    int scale = ls - 32;
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int sumi = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int wlo = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
        int whi = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
        sumi = dp4a(wlo, aLo[k], sumi);
        sumi = dp4a(whi, aHi[k], sumi);
    }
    return d_sb * (float)(scale * sumi) * d8;
}

// ---- RP dot bodies: SAME dequant values, SAME dp4a/accumulate order — only addressing changes ----
// IQ3_S rp record (16B): [0..7]=qs [8..11]=signs [12]=qh [13]=sc_nib [14..15]=d(fp16)
__device__ __forceinline__ float expert_dot_iq3s_g_rp(const unsigned char* wrow_rp, int g,
                                                      const signed char* aqb, float d8) {
    const int4 rec = *(const int4*)(wrow_rp + (size_t)g * 16);
    const int qsx = rec.x, qsy = rec.y, sgn = rec.z, w3 = rec.w;
    float d = half_to_float((unsigned short)(((unsigned int)w3) >> 16));
    int qh = w3 & 0xff;
    int sc_nib = (w3 >> 8) & 0xff;
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int q0 = ((l0 < 4 ? qsx : qsy) >> (8 * (l0 & 3)))        & 0xff;
        int q1 = ((l0 + 1 < 4 ? qsx : qsy) >> (8 * ((l0 + 1) & 3))) & 0xff;
        int gl = iq3s_grid_d(q0 | ((qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(q1 | ((qh << (7 - l0)) & 0x100));
        unsigned char sb = (unsigned char)((sgn >> (8 * (l0 / 2))) & 0xff);
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        int grid_l = __vsub4(gl ^ signs0, signs0);
        int grid_h = __vsub4(gh ^ signs1, signs1);
        sumi = dp4a(grid_l, aq4[l0 + 0], sumi);
        sumi = dp4a(grid_h, aq4[l0 + 1], sumi);
    }
    return db * (float)sumi * d8;
}
// IQ4_XS rp: qs plane 16B/group + meta plane 4B/group [d fp16 | ls | pad]
__device__ __forceinline__ float expert_dot_iq4xs_g_rp(const unsigned char* qsrow, int g,
                                                       const unsigned char* metarow,
                                                       const signed char* aqb, float d8) {
    const int4 q = *(const int4*)(qsrow + (size_t)g * 16);
    const int m  = *(const int*)(metarow + (size_t)g * 4);
    float d_sb = half_to_float((unsigned short)(m & 0xffff));
    int scale = ((m >> 16) & 0xff) - 32;
    int qw[4] = { q.x, q.y, q.z, q.w };
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int sumi = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int b0 = (qw[k] >>  0) & 0xff, b1 = (qw[k] >>  8) & 0xff,
            b2 = (qw[k] >> 16) & 0xff, b3 = (qw[k] >> 24) & 0xff;
        int wlo = (kvalues_iq4nl_d[b0&0xf]&0xff) | ((kvalues_iq4nl_d[b1&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[b2&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[b3&0xf]&0xff)<<24);
        int whi = (kvalues_iq4nl_d[b0>>4]&0xff) | ((kvalues_iq4nl_d[b1>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[b2>>4]&0xff)<<16) | ((kvalues_iq4nl_d[b3>>4]&0xff)<<24);
        sumi = dp4a(wlo, aLo[k], sumi);
        sumi = dp4a(whi, aHi[k], sumi);
    }
    return d_sb * (float)(scale * sumi) * d8;
}

// ---- RP2 dot bodies: ZERO-INFLATION split planes (v1 showed +23%/+21% raw stream throughput
// but the 16-17% byte inflation from duplicated d/scale ate it to +5.7%/+3.3%). v2 keeps bytes
// ~equal to GGUF: per-group planes for the group-varying streams + ONE per-block meta record
// that the 8 lanes sharing a block load broadcast-style (same address -> one L1 transaction).
// IQ3_S v2: qs plane 8B/group + signs plane 4B/group + meta plane 16B/block [qh8|scales4|d2|pad2]
//           = 112B/block vs 110 GGUF (+1.8%). 3 aligned loads/group vs 5 scattered.
// IQ4_XS v2: qs plane 16B/group (int4) + meta plane 8B/block [d2|sh2|sl4] (the GGUF header
//           VERBATIM) = 136B/block vs 136 GGUF (+0.0%). 2 aligned loads/group.
__device__ __forceinline__ float expert_dot_iq3s_g_rp2(
        const unsigned char* qs_row, const unsigned char* sg_row, const unsigned char* mt_row,
        int g, const signed char* aqb, float d8) {
    int sblk = g >> 3, ib32 = g & 7;
    const int2 qsw = *(const int2*)(qs_row + (size_t)g * 8);
    const int sgn  = *(const int*)(sg_row + (size_t)g * 4);
    const int4 mt  = *(const int4*)(mt_row + (size_t)sblk * 16);   // 8 lanes/block: broadcast
    int qh = ((ib32 < 4 ? mt.x : mt.y) >> (8 * (ib32 & 3))) & 0xff;
    int scb = (mt.z >> (8 * (ib32 >> 1))) & 0xff;                  // scales[ib32/2]
    int sc_nib = (ib32 & 1) ? (scb >> 4) : (scb & 0xf);
    float d = half_to_float((unsigned short)(mt.w & 0xffff));
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int q0 = ((l0 < 4 ? qsw.x : qsw.y) >> (8 * (l0 & 3))) & 0xff;
        int q1 = ((l0 + 1 < 4 ? qsw.x : qsw.y) >> (8 * ((l0 + 1) & 3))) & 0xff;
        int gl = iq3s_grid_d(q0 | ((qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(q1 | ((qh << (7 - l0)) & 0x100));
        unsigned char sb = (unsigned char)((sgn >> (8 * (l0 / 2))) & 0xff);
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        int grid_l = __vsub4(gl ^ signs0, signs0);
        int grid_h = __vsub4(gh ^ signs1, signs1);
        sumi = dp4a(grid_l, aq4[l0 + 0], sumi);
        sumi = dp4a(grid_h, aq4[l0 + 1], sumi);
    }
    return db * (float)sumi * d8;
}
__device__ __forceinline__ float expert_dot_iq4xs_g_rp2(
        const unsigned char* qs_row, const unsigned char* mt_row,
        int g, const signed char* aqb, float d8) {
    int sblk = g >> 3, ib = g & 7;
    const int4 q = *(const int4*)(qs_row + (size_t)g * 16);
    const int2 mt = *(const int2*)(mt_row + (size_t)sblk * 8);     // [d2|sh2 | sl4] GGUF header
    float d_sb = half_to_float((unsigned short)(mt.x & 0xffff));
    unsigned short sh = (unsigned short)(((unsigned int)mt.x) >> 16);
    int slb = (mt.y >> (8 * (ib >> 1))) & 0xff;                    // sl[ib>>1]
    int ls = ((slb >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    int scale = ls - 32;
    int qw[4] = { q.x, q.y, q.z, q.w };
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int sumi = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int b0 = (qw[k] >>  0) & 0xff, b1 = (qw[k] >>  8) & 0xff,
            b2 = (qw[k] >> 16) & 0xff, b3 = (qw[k] >> 24) & 0xff;
        int wlo = (kvalues_iq4nl_d[b0&0xf]&0xff) | ((kvalues_iq4nl_d[b1&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[b2&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[b3&0xf]&0xff)<<24);
        int whi = (kvalues_iq4nl_d[b0>>4]&0xff) | ((kvalues_iq4nl_d[b1>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[b2>>4]&0xff)<<16) | ((kvalues_iq4nl_d[b3>>4]&0xff)<<24);
        sumi = dp4a(wlo, aLo[k], sumi);
        sumi = dp4a(whi, aHi[k], sumi);
    }
    return d_sb * (float)(scale * sumi) * d8;
}

// ---- gate_up kernels: moe_gate_up_silu8_dev_q8 geometry (grid (n_ff, n_used), block 32) ----
extern "C" __global__ void gu_base(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act, int in_f, int n_ff, int n_expert, long rb_g, long rb_u) {
    int o = blockIdx.x, j = blockIdx.y, lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_iq3s_g(grow, g, aqb, d8);
        accu += expert_dot_iq3s_g(urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
extern "C" __global__ void gu_rp(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act, int in_f, int n_ff, int n_expert, long rb_g, long rb_u) {
    int o = blockIdx.x, j = blockIdx.y, lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_iq3s_g_rp(grow, g, aqb, d8);
        accu += expert_dot_iq3s_g_rp(urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

extern "C" __global__ void gu_rp2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act, int in_f, int n_ff, int n_expert,
        long sg_off, long mt_off, long rb_qs, long rb_sg, long rb_mt) {
    int o = blockIdx.x, j = blockIdx.y, lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* gbase = (const unsigned char*)table[ex];
    const unsigned char* ubase = (const unsigned char*)table[n_expert + ex];
    const unsigned char* gqs = gbase + (long)o * rb_qs;
    const unsigned char* gsg = gbase + sg_off + (long)o * rb_sg;
    const unsigned char* gmt = gbase + mt_off + (long)o * rb_mt;
    const unsigned char* uqs = ubase + (long)o * rb_qs;
    const unsigned char* usg = ubase + sg_off + (long)o * rb_sg;
    const unsigned char* umt = ubase + mt_off + (long)o * rb_mt;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_iq3s_g_rp2(gqs, gsg, gmt, g, aqb, d8);
        accu += expert_dot_iq3s_g_rp2(uqs, usg, umt, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// ---- down kernels: moe_down8_fma_dev_q8_w8h2 geometry (grid out_f/2, block (32, n_used)) ----
__device__ __forceinline__ float2 down_h2_dot_base(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, long rb, int lane) {
    int nsb = in_f >> 5;                       // == 16 (shape-gated)
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                              + (long)(o0 + half) * rb;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    float acc = expert_dot_iq4xs_g(wrow, l16, arow + (size_t)l16 * 32, adrow[l16]);
    float accA = (half == 0) ? acc : 0.0f;
    float a0 = warp_reduce_sum(accA);
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f;
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
__device__ __forceinline__ float2 down_h2_dot_rp(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, long rb_qs, long meta_off, int lane) {
    int nsb = in_f >> 5;
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* base = (const unsigned char*)table[2 * n_expert + ex];
    const unsigned char* qsrow   = base + (long)(o0 + half) * rb_qs;
    const unsigned char* metarow = base + meta_off + (long)(o0 + half) * (nsb * 4);
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    float acc = expert_dot_iq4xs_g_rp(qsrow, l16, metarow, arow + (size_t)l16 * 32, adrow[l16]);
    float accA = (half == 0) ? acc : 0.0f;
    float a0 = warp_reduce_sum(accA);
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f;
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void down_base(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_base(table, sel, aq2, ad2, j, o0, in_f, n_expert, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
__device__ __forceinline__ float2 down_h2_dot_rp2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, long rb_qs, long mt_off, long rb_mt, int lane) {
    int nsb = in_f >> 5;
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* base = (const unsigned char*)table[2 * n_expert + ex];
    const unsigned char* qsrow = base + (long)(o0 + half) * rb_qs;
    const unsigned char* mtrow = base + mt_off + (long)(o0 + half) * rb_mt;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    float acc = expert_dot_iq4xs_g_rp2(qsrow, mtrow, l16, arow + (size_t)l16 * 32, adrow[l16]);
    float accA = (half == 0) ? acc : 0.0f;
    float a0 = warp_reduce_sum(accA);
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f;
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void down_rp2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, long rb_qs, long mt_off, long rb_mt) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_rp2(table, sel, aq2, ad2, j, o0, in_f, n_expert, rb_qs, mt_off, rb_mt, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
extern "C" __global__ void down_rp(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, long rb_qs, long meta_off) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_rp(table, sel, aq2, ad2, j, o0, in_f, n_expert, rb_qs, meta_off, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}

// ================================ host ================================
static uint16_t rand_half_small() {
    // fp16 patterns in a small-normal band (~1e-3..1e-2); any bits would preserve bit-identity,
    // sane values keep the outputs finite for eyeballing.
    return (uint16_t)(0x1400 | (rand() & 0x03FF));
}

// host repack IQ3_S row (in_f elems, in_f/256 superblocks of 110B) -> nsb 16B records
static void repack_iq3s_row(const uint8_t* row, uint8_t* rp, int in_f) {
    int nsblk = in_f / 256;
    for (int sblk = 0; sblk < nsblk; sblk++) {
        const uint8_t* b = row + (size_t)sblk * 110;
        for (int ib = 0; ib < 8; ib++) {
            uint8_t* rec = rp + ((size_t)sblk * 8 + ib) * 16;
            memcpy(rec, b + 2 + ib * 8, 8);              // qs
            memcpy(rec + 8, b + 74 + ib * 4, 4);         // signs
            rec[12] = b[66 + ib];                        // qh
            rec[13] = (ib & 1) ? (uint8_t)(b[106 + ib / 2] >> 4)
                               : (uint8_t)(b[106 + ib / 2] & 0xf); // sc_nib
            rec[14] = b[0]; rec[15] = b[1];              // d (fp16 bits, duplicated per group)
        }
    }
}
// host repack IQ3_S row v2 -> qs plane (nsb*8B) + signs plane (nsb*4B) + meta plane (nsblk*16B)
static void repack_iq3s_row_v2(const uint8_t* row, uint8_t* qsp, uint8_t* sgp, uint8_t* mtp, int in_f) {
    int nsblk = in_f / 256;
    for (int sblk = 0; sblk < nsblk; sblk++) {
        const uint8_t* b = row + (size_t)sblk * 110;
        for (int ib = 0; ib < 8; ib++) {
            memcpy(qsp + ((size_t)sblk * 8 + ib) * 8, b + 2 + ib * 8, 8);
            memcpy(sgp + ((size_t)sblk * 8 + ib) * 4, b + 74 + ib * 4, 4);
        }
        uint8_t* m = mtp + (size_t)sblk * 16;
        memcpy(m, b + 66, 8);        // qh[8]
        memcpy(m + 8, b + 106, 4);   // scales[4]
        m[12] = b[0]; m[13] = b[1];  // d
        m[14] = 0; m[15] = 0;
    }
}
// host repack IQ4_XS row v2 -> qs plane (nsb*16B) + meta plane (nsblk*8B = GGUF header verbatim)
static void repack_iq4xs_row_v2(const uint8_t* row, uint8_t* qsp, uint8_t* mtp, int in_f) {
    int nsblk = in_f / 256;
    for (int sblk = 0; sblk < nsblk; sblk++) {
        const uint8_t* b = row + (size_t)sblk * 136;
        memcpy(qsp + (size_t)sblk * 128, b + 8, 128);
        memcpy(mtp + (size_t)sblk * 8, b, 8);         // [d2|sh2|sl4]
    }
}
// host repack IQ4_XS row -> qs plane (nsb*16B) + meta plane (nsb*4B), planes packed per EXPERT
static void repack_iq4xs_row(const uint8_t* row, uint8_t* qsp, uint8_t* metap, int in_f) {
    int nsblk = in_f / 256;
    for (int sblk = 0; sblk < nsblk; sblk++) {
        const uint8_t* b = row + (size_t)sblk * 136;
        uint16_t sh; memcpy(&sh, b + 2, 2);
        const uint8_t* sl = b + 4;
        for (int ib = 0; ib < 8; ib++) {
            memcpy(qsp + ((size_t)sblk * 8 + ib) * 16, b + 8 + ib * 16, 16);
            uint8_t ls = (uint8_t)((((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf)
                                   | (((sh >> (2 * ib)) & 3) << 4)));
            uint8_t* m = metap + ((size_t)sblk * 8 + ib) * 4;
            m[0] = b[0]; m[1] = b[1]; m[2] = ls; m[3] = 0;
        }
    }
}

struct Timing { float med_us; };
static Timing time_launches(void (*launch)(int set, void*), void* ctx, int nsets, int niter, int reps) {
    std::vector<float> t(reps);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0));
        for (int it = 0; it < niter; it++) launch(it % nsets, ctx);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        t[r] = ms * 1000.0f / niter;
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    std::sort(t.begin(), t.end());
    return Timing{ t[reps / 2] };
}

// ---- gate_up experiment ----
static const int GU_INF = 2048, GU_NFF = 512, N_USED = 8, N_EXP = 8, NSETS = 40;
static const int GU_NSB = GU_INF / 32;                       // 64 groups
static const long GU_RB   = (long)(GU_INF / 256) * 110;      // 880
static const long GU_RB_RP = (long)GU_NSB * 16;              // 1024
// rp2 plane strides (per expert): qs 8B/g, signs 4B/g, meta 16B/block
static const long GU_RB_QS2 = (long)GU_NSB * 8;              // 512
static const long GU_RB_SG2 = (long)GU_NSB * 4;              // 256
static const long GU_RB_MT2 = (long)(GU_INF / 256) * 16;     // 128
static const long GU_SG_OFF2 = (long)GU_NFF * GU_RB_QS2;
static const long GU_MT_OFF2 = GU_SG_OFF2 + (long)GU_NFF * GU_RB_SG2;
struct GuCtx {
    unsigned long long* d_tables[NSETS];    // device table ptr per set
    int* d_sel; signed char* d_aq; float* d_ad; float* d_act;
    int mode;                               // 0=base 1=rp 2=rp2
};
static void gu_launch(int set, void* vctx) {
    GuCtx* c = (GuCtx*)vctx;
    dim3 grid(GU_NFF, N_USED), block(32);
    if (c->mode == 2)
        gu_rp2<<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_aq, c->d_ad, c->d_act,
                                GU_INF, GU_NFF, N_EXP, GU_SG_OFF2, GU_MT_OFF2,
                                GU_RB_QS2, GU_RB_SG2, GU_RB_MT2);
    else if (c->mode == 1)
        gu_rp <<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_aq, c->d_ad, c->d_act,
                                GU_INF, GU_NFF, N_EXP, GU_RB_RP, GU_RB_RP);
    else
        gu_base<<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_aq, c->d_ad, c->d_act,
                                 GU_INF, GU_NFF, N_EXP, GU_RB, GU_RB);
}

// ---- down experiment ----
static const int DN_INF = 512, DN_OUTF = 2048;
static const int DN_NSB = DN_INF / 32;                        // 16 groups
static const long DN_RB    = (long)(DN_INF / 256) * 136;      // 272
static const long DN_RB_QS = (long)DN_NSB * 16;               // 256
static const long DN_META_OFF = (long)DN_OUTF * DN_RB_QS;     // meta plane after qs plane (per expert)
// rp2 plane strides: qs 16B/g (GGUF qs verbatim), meta 8B/block (GGUF header verbatim)
static const long DN_RB_QS2 = (long)DN_NSB * 16;             // 256
static const long DN_RB_MT2 = (long)(DN_INF / 256) * 8;      // 16
static const long DN_MT_OFF2 = (long)DN_OUTF * DN_RB_QS2;
struct DnCtx {
    unsigned long long* d_tables[NSETS];
    int* d_sel; float* d_w; signed char* d_aq2; float* d_ad2; float* d_dst;
    int mode;                               // 0=base 1=rp 2=rp2
};
static void dn_launch(int set, void* vctx) {
    DnCtx* c = (DnCtx*)vctx;
    dim3 grid(DN_OUTF / 2), block(32, N_USED);
    if (c->mode == 2)
        down_rp2<<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_w, c->d_aq2, c->d_ad2,
                                  c->d_dst, DN_INF, DN_OUTF, N_USED, N_EXP,
                                  DN_RB_QS2, DN_MT_OFF2, DN_RB_MT2);
    else if (c->mode == 1)
        down_rp <<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_w, c->d_aq2, c->d_ad2,
                                  c->d_dst, DN_INF, DN_OUTF, N_USED, N_EXP, DN_RB_QS, DN_META_OFF);
    else
        down_base<<<grid, block>>>(c->d_tables[set], c->d_sel, c->d_w, c->d_aq2, c->d_ad2,
                                   c->d_dst, DN_INF, DN_OUTF, N_USED, N_EXP, DN_RB);
}

int main() {
    srand(24);
    int niter = 2000, reps = 5;
    printf("SLAB-REPACK probe: 35B expert shapes, %d layer-sets cycled, %d iters, %d reps (median)\n",
           NSETS, niter, reps);

    // ======================= gate_up IQ3_S =======================
    {
        size_t wrow = GU_RB, wrow_rp = GU_RB_RP;
        size_t exp_bytes = (size_t)GU_NFF * wrow, exp_bytes_rp = (size_t)GU_NFF * wrow_rp;
        // host weights for ONE set (bit-identity check); device: NSETS sets, random content per set
        std::vector<uint8_t> hw((size_t)2 * N_EXP * exp_bytes);
        for (size_t i = 0; i < hw.size(); i++) hw[i] = (uint8_t)rand();
        // patch every superblock's d to a sane fp16
        for (size_t p = 0; p < (size_t)2 * N_EXP; p++)
            for (int r = 0; r < GU_NFF; r++)
                for (int sb = 0; sb < GU_INF / 256; sb++) {
                    uint16_t d = rand_half_small();
                    memcpy(&hw[p * exp_bytes + (size_t)r * wrow + (size_t)sb * 110], &d, 2);
                }
        std::vector<uint8_t> hw_rp((size_t)2 * N_EXP * exp_bytes_rp);
        for (size_t p = 0; p < (size_t)2 * N_EXP; p++)
            for (int r = 0; r < GU_NFF; r++)
                repack_iq3s_row(&hw[p * exp_bytes + (size_t)r * wrow],
                                &hw_rp[p * exp_bytes_rp + (size_t)r * wrow_rp], GU_INF);
        // rp2 planes per expert: qs | signs | meta
        size_t exp_bytes_rp2 = (size_t)GU_NFF * (GU_RB_QS2 + GU_RB_SG2 + GU_RB_MT2);
        std::vector<uint8_t> hw_rp2((size_t)2 * N_EXP * exp_bytes_rp2);
        for (size_t p = 0; p < (size_t)2 * N_EXP; p++)
            for (int r = 0; r < GU_NFF; r++)
                repack_iq3s_row_v2(&hw[p * exp_bytes + (size_t)r * wrow],
                                   &hw_rp2[p * exp_bytes_rp2 + (size_t)r * GU_RB_QS2],
                                   &hw_rp2[p * exp_bytes_rp2 + GU_SG_OFF2 + (size_t)r * GU_RB_SG2],
                                   &hw_rp2[p * exp_bytes_rp2 + GU_MT_OFF2 + (size_t)r * GU_RB_MT2],
                                   GU_INF);

        // device: allocate NSETS x (2*N_EXP experts) both layouts; set 0 = the checked content
        GuCtx cb{}, cr{}, c2{};
        for (int s = 0; s < NSETS; s++) {
            uint8_t *db, *dr, *d2;
            CK(cudaMalloc(&db, 2 * N_EXP * exp_bytes));
            CK(cudaMalloc(&dr, 2 * N_EXP * exp_bytes_rp));
            CK(cudaMalloc(&d2, 2 * N_EXP * exp_bytes_rp2));
            // content identical per set; distinct ADDRESSES defeat L2 across the cycled sets
            CK(cudaMemcpy(db, hw.data(), hw.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dr, hw_rp.data(), hw_rp.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d2, hw_rp2.data(), hw_rp2.size(), cudaMemcpyHostToDevice));
            std::vector<unsigned long long> tb(2 * N_EXP), tr(2 * N_EXP), t2(2 * N_EXP);
            for (int e = 0; e < 2 * N_EXP; e++) {
                tb[e] = (unsigned long long)(db + (size_t)e * exp_bytes);
                tr[e] = (unsigned long long)(dr + (size_t)e * exp_bytes_rp);
                t2[e] = (unsigned long long)(d2 + (size_t)e * exp_bytes_rp2);
            }
            CK(cudaMalloc(&cb.d_tables[s], 2 * N_EXP * 8));
            CK(cudaMalloc(&cr.d_tables[s], 2 * N_EXP * 8));
            CK(cudaMalloc(&c2.d_tables[s], 2 * N_EXP * 8));
            CK(cudaMemcpy(cb.d_tables[s], tb.data(), 2 * N_EXP * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cr.d_tables[s], tr.data(), 2 * N_EXP * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(c2.d_tables[s], t2.data(), 2 * N_EXP * 8, cudaMemcpyHostToDevice));
        }
        std::vector<int> hsel = {3, 1, 7, 0, 5, 2, 6, 4};
        std::vector<int8_t> haq(GU_INF); for (auto& v : haq) v = (int8_t)(rand() % 255 - 127);
        std::vector<float> had(GU_NSB);  for (auto& v : had) v = 0.001f + 0.01f * (rand() / (float)RAND_MAX);
        CK(cudaMalloc(&cb.d_sel, hsel.size() * 4));
        CK(cudaMemcpy(cb.d_sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_aq, haq.size()));
        CK(cudaMemcpy(cb.d_aq, haq.data(), haq.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_ad, had.size() * 4));
        CK(cudaMemcpy(cb.d_ad, had.data(), had.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_act, (size_t)N_USED * GU_NFF * 4));
        // cr/c2 share sel/aq/ad/act with cb but keep their OWN tables (saved across struct copies)
        unsigned long long* sav_r[NSETS]; memcpy(sav_r, cr.d_tables, sizeof(sav_r));
        unsigned long long* sav_2[NSETS]; memcpy(sav_2, c2.d_tables, sizeof(sav_2));
        cr = cb; memcpy(cr.d_tables, sav_r, sizeof(sav_r)); cr.mode = 1;
        c2 = cb; memcpy(c2.d_tables, sav_2, sizeof(sav_2)); c2.mode = 2;
        printf("gate_up IQ3_S: shapes n_ff=%d in_f=%d n_used=%d rb=%ld rb_rp=%ld\n",
               GU_NFF, GU_INF, N_USED, GU_RB, GU_RB_RP);
        // bit-identity on set 0
        std::vector<float> a0((size_t)N_USED * GU_NFF), a1((size_t)N_USED * GU_NFF);
        cb.mode = 0; gu_launch(0, &cb); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), cb.d_act, a0.size() * 4, cudaMemcpyDeviceToHost));
        for (GuCtx* c : { &cr, &c2 }) {
            gu_launch(0, c); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(a1.data(), c->d_act, a1.size() * 4, cudaMemcpyDeviceToHost));
            int mism = memcmp(a0.data(), a1.data(), a0.size() * 4) ? 1 : 0;
            if (mism) {
                for (size_t i = 0; i < a0.size(); i++)
                    if (memcmp(&a0[i], &a1[i], 4)) {
                        printf("  rp%d FIRST MISMATCH @ %zu: base %.9g rp %.9g\n",
                               c->mode, i, a0[i], a1[i]); break;
                    }
            }
            printf("  rp%d bit-identity: %s\n", c->mode, mism ? "FAIL" : "PASS (0 mismatched bytes)");
        }
        // warmup + interleaved timing A/B/C/A
        for (int it = 0; it < 200; it++) { gu_launch(it % NSETS, &cb); gu_launch(it % NSETS, &cr); gu_launch(it % NSETS, &c2); }
        CK(cudaDeviceSynchronize());
        Timing tb1 = time_launches(gu_launch, &cb, NSETS, niter, reps);
        Timing tr1 = time_launches(gu_launch, &cr, NSETS, niter, reps);
        Timing t21 = time_launches(gu_launch, &c2, NSETS, niter, reps);
        Timing tb2 = time_launches(gu_launch, &cb, NSETS, niter, reps);
        double base_us = 0.5 * (tb1.med_us + tb2.med_us);
        double wb = 2.0 * N_EXP * exp_bytes, wr = 2.0 * N_EXP * exp_bytes_rp,
               w2 = 2.0 * N_EXP * exp_bytes_rp2;
        printf("  base: %.2f us (A %.2f / A' %.2f) = %.0f GB/s wt-stream\n",
               base_us, tb1.med_us, tb2.med_us, wb / base_us / 1e3);
        printf("  rp1:  %.2f us = %.0f GB/s wt-stream (%+.1f%% bytes)  SPEEDUP %.3fx (%+.1f%%)\n",
               tr1.med_us, wr / tr1.med_us / 1e3, 100.0 * (wr / wb - 1.0),
               base_us / tr1.med_us, 100.0 * (base_us / tr1.med_us - 1.0));
        printf("  rp2:  %.2f us = %.0f GB/s wt-stream (%+.1f%% bytes)  SPEEDUP %.3fx (%+.1f%%)\n",
               t21.med_us, w2 / t21.med_us / 1e3, 100.0 * (w2 / wb - 1.0),
               base_us / t21.med_us, 100.0 * (base_us / t21.med_us - 1.0));
    }

    // ======================= down IQ4_XS =======================
    {
        size_t wrow = DN_RB;
        size_t exp_bytes = (size_t)DN_OUTF * wrow;
        size_t exp_bytes_rp = (size_t)DN_OUTF * (DN_RB_QS + DN_NSB * 4);  // qs plane + meta plane
        std::vector<uint8_t> hw((size_t)N_EXP * exp_bytes);
        for (size_t i = 0; i < hw.size(); i++) hw[i] = (uint8_t)rand();
        for (size_t e = 0; e < (size_t)N_EXP; e++)
            for (int r = 0; r < DN_OUTF; r++)
                for (int sb = 0; sb < DN_INF / 256; sb++) {
                    uint16_t d = rand_half_small();
                    memcpy(&hw[e * exp_bytes + (size_t)r * wrow + (size_t)sb * 136], &d, 2);
                }
        std::vector<uint8_t> hw_rp((size_t)N_EXP * exp_bytes_rp);
        for (size_t e = 0; e < (size_t)N_EXP; e++)
            for (int r = 0; r < DN_OUTF; r++)
                repack_iq4xs_row(&hw[e * exp_bytes + (size_t)r * wrow],
                                 &hw_rp[e * exp_bytes_rp + (size_t)r * DN_RB_QS],
                                 &hw_rp[e * exp_bytes_rp + DN_META_OFF + (size_t)r * DN_NSB * 4],
                                 DN_INF);
        size_t exp_bytes_rp2 = (size_t)DN_OUTF * (DN_RB_QS2 + DN_RB_MT2);
        std::vector<uint8_t> hw_rp2((size_t)N_EXP * exp_bytes_rp2);
        for (size_t e = 0; e < (size_t)N_EXP; e++)
            for (int r = 0; r < DN_OUTF; r++)
                repack_iq4xs_row_v2(&hw[e * exp_bytes + (size_t)r * wrow],
                                    &hw_rp2[e * exp_bytes_rp2 + (size_t)r * DN_RB_QS2],
                                    &hw_rp2[e * exp_bytes_rp2 + DN_MT_OFF2 + (size_t)r * DN_RB_MT2],
                                    DN_INF);
        DnCtx cb{}, cr{}, c2{};
        for (int s = 0; s < NSETS; s++) {
            uint8_t *db, *dr, *d2;
            CK(cudaMalloc(&db, N_EXP * exp_bytes));
            CK(cudaMalloc(&dr, N_EXP * exp_bytes_rp));
            CK(cudaMalloc(&d2, N_EXP * exp_bytes_rp2));
            CK(cudaMemcpy(db, hw.data(), hw.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dr, hw_rp.data(), hw_rp.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d2, hw_rp2.data(), hw_rp2.size(), cudaMemcpyHostToDevice));
            // table layout matches kernel: index 2*n_expert+ex
            std::vector<unsigned long long> tb(3 * N_EXP, 0), tr(3 * N_EXP, 0), t2(3 * N_EXP, 0);
            for (int e = 0; e < N_EXP; e++) {
                tb[2 * N_EXP + e] = (unsigned long long)(db + (size_t)e * exp_bytes);
                tr[2 * N_EXP + e] = (unsigned long long)(dr + (size_t)e * exp_bytes_rp);
                t2[2 * N_EXP + e] = (unsigned long long)(d2 + (size_t)e * exp_bytes_rp2);
            }
            CK(cudaMalloc(&cb.d_tables[s], 3 * N_EXP * 8));
            CK(cudaMalloc(&cr.d_tables[s], 3 * N_EXP * 8));
            CK(cudaMalloc(&c2.d_tables[s], 3 * N_EXP * 8));
            CK(cudaMemcpy(cb.d_tables[s], tb.data(), 3 * N_EXP * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cr.d_tables[s], tr.data(), 3 * N_EXP * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(c2.d_tables[s], t2.data(), 3 * N_EXP * 8, cudaMemcpyHostToDevice));
        }
        std::vector<int> hsel = {6, 2, 0, 4, 7, 1, 3, 5};
        std::vector<float> hwts(N_USED); for (auto& v : hwts) v = 0.05f + 0.2f * (rand() / (float)RAND_MAX);
        std::vector<int8_t> haq((size_t)N_USED * DN_INF);
        for (auto& v : haq) v = (int8_t)(rand() % 255 - 127);
        std::vector<float> had((size_t)N_USED * DN_NSB);
        for (auto& v : had) v = 0.001f + 0.01f * (rand() / (float)RAND_MAX);
        CK(cudaMalloc(&cb.d_sel, hsel.size() * 4));
        CK(cudaMemcpy(cb.d_sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_w, hwts.size() * 4));
        CK(cudaMemcpy(cb.d_w, hwts.data(), hwts.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_aq2, haq.size()));
        CK(cudaMemcpy(cb.d_aq2, haq.data(), haq.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_ad2, had.size() * 4));
        CK(cudaMemcpy(cb.d_ad2, had.data(), had.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.d_dst, (size_t)DN_OUTF * 4));
        unsigned long long* sav_r[NSETS]; memcpy(sav_r, cr.d_tables, sizeof(sav_r));
        unsigned long long* sav_2[NSETS]; memcpy(sav_2, c2.d_tables, sizeof(sav_2));
        cr = cb; memcpy(cr.d_tables, sav_r, sizeof(sav_r)); cr.mode = 1;
        c2 = cb; memcpy(c2.d_tables, sav_2, sizeof(sav_2)); c2.mode = 2;
        printf("down IQ4_XS: shapes out_f=%d in_f=%d n_used=%d rb=%ld rb_qs=%ld meta_off=%ld\n",
               DN_OUTF, DN_INF, N_USED, DN_RB, DN_RB_QS, DN_META_OFF);
        std::vector<float> a0(DN_OUTF), a1(DN_OUTF);
        cb.mode = 0; dn_launch(0, &cb); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), cb.d_dst, a0.size() * 4, cudaMemcpyDeviceToHost));
        for (DnCtx* c : { &cr, &c2 }) {
            dn_launch(0, c); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(a1.data(), c->d_dst, a1.size() * 4, cudaMemcpyDeviceToHost));
            int mism = memcmp(a0.data(), a1.data(), a0.size() * 4) ? 1 : 0;
            if (mism) {
                for (size_t i = 0; i < a0.size(); i++)
                    if (memcmp(&a0[i], &a1[i], 4)) {
                        printf("  rp%d FIRST MISMATCH @ %zu: base %.9g rp %.9g\n",
                               c->mode, i, a0[i], a1[i]); break;
                    }
            }
            printf("  rp%d bit-identity: %s\n", c->mode, mism ? "FAIL" : "PASS (0 mismatched bytes)");
        }
        for (int it = 0; it < 200; it++) { dn_launch(it % NSETS, &cb); dn_launch(it % NSETS, &cr); dn_launch(it % NSETS, &c2); }
        CK(cudaDeviceSynchronize());
        Timing tb1 = time_launches(dn_launch, &cb, NSETS, niter, reps);
        Timing tr1 = time_launches(dn_launch, &cr, NSETS, niter, reps);
        Timing t21 = time_launches(dn_launch, &c2, NSETS, niter, reps);
        Timing tb2 = time_launches(dn_launch, &cb, NSETS, niter, reps);
        double base_us = 0.5 * (tb1.med_us + tb2.med_us);
        double wb = 1.0 * N_EXP * exp_bytes, wr = 1.0 * N_EXP * exp_bytes_rp,
               w2 = 1.0 * N_EXP * exp_bytes_rp2;
        printf("  base: %.2f us (A %.2f / A' %.2f) = %.0f GB/s wt-stream\n",
               base_us, tb1.med_us, tb2.med_us, wb / base_us / 1e3);
        printf("  rp1:  %.2f us = %.0f GB/s wt-stream (%+.1f%% bytes)  SPEEDUP %.3fx (%+.1f%%)\n",
               tr1.med_us, wr / tr1.med_us / 1e3, 100.0 * (wr / wb - 1.0),
               base_us / tr1.med_us, 100.0 * (base_us / tr1.med_us - 1.0));
        printf("  rp2:  %.2f us = %.0f GB/s wt-stream (%+.1f%% bytes)  SPEEDUP %.3fx (%+.1f%%)\n",
               t21.med_us, w2 / t21.med_us / 1e3, 100.0 * (w2 / wb - 1.0),
               base_us / t21.med_us, 100.0 * (base_us / t21.med_us - 1.0));
    }
    return 0;
}
