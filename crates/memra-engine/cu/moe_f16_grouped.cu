// moe_f16_grouped.cu — per-layer expert dequant to f16 + ONE grouped f16 GEMM per projection
// over the CSR expert->token groups (round 46 arc 2, the "vLLM shape" lever from round 44).
//
// WHY: the expert-segmented MMQ kernel re-dequants the same expert weights per
// (out-tile x token-tile x k-block) — at ~65-145 tokens/expert that is up to 12x redundant
// dequant, 13x scalar instructions per mma, and ~50% tile-padding waste (ncu round 46).
// Here the active experts dequant ONCE per (layer, projection) into an f16 workspace and
// cublasGemmGroupedBatchedEx runs every expert's [m_e x out_f x in_f] GEMM in one call at
// tensor-core f16 rate.
//
// NUMERICS: the f16-mirror class (campaign A / battery-adopted): dequanted weight values
// round to f16, f32 accumulate. NOT byte-identical to the MMQ path — argmax + spec gates
// arbitrate, per-model, like every numeric-class promotion. Experimental door
// MEMRA_MOE_F16G=1 until gated.
//
// Column-major mapping (derivation in the round-46 ledger): per group g,
//   C(m=out_f, n=m_e, k=in_f), opA=T opB=N,
//   A = Wf16[g]   (row-major [out_f][in_f], lda=in_f),
//   B = actf16 + pair_off[g]*in_f (pair-major row-major, ldb=in_f),
//   C = y + pair_off[g]*out_f     (f32 pair-major row-major, ldc=out_f).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>

#define QT_IQ4_XS 5
#define QT_IQ3_S  6
#define QT_Q4_0   12
#define QT_Q4_K   1
#define QT_Q6_K   2
#define QT_Q3_K   4

__device__ __forceinline__ float g_half_to_float(uint16_t h){ return __half2float(*reinterpret_cast<const __half*>(&h)); }
__constant__ signed char g_kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// iq3s codebook — verbatim copy of iq3s_grid_const (cu/mmq_iq_experts.cu, itself verbatim from
// ggml-common.h). Duplicated rather than shared via header so the default MMQ TU stays untouched.
static __device__ __constant__ unsigned int g_iq3s_grid[512] = {
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

// ---- dequant: one expert row slice -> f16. grid.x = out-row, grid.y = active-expert seg.
// 256 threads walk the row's k. Layout out: dst[seg][row][k] row-major (lda = in_f).
static __global__ void dequant_q4_0_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // q4_0: 18B block = f16 d + 16B nibbles for 32 values (lo nibbles vals 0..15, hi 16..31).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int blk=v>>5, l=v&31;
        const uint8_t* b = r + (size_t)blk*18;
        float dscale = g_half_to_float(*(const uint16_t*)b);
        uint8_t q = b[2 + (l&15)];
        int nib = (l<16) ? (q & 0xF) : (q >> 4);
        d[v] = __float2half(dscale * (float)(nib - 8));
    }
}

static __global__ void dequant_iq4xs_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // iq4_xs: 136B superblock = f16 d, u16 scales_h, 4B scales_l, 128B nibbles (256 vals).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*136;
        float d_sb = g_half_to_float(*(const uint16_t*)b);
        uint16_t sh = *(const uint16_t*)(b+2);
        const uint8_t* sl = b+4; const uint8_t* qs = b+8;
        int g = l>>5, lg = l&31;
        int ls = ((sl[g>>1]>>(4*(g&1)))&0xf) | (((sh>>(2*g))&3)<<4);
        const uint8_t* gqs = qs + g*16;
        int code = (lg<16) ? (gqs[lg]&0xf) : (gqs[lg-16]>>4);
        d[v] = __float2half(d_sb * (float)(ls-32) * (float)g_kvalues_iq4nl[code]);
    }
}

// ---- round-49 dequant coverage: q35's UD-IQ4_XS bank is a MIX (gate/up IQ3_S x39 + Q3_K x1 +
// IQ4_XS x1; down IQ4_XS x37 + Q6_K x3 + Q4_K x1). The round-47 IQ4_XS/Q4_0-only table admitted
// ~1 of 41 q35 layers — the actual reason the q35 f16g cell measured FLAT. Each kernel is a
// per-value port of the memra-gguf CPU dequant oracle (crates/memra-gguf/src/dequant.rs, itself
// diffed against ggml row dequant on real tensors).

// iq3_s: 110B superblock = f16 d, qs[64], qh[8], signs[32], scales[4]. Grid-codebook + per-byte
// signs; scale = d*(1+2*nibble) per 32-group.
static __global__ void dequant_iq3s_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*110;
        float dd = g_half_to_float(*(const uint16_t*)b);
        const uint8_t* qs = b+2; const uint8_t* qh = b+66;
        const uint8_t* sg = b+74; const uint8_t* sc = b+106;
        int ib32 = l>>5, lg = l&31, l4 = lg>>3, j = lg&7;
        float db = dd * (1.0f + 2.0f*(float)((sc[ib32>>1] >> (4*(ib32&1))) & 0xf));
        int qsb = qs[ib32*8 + 2*l4 + (j>>2)];
        int hshift = (j<4) ? (8-2*l4) : (7-2*l4);
        int idx = qsb | (((int)qh[ib32] << hshift) & 256);
        int gb = (g_iq3s_grid[idx] >> (8*(j&3))) & 0xff;
        float sgn = (sg[ib32*4 + l4] & (1<<j)) ? -1.0f : 1.0f;
        d[v] = __float2half(db * (float)gb * sgn);
    }
}

// q6_K: 210B superblock = ql[128], qh[64], i8 scales[16], f16 d (d at the END).
static __global__ void dequant_q6k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l255=v&255;
        const uint8_t* b = r + (size_t)sb*210;
        const uint8_t* ql = b; const uint8_t* qh = b+128;
        const int8_t* sc = (const int8_t*)(b+192);
        float dd = g_half_to_float(*(const uint16_t*)(b+208));
        int n2 = l255>>7, rr = l255&127, q4 = rr>>5, l = rr&31, is = l>>4;
        int qlb = ql[n2*64 + l + (q4&1)*32];
        int nib = (q4<2) ? (qlb & 0xF) : (qlb >> 4);
        int qv = (nib | (((qh[n2*32 + l] >> (2*q4)) & 3) << 4)) - 32;
        d[v] = __float2half(dd * (float)sc[n2*8 + is + 2*q4] * (float)qv);
    }
}

// q4_K: 144B superblock = f16 d, f16 dmin, scales[12] (6-bit packed), qs[128].
static __global__ void dequant_q4k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*144;
        float dd = g_half_to_float(*(const uint16_t*)b);
        float dmin = g_half_to_float(*(const uint16_t*)(b+2));
        const uint8_t* scs = b+4; const uint8_t* q = b+16;
        int g64 = l>>6, w64 = l&63, is = g64*2 + (w64>>5);
        int sc8, m8;
        if(is < 4){ sc8 = scs[is] & 63;                       m8 = scs[is+4] & 63; }
        else      { sc8 = (scs[is+4] & 0xF) | ((scs[is-4] >> 6) << 4);
                    m8  = (scs[is+4] >> 4)  | ((scs[is]   >> 6) << 4); }
        int qb = q[g64*32 + (w64&31)];
        int nib = (w64<32) ? (qb & 0xF) : (qb >> 4);
        d[v] = __float2half(dd * (float)sc8 * (float)nib - dmin * (float)m8);
    }
}

// q3_K: 110B superblock = hmask[32], qs[64], scales[12] (6-bit aux dance), f16 d.
static __global__ void dequant_q3k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*110;
        const uint8_t* hm = b; const uint8_t* qs = b+32; const uint8_t* scb = b+96;
        float dd = g_half_to_float(*(const uint16_t*)(b+108));
        int nn = l>>7, rr = l&127, jj = rr>>5, l16 = rr&31;
        int is = nn*8 + jj*2 + (l16>>4);
        // 6-bit scale unpack (ggml aux-word dance; scb byte-loaded — 110B stride is unaligned)
        uint32_t aux0 = scb[0] | (scb[1]<<8) | (scb[2]<<16) | ((uint32_t)scb[3]<<24);
        uint32_t aux1 = scb[4] | (scb[5]<<8) | (scb[6]<<16) | ((uint32_t)scb[7]<<24);
        uint32_t aux2 = scb[8] | (scb[9]<<8) | (scb[10]<<16) | ((uint32_t)scb[11]<<24);
        const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        uint32_t w;
        switch(is>>2){
            case 0:  w = (aux0 & km2)        | (((aux2 >> 0) & km1) << 4); break;
            case 1:  w = (aux1 & km2)        | (((aux2 >> 2) & km1) << 4); break;
            case 2:  w = ((aux0 >> 4) & km2) | (((aux2 >> 4) & km1) << 4); break;
            default: w = ((aux1 >> 4) & km2) | (((aux2 >> 6) & km1) << 4); break;
        }
        int s6 = (int)((w >> (8*(is&3))) & 0xff) - 32;
        int q2 = (qs[nn*32 + l16] >> (2*jj)) & 3;
        int hb = (hm[l16] & (1 << (nn*4 + jj))) ? 0 : 4;
        d[v] = __float2half(dd * (float)s6 * (float)(q2 - hb));
    }
}

// ---- f16 -> f32 elementwise (the grouped GEMM emits f16 C: the grouped API's type
// matrix has no 16F-in/32F-out combo — rc 20015 on H100 cublas 13).
static __global__ void h2f_kernel(const __half* __restrict__ s, float* __restrict__ d, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) d[i] = __half2float(s[i]);
}
// f16 -> f32 with the per-row activation scale folded back (row = pair, C row-major).
static __global__ void h2f_rows_scale_kernel(const __half* __restrict__ src, float* __restrict__ d,
                                             const float* __restrict__ s, int ncols, int nrows){
    int r = blockIdx.x; if(r>=nrows) return;
    float sc = s[r];
    const __half* sr = src + (size_t)r*ncols;
    float* dr = d + (size_t)r*ncols;
    for(int c=threadIdx.x;c<ncols;c+=blockDim.x) dr[c] = __half2float(sr[c]) * sc;
}

// ---- activation gather+convert: f32 x[token][in_f] -> f16 B[pair][in_f] via pair_tok,
// with PER-ROW amax normalization (gemma's late-layer activation spikes overflow raw f16
// — the round-46 NaN find; the MMQ path survives on per-32 q8 scales). The row scale
// folds back into the GEMM output (y row *= s[pair]). pair_tok == nullptr => identity.
static __global__ void gather_act_f16_kernel(
        const float* __restrict__ x, const int* __restrict__ pair_tok,
        __half* __restrict__ dst, float* __restrict__ s, int in_f, int n_pairs){
    int p = blockIdx.x; if(p>=n_pairs) return;
    int src = pair_tok ? pair_tok[p] : p;
    const float* xs = x + (size_t)src*in_f;
    __half* d = dst + (size_t)p*in_f;
    __shared__ float red[256];
    float amax = 0.0f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) amax = fmaxf(amax, fabsf(xs[v]));
    red[threadIdx.x] = amax; __syncthreads();
    for(int off=128; off>0; off>>=1){
        if(threadIdx.x<off) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+off]);
        __syncthreads();
    }
    amax = red[0];
    float inv = (amax>0.0f) ? 1.0f/amax : 0.0f;
    if(threadIdx.x==0) s[p] = amax;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) d[v] = __float2half(xs[v]*inv);
}

// ================= single-kernel grouped GEMM (MEMRA_MOE_F16G=2, round 49) =================
//
// cublasGemmGroupedBatchedEx issues through cublas-INTERNAL streams that are not ordered with
// the engine stream (round 47: deterministic NaN race; v1 pays a full stream sync per
// projection — the tax that capped the door at +4-11% g26 / flat q35). This kernel is the
// structural fix: ONE launch on OUR stream runs every CSR group's [m_e x out_f x in_f] GEMM —
// ordered by construction, zero syncs — and C emits f32 DIRECTLY with the per-row act amax
// scale folded in (kills the f16-C + h2f pass the grouped API's type matrix forced).
//
// Form: plain tiled m16n8k16 f16 mma, f32 accumulate (same numeric class as the cublas
// COMPUTE_32F arm). grid.z = CSR group, grid.y = m-tile inside the group (grid is sized for
// the LARGEST group; CTAs past a group's pair count exit in ~2 loads), grid.x = out-tile.
// Dispatch order (x fastest, z slowest) keeps one group's tiles adjacent so the W-tile
// re-reads across its m-tiles hit L2. BM=32 BN=64 BK=32, 4 warps (2x2), cp.async
// double-buffered smem, ldmatrix operand loads. All sm_80-class — portable to every memra
// arch (89/90a/100a/120a), unlike the sm_120a-only CUTLASS static lib.

#define SK_BM 32
#define SK_BN 64
#define SK_BK 32
#define SK_STRIDE (SK_BK + 8)   // +8 halves de-banks ldmatrix rows; keeps 16B alignment

__device__ __forceinline__ void sk_cp16(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
// ldmatrix x4: a 16x16 half tile from k-contiguous smem rows. Register i = the 8x8 submatrix
// (rows i&1 ? 8-15 : 0-7, k i&2 ? 8-15 : 0-7) — exactly the m16n8k16 A-operand register order;
// the same load on the [n][k] W tile yields the B operand as n-blocks {r0,r2} (n 0-7) and
// {r1,r3} (n 8-15) (both operands are k-contiguous, so no .trans needed).
__device__ __forceinline__ void sk_ldm16x16(unsigned (&r)[4], const __half* base, int stride){
    const __half* p = base + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}
__device__ __forceinline__ void sk_mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __global__ void moe_f16g_sk_kernel(
        const __half* __restrict__ W,        // [n_active][out_f][in_f] dequant workspace
        const __half* __restrict__ A,        // [n_pairs][in_f] act, CSR pair-major
        float* __restrict__ Y,               // [n_pairs][out_f] f32 out, CSR pair-major
        const float* __restrict__ row_scale, // [n_pairs] act amax (folds back here)
        const int* __restrict__ ex_off,      // [n_active+1] CSR group offsets (device)
        int in_f, int out_f){
    const int g  = blockIdx.z;
    const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
    const int m0 = blockIdx.y * SK_BM;
    if(m0 >= m_e) return;
    const int n0 = blockIdx.x * SK_BN;

    __shared__ __half As[2][SK_BM][SK_STRIDE];
    __shared__ __half Bs[2][SK_BN][SK_STRIDE];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const __half* Ag = A + (size_t)lo * in_f;
    const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;

    // Stage loads: A tile 32 rows x 64B = 1 cp.async16/thread; B tile 64 rows x 64B = 2/thread.
    // Rows past the group's pairs / past out_f clamp to the last valid row — real bytes load,
    // the store guards discard the results.
    const int ar = tid >> 2,  ac  = (tid & 3) * 8;
    const int am = min(m0 + ar, m_e - 1);
    const int br = tid >> 1,  bc  = (tid & 1) * 16;
    const int bn = min(n0 + br, out_f - 1);
    const __half* aga = Ag + (size_t)am * in_f + ac;
    const __half* bga = Wg + (size_t)bn * in_f + bc;

    const int nkb = in_f / SK_BK;
    // prologue: stage 0
    sk_cp16(&As[0][ar][ac],    aga);
    sk_cp16(&Bs[0][br][bc],     bga);
    sk_cp16(&Bs[0][br][bc + 8], bga + 8);
    asm volatile("cp.async.commit_group;");

    float acc[4][4] = {};
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;

    for(int kb = 0; kb < nkb; kb++){
        const int cur = kb & 1;
        if(kb + 1 < nkb){
            const int nxt = cur ^ 1, k0 = (kb + 1) * SK_BK;
            sk_cp16(&As[nxt][ar][ac],    aga + k0);
            sk_cp16(&Bs[nxt][br][bc],     bga + k0);
            sk_cp16(&Bs[nxt][br][bc + 8], bga + k0 + 8);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        #pragma unroll
        for(int kk = 0; kk < 2; kk++){
            unsigned a[4], b0[4], b1[4];
            sk_ldm16x16(a,  &As[cur][wm][kk*16],      SK_STRIDE);
            sk_ldm16x16(b0, &Bs[cur][wn][kk*16],      SK_STRIDE);
            sk_ldm16x16(b1, &Bs[cur][wn + 16][kk*16], SK_STRIDE);
            sk_mma(acc[0], a, b0[0], b0[2]);
            sk_mma(acc[1], a, b0[1], b0[3]);
            sk_mma(acc[2], a, b1[0], b1[2]);
            sk_mma(acc[3], a, b1[1], b1[3]);
        }
        __syncthreads();
    }

    // epilogue: f32 store with the per-pair act scale folded back (mode 1 pays a separate
    // h2f_rows_scale pass for this). C frag: c0/c1 = (row lane/4, col 2*(lane%4)+{0,1});
    // c2/c3 = row+8. acc[nb] covers warp cols wn + nb*8.
    const int r0 = m0 + wm + lane / 4;
    const int cb = n0 + wn + (lane % 4) * 2;
    const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
    const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
    float* y0 = Y + (size_t)(lo + r0) * out_f;
    #pragma unroll
    for(int nb = 0; nb < 4; nb++){
        const int c = cb + nb * 8;
        if(r0 < m_e){
            if(c     < out_f) y0[c]     = acc[nb][0] * s0;
            if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
        }
        if(r0 + 8 < m_e){
            if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
            if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
        }
    }
}

// ============== round-51 (lane/sk-bm128): persistent problem-visitor + BM128 form ==============
//
// The round-49 sk kernel launches grid (ntx, ceil(max_m/BM), n_active) — under q35's ~17x routing
// skew that is ~92% early-exit churn (H100 ncu, research/sk-vs-cublas-20260801: grid (32,36,252)
// = 290k blocks, most exit at `m0 >= m_e` after ~2 loads), and BM=32 re-fetches each expert's W
// tile 4x more through L2 than cutlass's 128x64 (L2 SOL 75-88% — the wall that let cublas win
// 1.32x on H100 while HBM sat at 30%). Two fixes:
//   1. PROBLEM VISITOR (both forms): CTAs grid-stride a flat list of REAL tiles. Per-group tile
//      counts are prefix-summed into smem from the device CSR offsets (thread 0, n_active <=
//      SK_MAX_G); each flat id binary-searches its group. Zero early-exit blocks; the grid is
//      min(total_tiles, SMs x occupancy-API blocks) persistent CTAs — the cutlass waves=1 shape.
//   2. BM=128/BN=64/BK=64 3-STAGE cp.async form (sk128) for large-m groups: 4x fewer W-tile
//      re-fetches through L2 with the SAME sm_80-portable ldmatrix + mma.sync m16n8k16 pipeline
//      (no wgmma/TMA — runs on every memra arch). Small-m groups keep the 32x64 form (a 128-row
//      tile on a ~17-pair group is ~87% padding); the crossover is swept, not guessed
//      (MEMRA_F16G_SK_CROSS, read on the Rust side).
//
// NUMERICS: bit-identical to the round-49 kernel by construction — each output element's k-chain
// is the same ascending sequence of m16n8k16 f32-accumulate steps on the same f16 operands; only
// WHICH CTA computes a tile changes. Gated byte-identical in kernel-check ("f16g-sk" section).
//
// SMEM (sk128): A 128x(64+8) + B 64x(64+8) halves = 18432 + 9216 = 27648 B/stage x3 stages =
// 82944 B dynamic (+ 2052 B static tile-prefix) — needs the >48KB opt-in and fits sm_120a's
// ~99KB. cudaFuncSetAttribute is CHECKED with a device-fit fallback to the 32x64 visitor form
// (the round-49 mmq_iq rc=1001 lesson: never unchecked smem growth). 2 CTA/SM is smem-impossible
// for sk128 on sm_120a (2 x 82944 > ~100KB/SM): it runs 1 CTA/SM x 8 warps — occupancy is not
// the target, pipeline depth is (cutlass wins at 12.5% occupancy).

#define SK_MAX_G 512

// Per-group tile-count prefix into s_pre[0..n_active] (groups whose pair count lies outside
// [mlo, mhi) contribute 0 tiles). Thread 0 serial: n_active <= SK_MAX_G, one cheap pass per CTA.
__device__ __forceinline__ void sk_tile_prefix(int* s_pre, const int* ex_off, int n_active,
                                               int ntx, int bm, int mlo, int mhi){
    if(threadIdx.x == 0 && threadIdx.y == 0){
        int acc = 0; s_pre[0] = 0;
        for(int g = 0; g < n_active; g++){
            int m_e = ex_off[g+1] - ex_off[g];
            if(m_e >= mlo && m_e < mhi) acc += ((m_e + bm - 1)/bm)*ntx;
            s_pre[g+1] = acc;
        }
    }
    __syncthreads();
}
// Largest g with s_pre[g] <= t (empty groups have s_pre[g] == s_pre[g+1] and are skipped over).
__device__ __forceinline__ int sk_tile_group(const int* s_pre, int n_active, int t){
    int lo = 0, hi = n_active - 1;
    while(lo < hi){ int mid = (lo + hi + 1) >> 1; if(s_pre[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
}

// 32x64 visitor form: the round-49 tile body verbatim, driven by the flat tile list.
static __global__ void __launch_bounds__(128)
moe_f16g_sk32v_kernel(
        const __half* __restrict__ W, const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    __shared__ int s_pre[SK_MAX_G + 1];
    // explicit 16B alignment: cp.async.cg 16 + ldmatrix require it, and __half arrays placed
    // after the 2052-B s_pre otherwise land 4-aligned (the exact miss found on first run).
    __shared__ __align__(16) __half As[2][SK_BM][SK_STRIDE];
    __shared__ __align__(16) __half Bs[2][SK_BN][SK_STRIDE];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, mlo, mhi);

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nkb  = in_f / SK_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int ar = tid >> 2,  ac  = (tid & 3) * 8;
    const int br = tid >> 1,  bc  = (tid & 1) * 16;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;
        const int am = min(m0 + ar, m_e - 1);
        const int bn = min(n0 + br, out_f - 1);
        const __half* aga = Ag + (size_t)am * in_f + ac;
        const __half* bga = Wg + (size_t)bn * in_f + bc;

        sk_cp16(&As[0][ar][ac],    aga);
        sk_cp16(&Bs[0][br][bc],     bga);
        sk_cp16(&Bs[0][br][bc + 8], bga + 8);
        asm volatile("cp.async.commit_group;");

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb & 1;
            if(kb + 1 < nkb){
                const int nxt = cur ^ 1, k0 = (kb + 1) * SK_BK;
                sk_cp16(&As[nxt][ar][ac],    aga + k0);
                sk_cp16(&Bs[nxt][br][bc],     bga + k0);
                sk_cp16(&Bs[nxt][br][bc + 8], bga + k0 + 8);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();
            #pragma unroll
            for(int kk = 0; kk < 2; kk++){
                unsigned a[4], b0[4], b1[4];
                sk_ldm16x16(a,  &As[cur][wm][kk*16],      SK_STRIDE);
                sk_ldm16x16(b0, &Bs[cur][wn][kk*16],      SK_STRIDE);
                sk_ldm16x16(b1, &Bs[cur][wn + 16][kk*16], SK_STRIDE);
                sk_mma(acc[0], a, b0[0], b0[2]);
                sk_mma(acc[1], a, b0[1], b0[3]);
                sk_mma(acc[2], a, b1[0], b1[2]);
                sk_mma(acc[3], a, b1[1], b1[3]);
            }
            __syncthreads();
        }

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
        const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
        float* y0 = Y + (size_t)(lo + r0) * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                if(c     < out_f) y0[c]     = acc[nb][0] * s0;
                if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
            }
            if(r0 + 8 < m_e){
                if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
                if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
            }
        }
    }
}

// 128x64x64 3-stage visitor form. 8 warps in a 4(m) x 2(n) grid — each warp owns a 32x32
// sub-tile (2 m-frags x 2 b-pairs of the same m16n8k16 pipeline). Dynamic smem (>48KB static
// limit): stages-major A then B.
#define SK128_BM 128
#define SK128_BN 64
#define SK128_BK 64
#define SK128_STRIDE (SK128_BK + 8)   // 72 halves = 144 B rows: 16B-aligned, de-banked
#define SK128_STAGES 3
#define SK128_A_ELEMS (SK128_BM * SK128_STRIDE)
#define SK128_B_ELEMS (SK128_BN * SK128_STRIDE)
#define SK128_SMEM_BYTES ((SK128_STAGES * (SK128_A_ELEMS + SK128_B_ELEMS)) * (int)sizeof(__half))

static __global__ void __launch_bounds__(256)
moe_f16g_sk128v_kernel(
        const __half* __restrict__ W, const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    extern __shared__ __align__(16) __half sk128_sm[];
    __half* Asm = sk128_sm;                                   // [stage][128][72]
    __half* Bsm = sk128_sm + SK128_STAGES * SK128_A_ELEMS;    // [stage][64][72]
    __shared__ int s_pre[SK_MAX_G + 1];
    const int ntx = (out_f + SK128_BN - 1) / SK128_BN;
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK128_BM, mlo, mhi);

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..255
    const int nkb  = in_f / SK128_BK;
    const int wm   = (warp & 3) * 32, wn = (warp >> 2) * 32;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK128_BM;
        const int n0 = (local % ntx) * SK128_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;

        // Stage-load geometry: A = 128 rows x 128B = 1024 cp.async16 (4/thread); B = 64 rows x
        // 128B = 512 (2/thread). Chunk c -> row c>>3, col (c&7)*8 halves. Rows past the group's
        // pairs / past out_f clamp to the last valid row — real bytes load, store guards discard.
        const __half* agp[4]; int asr[4], asc[4];
        #pragma unroll
        for(int i = 0; i < 4; i++){
            const int c = tid + i * 256;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        const __half* bgp[2]; int bsr[2], bsc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 256;
            bsr[i] = c >> 3; bsc[i] = (c & 7) * 8;
            const int bn = min(n0 + bsr[i], out_f - 1);
            bgp[i] = Wg + (size_t)bn * in_f + bsc[i];
        }
        #define SK128_LOAD(st, k0) do { \
            __half* a_s = Asm + (st) * SK128_A_ELEMS; \
            __half* b_s = Bsm + (st) * SK128_B_ELEMS; \
            _Pragma("unroll") \
            for(int i = 0; i < 4; i++) sk_cp16(a_s + asr[i]*SK128_STRIDE + asc[i], agp[i] + (k0)); \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) sk_cp16(b_s + bsr[i]*SK128_STRIDE + bsc[i], bgp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        // prologue: stages 0 (+1 when the k extent has a second block)
        SK128_LOAD(0, 0);
        if(nkb > 1) SK128_LOAD(1, SK128_BK);

        float acc[2][4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SK128_STAGES;
            if(kb + 2 < nkb){
                SK128_LOAD((kb + 2) % SK128_STAGES, (kb + 2) * SK128_BK);
                asm volatile("cp.async.wait_group 2;");
            } else if(kb + 1 < nkb){
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();
            const __half* Ab = Asm + cur * SK128_A_ELEMS;
            const __half* Bb = Bsm + cur * SK128_B_ELEMS;
            #pragma unroll
            for(int kk = 0; kk < 4; kk++){
                unsigned a0[4], a1[4], b0[4], b1[4];
                sk_ldm16x16(a0, Ab + (wm     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(a1, Ab + (wm + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b0, Bb + (wn     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b1, Bb + (wn + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_mma(acc[0][0], a0, b0[0], b0[2]);
                sk_mma(acc[0][1], a0, b0[1], b0[3]);
                sk_mma(acc[0][2], a0, b1[0], b1[2]);
                sk_mma(acc[0][3], a0, b1[1], b1[3]);
                sk_mma(acc[1][0], a1, b0[0], b0[2]);
                sk_mma(acc[1][1], a1, b0[1], b0[3]);
                sk_mma(acc[1][2], a1, b1[0], b1[2]);
                sk_mma(acc[1][3], a1, b1[1], b1[3]);
            }
            __syncthreads();
        }
        #undef SK128_LOAD

        #pragma unroll
        for(int mi = 0; mi < 2; mi++){
            const int r0 = m0 + wm + mi * 16 + lane / 4;
            const int cb = n0 + wn + (lane % 4) * 2;
            const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
            const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
            float* y0 = Y + (size_t)(lo + r0) * out_f;
            #pragma unroll
            for(int nb = 0; nb < 4; nb++){
                const int c = cb + nb * 8;
                if(r0 < m_e){
                    if(c     < out_f) y0[c]     = acc[mi][nb][0] * s0;
                    if(c + 1 < out_f) y0[c + 1] = acc[mi][nb][1] * s0;
                }
                if(r0 + 8 < m_e){
                    if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[mi][nb][2] * s1;
                    if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[mi][nb][3] * s1;
                }
            }
        }
    }
}

extern "C" {

size_t memra_moe_f16g_w_bytes(int n_active, int out_f, int in_f){
    return (size_t)n_active*out_f*in_f*sizeof(__half);
}
size_t memra_moe_f16g_act_bytes(int n_pairs, int in_f){
    return (size_t)n_pairs*in_f*sizeof(__half);
}

int memra_moe_f16g_dequant(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, void* w_f16, int in_f, int out_f, int n_active,
        int qtype, long row_bytes, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)out_f, (unsigned)n_active, 1), blk(256,1,1);
    if(qtype==QT_Q4_0)
        dequant_q4_0_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_IQ4_XS)
        dequant_iq4xs_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_IQ3_S)
        dequant_iq3s_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q6_K)
        dequant_q6k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q4_K)
        dequant_q4k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q3_K)
        dequant_q3k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else return 2;   // unsupported qtype => caller falls back to the MMQ arm
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// Single-kernel grouped GEMM (MEMRA_MOE_F16G=2): every CSR group's GEMM on the caller's stream,
// f32 C with the per-pair act scale folded in. ex_off_dev = DEVICE CSR offsets (n_active+1);
// ex_off_host = the same offsets on the HOST (already there at the call site — sizes the visitor
// grids and skips empty arms with no extra transfer); max_m = host-side max group size.
// in_f must be a multiple of 32.
// shape_sel < 0  -> the round-49 grid-scan kernel (rollback arm, MEMRA_F16G_SK=0).
// shape_sel >= 0 -> problem-visitor: groups with m_e >= cross ride the 128x64x64 3-stage form,
//                   smaller groups the 32x64 form (cross=1 forces all-128, INT_MAX all-32).
//                   Policy/env parsing lives on the Rust side (moe_f16g_sk_params).
int memra_moe_f16g_gemm_sk(const void* w_f16, const void* act_f16, float* y,
        const float* row_scale, const int* ex_off_dev, const int* ex_off_host,
        int n_active, int max_m, int in_f, int out_f, int shape_sel, int cross,
        void* stream){
    if(in_f % SK_BK) return 2;
    if(n_active <= 0 || max_m <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    if(shape_sel < 0 || n_active > SK_MAX_G || ex_off_host == nullptr){
        // round-49 form: grid (ntx, ceil(max_m/BM), n_active), early-exit past a group's pairs.
        dim3 grid((unsigned)((out_f + SK_BN - 1) / SK_BN),
                  (unsigned)((max_m + SK_BM - 1) / SK_BM),
                  (unsigned)n_active);
        dim3 blk(32, 4, 1);
        moe_f16g_sk_kernel<<<grid, blk, 0, st>>>(
            (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev, in_f, out_f);
        cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
    }
    // Device caps + per-form occupancy, once. occ128: -2 unprobed, -1 device-unfit (fallback to
    // the 32x64 form — the round-49 rc=1001 lesson: SetAttribute CHECKED, never assumed).
    static int sms = 0, occ32 = 0, occ128 = -2;
    if(sms == 0){
        int dev = 0; cudaGetDevice(&dev);
        if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
            sms = 1;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ32, moe_f16g_sk32v_kernel, 128, 0)
           != cudaSuccess || occ32 < 1) occ32 = 1;
    }
    if(occ128 == -2){
        int optin = 0, dev = 0; cudaGetDevice(&dev);
        if(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
            optin = 48*1024;
        if((size_t)SK128_SMEM_BYTES > (size_t)optin
           || cudaFuncSetAttribute(moe_f16g_sk128v_kernel,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, SK128_SMEM_BYTES) != cudaSuccess
           || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ128, moe_f16g_sk128v_kernel, 256,
                  SK128_SMEM_BYTES) != cudaSuccess
           || occ128 < 1)
            occ128 = -1;
    }
    int xcross = cross;
    if(occ128 < 1 || (in_f % SK128_BK) != 0) xcross = 0x7fffffff;  // every group rides 32x64
    if(xcross < 1) xcross = 1;
    const int ntx = (out_f + SK_BN - 1) / SK_BN;                   // BN identical in both forms
    long t32 = 0, t128 = 0;
    for(int g = 0; g < n_active; g++){
        const int m_e = ex_off_host[g+1] - ex_off_host[g];
        if(m_e <= 0) continue;
        if(m_e >= xcross) t128 += (long)((m_e + SK128_BM - 1)/SK128_BM) * ntx;
        else              t32  += (long)((m_e + SK_BM - 1)/SK_BM) * ntx;
    }
    if(t128 > 0){
        const long cap = (long)sms * occ128;
        const int grid = (int)(t128 < cap ? t128 : cap);
        moe_f16g_sk128v_kernel<<<grid, dim3(32,8,1), SK128_SMEM_BYTES, st>>>(
            (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev,
            n_active, in_f, out_f, xcross, 0x7fffffff, (int)t128);
    }
    if(t32 > 0){
        const long cap = (long)sms * occ32;
        const int grid = (int)(t32 < cap ? t32 : cap);
        moe_f16g_sk32v_kernel<<<grid, dim3(32,4,1), 0, st>>>(
            (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev,
            n_active, in_f, out_f, 1, xcross, (int)t32);
    }
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_gather_act(const float* x, const int* pair_tok_or_null, void* act_f16,
        float* row_scale, int in_f, int n_pairs, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_pairs,1,1), blk(256,1,1);
    gather_act_f16_kernel<<<grid,blk,0,st>>>(x, pair_tok_or_null, (__half*)act_f16, row_scale,
                                             in_f, n_pairs);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f_scaled(const void* src_f16, float* dst, const float* row_scale,
        int ncols, int nrows, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    h2f_rows_scale_kernel<<<(unsigned)nrows,256,0,st>>>((const __half*)src_f16, dst, row_scale,
                                                        ncols, nrows);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f(const void* src_f16, float* dst, size_t n, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    size_t blocks = (n + 255)/256;
    h2f_kernel<<<(unsigned)blocks,256,0,st>>>((const __half*)src_f16, dst, n);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// One grouped GEMM over the CSR groups. ex_off_host = HOST copy of the CSR offsets
// (n_active+1 ints). y = f16 [n_pairs, out_f] (pair-major; caller converts via h2f —
// the grouped API's type matrix has no 16F-in/32F-out combo).
int memra_moe_f16g_gemm(const void* w_f16, const void* act_f16, void* y,
        const int* ex_off_host, int n_active, int in_f, int out_f, void* stream){
    static cublasHandle_t handle = nullptr;
    if(!handle){
        if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 3;
    }
    cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream));

    // per-group host arrays (n_active <= a few hundred; stack-scale, heap for safety)
    const int G = n_active;
    static thread_local int cap = 0;
    static thread_local cublasOperation_t *ta=nullptr,*tb=nullptr;
    static thread_local int *ma=nullptr,*na=nullptr,*ka=nullptr,*lda=nullptr,*ldb=nullptr,*ldc=nullptr,*gsz=nullptr;
    static thread_local float *al=nullptr,*be=nullptr;
    static thread_local const void **Aa=nullptr,**Ba=nullptr; static thread_local void **Ca=nullptr;
    if(G>cap){
        delete[] ta; delete[] tb; delete[] ma; delete[] na; delete[] ka; delete[] lda; delete[] ldb;
        delete[] ldc; delete[] gsz; delete[] al; delete[] be; delete[] Aa; delete[] Ba; delete[] Ca;
        cap=G;
        ta=new cublasOperation_t[G]; tb=new cublasOperation_t[G];
        ma=new int[G]; na=new int[G]; ka=new int[G]; lda=new int[G]; ldb=new int[G]; ldc=new int[G];
        gsz=new int[G]; al=new float[G]; be=new float[G];
        Aa=new const void*[G]; Ba=new const void*[G]; Ca=new void*[G];
    }
    int g_used = 0;
    for(int g=0; g<G; g++){
        int lo=ex_off_host[g], hi=ex_off_host[g+1], m_e=hi-lo;
        if(m_e<=0) continue;
        ta[g_used]=CUBLAS_OP_T; tb[g_used]=CUBLAS_OP_N;
        ma[g_used]=out_f; na[g_used]=m_e; ka[g_used]=in_f;
        lda[g_used]=in_f; ldb[g_used]=in_f; ldc[g_used]=out_f;
        al[g_used]=1.0f; be[g_used]=0.0f; gsz[g_used]=1;
        Aa[g_used]=(const uint8_t*)w_f16 + (size_t)g*out_f*in_f*sizeof(__half);
        Ba[g_used]=(const uint8_t*)act_f16 + (size_t)lo*in_f*sizeof(__half);
        Ca[g_used]=(uint8_t*)y + (size_t)lo*out_f*sizeof(__half);
        g_used++;
    }
    if(g_used==0) return 0;
    cublasStatus_t s = cublasGemmGroupedBatchedEx(handle, ta, tb, ma, na, ka,
        al, Aa, CUDA_R_16F, lda, Ba, CUDA_R_16F, ldb,
        be, Ca, CUDA_R_16F, ldc, g_used, gsz, CUBLAS_COMPUTE_32F);
    if(s != CUBLAS_STATUS_SUCCESS) return 20000 + (int)s;
    return 0;
}

} // extern "C"
