// Resident-quantized matmul: weights stay in GGUF block format in VRAM, dequantized in-register
// inside the kernel (never materialized as f32/f16). Fixes the OOM. Activations are f32 (Stage A:
// correctness-first; Stage B will quantize activations to q8_1 + int8 dp4a like llama.cpp MMVQ/MMQ).
//
// y[m, out] = x[m, in] @ W[out, in]^T,  W is quantized (ggml block layout), x/y are f32.
// Layout: x token-major [m, in] (x[t*in + i]); W row o = out-feature o, `in` elements quantized;
//         y token-major [m, out] (y[t*out + o]). One block per (token, out-row); threads reduce over `in`.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// IQ3_S grid: 512 u32 entries, each packs 4 unsigned bytes. Verbatim from ggml-common.h:1042.
// STORAGE CLASS (2026-07-06): plain __device__ (global mem, L1-cached), NOT __constant__ —
// the constant cache broadcasts only on uniform addresses and SERIALIZES divergent reads, and
// these grid lookups are divergent by construction (every lane decodes different codes).
// llama's GGML_TABLE_BEGIN is `static const __device__` for exactly this reason.
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
__device__ __forceinline__ unsigned int iq3s_grid_d(int idx) { return iq3s_grid_const[idx]; }

// ---- per-dtype: dequantize element j of weight-row `wrow` (raw bytes) and return its f32 value ----
// Q8_0: block=32, bytes=34 (fp16 d + int8[32]).
__device__ __forceinline__ float deq_q8_0(const uint8_t* row, int j) {
    int blk = j >> 5, off = j & 31;
    const uint8_t* b = row + blk * 34;
    float d = half_to_float(*(const uint16_t*)b);
    int8_t q = (int8_t)b[2 + off];
    return d * (float)q;
}
// Q4_K: superblock=256, bytes=144. {fp16 d, fp16 dmin, u8 scales[12], u8 qs[128]}.
// 8 sub-blocks of 32; 6-bit scale/min via get_scale_min_k4.
__device__ __forceinline__ void q4k_scale_min(const uint8_t* sc, int j, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); *m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}
__device__ __forceinline__ float deq_q4_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;          // which superblock, idx within
    const uint8_t* b = row + blk * 144;
    float d = half_to_float(*(const uint16_t*)b);
    float dmin = half_to_float(*(const uint16_t*)(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qs = b + 16;
    // ggml q4_K layout: for is in 0..7, group of 32. j = group*32 + l (l 0..31).
    // qs are nibble-packed: 64-elem chunk uses 32 bytes; low nibble first 32, high nibble next 32.
    int group = jj >> 5;       // 0..7
    int l = jj & 31;
    // each 64-block (2 groups) shares 32 qs bytes: group even -> low nibble, odd -> high nibble
    int chunk = group >> 1;    // 0..3  (which 32-byte qs run)
    const uint8_t* q = qs + chunk * 32;
    uint8_t sc, mn;
    q4k_scale_min(scales, group, &sc, &mn);
    float val;
    if ((group & 1) == 0) val = d * (float)sc * (float)(q[l] & 0xF) - dmin * (float)mn;
    else                  val = d * (float)sc * (float)(q[l] >> 4)  - dmin * (float)mn;
    return val;
}
// Q6_K: superblock=256, bytes=210. {u8 ql[128], u8 qh[64], i8 scales[16], fp16 d}.
__device__ __forceinline__ float deq_q6_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 210;
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales = (const int8_t*)(b + 192);
    float d = half_to_float(*(const uint16_t*)(b + 208));
    // ggml q6_K: two halves of 128. n = jj/128 (0/1); within half l=jj%128 ; sub group of 16 -> scale.
    int n = jj >> 7;           // 0 or 1
    int l = jj & 127;          // 0..127
    int il = l & 31;           // position within 32-run
    int run = l >> 5;          // 0..3 within half
    const uint8_t* qlh = ql + n * 64;
    const uint8_t* qhh = qh + n * 32;
    // reconstruct q like ggml dequantize_row_q6_K
    int ql_bits, qh_bits;
    if (run == 0)      { ql_bits = qlh[il] & 0xF;        qh_bits = (qhh[il] >> 0) & 3; }
    else if (run == 1) { ql_bits = qlh[il + 32] & 0xF;   qh_bits = (qhh[il] >> 2) & 3; }
    else if (run == 2) { ql_bits = qlh[il] >> 4;         qh_bits = (qhh[il] >> 4) & 3; }
    else               { ql_bits = qlh[il + 32] >> 4;    qh_bits = (qhh[il] >> 6) & 3; }
    int q = (ql_bits | (qh_bits << 4)) - 32;
    int is = n * 8 + run * 2 + (il >> 4);   // scale index 0..15
    return d * (float)scales[is] * (float)q;
}

// device codebook tables — plain __device__ (L1), NOT __constant__: per-lane indices diverge
// (expert_dot_iq4xs_g does 8 byte-lookups per group per lane), and the constant cache serializes
// divergent reads. Same fix class as iq3s_grid_const (+11.8% 35B decode, 2026-07-06).
// mxfp4 stays __constant__: its consumers go through get_int_from_table_16_d (byte_perm on two
// uniform 8B halves — broadcast-friendly, the constant cache's good case).
// __align__(16): expert_dot_iq4xs_g_v reads this table as four u32 words for the byte_perm
// lookup (get_int_from_table_16_d) — same 16 byte VALUES, alignment attribute only.
__device__ __align__(16) signed char kvalues_iq4nl_d[16] =
    {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ __constant__ signed char kvalues_mxfp4_d[16] =
    {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

// Fast 4-bit codebook lookup (llama.cpp vecdotq.cuh get_int_from_table_16). Takes 4 packed
// bytes (8 nibbles) in q4; returns int2 where .x = the 4 codebook int8s of the LOW nibbles
// (one per byte, packed) and .y = the 4 codebook int8s of the HIGH nibbles. ~5 __byte_perm
// vs 8 scalar table[] loads — the NVFP4/MXFP4/IQ4 decode hot loop is ALU-bound otherwise.
// CUDA __byte_perm selects bytes by 3-bit indices; the 4th index bit is handled by a 2nd perm.
__device__ __forceinline__ int2 get_int_from_table_16_d(int q4, const signed char* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210u | ((q4 & 0x88888888u) >> 1));
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16u * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], (uint32_t)q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], (uint32_t)q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// UE4M3 -> f32, software fallback (ggml_cuda_ue4m3_to_fp32 common.cuh:843-854). NaN 0/0x7F -> 0.
__device__ __forceinline__ float ue4m3_to_f32_d(unsigned char x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int   exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? ldexpf(man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}
// HW UE4M3 -> f32 (OCP E4M3, bias 7, NO x0.5). This is what the mxf4nvf4 block_scale MMA decodes
// its sa/sb operand as (verified by probe/fp4_4x_final.cu, maxrel=0). The GGUF NVFP4 micro-scale
// byte fed RAW here decodes to exactly 2x the GGUF value — which is cancelled by the e2m1 nibble
// being GGUF-codebook/2 (GGUF dequant = (2*e2m1_hw)*(0.5*ue4m3_hw) = e2m1_hw*ue4m3_hw). So GGUF
// scale bytes + GGUF e2m1 nibbles fed verbatim == GGUF dequant exactly. (used by quantize_fp4_act).
__device__ __forceinline__ float ue4m3_to_f32_hw(unsigned char x) {
    int   exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    return (exp == 0) ? ldexpf(man / 8.0f, -6) : ldexpf(1.0f + man / 8.0f, exp - 7);
}

// ---- Q5_K f32 deq (oracle for the dp4a kernel) ----
__device__ __forceinline__ float deq_q5_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 176;
    float d    = half_to_float(*(const uint16_t*)b);
    float dmin = half_to_float(*(const uint16_t*)(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int group = jj >> 5;          // 0..7
    int l = jj & 31;
    int chunk = group >> 1;       // shares 32 qs bytes
    const uint8_t* q = ql + chunk * 32;
    uint8_t sc, mn;
    q4k_scale_min(scales, group, &sc, &mn);       // identical 6-bit unpack to Q4_K
    int g64 = group >> 1;
    int half = group & 1;
    int hbit = 2 * g64 + half;
    int nib = (half == 0) ? (q[l] & 0xF) : (q[l] >> 4);
    int h = (qh[l] >> hbit) & 1;
    int w = nib | (h << 4);                        // unsigned 0..31
    return d * (float)sc * (float)w - dmin * (float)mn;
}

// ---- Q3_K f32 deq ----
__device__ __forceinline__ float deq_q3_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    const uint8_t* hmask  = b;
    const uint8_t* qs     = b + 32;
    const uint8_t* scbyte = b + 96;
    float d = half_to_float(*(const uint16_t*)(b + 108));
    // unpack 16 6-bit signed scales (aux dance)
    unsigned int aux0 = (scbyte[0]) | (scbyte[1]<<8) | (scbyte[2]<<16) | (scbyte[3]<<24);
    unsigned int aux1 = (scbyte[4]) | (scbyte[5]<<8) | (scbyte[6]<<16) | (scbyte[7]<<24);
    unsigned int aux2 = (scbyte[8]) | (scbyte[9]<<8) | (scbyte[10]<<16) | (scbyte[11]<<24);
    const unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux2;
    unsigned int n0 = (aux0 & km2) | (((tmp>>0)&km1)<<4);
    unsigned int n1 = (aux1 & km2) | (((tmp>>2)&km1)<<4);
    unsigned int n2 = ((aux0>>4)&km2) | (((tmp>>4)&km1)<<4);
    unsigned int n3 = ((aux1>>4)&km2) | (((tmp>>6)&km1)<<4);
    signed char sc[16];
    { unsigned int w[4] = {n0,n1,n2,n3};
      for (int k=0;k<4;k++){ sc[k*4+0]=(signed char)(w[k]); sc[k*4+1]=(signed char)(w[k]>>8);
                             sc[k*4+2]=(signed char)(w[k]>>16); sc[k*4+3]=(signed char)(w[k]>>24);} }
    // map jj (0..255) back to (half, j-iter, l, shift, m_bit, scale index)
    int half = jj >> 7;             // 0/1 (which 128)
    int rem  = jj & 127;            // 0..127
    int jiter = rem >> 5;           // 0..3 (which of the 4 j-iterations within the half)
    int within = rem & 31;          // 0..31 within the 32-wide j-iteration
    int sublo = within >> 4;        // 0 -> low 16 (sc index is_base), 1 -> high 16 (is_base+1)
    int l = within & 15;
    int shift = 2 * jiter;
    int m_bit_idx = half * 4 + jiter;          // running bit position (0..7)
    int is = (half * 8) + jiter * 2 + sublo;   // scale index 0..15
    const uint8_t* q = qs + half * 32;
    int qidx = sublo * 16 + l;                 // q[l] or q[l+16]
    int hidx = sublo * 16 + l;                 // hmask[l] or hmask[l+16]
    int q2 = (q[qidx] >> shift) & 3;
    int hb = (hmask[hidx] & (1 << m_bit_idx)) ? 0 : 4;
    int w = q2 - hb;
    return d * (float)((int)sc[is] - 32) * (float)w;
}

// ---- IQ4_XS f32 deq ----
__device__ __forceinline__ float deq_iq4_xs(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 136;
    float d = half_to_float(*(const uint16_t*)b);
    unsigned short sh = *(const uint16_t*)(b + 2);
    const uint8_t* sl = b + 4;
    const uint8_t* qs = b + 8;
    int ib = jj >> 5;               // 0..7
    int within = jj & 31;           // 0..31
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    float dl = d * (float)(ls - 32);
    const uint8_t* q = qs + ib * 16;
    int code = (within < 16) ? (q[within] & 0xf) : (q[within - 16] >> 4);
    return dl * (float)kvalues_iq4nl_d[code];
}

// ---- IQ3_S f32 deq ----
__device__ __forceinline__ float deq_iq3_s(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    float d = half_to_float(*(const uint16_t*)b);
    const uint8_t* qs    = b + 2;     // [64]
    const uint8_t* qh    = b + 66;    // [8]
    const uint8_t* signs = b + 74;    // [32]
    const uint8_t* scales= b + 106;   // [4]
    // Each ib32 group (32 elems) = qh[ib32], 4 sign bytes, 8 qs bytes. 8 elems per l (grid1/grid2).
    int ib32   = jj >> 5;             // 0..7
    int within = jj & 31;             // 0..31
    int l      = within >> 3;         // 0..3  (which qs pair)
    int e      = within & 7;          // 0..7  (grid byte slot)
    // ggml: db for even ib32 uses &0xf, odd uses >>4 of scales[ib32/2]
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const uint8_t* qsb = qs + ib32 * 8;       // 8 qs bytes per ib32
    unsigned char qhb = qh[ib32];
    const uint8_t* sgn = signs + ib32 * 4;
    int qpair = (e < 4) ? (2 * l + 0) : (2 * l + 1);
    int shamt = (e < 4) ? (8 - 2 * l) : (7 - 2 * l);
    int gidx = qsb[qpair] | (((int)qhb << shamt) & 256);
    int jb = e & 3;                            // grid byte 0..3
    unsigned int gw = iq3s_grid_d(gidx);
    int gval = (gw >> (8 * jb)) & 0xff;
    int sbit = (e < 4) ? jb : (jb + 4);
    float sign = (sgn[l] & (1 << sbit)) ? -1.0f : 1.0f;
    return db * (float)gval * sign;
}

// ---- NVFP4 f32 deq ----
__device__ __forceinline__ float deq_nvfp4(const uint8_t* row, int j) {
    int blk = j / 64, jj = j & 63;
    const uint8_t* b = row + blk * 36;
    const uint8_t* d_bytes = b;
    const uint8_t* qs = b + 4;
    int s = jj >> 4;            // sub-block 0..3
    int within = jj & 15;
    int byte = qs[s * 8 + (within & 7)];
    int code = (within < 8) ? (byte & 0xF) : (byte >> 4);
    return (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(d_bytes[s]);
}

// ---- Q4_0 f32 deq (gemma4 QAT-Q4_0 checkpoints): 18B block per 32 elems = fp16 d + 16 nibble
// bytes; x[j] = d * (nib - 8); qs[i] holds elems i (lo nibble) and i+16 (hi nibble). ----
__device__ __forceinline__ float deq_q4_0(const uint8_t* row, int j) {
    const uint8_t* blk = row + (j / 32) * 18;
    float d = __half2float(*(const __half*)blk);
    const uint8_t* qs = blk + 2;
    int i = j % 32;
    int q = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
    return d * (float)(q - 8);
}

// ---- Q2_K f32 deq ----
__device__ __forceinline__ float deq_q2_k(const uint8_t* row, int j) {
    int blk = j >> 8;
    int jj = j & 255;
    const uint8_t* b = row + (long)blk * 84;
    const uint8_t* scales = b;
    const uint8_t* qs = b + 16;
    float d = half_to_float(*(const unsigned short*)(b + 80));
    float dmin = half_to_float(*(const unsigned short*)(b + 82));
    int within = jj & 127;
    int shift = 2 * (within >> 5);
    int q = (qs[(jj >> 7) * 32 + (within & 31)] >> shift) & 3;
    int sc = scales[jj >> 4];
    return d * (float)(sc & 0xf) * (float)q - dmin * (float)(sc >> 4);
}

enum QType { QT_Q8_0 = 0, QT_Q4_K = 1, QT_Q6_K = 2,
             QT_Q5_K = 3, QT_Q3_K = 4, QT_IQ4_XS = 5, QT_IQ3_S = 6, QT_NVFP4 = 7,
             QT_F32 = 8,
             // SPLIT-PLANE repacked NVFP4 (A6 walk-order repack): quant plane
             // [out_f x in_f/64 x 32B] followed by scale plane [out_f x in_f/64 x 4B].
             // Host-side tag only for the Stage-A generic kernel (GpuTensor keeps QT_NVFP4 +
             // an rp flag); deq() cannot express it (needs tensor base + out_f, not a row ptr).
             QT_NVFP4_RP = 9,
             // CHECKPOINT-NATIVE FP8-E4M3 (BW24_ST_E4M3, lane e4m3dec 2026-07-08): the raw
             // safetensors e4m3 weight bytes [out_f, in_f] row-major (row_bytes == in_f), NO
             // per-32 scales — the per-tensor f32 weight_scale rides the host GpuTensor `scale`
             // (fused at the mmvq write, like the NVFP4 macro-scale). Weight side is EXACT
             // (the checkpoint's own precision; the Q8_0 re-encode this replaces was lossy).
             QT_F8_E4M3 = 10,
             // Raw BF16 row (FULL_PREC embed gather): 2 B/elem; f32 = bits << 16, exact.
             QT_BF16 = 11,
             // gemma4 QAT checkpoints (Q4_0 blocks, host tag 12 = lib.rs QT_Q4_0).
             QT_Q4_0 = 12,
             // Q2_K expert blocks. Generic staged f32-dequant path only for now.
             QT_Q2_K = 13 };

// e4m3 (OCP FP8, signed, bias 7) -> f32 via the native sm_89+ cvt (e4m3x2 -> f16x2 -> f32x2;
// every e4m3 value is exactly representable in f16, and f16 -> f32 is exact, so this chain is
// EXACT). Byte 0 of the ushort -> .x, byte 1 -> .y (little-endian, matches the cvt semantics).
__device__ __forceinline__ float2 e4m3x2_to_f32x2(unsigned short w2) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)w2, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&hr));
}
// Single-byte e4m3 -> f32 (Stage-A deq + scalar tails). Same exact chain as the x2 form.
__device__ __forceinline__ float e4m3_to_f32_d(unsigned char b) {
    __nv_fp8_e4m3 v; v.__x = (__nv_fp8_storage_t)b;
    return (float)v;
}

__device__ __forceinline__ float deq(int qtype, const uint8_t* row, int j) {
    switch (qtype) {
        case QT_Q8_0:   return deq_q8_0(row, j);
        case QT_Q4_K:   return deq_q4_k(row, j);
        case QT_Q6_K:   return deq_q6_k(row, j);
        case QT_Q5_K:   return deq_q5_k(row, j);
        case QT_Q3_K:   return deq_q3_k(row, j);
        case QT_IQ4_XS: return deq_iq4_xs(row, j);
        case QT_IQ3_S:  return deq_iq3_s(row, j);
        case QT_NVFP4:  return deq_nvfp4(row, j);
        case QT_Q2_K:   return deq_q2_k(row, j);
        // Unquantized f32 weight row (safetensors MoE Path A: experts gathered + dequantized to f32
        // host-resident, staged verbatim). `row` is the start of one out-row of `in_f` contiguous f32s.
        case QT_F32:    return ((const float*)row)[j];
        // Checkpoint-native e4m3 (BW24_ST_E4M3): 1 byte/element, row_bytes == in_f. The per-tensor
        // weight_scale is applied POST-matmul by the host (scale_inplace), like the NVFP4 macro-scale.
        case QT_F8_E4M3: return e4m3_to_f32_d(row[j]);
        // Raw bf16 (FULL_PREC embed): exact expansion, bit-identical to the host
        // f32::from_bits((bits as u32) << 16) contract.
        case QT_BF16: {
            unsigned int bits = ((const unsigned short*)row)[j];
            return __uint_as_float(bits << 16);
        }
        case QT_Q4_0:   return deq_q4_0(row, j);
    }
    return 0.0f;
}

// ---- Embed-from-device (CUDA-GRAPH-PLAN Phase 1): gather + dequant ONE token row whose id lives
// in a device u32 buffer (the argmax output), so the token never round-trips to host in steady
// state. x_out[j] = deq(qtype, embd_row(token_d[0]), j) for j in [0,n_embd). Bit-identical to host
// EmbedHost::gather (same per-dtype deq path). Single token (decode T=1). row_bytes = bytes/embed-row.
extern "C" __global__ void embed_gather_u32(
        const unsigned char* __restrict__ embd, const unsigned int* __restrict__ token_d,
        float* __restrict__ x_out, int n_embd, int qtype, long row_bytes) {
    unsigned int tok = token_d[0];
    const unsigned char* row = embd + (size_t)tok * row_bytes;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_embd; j += gridDim.x * blockDim.x)
        x_out[j] = deq(qtype, row, j);
}
// T-token variant (spec verify/replay): tokens_d[T] device ids -> x_out[T, n_embd]. grid.y = t.
// Replaces the host-side per-row dequant + ~T*14KB HtoD of EmbedHost::gather on the spec hot loop
// (nsys: cuMemcpyHtoDAsync was 84% of spec API time). Same per-dtype deq -> bit-identical rows.
extern "C" __global__ void embed_gather_u32_t(
        const unsigned char* __restrict__ embd, const unsigned int* __restrict__ tokens_d,
        float* __restrict__ x_out, int n_embd, int qtype, long row_bytes, int t) {
    int ti = blockIdx.y;
    if (ti >= t) return;
    unsigned int tok = tokens_d[ti];
    const unsigned char* row = embd + (size_t)tok * row_bytes;
    float* xr = x_out + (size_t)ti * n_embd;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_embd; j += gridDim.x * blockDim.x)
        xr[j] = deq(qtype, row, j);
}

// ---- Device i32 increment (CUDA-GRAPH-PLAN Phase 1): pos_d[0] += 1 inside the captured graph,
// replacing the per-step host htod_i32(&[pos]). One thread.
extern "C" __global__ void inc_i32(int* __restrict__ p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1; }

// ================= Stage-B: int8 dp4a MMVQ (decode hot path) =================
// Quantize activation row to q8_1 blocks (32 vals -> int8 + fp16 scale d), then weight-int8 dot.
// Activation buffer layout per block i: [32 int8 qs][1 float d]. We pack as: int8 qs in a byte array
// + a parallel float array of per-block d. Done in a tiny kernel below.

// dp4a: 4x int8 dot accumulate (sm_61+). Available on sm_120.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int r = c;
    for (int i = 0; i < 4; i++) { int8_t x = (a >> (i*8)) & 0xff, y = (b >> (i*8)) & 0xff; r += x*y; }
    return r;
#endif
}

// Quantize an [m, in] f32 activation matrix to q8_1: out_q (int8 [m, in]) + out_d (f32 [m, in/32]).
// One block per (token, block-of-32). amax over 32, d=amax/127, qs=round(x/d).
// WARP-PER-BLOCK (decode elementwise-soup fix, ncu 2026-07-03): lane j owns element j of one
// 32-block -> coalesced 128B read + 32B write, vs the old thread-owns-block 32-way strided form.
// __shfl_xor max reduce is order-independent -> d and q8 values BIT-IDENTICAL to the old kernel.
extern "C" __global__ void quantize_q8_1(const float* __restrict__ x, signed char* __restrict__ out_q,
                                         float* __restrict__ out_d, int in_f, int m) {
    int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global block-of-32 index
    int lane = threadIdx.x & 31;
    int nblk_row = in_f / 32;
    if (blk >= m * nblk_row) return;
    int t = blk / nblk_row;
    int b = blk % nblk_row;
    size_t off = (size_t)t * in_f + b * 32 + lane;
    float v = x[off];
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[off] = (signed char)__float2int_rn(v * id);
    if (lane == 0) out_d[(size_t)t * nblk_row + b] = d;
}

// ================= Stage-C: FP4 (e2m1) activation quantize for the mxf4 block-scale GEMM =========
// Quantize an [m, in] f32 activation to NVFP4-style e2m1 nibbles + per-16 UE4M3 scale, in the EXACT
// layout the mxf4nvf4 GEMM B-fragment gather wants (verified by probe/fp4_4x_*.cu):
//   aq4 : u32 [m][in_f/8]  — nibble (k&7) of word (k/8) = e2m1 code of activation element k
//   ad4 : u8  [m][in_f/16] — one UE4M3 scale byte per 16-elem K block
// e2m1 magnitudes: {0,0.5,1,1.5,2,3,4,6}; HW value of a nibble == kvalues here are GGUF-codebook
// (=2x HW e2m1); but for the B operand we feed RAW e2m1 nibbles whose HW value is the GGUF/2. So we
// must encode x/d to the *HW* e2m1 grid (max 6.0). The UE4M3 d is chosen so amax/d <= 6.
// HW UE4M3 (OCP E4M3, bias 7, NO x0.5): enc/dec below. Scale stored as the HW byte (NOT the GGUF
// 0.5x form) — the GEMM treats sb as HW UE4M3.
__device__ __forceinline__ int e2m1_encode_hw(float v) {
    // nearest of the 8 signed e2m1 grid points {0,.5,1,1.5,2,3,4,6}. sign bit = bit3.
    float a = fabsf(v);
    int code;
    // round-to-nearest on the irregular grid
    if (a < 0.25f) code = 0;            // 0
    else if (a < 0.75f) code = 1;       // 0.5
    else if (a < 1.25f) code = 2;       // 1.0
    else if (a < 1.75f) code = 3;       // 1.5
    else if (a < 2.5f) code = 4;        // 2.0
    else if (a < 3.5f) code = 5;        // 3.0
    else if (a < 5.0f) code = 6;        // 4.0
    else code = 7;                      // 6.0
    if (code != 0 && v < 0.0f) code |= 0x8;
    return code;
}
// HW UE4M3 encode of a NON-NEGATIVE scale s: round to nearest E4M3 (bias 7, no x0.5). Clamp [2^-9, 448].
__device__ __forceinline__ unsigned char ue4m3_encode_hw(float s) {
    if (!(s > 0.0f)) return 0;
    s = fminf(s, 448.0f);
    int e; float m = frexpf(s, &e);    // s = m*2^e, m in [0.5,1)
    // normalized: s = 2^(E-7)*(1+man/8), E = exponent field (1..15), man 0..7
    int E = e - 1 + 7;                 // since m in [0.5,1): s = 2^(e-1)*(2m), 2m in [1,2)
    float frac = 2.0f * m - 1.0f;      // in [0,1)
    if (E <= 0) {                      // subnormal: s = (man/8)*2^-6
        float q = s * 64.0f * 8.0f;    // man = round(s / 2^-9)
        int man = (int)(q + 0.5f);
        if (man > 7) man = 7;
        return (unsigned char)man;     // E=0
    }
    int man = (int)(frac * 8.0f + 0.5f);
    if (man == 8) { man = 0; E += 1; }
    if (E > 15) { E = 15; man = 7; }
    return (unsigned char)((E << 3) | man);
}
// One CTA-thread per (token, 16-block). amax over 16 -> UE4M3 d (so amax/d ~ 6) -> e2m1 encode.
extern "C" __global__ void quantize_fp4_act(const float* __restrict__ x, unsigned* __restrict__ aq4,
                                            unsigned char* __restrict__ ad4, int in_f, int m) {
    int b16 = blockIdx.x * blockDim.x + threadIdx.x;  // global 16-block index
    int nb16_row = in_f / 16;
    if (b16 >= m * nb16_row) return;
    int t = b16 / nb16_row;
    int blk = b16 % nb16_row;
    const float* xr = x + (size_t)t * in_f + blk * 16;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(xr[j]));
    // choose d so that amax/d == 6 (the e2m1 max). d ~ amax/6, quantized to UE4M3.
    float dwant = amax > 0.0f ? amax / 6.0f : 0.0f;
    unsigned char db = ue4m3_encode_hw(dwant);
    float d = ue4m3_to_f32_hw(db);
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    ad4[(size_t)t * nb16_row + blk] = db;
    // encode 16 nibbles into two u32 words (k/8 within the 16-block -> word blk*2 + (k/8)).
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned w = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int code = e2m1_encode_hw(xr[half * 8 + n] * id);
            w |= ((unsigned)code) << (4 * n);
        }
        aq4[((size_t)t * (in_f / 8)) + blk * 2 + half] = w;
    }
}

// Block reduce shared by the dp4a MMVQ kernels: full-warp shfl, then warp0 sums the per-warp
// partials. Correct for any blockDim.x that is a multiple of 32 (used with 128 = 4 warps).
__device__ __forceinline__ void mmvq_block_reduce_write(float acc, float* __restrict__ y,
                                                        size_t out_idx, int tid) {
    __shared__ float s[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[out_idx] = v;
    }
}

// Vectorized weight-int load: 4 int8 starting at `p` (only 2-byte aligned in Q8_0 -> uint16x2).
// Mirrors llama.cpp get_int_b2 (vecdotq.cuh:18-25). Safe for any 2-byte-aligned source.
__device__ __forceinline__ int get_int_b2(const void* p) {
    const unsigned short* u = (const unsigned short*)p;
    return (int)u[0] | ((int)u[1] << 16);
}

// Vectorized weight-int load: 4 int8 starting at `p`, single 32-bit LDG. Mirrors llama get_int_b4
// (vecdotq.cuh). Safe for any 4-byte-aligned source. NVFP4 qss is provably 4-aligned
// (row_bytes=(in_f/64)*36 -> mult of 4; qs=b+4; qss=qs+s*8) so the qs stream qualifies. Do NOT
// widen to int2/LDG.E.64 there: rows are only 8-aligned when in_f%128==0 -> faults on odd in_f/64.
__device__ __forceinline__ int get_int_b4(const void* p) {
    return *(const int*)p;
}

// ============================ Stage-B MMVQ (warp-per-row decode) ============================
// PERF-3 (DECODE-GEMV-PLAN): warp-per-row layout matching llama.cpp mmvq.cu. block=(32,ROWS,1):
// one WARP (threadIdx.y) owns one output row. Reduction is warp-only __shfl_xor_sync (no smem,
// no __syncthreads — removes the cross-warp barrier from the m=1 critical path). The per-element
// DEQUANT BODIES are LIFTED VERBATIM from the matching _dp4a kernels (same get_int_b2/codebook
// math), so the int sumi is bit-for-bit identical; only the layout + reduction order change.
// ROWS_PER_BLOCK = 4 (128 threads, 4 independent rows in flight) is llama's GENERIC ncols_dst=1.
#define BW24_MMVQ_ROWS 4

// Warp-only reduce: full-warp shfl-xor (butterfly), all lanes hold the sum. No smem/barrier.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ----- Q8_0 warp-per-row MMVQ. Body lifted from qmatvec_q8_0_dp4a (loop @ ~line 398). -----
extern "C" __global__ void qmatvec_q8_0_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;                              // 0..31
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {        // per-warp contiguous stride (32 lanes)
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);   // 2-byte aligned OK
        const unsigned char* wq = wb + 2;                       // qs: 2-byte aligned -> get_int_b2
        const int4* aq16 = (const int4*)(arow + blk * 32);      // 32-aligned -> 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q8_0 m=1 single-row body shared by the FUSED multi-tensor launches below. LIFTED VERBATIM
// from qmatvec_q8_0_mmvq with t pinned to 0 (decode m==1): same block walk, same dp4a order, same
// warp_reduce_sum, same write -> per (tensor,row) output bits identical to a separate m=1 launch. -----
__device__ __forceinline__ void q8_0_mmvq_row1(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);
        const unsigned char* wq = wb + 2;
        const int4* aq16 = (const int4*)(aq + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * ad[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc;
}

// ----- FUSED Q8_0 m=1 matvec PAIR, UNEQUAL out_f (trunk launch-fusion, 2026-07-05). The 35B trunk
// decode runs ~250 tiny q8_0 m=1 launches/token (2.4-8us, launch-latency class: 44k of the 48-tok
// window's 160k-class launches are this kernel). Same-input projections fold into ONE grid: blocks
// [0,nb0) compute tensor 0, [nb0,nb0+nb1) tensor 1 (the NVFP4 gate+up dual + beta/alpha dual proved
// the recipe; this variant lifts the same-out_f restriction via a block-offset split instead of
// blockIdx.y). Both tensors share in_f (Q8_0 row_bytes is a pure function of in_f -> ONE row_bytes)
// and the SAME q8_1 activation. Per (tensor,row) the body is qmatvec_q8_0_mmvq VERBATIM ->
// BIT-IDENTICAL to two separate m=1 launches. Targets: 35B wqkv+wqkv_gate (8192/4096),
// gate_shexp+up_shexp (512/512). Seam BW24_Q8_DUAL=0 (host-side). -----
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long row_bytes) {
    int nb0 = (out0 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0) { W = W0; y = y0; out_f = out0; }
    else         { W = W1; y = y1; out_f = out1; b -= nb0; }
    q8_0_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, b * BW24_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- FUSED Q8_0 m=1 matvec TRIPLE (wq+wk+wv: same input h, same in_f, out_f 8192/512/512 on
// the 35B full-attn layers). Same block-offset recipe as fused2 with three ranges. -----
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long row_bytes) {
    int nb0 = (out0 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int nb1 = (out1 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; b -= nb0 + nb1; }
    q8_0_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, b * BW24_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- Q4_K warp-per-row MMVQ. Body lifted from qmatvec_q4_K_dp4a (loop @ ~line 427). -----
extern "C" __global__ void qmatvec_q4_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        const int* q4 = (const int*)(qs + chunk * 32);          // 4-byte aligned
        bool hi = (grp & 1);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int raw = q4[k];
            int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d) * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q5_K warp-per-row MMVQ. Body lifted from qmatvec_q5_K_dp4a (the only major decode matvec that
// still fell to the slow dp4a path at m=1 — 7% of 9B decode). One warp owns one output row; lane
// strides the 32-blocks; warp-only shfl reduce (no smem barrier). Bit-equivalent to qmatvec_q5_K_dp4a
// up to f32 reduction order (same vectorized q5_K unpack + dp4a + min-offset). -----
extern "C" __global__ void qmatvec_q5_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);
            int qh4 = get_int_b2(qh + k * 4);
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;
            int wpack = low | (h << 4);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d)   * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q5_K MULTI-ROW-PER-WARP MMVQ (the FR-Spec trimmed draft head is Q5_K 32768 rows — 8% of
// the 27B p3 spec wall at 1.02ms/draft launch, memory-latency bound like the other k-quants). Same
// multirow recipe as q4k_mmvq_multirow: activation int8 loaded ONCE (2x int4), reused across RPW
// rows; RPW weight rows in flight hide the load latency. BIT-IDENTICAL per row to qmatvec_q5_K_mmvq
// (same scale/min unpack, same qh bit extraction, same dp4a order, same warp_reduce_sum). -----
template<int RPW>
__device__ __forceinline__ void q5k_mmvq_multirow(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        // activation loaded ONCE, reused across RPW rows (+ the min-sum, row-independent).
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 176;
            float d_sb    = half_to_float(*(const unsigned short*)b);
            float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qh = b + 16;
            const unsigned char* qs = b + 48;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            const unsigned char* q = qs + g64 * 32;
            int sumi_d = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int q4  = get_int_b2(q  + k * 4);
                int qh4 = get_int_b2(qh + k * 4);
                int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
                int h   = (qh4 >> hbit) & 0x01010101;
                int wpack = low | (h << 4);
                sumi_d = dp4a(wpack, aq4[k], sumi_d);
            }
            acc[r] += d_sb * (float)((int)sc * sumi_d) * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- Q5_K ISSUE-REDUCED MMVQ family (`_il`, q5issue lane 2026-07-08, BW24_Q5K_ISSUE=1) ----
// WHY: the q5_K mmvq family sits at ~61% of the bandwidth wall on the 27B lm_head (1030us) —
// the k-quant mmvq ceiling is instruction-ISSUE, not loads (q6krp repack: exactly 0 gain; the
// down8 byte_perm decode: +2.5% e2e). Per 32-elem group the reference body issues 2 LDG.U16
// (d/dmin) + a warp-DIVERGENT grp<4 / grp>=4 scale unpack (2-4 LDG.U8 byte loads, both paths
// serialized every iteration since grp = lane&7 splits the warp) + 16 LDG.32 (get_int_b2 qs/qh)
// = ~21 LSU ops + divergence against 16 dp4a of real work. This body produces the IDENTICAL
// packed ints from the IDENTICAL bytes with 5 LDG.128: one uint4 header (d|dmin|scales[12]),
// 2x uint4 qh (the whole 32B plane), 2x uint4 qs — and replaces the divergent scale branch with
// branchless register extraction (both paths computed + select on the loop-invariant grp>=4).
// BIT-IDENTITY (value-level): the uint4 components ARE the little-endian 32-bit words
// get_int_b2 builds (q5_K block=176B: b, b+16, b+48+g64*32 all 16-aligned when W is);
// (q4 >> sh4) & M with sh4 = hi*4 == the `hi ? (q4>>4)&M : q4&M` select (>>0 is identity);
// the scale/min register math lands the exact scales[] bytes the branchy path loads
// (hdr.y/z/w = scales[0..3]/[4..7]/[8..11], byte j via >>8j). The dp4a chain order (k
// ascending, sumi_d/sumi_sum separate integer chains) and the closing float expression are
// UNCHANGED, so outputs are bit-identical per (token,row).
// ALIGNMENT: q5_K row_bytes = (in_f/256)*176, a multiple of 16 -> every superblock pointer is
// 16-aligned iff W is (cudaMalloc slabs are 256B-aligned; every real dispatch passes the
// tensor-base slice). A GRID-UNIFORM guard falls back to the reference body for exotic bases.
template<int RPW>
__device__ __forceinline__ void q5k_mmvq_multirow_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    if ((((unsigned long long)W | (unsigned long long)row_bytes) & 15ull) != 0ull) {
        q5k_mmvq_multirow<RPW>(W, aq, ad, y, in_f, out_f, m, row_bytes);  // reference fallback
        return;
    }
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    // decode geometry is loop-invariant per lane: g = lane + 32*i -> g&7 == lane&7.
    int grp  = lane & 7;
    int g64  = grp >> 1;
    int sh4  = (grp & 1) * 4;        // 0 for the low-nibble plane, 4 for the high
    int hbit = 2 * g64 + (grp & 1);
    bool hi4 = grp >= 4;
    int sh8  = 8 * (grp & 3);        // byte j of the scale words, j = grp or grp-4
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sumi_sum = 0;                        // row-independent activation sum (shared)
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 176;
            uint4 hdr = *(const uint4*)b;        // d|dmin (4B) + scales[12] in one LDG.128
            float d_sb    = half_to_float((unsigned short)(hdr.x & 0xffffu));
            float dmin_sb = half_to_float((unsigned short)(hdr.x >> 16));
            unsigned by = (hdr.y >> sh8) & 0xffu;   // scales[j]
            unsigned bz = (hdr.z >> sh8) & 0xffu;   // scales[j+4]
            unsigned bw = (hdr.w >> sh8) & 0xffu;   // scales[j+8]
            int sc = hi4 ? (int)((bw & 0xFu) | ((by >> 6) << 4)) : (int)(by & 63u);
            int mn = hi4 ? (int)((bw >> 4)   | ((bz >> 6) << 4)) : (int)(bz & 63u);
            const uint4* qhv = (const uint4*)(b + 16);            // whole 32B qh plane
            uint4 h01 = qhv[0], h23 = qhv[1];
            const uint4* qsv = (const uint4*)(b + 48 + g64 * 32); // 32B nibble plane
            uint4 q01 = qsv[0], q23 = qsv[1];
            int qw[8]  = { (int)q01.x, (int)q01.y, (int)q01.z, (int)q01.w,
                           (int)q23.x, (int)q23.y, (int)q23.z, (int)q23.w };
            int qhw[8] = { (int)h01.x, (int)h01.y, (int)h01.z, (int)h01.w,
                           (int)h23.x, (int)h23.y, (int)h23.z, (int)h23.w };
            int sumi_d = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int low = (qw[k] >> sh4) & 0x0F0F0F0F;
                int h   = (qhw[k] >> hbit) & 0x01010101;
                int wpack = low | (h << 4);
                sumi_d = dp4a(wpack, aq4[k], sumi_d);
            }
            acc[r] += d_sb * (float)(sc * sumi_d) * d8
                    - dmin_sb * (float)(mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a;
    }
}
// Single-row twin: RPW=1 of the multirow body is bit-identical to qmatvec_q5_K_mmvq (the only
// difference is sumi_sum computed before sumi_d — separate exact integer chains; the float
// expression and per-g accumulation order are unchanged). Fallback likewise goes to
// q5k_mmvq_multirow<1>, bit-identical to the reference single-row kernel.
extern "C" __global__ void qmatvec_q5_K_mmvq_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow_il<1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_mr2_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow_il<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}



// ----- Q6_K warp-per-row MMVQ. Body lifted from qmatvec_q6_K_dp4a (loop @ ~line 476). -----
extern "C" __global__ void qmatvec_q6_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        // VECTORIZED unpack (was a scalar 4-byte inner loop = ~20 ALU ops/k starving DRAM to 19%).
        // For each k the 4 ql bytes (il=k*4..k*4+3) and 4 qh bytes are CONTIGUOUS -> read each as one
        // 32-bit word (get_int_b2: 2-aligned-safe, q6_K block=210 is even) and extract all 4 nibbles/
        // 2-bit groups with SIMD masks. BIT-IDENTICAL: get_int_b2 packs byte e at bit e*8, exactly the
        // old `<<(e*8)` order; per-byte ql_bits|(qh_bits<<4) and __vsubss4 are unchanged.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ql4 = get_int_b2(qlh + k * 4 + ql_off);          // 4 ql bytes
            int qh4 = get_int_b2(qhh + k * 4);                   // 4 qh bytes
            int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
            int qhn = (qh4 >> qh_sh) & 0x03030303;               // 2-bit group per byte, 0..3
            int vpack = qln | (qhn << 4);                        // per byte = ql_bits|(qh_bits<<4), 0..63
            int wpack = __vsubss4(vpack, 0x20202020);            // subtract 32 per byte (signed sat)
            int a = aq4[k];
            if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
            else       sumi1 = dp4a(wpack, a, sumi1);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- NVFP4 warp-per-row MMVQ. Body lifted from qmatvec_nvfp4_dp4a (loop @ ~line 674). -----
extern "C" __global__ void qmatvec_nvfp4_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;          // which 64-elem block_nvfp4 (36 bytes)
        int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2;
        // activation 32 int8 = 8 ints: load as 2x int4 (16B) -> cuts 8 LDG.E.32 to 2 LDG.E.128,
        // attacking lg_throttle (3.82, LSU queue full). aqb = arow + g*32 is 32-aligned -> int4 safe.
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];   // aq4[0..3]
        int4 a23 = aq16[1];   // aq4[4..7]
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;
            int q4a = get_int_b4(qss);      // P1: single LDG.E.32 (was 4x LDG.E.U8); qss 4-aligned
            int q4b = get_int_b4(qss + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * yscale;
}

// ---- NVFP4 MMVQ, MULTI-ROW-PER-WARP (MLP lever). The single-row mmvq above is m=1 LATENCY-bound
// (ncu: 30-46% DRAM, loads-in-flight starved — one acc chain per warp waits on each weight LDG
// before the next dp4a). This variant has ONE warp compute RPW output rows in ONE pass over the
// shared activation: the activation int8 (loaded once as 2x int4) is REUSED across all RPW rows, and
// RPW independent weight rows are loaded + RPW independent acc chains run per iteration -> RPW x the
// memory-level parallelism, hiding the weight-load latency WITHOUT a cross-warp reduce barrier (the
// barrier was why more-WARPS-per-row was slower; more-ROWS-per-warp has no barrier). Activation
// bytes leave HBM/L2 1x per warp instead of 1x per row. BIT-IDENTICAL per row to qmatvec_nvfp4_mmvq:
// same dp4a order, same ue4m3 scale, same warp_reduce_sum, same write. grid.x sized for RPW rows/warp.
// yscale = the per-tensor NVFP4 macro-scale, applied AT THE WRITE (y = reduced_acc * yscale).
// Bit-identical to the old separate scale_inplace pass (same single IEEE multiply on the same
// value); folding it removes one launch per matvec (53 scale_f32 launches/token on the 9B).
template<int RPW>
__device__ __forceinline__ void nvfp4_mmvq_multirow(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * RPW;   // first of this warp's RPW rows
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        int s0 = whichHalf * 2;
        // activation loaded ONCE, reused across all RPW rows.
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];
        int4 a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}
// t=0-pinned single-token body of nvfp4_mmvq_multirow (blockIdx.y is repurposed by the dual
// kernel for tensor select). SAME dp4a order / scales / reduce as the multirow helper -> the
// dual kernel's per-element results are bit-identical to the mr2 kernel at m=1.
template<int RPW>
__device__ __forceinline__ void nvfp4_mmvq_dual_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * RPW;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq;
    const float*         adrow = ad;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        int s0 = whichHalf * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];
        int4 a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[o] = a * yscale;
    }
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    nvfp4_mmvq_multirow<2>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}
// DUAL gate+up matvec (mm-fusion, 2026-07-03): the FFN gate and up projections share the SAME
// activation and the same (in_f, out_f) shape; running them as two sequential launches leaves the
// tail of each under-filled and pays two launch latencies. ONE grid computes both: blockIdx.y
// selects the tensor (0=gate -> y0, 1=up -> y1). Per (tensor, row) the body is nvfp4_mmvq_multirow
// verbatim -> BIT-IDENTICAL per output element to two separate launches. (The reference engine
// runs the same fusion as its top 27B decode kernel at 47-50% DRAM vs ~40% for singles.)
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_mr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes, float y0scale, float y1scale) {
    const unsigned char* W = (blockIdx.y == 0) ? W0 : W1;
    float* y = (blockIdx.y == 0) ? y0 : y1;
    float yscale = (blockIdx.y == 0) ? y0scale : y1scale;
    // nvfp4_mmvq_multirow reads blockIdx.y as the token index; decode m==1 -> token 0. Inline the
    // call with t forced to 0 via a shifted grid: we reuse the body by passing m=1 and mapping
    // blockIdx.y ourselves — the helper uses blockIdx.y for t, so temporarily this kernel only
    // supports m==1 (asserted host-side).
    nvfp4_mmvq_dual_row<2>(W, aq, ad, y, in_f, out_f, row_bytes, yscale);
}

// ---- NVFP4 BATCHED matvec, WEIGHT-TILE-RESIDENT across M token columns (the m=2-4 concurrent-decode
// win). The current mmvq launches grid.y=m INDEPENDENT blocks per output row -> the weight row is
// re-read m times from HBM/L2. Here ONE warp owns ONE output row and walks the weight ONCE, doing
// dp4a against ALL m activation columns (m independent accumulators in regs). The weight quant
// bytes + decoded e2m1 values leave HBM/L2 ONCE and serve all m tokens (the activation is tiny: m*32
// int8 per group). So m tokens cost ~1 weight-read instead of m. y is [m, out_f] (token-major, same
// as the per-m kernel writes y[t*out_f+o]). MCOLS is the compile-time batch (2 or 4). For m<MCOLS the
// extra columns are computed against zero-padded activation (caller sizes y for exactly m; we guard).
// BIT-IDENTICAL per (token,row) to qmatvec_nvfp4_mmvq: same dp4a order, same ue4m3 scale, same reduce.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2;
        // decode the weight nibbles ONCE for this group (reused across all m token columns).
        int2 wv[2][2];   // [sl][0]=va, [sl][1]=vb
        float wscale[2];
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;
            int q4a = get_int_b4(qss);
            int q4b = get_int_b4(qss + 4);
            wv[sl][0] = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            wv[sl][1] = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            wscale[sl] = ue4m3_to_f32_d(d_bytes[s]);
        }
        // for each token column: load its 32 int8 activation + per-group scale, dp4a vs the decoded W.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int base = sl * 4;
                int2 va = wv[sl][0], vb = wv[sl][1];
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += wscale[sl] * (float)sumi;
            }
            acc[c] += adg * partial;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 (K=4..7 spec verify, T=5..8). Same template; columns c >= m break out, so m=5 does the
// b4+b1 split's total dp4a work with ONE weight read/decode instead of five (the pre-b8 T=5 path
// was grid.y=m per-row MMVQ — 5 full weight reads — measured as the 27B K=4 cliff).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, WEIGHT-PREFETCH double-buffer (b4 long_scoreboard fix, 2026-07-03).
// ncu --set full on the REAL 27B verify (12 steady launches): the batched kernel is memory-LATENCY
// bound, not bandwidth bound — long_scoreboard 18-30 stalls/issue (every other reason <=1.7),
// DRAM only 41-51% active, lg_throttle 0.7 (LSU queue fine), L1 hit 94% (activations), L2 hit 18%
// (weights stream from DRAM). Cause: ONE weight-load wavefront (6 LDGs, 18B) in flight per warp per
// k-iteration — half the m=1 mr2 kernel's per-warp weight MLP. Fix: stage the NEXT g-iteration's
// weight words in registers, issuing its 5 LDGs BEFORE consuming the current ones -> 2 weight
// wavefronts in flight per warp. Also folds the 2 scale byte-loads into the superblock's one
// 4-byte scale word (b is 4-aligned; extracted bytes feed the SAME ue4m3_to_f32_d) and the 4 quant
// words are the SAME 16 bytes the reference reads via get_int_b4 x4. BIT-IDENTICAL per (token,row):
// identical dp4a order, scales, adg factor, warp_reduce_sum — only load ISSUE TIME changes.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    // staged weight words for the CURRENT g: 4 quant words (16B at qs + whichHalf*16) + the
    // superblock's 4-byte scale word.
    int q0 = 0, q1 = 0, q2 = 0, q3 = 0, scw = 0;
    int g = lane;
    if (g < nsb) {
        const unsigned char* b = wrow + (long)(g >> 1) * 36;
        const unsigned char* qp = b + 4 + (g & 1) * 16;
        q0 = get_int_b4(qp);      q1 = get_int_b4(qp + 4);
        q2 = get_int_b4(qp + 8);  q3 = get_int_b4(qp + 12);
        scw = get_int_b4(b);
    }
    while (g < nsb) {
        int cq0 = q0, cq1 = q1, cq2 = q2, cq3 = q3, cscw = scw;
        int gn = g + 32;
        if (gn < nsb) {   // issue the NEXT wavefront before consuming the current one
            const unsigned char* bn = wrow + (long)(gn >> 1) * 36;
            const unsigned char* qpn = bn + 4 + (gn & 1) * 16;
            q0 = get_int_b4(qpn);      q1 = get_int_b4(qpn + 4);
            q2 = get_int_b4(qpn + 8);  q3 = get_int_b4(qpn + 12);
            scw = get_int_b4(bn);
        }
        int s0 = (g & 1) * 2;
        // decode ONCE per group, exactly like the reference (sl=0 -> cq0/cq1, sl=1 -> cq2/cq3).
        int2 wv[2][2];
        float wscale[2];
        wv[0][0] = get_int_from_table_16_d(cq0, kvalues_mxfp4_d);
        wv[0][1] = get_int_from_table_16_d(cq1, kvalues_mxfp4_d);
        wv[1][0] = get_int_from_table_16_d(cq2, kvalues_mxfp4_d);
        wv[1][1] = get_int_from_table_16_d(cq3, kvalues_mxfp4_d);
        wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
        wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int base = sl * 4;
                int2 va = wv[sl][0], vb = wv[sl][1];
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += wscale[sl] * (float)sumi;
            }
            acc[c] += adg * partial;
        }
        g = gn;
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, TWO ROWS PER WARP (same long_scoreboard fix by the mr2 route: 2
// independent weight-row streams per warp = 12 weight LDGs in flight instead of 6, and the m
// activation columns are loaded once per warp and reused across BOTH rows). Per (token,row) the
// body is the reference nvfp4_mmvq_batched verbatim -> bit-identical; only the row->warp mapping
// (grid shape) and cross-row interleave change, both exactness-free. Costs ~+14 regs -> one fewer
// resident block; measured against _pf on the DRAM-cold sweep before defaulting.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        // decode BOTH rows' weight groups first (both wavefronts issued together).
        int2 wv[2][2][2];    // [row][sl][a/b]
        float wscale[2][2];  // [row][sl]
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 36;
            const unsigned char* qs = b + 4;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                wv[r][sl][0] = get_int_from_table_16_d(get_int_b4(qss),     kvalues_mxfp4_d);
                wv[r][sl][1] = get_int_from_table_16_d(get_int_b4(qss + 4), kvalues_mxfp4_d);
                wscale[r][sl] = ue4m3_to_f32_d(b[s]);
            }
        }
        // each token column's activation loaded ONCE, dp4a vs both rows.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// 8-RESIDENT-BLOCK twin of b4_r2: __launch_bounds__(128, 8) squeezes 67 -> 64 regs (STACK:8, no
// LOCAL spill) so 8 blocks fit per SM instead of 7. Same template, bit-identical per (token,row).
// Only wins when the extra residency DROPS the integer wave count of the halved grid — measured
// DRAM-cold m=4: ffn_down 640 blocks 1.11 -> 0.98 waves = 112.5 -> 81.6us (beats pf 90.1);
// ssm_out 44.9 -> 34.1; qkv 1280 blocks 2.23 -> 1.95 waves = 58.1 -> 51.1. When ceil(waves) does
// NOT drop, the reg squeeze is a pure ~3-4% per-block tax (gate/up 81.1 -> 83.9, attn_q 12288
// 61.0 -> 63.4) — the dispatcher compares ceil(waves) at both residencies and picks.
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 twins (K=4..7 spec verify T=5..8). acc[2][8] costs ~+8 regs over b4_r2 — the r2w8
// residency squeeze may spill at MCOLS=8; measured per shape before defaulting (msweep m=5..8).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b8_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, PREFETCH x TWO-ROWS combined (4 weight wavefronts in flight/warp:
// 2 rows x double-buffer). Highest register pressure of the family; measured, not assumed.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    int q[2][4]; int scw[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) { q[r][0]=q[r][1]=q[r][2]=q[r][3]=0; scw[r]=0; }
    int g = lane;
    if (g < nsb) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)(g >> 1) * 36;
            const unsigned char* qp = b + 4 + (g & 1) * 16;
            q[r][0] = get_int_b4(qp);      q[r][1] = get_int_b4(qp + 4);
            q[r][2] = get_int_b4(qp + 8);  q[r][3] = get_int_b4(qp + 12);
            scw[r] = get_int_b4(b);
        }
    }
    while (g < nsb) {
        int cq[2][4]; int cscw[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            cq[r][0]=q[r][0]; cq[r][1]=q[r][1]; cq[r][2]=q[r][2]; cq[r][3]=q[r][3];
            cscw[r]=scw[r];
        }
        int gn = g + 32;
        if (gn < nsb) {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                const unsigned char* bn = wrow0 + (long)r * row_bytes + (long)(gn >> 1) * 36;
                const unsigned char* qpn = bn + 4 + (gn & 1) * 16;
                q[r][0] = get_int_b4(qpn);      q[r][1] = get_int_b4(qpn + 4);
                q[r][2] = get_int_b4(qpn + 8);  q[r][3] = get_int_b4(qpn + 12);
                scw[r] = get_int_b4(bn);
            }
        }
        int s0 = (g & 1) * 2;
        int2 wv[2][2][2];
        float wscale[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            wv[r][0][0] = get_int_from_table_16_d(cq[r][0], kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(cq[r][1], kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(cq[r][2], kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(cq[r][3], kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw[r] >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw[r] >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        g = gn;
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pfr2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pfr2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, cp.async SMEM WEIGHT RING (A5, 2026-07-04 — Marlin/CUTLASS multi-stage
// staging pattern). ncu on the post-pf/r2w8 dispatch showed the residual stall is STILL memory
// latency (long_scoreboard 8.8-16.4/issue vs FMA-dep wait <=1.9, DRAM only 48-69%): the register
// double-buffer (pf) and 2-row ILP (r2) top out at 1-2 weight wavefronts in flight per warp because
// every extra wavefront costs ~20 registers. cp.async.cg stages weight bytes global->smem WITHOUT
// register cost, so a STAGES-deep ring holds (STAGES-1) full 576B warp-windows in flight per warp.
// Layout law: one warp k-iteration consumes a CONTIGUOUS 576-byte window (16 NVFP4 36B blocks —
// 32 lanes x half-block) at window g-iter*576 of the row; when row_bytes%16==0 (all trunk shapes:
// in_f%256==0 -> (in_f/64)*36 % 16 == 0) every window is 16B-aligned in GLOBAL space, so the ring
// copies 36 16B cp.async.cg chunks per window. Lanes then read their 5 words (4 quant + 1 scale)
// from smem (LDS, no long_scoreboard). Host dispatch gates _ca on row_bytes%16==0 && nsb%32==0;
// otherwise falls back to pf/r2. BIT-IDENTICAL per (token,row): the staged bytes ARE the global
// bytes (cp.async is a byte copy); identical dp4a order, scales, adg factor, warp_reduce_sum —
// only WHERE the bytes stage changes, not the dot order.
#define CA_WIN 576   // bytes per warp-window: 16 blocks x 36B
__device__ __forceinline__ void cp_async16_g(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Issue one row-window (36 x 16B chunks) into `dst`. Lane L copies chunk L, lanes 0..3 also copy
// chunk 32+L. `src` = wrow + iter*CA_WIN, 16B-aligned by the dispatch gate.
__device__ __forceinline__ void ca_issue_window(unsigned char* dst, const unsigned char* src, int lane) {
    cp_async16_g(dst + lane * 16, src + lane * 16);
    if (lane < 4) cp_async16_g(dst + (32 + lane) * 16, src + (32 + lane) * 16);
}

// WROWS=1: one row/warp, STAGES-deep ring (smem 4 warps x STAGES x 576B).
// WROWS=2: two rows/warp (r2's activation-reuse + halved grid) x STAGES ring on both row streams.
template<int MCOLS, int WROWS, int STAGES>
__device__ __forceinline__ void nvfp4_mmvq_batched_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int niter = nsb >> 5;                       // dispatch gate: nsb%32==0
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    __shared__ __align__(16) unsigned char smw[BW24_MMVQ_ROWS][STAGES][WROWS][CA_WIN];
    unsigned char (*ring)[WROWS][CA_WIN] = smw[threadIdx.y];
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    // prologue: ALWAYS commit STAGES-1 groups (empty commits keep the per-thread group count
    // uniform when niter < STAGES, so wait<STAGES-2> below really completes the oldest stage).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < niter) {
            ca_issue_window(&ring[s][0][0], wrow0 + s * CA_WIN, lane);
            if (WROWS == 2 && has1)
                ca_issue_window(&ring[s][1][0], wrow0 + row_bytes + s * CA_WIN, lane);
        }
        cp_async_commit();
    }
    for (int it = 0; it < niter; it++) {
        cp_async_wait<STAGES - 2>();            // oldest committed stage (it) landed
        __syncwarp();
        const unsigned char* wnd0 = &ring[it % STAGES][0][0];
        int g = it * 32 + lane;
        int loff = (lane >> 1) * 36;            // this lane's block within the window
        int qoff = loff + 4 + (lane & 1) * 16;  // its 16B quant half (4B-aligned in smem)
        int s0 = (lane & 1) * 2;
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const unsigned char* wnd = wnd0 + (WROWS == 2 ? r * CA_WIN : 0);
            int cscw = *(const int*)(wnd + loff);
            int cq0 = *(const int*)(wnd + qoff);
            int cq1 = *(const int*)(wnd + qoff + 4);
            int cq2 = *(const int*)(wnd + qoff + 8);
            int cq3 = *(const int*)(wnd + qoff + 12);
            // decode ONCE per group, exactly like pf (sl=0 -> cq0/cq1, sl=1 -> cq2/cq3).
            int2 wv[2][2];
            float wscale[2];
            wv[0][0] = get_int_from_table_16_d(cq0, kvalues_mxfp4_d);
            wv[0][1] = get_int_from_table_16_d(cq1, kvalues_mxfp4_d);
            wv[1][0] = get_int_from_table_16_d(cq2, kvalues_mxfp4_d);
            wv[1][1] = get_int_from_table_16_d(cq3, kvalues_mxfp4_d);
            wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[sl][0], vb = wv[sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        // refill: consume slot (it%STAGES) is done for THIS warp's lanes after the reads above
        // retire; the overwrite targets slot (it+STAGES-1)%STAGES = (it-1)%STAGES, whose reads
        // finished an iteration ago (separated by the next iter's __syncwarp).
        int itn = it + STAGES - 1;
        if (itn < niter) {
            ca_issue_window(&ring[itn % STAGES][0][0], wrow0 + (size_t)itn * CA_WIN, lane);
            if (WROWS == 2 && has1)
                ca_issue_window(&ring[itn % STAGES][1][0], wrow0 + row_bytes + (size_t)itn * CA_WIN, lane);
        }
        cp_async_commit();
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<2, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<4, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_car2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<2, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_car2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<4, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 split-plane matvec, cp.async SOFTWARE-PIPELINED (2026-07-05). The _rp kernels below
// issue a SYNCHRONOUS int4 quant load per g-iter; ncu on 27B decode (b4_rpr2) showed the dominant
// stall is long_scoreboard 7.8-9.0 inst/issue (global-load latency), occupancy only 55% (67 regs
// -> 7 blocks/SM), DRAM 40% — memory-LATENCY bound, not wall-bound. This variant pipelines the
// split-plane windows with cp.async so weight loads for iter it+STAGES-1 are in flight while iter
// it computes. Split-plane makes the window trivially aligned: per warp-iter the quant read is
// 512B contiguous (32 lanes x 16B at rowq + it*512) and the scale read is 64B (16 words at
// rows + it*64). Window = 512B quant + 64B scale = 576B (== CA_WIN). BIT-IDENTICAL to _rp: the
// staged bytes ARE the global bytes, same word order (qw.x..qw.w = cq0..cq3), same scale byte
// extraction, same dp4a order + adg + warp_reduce_sum — only WHERE the bytes stage changes.
#define RP_WIN 576   // 512B quant (32x16B) + 64B scale (4x16B)
__device__ __forceinline__ void ca_issue_window_rp(unsigned char* dst,
        const unsigned char* qsrc, const unsigned char* ssrc, int lane) {
    cp_async16_g(dst + lane * 16, qsrc + lane * 16);          // quant: 32 lanes x 16B = 512B
    if (lane < 4) cp_async16_g(dst + 512 + lane * 16, ssrc + lane * 16);  // scale: 4 lanes x 16B = 64B
}
template<int MCOLS, int WROWS, int STAGES>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    int niter = nsb >> 5;                       // dispatch gate: nsb%32==0
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;   // this warp's row0 quant base
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;    // this warp's row0 scale base
    long qstride = (long)nsb64 * 32;            // +1 row in the quant plane
    long sstride = (long)nsb64 * 4;             // +1 row in the scale plane
    __shared__ __align__(16) unsigned char smw[BW24_MMVQ_ROWS][STAGES][WROWS][RP_WIN];
    unsigned char (*ring)[WROWS][RP_WIN] = smw[threadIdx.y];
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < niter) {
            ca_issue_window_rp(&ring[s][0][0], rowq0 + (size_t)s * 512, rows0 + (size_t)s * 64, lane);
            if (WROWS == 2 && has1)
                ca_issue_window_rp(&ring[s][1][0], rowq0 + qstride + (size_t)s * 512,
                                   rows0 + sstride + (size_t)s * 64, lane);
        }
        cp_async_commit();
    }
    for (int it = 0; it < niter; it++) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        int g = it * 32 + lane;
        int s0 = (lane & 1) * 2;
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const unsigned char* wnd = &ring[it % STAGES][r][0];
            int4 qw = *(const int4*)(wnd + lane * 16);
            int cscw = *(const int*)(wnd + 512 + (lane >> 1) * 4);
            int2 wv[2][2];
            float wscale[2];
            wv[0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[sl][0], vb = wv[sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        int itn = it + STAGES - 1;
        if (itn < niter) {
            ca_issue_window_rp(&ring[itn % STAGES][0][0], rowq0 + (size_t)itn * 512,
                               rows0 + (size_t)itn * 64, lane);
            if (WROWS == 2 && has1)
                ca_issue_window_rp(&ring[itn % STAGES][1][0], rowq0 + qstride + (size_t)itn * 512,
                                   rows0 + sstride + (size_t)itn * 64, lane);
        }
        cp_async_commit();
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<4, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpcar2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<4, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<2, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpcar2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<2, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, SPLIT-PLANE WALK-ORDER REPACK (A6, 2026-07-04 — Marlin-style offline
// repack). The GGUF 36B block interleaves a 4B scale word with 32B of quants, so a lane's per-g
// weight read is 5 scattered 4B LDGs at 36B stride (the "18B straggle": 4x LDG.32 quants at a
// 4B-aligned address + 1 scale LDG). The repacked layout splits the tensor into two planes:
//   quant plane: out_f rows x (in_f/64) x 32B  — lane's 16B half at row_q + g*16, PERFECTLY
//                16B-aligned; the warp reads 512B contiguous per g-iter = one LDG.128 wavefront;
//   scale plane: out_f rows x (in_f/64) x 4B   — block's scale word at row_s + (g>>1)*4 (the
//                warp reads 64B contiguous; lane pairs broadcast-share a word).
// Same total bytes (36/block), byte-for-byte the same values — only their ADDRESSES move, so the
// decode (same word order cq0..cq3 + same scale-byte extraction as _pf) is BIT-IDENTICAL per
// (token,row). W points at the repacked tensor base; the scale plane starts at
// out_f*(in_f/64)*32 (32B-multiple -> aligned). row_bytes is unused (kept for ABI parity).
template<int MCOLS, int WROWS>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;         // 32-elem groups
    int nsb64 = in_f >> 6;       // 64-elem NVFP4 blocks
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            // ONE 16B load for the quant half (vs 4x LDG.32 at a 36B-stride address), one 4B
            // scale-word load from the dense plane. Word order cq0..cq3 identical to _pf.
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (WROWS == 2 && r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<2, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<2, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// 8-resident-block twin of b4_rpr2 (r2w8 precedent: wins when the extra residency deletes a
// straggler wave of the halved grid).
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b4_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 rp twins (K=4..7 spec verify T=5..8 on the default split-plane layout).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b8_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- rpsc: _rp + per-warp SMEM SCALE PRESTAGE (2026-07-06). ncu on the 27B verify (b4_rpr2):
// 62.4% long_scoreboard, DRAM 36-52%, reg-limited 7 blocks — latency-bound. The steady loop has
// TWO outstanding global dependencies per (row, g-iter): the 16B quant half and the 4B scale word,
// both streaming from DRAM. This twin coalesced-loads each warp's FULL scale rows (nsb64 words,
// <=272 = 1088B/row) into smem ONCE before the loop, so the loop keeps ONE global dependency (the
// quant stream). No register growth (the unroll-2/rpca occupancy trap does not apply — staging is
// a pointer swap). Same values, same dp4a + warp_reduce_sum order -> BIT-IDENTICAL to _rp per
// (token,row). Dispatch gates: in_f % 512 == 0 && in_f/64 <= RP_MAX_NSB64 (all 27B/9B shapes).
#define RP_MAX_NSB64 272   // in_f <= 17408
template<int MCOLS, int WROWS>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_sc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    __shared__ __align__(16) int ssc[BW24_MMVQ_ROWS][WROWS][RP_MAX_NSB64];
    // prestage this warp's scale rows (warp-private smem -> __syncwarp, no block barrier).
    int n4 = nsb64 >> 2;                    // dispatch gate: nsb64 % 4 == 0
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4);
        int4* dst = (int4*)&ssc[threadIdx.y][r][0];
        for (int i = lane; i < n4; i += 32) dst[i] = src[i];
    }
    __syncwarp();
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = ssc[threadIdx.y][r][sblk];          // smem, no global dependency
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (WROWS == 2 && r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<2, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- rpks: K-SPLIT x2 ACROSS WARP PAIRS (2026-07-06). block (32,4) = TWO warp-pairs; a pair
// owns 2 output rows (same 2 independent weight streams per warp as rpr2), the pair's two warps
// split the k-range in half. grid.x = ceil(out_f/4) — 2x rpr2's blocks with the same regs/thread:
// latency hidden by BLOCK-level parallelism instead of per-thread ILP (the LAW fix; unroll-2 and
// rpca lost by growing registers). Reduce order: per-lane serial accumulation over the chunk's
// g's + warp_reduce_sum per (row,col) — identical WITHIN a chunk to _rp — then ONE cross-warp add
// in fixed chunk order (chunk0 + chunk1) via smem. DETERMINISTIC but NOT bit-identical to _rp
// (k-order differs); verify arbitrates exactness — gates are acceptance parity + argmax MATCH.
// SC=true additionally prestages each warp's scale-row HALF into smem (rpsc mechanism).
// Dispatch gates: in_f % 512 == 0 (nsb%16==0 -> aligned half-plane staging) && nsb64 <= 272.
template<int MCOLS, bool SC>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int pair = threadIdx.y >> 1;            // 0..1: which 2-row group of the block
    int kc = threadIdx.y & 1;               // 0..1: which k-chunk of the pair
    int o0 = (blockIdx.x * 2 + pair) * 2;
    const bool act = o0 < out_f;            // inactive warps still reach __syncthreads
    const bool has1 = act && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    int half = nsb >> 1;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    // per-warp scale half-rows: [warp][row][nsb64/2 words] (chunk kc reads sblk in
    // [kc*half/2, kc*half/2 + half/2); local index = sblk - kc*(half>>1)). Sized 1 when SC=false
    // so the plain rpks twin doesn't pay 4.3KB of dead smem per block.
    __shared__ __align__(16) int ssc[4][2][SC ? RP_MAX_NSB64 / 2 : 1];
    int sbase = kc * (half >> 1);
    if (SC && act) {
        int n4 = (half >> 1) >> 2;          // dispatch gate: (nsb/4) % 4 == 0 (in_f % 512 == 0)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sbase * 4);
            int4* dst = (int4*)&ssc[threadIdx.y][r][0];
            for (int i = lane; i < n4; i += 32) dst[i] = src[i];
        }
        __syncwarp();
    }
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    if (act) {
        int gend = kc * half + half;
        for (int g = kc * half + lane; g < gend; g += 32) {
            int sblk = g >> 1;
            int s0 = (g & 1) * 2;
            int2 wv[2][2][2];
            float wscale[2][2];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
                int4 qw = *qh;
                int cscw = SC ? ssc[threadIdx.y][r][sblk - sbase]
                              : *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
                wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
                wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
                wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
                wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
                wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
                wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            }
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    if (r == 1 && !has1) break;
                    float partial = 0.0f;
                    #pragma unroll
                    for (int sl = 0; sl < 2; sl++) {
                        int base = sl * 4;
                        int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                        int sumi = 0;
                        sumi = dp4a(va.x, aq4[base + 0], sumi);
                        sumi = dp4a(vb.x, aq4[base + 1], sumi);
                        sumi = dp4a(va.y, aq4[base + 2], sumi);
                        sumi = dp4a(vb.y, aq4[base + 3], sumi);
                        partial += wscale[r][sl] * (float)sumi;
                    }
                    acc[r][c] += adg * partial;
                }
            }
        }
    }
    // reduce: butterfly per (row,col) inside each chunk-warp, then chunk0 + chunk1 in FIXED order.
    float asum[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) asum[r][c] = warp_reduce_sum(acc[r][c]);
    __shared__ float part[2][2][MCOLS];     // [pair][row][col], written by the kc==1 warp
    if (kc == 1 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) part[pair][r][c] = asum[r][c];
    }
    __syncthreads();
    if (act && kc == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                y[(size_t)c * out_f + o0 + r] = asum[r][c] + part[pair][r][c];
            }
        }
    }
}
// ---- rpms: M-SPLIT ACROSS WARP PAIRS (2026-07-06). Same occupancy goal as rpks (2x rpr2's
// blocks: grid = ceil(out_f/4), block (32,4) = 2 pairs x 2 rows) WITHOUT touching the k-reduce
// order: the pair's two warps both walk the FULL k-range of the SAME 2 rows but each owns half
// the m columns (warp kc computes cols [kc*MCOLS/2, (kc+1)*MCOLS/2), c>=m masked). Every
// (token,row) dot keeps the reference per-lane serial chain + warp_reduce_sum -> BIT-IDENTICAL
// to _rp. (The rpks e2e self-consistency FAIL taught: verify logits MUST be bit-identical to the
// decode path — the k-order shift moves greedy argmax at tie margins and run-spec FAILs.) The
// twin warp re-reads the same weight bytes in near-lockstep -> L1/L2 serve the second copy; the
// per-warp column work halves and acc/act registers drop (acc[2][MCOLS/2]). No cross-warp
// reduce, no smem, no __syncthreads — warps fully independent. SC=true prestages the pair's
// scale rows to smem (rpsc mechanism, warp-private so no block barrier).
template<int MCOLS, bool SC>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    constexpr int CH = MCOLS / 2;           // columns per warp
    int pair = threadIdx.y >> 1;            // 0..1: which 2-row group of the block
    int kc = threadIdx.y & 1;               // 0..1: which column half
    int o0 = (blockIdx.x * 2 + pair) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int c0 = kc * CH;                       // this warp's first column
    if (c0 >= m) return;                    // whole column half masked
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    __shared__ __align__(16) int ssc[4][2][SC ? RP_MAX_NSB64 : 1];
    if (SC) {
        int n4 = nsb64 >> 2;                // dispatch gate: nsb64 % 4 == 0
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4);
            int4* dst = (int4*)&ssc[threadIdx.y][r][0];
            for (int i = lane; i < n4; i += 32) dst[i] = src[i];
        }
        __syncwarp();
    }
    float acc[2][CH];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < CH; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[2][2][2];
        float wscale[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = SC ? ssc[threadIdx.y][r][sblk]
                          : *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            if (c0 + c >= m) break;
            const signed char* arow = aq + (size_t)(c0 + c) * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)(c0 + c) * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            if (c0 + c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)(c0 + c) * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<4, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<8, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<2, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<4, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<8, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<4, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<8, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<2, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<4, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<8, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ============ SPLIT-PLANE rp twins of the m=1 NVFP4 decode family (A6 integration) ============
// Each is the matching kernel's body with the weight-group loads swapped to the split-plane
// addresses (ONE 16B quant load + one 4B scale word) — identical decode word order (qw.x..qw.w ==
// q4a/q4b of sl=0,1), identical scale-byte extraction, identical dp4a/reduce order per (token,row).

// m>=1 warp-per-row (grid.y = t). Twin of qmatvec_nvfp4_mmvq; also serves decode-exact grid.y=m.
extern "C" __global__ void qmatvec_nvfp4_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* rowq = W + (size_t)o * nsb64 * 32;
    const unsigned char* rows = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int4 qw = *(const int4*)(rowq + (size_t)g * 16);
        int cscw = *(const int*)(rows + (size_t)sblk * 4);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = (sl == 0) ? qw.x : qw.z;
            int q4b = (sl == 0) ? qw.y : qw.w;
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * yscale;
}

// multirow rp body (t from blockIdx.y unless pinned): twin of nvfp4_mmvq_multirow.
template<int RPW, bool PIN_T0>
__device__ __forceinline__ void nvfp4_mmvq_multirow_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = PIN_T0 ? 0 : blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            int4 qw = *(const int4*)(qplane + ((size_t)o * nsb64 + sblk) * 32 + (size_t)(g & 1) * 16);
            int cscw = *(const int*)(splane + ((size_t)o * nsb64 + sblk) * 4);
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int q4a = (sl == 0) ? qw.x : qw.z;
                int q4b = (sl == 0) ? qw.y : qw.w;
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    nvfp4_mmvq_multirow_rp<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}
// DUAL gate+up rp twin (blockIdx.y selects tensor; m==1 asserted host-side like the original).
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_mr2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes, float y0scale, float y1scale) {
    const unsigned char* W = (blockIdx.y == 0) ? W0 : W1;
    float* y = (blockIdx.y == 0) ? y0 : y1;
    float yscale = (blockIdx.y == 0) ? y0scale : y1scale;
    nvfp4_mmvq_multirow_rp<2, true>(W, aq, ad, y, in_f, out_f, 1, row_bytes, yscale);
}

// dp4a rp twin (128-thread two-level reduce, grid (out_f, m)). Twin of qmatvec_nvfp4_dp4a.
extern "C" __global__ void qmatvec_nvfp4_dp4a_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* rowq = W + (size_t)o * nsb64 * 32;
    const unsigned char* rows = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int4 qw = *(const int4*)(rowq + (size_t)g * 16);
        int cscw = *(const int*)(rows + (size_t)sblk * 4);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = (sl == 0) ? qw.x : qw.z;
            int q4b = (sl == 0) ? qw.y : qw.w;
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ============================ k-quant BATCHED weight-resident matvec ============================
// Same structure as nvfp4_mmvq_batched: ONE warp owns ONE output row and walks the weight ONCE,
// decoding each weight group's quant bytes a SINGLE time and dp4a-ing the decoded weight against
// ALL m activation columns (m independent reg accumulators). The weight bytes + decoded ints leave
// HBM/L2 ONCE and serve all m tokens — the m>1 verify/MTP win (vs grid.y=m _dp4a, which re-reads the
// weight m times). y is [m, out_f] token-major. BIT-IDENTICAL per (token,row) to the matching _mmvq
// kernel: the per-element dequant + dp4a order + warp_reduce_sum are lifted verbatim; only the loop
// nest order (group-outer, column-inner) changes, which does not alter any per-(token,row) f32 sum.
// MCOLS is the compile-time batch (2 or 4); m<=MCOLS, the c>=m columns are skipped.

// ----- Q8_0 batched. Per-group reusable: dw + 8 weight ints. Per-column: activation int8 + dp4a. -----
// Row-parameterized body (`o` = the output row this warp owns): LIFTED VERBATIM from the original
// q8_0_mmvq_batched so the per-(token,row) FP chain is unchanged. The plain _b2/_b4/_b8 kernels
// pass the standard blockIdx.x mapping; the FUSED multi-tensor twins below pass the
// block-offset-split mapping (the fused2/fused3 m=1 recipe applied to the batched tier).
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_batched_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);
        const unsigned char* wq = wb + 2;
        int wi[8];                               // decode weight ints ONCE for this block
        #pragma unroll
        for (int k = 0; k < 8; k++) wi[k] = get_int_b2(wq + k * 4);
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y);
}
// ---- Q4_0 batched twins (gemma verify t=2..8): per (token,row) chain BIT-IDENTICAL to
// qmatvec_q4_0_mmvq / _mr2 (same dp4a issue order, same d4*(sumi-8*sums)*d8 accumulate). ----
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* b = wrow + (long)blk * 18;
        float d4 = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs = b + 2;
        int lo[4], hi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
            lo[k] = (int)(raw & 0x0F0F0F0Fu);
            hi[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13, the L1TEX fix): same values, same dp4a order.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            const int al[4] = { a01.x, a01.y, a01.z, a01.w };
            const int ah[4] = { a23.x, a23.y, a23.z, a23.w };
            int sumi = 0, sums = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(lo[k], al[k], sumi);
                sumi = dp4a(hi[k], ah[k], sumi);
                sums = dp4a(0x01010101, al[k], sums);
                sums = dp4a(0x01010101, ah[k], sums);
            }
            acc[c] += d4 * (float)(sumi - 8 * sums) * ad[(size_t)c * nblk + blk];
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
// Q4_0 batched MULTIROW (verify trunk lever): 2 rows/warp — activation int4 loads AND the
// row-independent ones-sums computed ONCE per (col, group), reused across both rows. Per
// (token, row) float chain identical to q4_0_mmvq_batched (d4*(sumi-8*sums)*d8 in g order).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* w0 = W + (long)o0 * row_bytes;
    const unsigned char* w1 = W + (long)(o0 + 1) * row_bytes;
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            const unsigned char* b = w0 + (long)blk * 18;
            const unsigned char* qs = b + 2;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                lo0[k] = (int)(raw & 0x0F0F0F0Fu);
                hi0[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            const unsigned char* b = w1 + (long)blk * 18;
            const unsigned char* qs = b + 2;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                lo1[k] = (int)(raw & 0x0F0F0F0Fu);
                hi1[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(*(const unsigned short*)(w0 + (long)blk * 18));
        float d41 = two ? half_to_float(*(const unsigned short*)(w1 + (long)blk * 18)) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13): the 8 scalar int loads were 4x the L1TEX
            // transactions of the t=1 walk's two 16B loads — L1TEX measured 90% saturated
            // (the b-tier limiter). Same bytes, same order per k — bit-identical.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)c * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { s0 = dp4a(lo0[k], a[k], s0); s0 = dp4a(hi0[k], a[4 + k], s0); }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) { s1 = dp4a(lo1[k], a[k], s1); s1 = dp4a(hi1[k], a[4 + k], s1); }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float v0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = v0;
        if (two) {
            float v1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = v1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

extern "C" __global__ void qmatvec_q4_0_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


extern "C" __global__ void qmatvec_q8_0_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- FUSED Q8_0 BATCHED matvec PAIR/TRIPLE (verify t=2-4 trunk launch-fusion, BW24_SPEC_FUSED_T,
// lane/close35b): the m=1 fused2/fused3 block-offset split applied to the batched weight-resident
// tier. Blocks [0,nb0) compute tensor 0, [nb0,nb0+nb1) tensor 1 (fused3: a third range). Per
// (tensor,token,row) the body is q8_0_mmvq_batched_row VERBATIM with the identical row mapping
// (Q8_0 batched_variant is always "base", ROWS=4) -> BIT-IDENTICAL to the separate _b2/_b4
// launches the verify t=2-4 path otherwise runs via matmul_decode_exact, with ONE shared q8_1
// activation quantize and ONE launch instead of two/three. y per tensor is token-major [m, out_f],
// same as the per-tensor kernels. Targets: 35B wqkv+wqkv_gate (8192/4096), wq/wk/wv (8192/512/512),
// gate_shexp+up_shexp (512/512). -----
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_fused2_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    int nb0 = (out0 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0) { W = W0; y = y0; out_f = out0; }
    else         { W = W1; y = y1; out_f = out1; b -= nb0; }
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * BW24_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<2>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_fused3_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    int nb0 = (out0 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int nb1 = (out1 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; b -= nb0 + nb1; }
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * BW24_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q8_0_mmvq_fused3_b<2>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q8_0_mmvq_fused3_b<4>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}

// ==================== F8-E4M3 (checkpoint-native) warp-per-row MMVQ + batched ====================
// BW24_ST_E4M3 decode path (lane e4m3dec, 2026-07-08): F8-E4M3-origin safetensors projections keep
// their RAW checkpoint e4m3 bytes resident ([out_f, in_f] row-major, row_bytes == in_f) instead of
// the lossy Q8_0 re-encode — the weight side of this dot is EXACT w.r.t. the checkpoint. The
// activation is the SAME q8_1 (aq int8 [m,in] + per-32 f32 ad) every fast decode path rides, so the
// fused norm->quantize producer chain is untouched. Per 32-block:
//     bs   = sum_j f32(e4m3(w[j])) * f32(aq[j])      (fmaf chain, fixed j order 0..31)
//     acc += ad[blk] * bs                            (fmaf, lane-strided blk walk like q8_0_mmvq)
// f32 accumulate throughout (e4m3 max 448 * 127 * 32 fits comfortably). The per-tensor f32
// weight_scale is FUSED at the write (`ws` arg, the NVFP4 macro-scale convention).
//
// EXACTNESS LAW: per (token,row) the body is a pure function of (row bytes, that token's q8_1
// row) — grid.y=m verify launches are bit-identical to the m=1 decode launch by construction,
// and the batched _b2/_b4/_b8 twins below replay the IDENTICAL fmaf chain per column.

// One row x one token: the shared body (bit-contract anchor for the m=1, grid.y=m and batched forms).
__device__ __forceinline__ float e4m3_row_dot(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, int nblk, int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        // 32 e4m3 weight bytes: 2x LDG.128 (wrow is 32B-aligned: base alloc 256B, row stride
        // in_f % 32 == 0). 32 int8 activation: 2x LDG.128 (same as the q8_0 twin).
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            int a = au[k];
            bs = fmaf(wlo.x, (float)(signed char)(a & 0xff), bs);
            bs = fmaf(wlo.y, (float)(signed char)((a >> 8) & 0xff), bs);
            bs = fmaf(whi.x, (float)(signed char)((a >> 16) & 0xff), bs);
            bs = fmaf(whi.y, (float)(a >> 24), bs);   // arithmetic shift: already sign-extended
        }
        acc = fmaf(adrow[blk], bs, acc);
    }
    return acc;
}

extern "C" __global__ void qmatvec_e4m3_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float ws) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                             ad + (size_t)t * nblk, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * ws;
}

// ----- F8-E4M3 batched (b2/b4/b8): ONE warp owns ONE row, weight bytes leave HBM/L2 ONCE for all
// m token columns (the m=2..8 verify/MTP tier — without this the F8 class re-reads its ~GBs of
// weights m times per verify, the known K>=4 spec cliff). Per (token,row) the fmaf chain is the
// e4m3_row_dot body VERBATIM (weights re-converted per column from the SAME registers — cvt is
// deterministic, so the f32 inputs and order are identical) -> bit-identical to grid.y=m _mmvq.
// NOTE: 8-arg signature (no ws) like every other batched kernel — the host launcher applies the
// macro-scale via scale_inplace. -----
template<int MCOLS>
__device__ __forceinline__ void e4m3_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];                 // weight bytes read ONCE for all columns
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float bs = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
                float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
                int a = au[k];
                bs = fmaf(wlo.x, (float)(signed char)(a & 0xff), bs);
                bs = fmaf(wlo.y, (float)(signed char)((a >> 8) & 0xff), bs);
                bs = fmaf(whi.x, (float)(signed char)((a >> 16) & 0xff), bs);
                bs = fmaf(whi.y, (float)(a >> 24), bs);
            }
            acc[c] = fmaf(ad[(size_t)c * nblk + blk], bs, acc[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q4_K batched. Per-group reusable: d_sb, dmin_sb, sc, mn, 8 decoded wpack. Per-column: act + dp4a
// (incl. the per-column sumi_sum = dp4a(0x01010101, a) min-offset term, which depends on activation). -----
template<int MCOLS>
__device__ __forceinline__ void q4k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        bool hi = (grp & 1);
        const int* q4 = (const int*)(qs + chunk * 32);
        int wpack[8];                            // decode the 4-bit weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int raw = q4[k];
            wpack[k] = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_d = 0, sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sumi_d   = dp4a(wpack[k], aq4[k], sumi_d);
                sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d_sb   * (float)((int)sc * sumi_d) * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q5_K batched. Per-group reusable: d_sb, dmin_sb, sc, mn, 8 decoded 5-bit wpack. -----
template<int MCOLS>
__device__ __forceinline__ void q5k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        int wpack[8];                            // decode the 5-bit weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);
            int qh4 = get_int_b2(qh + k * 4);
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;
            wpack[k] = low | (h << 4);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_d = 0, sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sumi_d   = dp4a(wpack[k], aq4[k], sumi_d);
                sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d_sb   * (float)((int)sc * sumi_d)   * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q6_K batched. Per-group reusable: d, scales, 8 decoded signed wpack. Symmetric (no min). -----
template<int MCOLS>
__device__ __forceinline__ void q6k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        int wpack[8];                            // decode the 6-bit signed weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ql4 = get_int_b2(qlh + k * 4 + ql_off);
            int qh4 = get_int_b2(qhh + k * 4);
            int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
            int qhn = (qh4 >> qh_sh) & 0x03030303;
            int vpack = qln | (qhn << 4);
            wpack[k] = __vsubss4(vpack, 0x20202020);
        }
        int sc0 = (int)scn[is0], sc1 = (int)scn[is1];
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi0 = 0, sumi1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k < 4) sumi0 = dp4a(wpack[k], aq4[k], sumi0);
                else       sumi1 = dp4a(wpack[k], aq4[k], sumi1);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d * d8 * ( (float)(sumi0 * sc0) + (float)(sumi1 * sc1) );
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


// ============ k-quant batched TWO-ROWS-PER-WARP variants (2026-07-04, NVFP4 _r2 recipe port) ============
// ncu on the DRAM-cold msweep (9B real shapes, m=4): q4_K b4 long_scoreboard 19.6/issue at DRAM
// 47.7% (L2 weight hit 13%, occupancy 71%), q5_K 16.4/issue at DRAM 38.2% — memory-LATENCY bound
// exactly like the NVFP4 batched family pre-fix (ONE weight wavefront in flight/warp). Same fix:
// each warp owns TWO output rows — 2 independent weight-row streams in flight and the m activation
// columns loaded once, reused across both rows. q6_K gets the template too, but its dominant real
// shape (the 9B lm_head, out_f=248320, 75 waves) measured DRAM 90-91% = BANDWIDTH-bound at the
// wall — build-to-measure only. Q8_0 gets NO r2: its only real batched shapes are the tiny
// out_f=32 ssm_alpha/beta (8-block grids; halving a grid that never fills one SM cannot help).
// BIT-IDENTICAL per (token,row) to the matching base batched kernel: identical scale/min unpack,
// identical wpack decode, identical dp4a order (the per-column sumi_sum is INTEGER and
// row-independent — hoisting it out of the row loop is exact), identical warp_reduce_sum. Only
// the row->warp mapping (grid shape) and cross-row interleave change, both exactness-free.

// ----- Q4_K batched r2 -----
template<int MCOLS>
__device__ __forceinline__ void q4k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        int chunk = grp >> 1;
        bool hi = (grp & 1);
        // decode BOTH rows' weight groups first (both wavefronts issued together).
        float dsb[2], dmn[2];
        int   scv[2], mnv[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 144;
            dsb[r] = half_to_float(*(const unsigned short*)b);
            dmn[r] = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qs     = b + 16;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            scv[r] = sc; mnv[r] = mn;
            const int* q4 = (const int*)(qs + chunk * 32);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int raw = q4[k];
                wpack[r][k] = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            }
        }
        // each token column's activation loaded ONCE, dp4a vs both rows.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi_d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi_d = dp4a(wpack[r][k], aq4[k], sumi_d);
                acc[r][c] += dsb[r] * (float)(scv[r] * sumi_d)   * d8
                           - dmn[r] * (float)(mnv[r] * sumi_sum) * d8;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q4_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q5_K batched r2 -----
template<int MCOLS>
__device__ __forceinline__ void q5k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        float dsb[2], dmn[2];
        int   scv[2], mnv[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 176;
            dsb[r] = half_to_float(*(const unsigned short*)b);
            dmn[r] = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qh = b + 16;
            const unsigned char* qs = b + 48;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            scv[r] = sc; mnv[r] = mn;
            const unsigned char* q = qs + g64 * 32;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int q4  = get_int_b2(q  + k * 4);
                int qh4 = get_int_b2(qh + k * 4);
                int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
                int h   = (qh4 >> hbit) & 0x01010101;
                wpack[r][k] = low | (h << 4);
            }
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi_d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi_d = dp4a(wpack[r][k], aq4[k], sumi_d);
                acc[r][c] += dsb[r] * (float)(scv[r] * sumi_d)   * d8
                           - dmn[r] * (float)(mnv[r] * sumi_sum) * d8;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q5_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q6_K batched r2 (built to MEASURE; the 9B lm_head shape is DRAM-wall-bound, see header) -----
template<int MCOLS>
__device__ __forceinline__ void q6k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        int n   = grp >> 2;
        int run = grp & 3;
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        float dv[2];
        int   sc0v[2], sc1v[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 210;
            const unsigned char* qlh = b + n * 64;
            const unsigned char* qhh = b + 128 + n * 32;
            const signed char*   scn = (const signed char*)(b + 192) + n * 8;
            dv[r] = half_to_float(*(const unsigned short*)(b + 208));
            sc0v[r] = (int)scn[is0]; sc1v[r] = (int)scn[is1];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int ql4 = get_int_b2(qlh + k * 4 + ql_off);
                int qh4 = get_int_b2(qhh + k * 4);
                int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
                int qhn = (qh4 >> qh_sh) & 0x03030303;
                int vpack = qln | (qhn << 4);
                wpack[r][k] = __vsubss4(vpack, 0x20202020);
            }
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < 4) sumi0 = dp4a(wpack[r][k], aq4[k], sumi0);
                    else       sumi1 = dp4a(wpack[r][k], aq4[k], sumi1);
                }
                acc[r][c] += dv[r] * d8 * ( (float)(sumi0 * sc0v[r]) + (float)(sumi1 * sc1v[r]) );
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q6_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// Q8_0 weight x q8_1 activation, int8 dp4a. y[m,out] = sum_blocks d_w*d_a*dp4a(w_qs, a_qs).
// W: block_q8_0 rows (34 bytes/block). aq: int8 [m,in]; ad: f32 [m, in/32].
// grid (out, m); block 128 threads (4 warps), each warp strides the in/32 blocks.
extern "C" __global__ void qmatvec_q8_0_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = tid; blk < nblk; blk += blockDim.x) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);   // weight block scale (2-byte aligned OK)
        const unsigned char* wq = wb + 2;                       // qs: 2-byte aligned -> get_int_b2
        const int4* aq16 = (const int4*)(arow + blk * 32);      // 2x int4 (128-bit), 32-aligned
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// Q4_K decode MMVQ (int8 dp4a). Min-offset via the q8_1 activation-sum term.
// y = sum_subblock [ d*sc*d8*dp4a(nibble,a) - dmin*m*d8*sum(a) ]. d/dmin folded PER sub-block
// (a thread's stripe crosses superblocks). Nibble scheme matches deq_q4_k oracle.
extern "C" __global__ void qmatvec_q4_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;                 // total 32-blocks per row
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        // qs at byte off 16 in a 144B superblock -> 4-byte aligned; chunk*32 keeps it 4-byte aligned.
        const int* q4 = (const int*)(qs + chunk * 32);
        bool hi = (grp & 1);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // nibble-by-shift over 4 packed weights (llama.cpp vmmq style, vecdotq.cuh:514-515):
            // low nibbles for even groups, high nibbles for odd. 0x0F0F0F0F masks all 4 lanes.
            int raw = q4[k];
            int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d) * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// Q6_K decode MMVQ (symmetric, no min). w=(ql|qh<<4)-32 signed; per-16 signed scales; fp16 d.
// Matches deq_q6_k oracle: n=grp>>2 half, run=grp&3, is=run*2+(il>>4).
extern "C" __global__ void qmatvec_q6_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        // ql offset for low/high nibble (run 0/1 use bytes [il], run 2/3 use [il+32]);
        // Stage-A deq_q6_k: run0 qlh[il]&0xF, run1 qlh[il+32]&0xF, run2 qlh[il]>>4, run3 qlh[il+32]>>4.
        // => byte offset +32 on ODD runs (1,3); high nibble on runs >=2 (2,3). The offset is (run&1),
        //    NOT (run>=2) — the old (run>=2) swapped run-1<->run-2 ql bytes (rel 0.34 on Q6_K lm_head).
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);          // true -> high nibble of ql byte
        int qh_sh  = run * 2;             // 0,2,4,6
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // Build the 4 unsigned 6-bit weights (0..63) packed one per byte, then __vsubss4 the
            // -32 across all 4 lanes in one SIMD op (llama.cpp vecdotq.cuh:638). Saturating sub is
            // exact here: vals are 0..63 so result is -32..31, well within int8.
            unsigned int vpack = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int il = k * 4 + e;
                int ql_bits = ql_hi ? (qlh[il + ql_off] >> 4) : (qlh[il + ql_off] & 0xF);
                int qh_bits = (qhh[il] >> qh_sh) & 3;
                unsigned int w = (unsigned int)(ql_bits | (qh_bits << 4));   // 0..63
                vpack |= (w & 0xff) << (e * 8);
            }
            int wpack = __vsubss4((int)vpack, 0x20202020);   // subtract 32 per byte (signed sat)
            int a = aq4[k];
            if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
            else       sumi1 = dp4a(wpack, a, sumi1);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// ===== Q5_K decode MMVQ (int8 dp4a). Unsigned 5-bit weight + min-offset via q8_1 sum. =====
extern "C" __global__ void qmatvec_q5_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        // VECTORIZED unpack (was scalar 4-byte inner loop = ~16 ALU ops/k starving DRAM to 31%).
        // The 4 q bytes (idx=k*4..+3) and 4 qh bytes are contiguous -> one get_int_b2 each (2-aligned:
        // q5_K block=176, qs=b+48, qh=b+16 all even). SIMD-extract: low nibble per byte + bit hbit of
        // qh per byte. BIT-IDENTICAL: same byte->bit e*8 packing, same lowbits|(h<<4) per byte.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);                    // 4 q bytes
            int qh4 = get_int_b2(qh + k * 4);                    // 4 qh bytes
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;                // bit hbit per byte, 0/1
            int wpack = low | (h << 4);                          // per byte 0..31
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d)   * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== Q3_K decode MMVQ (symmetric, signed 3-bit weight, NO min term). =====
// 32-chunk grp covers TWO 16-elem sub-blocks => two scale indices (lo/hi 16).
extern "C" __global__ void qmatvec_q3_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 110;
        const unsigned char* hmask  = b;
        const unsigned char* qs     = b + 32;
        const unsigned char* scbyte = b + 96;
        float d = half_to_float(*(const unsigned short*)(b + 108));
        // unpack 16 6-bit signed scales
        unsigned int aux0 = scbyte[0]|(scbyte[1]<<8)|(scbyte[2]<<16)|(scbyte[3]<<24);
        unsigned int aux1 = scbyte[4]|(scbyte[5]<<8)|(scbyte[6]<<16)|(scbyte[7]<<24);
        unsigned int aux2 = scbyte[8]|(scbyte[9]<<8)|(scbyte[10]<<16)|(scbyte[11]<<24);
        const unsigned int km1=0x03030303u, km2=0x0f0f0f0fu, tmp=aux2;
        unsigned int nA[4]={ (aux0&km2)|(((tmp>>0)&km1)<<4), (aux1&km2)|(((tmp>>2)&km1)<<4),
                             ((aux0>>4)&km2)|(((tmp>>4)&km1)<<4), ((aux1>>4)&km2)|(((tmp>>6)&km1)<<4) };
        signed char sc[16];
        for(int kk=0;kk<4;kk++){ sc[kk*4+0]=(signed char)nA[kk]; sc[kk*4+1]=(signed char)(nA[kk]>>8);
                                 sc[kk*4+2]=(signed char)(nA[kk]>>16); sc[kk*4+3]=(signed char)(nA[kk]>>24); }
        // grp -> half/jiter/shift/m_bit/scale-base. half=grp>>2, jiter=grp&3.
        int half = grp >> 2, jiter = grp & 3;
        int shift = 2 * jiter;
        int m_bit_idx = half * 4 + jiter;
        const unsigned char* q  = qs    + half * 32;   // 32-byte qs run for this half
        const unsigned char* hm = hmask;               // hmask not chunked: index by element directly
        int is_lo = half * 8 + jiter * 2 + 0;          // scale for lo 16 elems
        int is_hi = half * 8 + jiter * 2 + 1;          // scale for hi 16 elems
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumlo = 0, sumhi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = 0; bool hiHalf = (k >= 4);     // k0..3 -> lo16, k4..7 -> hi16
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = k * 4 + e;                   // 0..31 within chunk
                int l = idx & 15;
                int sub = idx >> 4;                    // 0 -> q[l], 1 -> q[l+16]
                int q2 = (q[sub * 16 + l] >> shift) & 3;
                int hb = (hm[sub * 16 + l] & (1 << m_bit_idx)) ? 0 : 4;
                int w = q2 - hb;                       // signed -4..3
                wpack |= (w & 0xff) << (e * 8);
            }
            int a = aq4[k];
            if (!hiHalf) sumlo = dp4a(wpack, a, sumlo);
            else         sumhi = dp4a(wpack, a, sumhi);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)sumlo * (float)((int)sc[is_lo] - 32)
                        + (float)sumhi * (float)((int)sc[is_hi] - 32) );
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== NVFP4 decode MMVQ (codebook->int8 dp4a, symmetric, no min). =====
// 32-elem activation block g covers TWO 16-elem NVFP4 sub-blocks (own UE4M3 scale each).
extern "C" __global__ void qmatvec_nvfp4_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;          // which 64-elem block_nvfp4 (36 bytes)
        int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2, s1 = s0 + 1;
        (void)s1;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        // sub-block s_local=0 -> activation ints aq4[0..3], s_local=1 -> aq4[4..7]
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;       // 8 qs bytes for this sub-block
            // Codebook the 16 packed 4-bit weights via __byte_perm (get_int_from_table_16_d) instead
            // of 16 scalar kvalues_mxfp4_d[] loads — this loop was ALU-bound (19% of BW ceiling).
            // For 4 packed bytes, .x = low-nibble codes (4 int8s packed) = old wlo*, .y = high-nibble
            // codes = old whi*. P1: qss is 4-aligned (row_bytes=(in_f/64)*36 mult of 4; qs=b+4; qss=+s*8)
            // -> single LDG.E.32 each via get_int_b4 (was 4x LDG.E.U8). int2/64-bit NOT safe: rows only
            // 8-aligned when in_f%128==0.
            int q4a = get_int_b4(qss);
            int q4b = get_int_b4(qss + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);  // .x=wlo0 (elems0..3) .y=whi0 (elems8..11)
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);  // .x=wlo1 (elems4..7) .y=whi1 (elems12..15)
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);   // elems 0..3
            sumi = dp4a(vb.x, aq4[base + 1], sumi);   // elems 4..7
            sumi = dp4a(va.y, aq4[base + 2], sumi);   // elems 8..11
            sumi = dp4a(vb.y, aq4[base + 3], sumi);   // elems 12..15
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== IQ4_XS decode MMVQ (OPTIONAL perf path; codebook->int8 dp4a, symmetric, no min). =====
// nibble->position split: low nibbles qs[0..15] -> elems 0..15, high -> elems 16..31.
// ---- MoE EXPERT dp4a DOT BODIES (2026-07-06 dp4a arc; HANDOVER "MoE expert dp4a upgrade") ----
// Per-32-elem-group int dots vs a q8_1 activation, separable block scales OUTSIDE the int dot —
// the q4k/q5k mmvq structure. IQ3_S sign trick ported from llama vec_dot_iq3_s_q8_1
// (vecdotq.cuh:1148): signs expand via __vcmpne4 mask -> XOR-sub negation on the packed grid
// bytes. Layout matches bw24 deq_iq3_s (block 110B: d@0 qs@2 qh@66 signs@74 scales@106).
// Group g = 32 elems; IQ3_S block = 256 elems = 8 groups; IQ4_XS block = 256 elems = 8 groups.
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
// K-QUANT expert dot bodies (2026-07-06): the UD-IQ4_XS 35B mix puts Q3_K/Q4_K/Q6_K experts on
// the tail layers (blk.38-40) — those layers fell to the f32-dequant _dev arm (80us vs 15us
// launches = 8.5% of the fixed-build decode window). Group-g bodies lifted VERBATIM from
// qmatvec_q3_K_dp4a / qmatvec_q4_K_mmvq / qmatvec_q6_K_mmvq (same unpack, same dp4a order,
// same accumulate expression), so per (row,group) the math matches those kernels bit-for-bit.
__device__ __forceinline__ float expert_dot_q3k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    const unsigned char* hmask  = b;
    const unsigned char* qs     = b + 32;
    const unsigned char* scbyte = b + 96;
    float d = half_to_float(*(const unsigned short*)(b + 108));
    unsigned int aux0 = scbyte[0]|(scbyte[1]<<8)|(scbyte[2]<<16)|(scbyte[3]<<24);
    unsigned int aux1 = scbyte[4]|(scbyte[5]<<8)|(scbyte[6]<<16)|(scbyte[7]<<24);
    unsigned int aux2 = scbyte[8]|(scbyte[9]<<8)|(scbyte[10]<<16)|(scbyte[11]<<24);
    const unsigned int km1=0x03030303u, km2=0x0f0f0f0fu, tmp=aux2;
    unsigned int nA[4]={ (aux0&km2)|(((tmp>>0)&km1)<<4), (aux1&km2)|(((tmp>>2)&km1)<<4),
                         ((aux0>>4)&km2)|(((tmp>>4)&km1)<<4), ((aux1>>4)&km2)|(((tmp>>6)&km1)<<4) };
    signed char sc[16];
    for(int kk=0;kk<4;kk++){ sc[kk*4+0]=(signed char)nA[kk]; sc[kk*4+1]=(signed char)(nA[kk]>>8);
                             sc[kk*4+2]=(signed char)(nA[kk]>>16); sc[kk*4+3]=(signed char)(nA[kk]>>24); }
    int half = grp >> 2, jiter = grp & 3;
    int shift = 2 * jiter;
    int m_bit_idx = half * 4 + jiter;
    const unsigned char* q  = qs + half * 32;
    const unsigned char* hm = hmask;
    int is_lo = half * 8 + jiter * 2 + 0;
    int is_hi = half * 8 + jiter * 2 + 1;
    const int* aq4 = (const int*)aqb;
    int sumlo = 0, sumhi = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int wpack = 0; bool hiHalf = (k >= 4);
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx = k * 4 + e;
            int l = idx & 15;
            int sub = idx >> 4;
            int q2 = (q[sub * 16 + l] >> shift) & 3;
            int hb = (hm[sub * 16 + l] & (1 << m_bit_idx)) ? 0 : 4;
            int w = q2 - hb;
            wpack |= (w & 0xff) << (e * 8);
        }
        int a = aq4[k];
        if (!hiHalf) sumlo = dp4a(wpack, a, sumlo);
        else         sumhi = dp4a(wpack, a, sumhi);
    }
    return d * d8 * ( (float)sumlo * (float)((int)sc[is_lo] - 32)
                    + (float)sumhi * (float)((int)sc[is_hi] - 32) );
}
__device__ __forceinline__ float expert_dot_q4k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 144;
    float d_sb    = half_to_float(*(const unsigned short*)b);
    float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs     = b + 16;
    unsigned char sc, mn;
    if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
    else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
           mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
    int chunk = grp >> 1;
    const int* q4 = (const int*)(qs + chunk * 32);
    bool hi = (grp & 1);
    const int* aq4 = (const int*)aqb;
    int sumi_d = 0, sumi_sum = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int raw = q4[k];
        int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
        int a = aq4[k];
        sumi_d   = dp4a(wpack, a, sumi_d);
        sumi_sum = dp4a(0x01010101, a, sumi_sum);
    }
    return d_sb * (float)((int)sc * sumi_d) * d8 - dmin_sb * (float)((int)mn * sumi_sum) * d8;
}
__device__ __forceinline__ float expert_dot_q6k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 210;
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char*   scales = (const signed char*)(b + 192);
    float d = half_to_float(*(const unsigned short*)(b + 208));
    int n   = grp >> 2;
    int run = grp & 3;
    const unsigned char* qlh = ql + n * 64;
    const unsigned char* qhh = qh + n * 32;
    const signed char*   scn = scales + n * 8;
    const int* aq4 = (const int*)aqb;
    int is0 = run * 2 + 0;
    int is1 = run * 2 + 1;
    int sumi0 = 0, sumi1 = 0;
    int ql_off = (run & 1) ? 32 : 0;
    int ql_hi  = (run >= 2);
    int qh_sh  = run * 2;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int il = k * 4;
        int qlw = get_int_b2(qlh + ql_off + il);
        int qhw = get_int_b2(qhh + il);
        int qln = ql_hi ? ((qlw >> 4) & 0x0F0F0F0F) : (qlw & 0x0F0F0F0F);
        int qhn = (qhw >> qh_sh) & 0x03030303;
        int vpack = qln | (qhn << 4);
        int wpack = __vsubss4(vpack, 0x20202020);
        int a = aq4[k];
        if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
        else       sumi1 = dp4a(wpack, a, sumi1);
    }
    return d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
}

// group-dispatching wrapper: qtype -> dot body (compile-time-hot switch, bodies inlined)
// NVFP4 expert dot (2026-07-07, MiniMax-M3): group-g body lifted VERBATIM from qmatvec_nvfp4_mmvq
// (same 36B GGUF block walk, same get_int_from_table_16_d + ue4m3 sub-scales, same dp4a order) —
// bit-identical per (row, group) to the m=1 kernel. The per-expert weight_scale_2 macro is applied
// by the CALLER (ffn_act_scaled / axpy fold), matching the dense-path contract.
__device__ __forceinline__ float expert_dot_nvfp4_g(const unsigned char* wrow, int g,
                                                    const signed char* aqb, float d8) {
    int sblk = g >> 1;
    int whichHalf = g & 1;
    const unsigned char* b = wrow + (long)sblk * 36;
    const unsigned char* d_bytes = b;
    const unsigned char* qs = b + 4;
    int s0 = whichHalf * 2;
    const int* aq4 = (const int*)aqb;
    float partial = 0.0f;
    #pragma unroll
    for (int sl = 0; sl < 2; sl++) {
        int s = s0 + sl;
        const unsigned char* qss = qs + s * 8;
        int q4a = get_int_b4(qss);
        int q4b = get_int_b4(qss + 4);
        int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
        int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
        int base = sl * 4;
        int sumi = 0;
        sumi = dp4a(va.x, aq4[base + 0], sumi);
        sumi = dp4a(vb.x, aq4[base + 1], sumi);
        sumi = dp4a(va.y, aq4[base + 2], sumi);
        sumi = dp4a(vb.y, aq4[base + 3], sumi);
        partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
    }
    return d8 * partial;
}

// ---- Q4_0 group dot (gemma4 QAT experts): one 18B block per 32-elem group; the exact
// qmatvec_q4_0_mmvq accumulation chain (dp4a nibbles + inline ones-sum, d*(sumi-8*sums)*d8). ----
__device__ __forceinline__ float expert_dot_q4_0_g(const unsigned char* wrow, int g,
                                                   const signed char* aqb, float d8) {
    const unsigned char* b = wrow + (long)g * 18;
    float d4 = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs = b + 2;
    const int* aq4 = (const int*)aqb;
    int sumi = 0, sums = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t raw;
        memcpy(&raw, qs + 4 * k, 4);
        int lo = (int)(raw & 0x0F0F0F0Fu);
        int hi = (int)((raw >> 4) & 0x0F0F0F0Fu);
        int a_lo = aq4[k];
        int a_hi = aq4[4 + k];
        sumi = dp4a(lo, a_lo, sumi);
        sumi = dp4a(hi, a_hi, sumi);
        sums = dp4a(0x01010101, a_lo, sums);
        sums = dp4a(0x01010101, a_hi, sums);
    }
    return d4 * (float)(sumi - 8 * sums) * d8;
}

__device__ __forceinline__ float expert_dot_g(int qtype, const unsigned char* wrow, int g,
                                              const signed char* aqb, float d8) {
    if (qtype == QT_IQ3_S)  return expert_dot_iq3s_g(wrow, g, aqb, d8);
    if (qtype == QT_IQ4_XS) return expert_dot_iq4xs_g(wrow, g, aqb, d8);
    if (qtype == QT_Q3_K)   return expert_dot_q3k_g(wrow, g, aqb, d8);
    if (qtype == QT_Q4_K)   return expert_dot_q4k_g(wrow, g, aqb, d8);
    if (qtype == QT_Q6_K)   return expert_dot_q6k_g(wrow, g, aqb, d8);
    if (qtype == QT_NVFP4)  return expert_dot_nvfp4_g(wrow, g, aqb, d8);
    if (qtype == QT_Q4_0)   return expert_dot_q4_0_g(wrow, g, aqb, d8);
    return 0.0f; // caller gates on supported qtypes
}

// ---- IQ4_XS WIDE-LOAD group dot (down8 lane 2026-07-08) ----
// WHY: the 35B down kernel (w8h2) runs at 47% of the byte-math wall (11.1us vs 5.2us) — NOT
// bandwidth-bound. The issue count is: expert_dot_iq4xs_g spends 16 LDG.U8 (qs bytes) + 32
// divergent byte lookups into kvalues_iq4nl_d + ~60 shift/or pack ALU per 32-elem group,
// against just 8 dp4a of real work. This body computes the SAME packed ints from the SAME
// bytes with 2 LDG.64 (qs) + 1 LDG.64 (d/sh/sl header) + 4 uniform u32 table words through
// get_int_from_table_16_d (~5 byte_perm per int pair — the llama.cpp vecdotq recipe, already
// proven bit-clean on the NVFP4/MXFP4 path here).
// BIT-IDENTITY: value-level, not order-level — wlo/whi/scale/d_sb are the exact same values
// expert_dot_iq4xs_g produces (little-endian byte extraction == the scalar byte loads; .x/.y
// of the table lookup == the low/high-nibble scalar packs), the dp4a issue order is unchanged
// (lo,hi per k), and the closing float expression is the same. sumi is exact integer math.
// ALIGNMENT: block=136B and every IQ4_XS row/expert stride here is a multiple of 8, so b is
// 8-aligned whenever the expert slab base is (cudaMalloc slabs are 256B-aligned). A warp-
// uniform guard falls back to the scalar body for any exotic base — same values either way.
__device__ __forceinline__ float expert_dot_iq4xs_g_v(const unsigned char* wrow, int g,
                                                      const signed char* aqb, float d8) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    if (((unsigned long long)b & 7ull) != 0ull)
        return expert_dot_iq4xs_g(wrow, g, aqb, d8);      // non-8-aligned slab: scalar body
    uint2 hdr = *(const uint2*)b;                         // d(2B) | sh(2B) | sl(4B), one LDG.64
    float d_sb = half_to_float((unsigned short)(hdr.x & 0xffffu));
    unsigned short sh = (unsigned short)(hdr.x >> 16);
    // sl[ib>>1] is byte (ib>>1) of hdr.y (little-endian); fold the byte+nibble shifts.
    int ls = ((hdr.y >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    int scale = ls - 32;
    const int2* qs2 = (const int2*)(b + 8 + ib * 16);     // (8 + ib*16) % 8 == 0 -> 8-aligned
    int2 q01 = qs2[0], q23 = qs2[1];
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int2 v0 = get_int_from_table_16_d(q01.x, kvalues_iq4nl_d);
    int2 v1 = get_int_from_table_16_d(q01.y, kvalues_iq4nl_d);
    int2 v2 = get_int_from_table_16_d(q23.x, kvalues_iq4nl_d);
    int2 v3 = get_int_from_table_16_d(q23.y, kvalues_iq4nl_d);
    int sumi = 0;                                          // same lo,hi dp4a order per k as scalar
    sumi = dp4a(v0.x, aLo[0], sumi); sumi = dp4a(v0.y, aHi[0], sumi);
    sumi = dp4a(v1.x, aLo[1], sumi); sumi = dp4a(v1.y, aHi[1], sumi);
    sumi = dp4a(v2.x, aLo[2], sumi); sumi = dp4a(v2.y, aHi[2], sumi);
    sumi = dp4a(v3.x, aLo[3], sumi); sumi = dp4a(v3.y, aHi[3], sumi);
    return d_sb * (float)(scale * sumi) * d8;
}
// ---- Q4_0 WIDE-LOAD group dot (gemma A4B lane 2026-07-12) ----
// WHY: expert_dot_q4_0_g issues 1 U16 + 16 byte-class loads per 18B block (the q4_0 stride
// is 2 mod 4 — nothing aligns); the gemma MoE pair reads DRAM 50% / SM 40% with the load
// chain as the critical path (same disease the trunk q4rp split-plane cured). This body
// reads the SAME 18 bytes as 6 aligned LDG.32 + funnelshift extraction (REVISION 4b recipe,
// SASS-proven on fa_v4_stage_k): same bytes -> same lo/hi ints -> same dp4a order ->
// bit-identical result.
// OVERREAD: the aligned 24B window reads up to 6B past the block — within a row/slab that is
// the next block's bytes; the LAST block of an expert slab needs the 8B tail pad at the moe
// alloc site (see moe slab alloc).
__device__ __forceinline__ float expert_dot_q4_0_g_v(const unsigned char* wrow, int g,
                                                     const signed char* aqb, float d8) {
    const unsigned char* b = wrow + (long)g * 18;
    const unsigned sh8 = ((unsigned)(size_t)b & 3u) * 8u;
    const uint32_t* ap = (const uint32_t*)((size_t)b & ~(size_t)3);
    uint32_t w0 = ap[0], w1 = ap[1], w2 = ap[2], w3 = ap[3], w4 = ap[4], w5 = ap[5];
    uint32_t s0 = __funnelshift_r(w0, w1, sh8);   // bytes b[0..3]
    uint32_t s1 = __funnelshift_r(w1, w2, sh8);   // b[4..7]
    uint32_t s2 = __funnelshift_r(w2, w3, sh8);   // b[8..11]
    uint32_t s3 = __funnelshift_r(w3, w4, sh8);   // b[12..15]
    uint32_t s4 = __funnelshift_r(w4, w5, sh8);   // b[16..19] (2B past the block)
    float d4 = half_to_float((unsigned short)(s0 & 0xffffu));
    // qs word k = bytes b[2+4k .. 5+4k] — one more 16-bit funnel over the byte stream.
    uint32_t q0 = __funnelshift_r(s0, s1, 16), q1 = __funnelshift_r(s1, s2, 16);
    uint32_t q2 = __funnelshift_r(s2, s3, 16), q3 = __funnelshift_r(s3, s4, 16);
    const uint32_t qw[4] = { q0, q1, q2, q3 };
    const int* aq4 = (const int*)aqb;
    int sumi = 0, sums = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int lo = (int)(qw[k] & 0x0F0F0F0Fu);
        int hi = (int)((qw[k] >> 4) & 0x0F0F0F0Fu);
        int a_lo = aq4[k];
        int a_hi = aq4[4 + k];
        sumi = dp4a(lo, a_lo, sumi);
        sumi = dp4a(hi, a_hi, sumi);
        sums = dp4a(0x01010101, a_lo, sums);
        sums = dp4a(0x01010101, a_hi, sums);
    }
    return d4 * (float)(sumi - 8 * sums) * d8;
}

// qtype wrapper: IQ4_XS and Q4_0 take the wide-load bodies; every other qtype = expert_dot_g
// verbatim.
__device__ __forceinline__ float expert_dot_g_v(int qtype, const unsigned char* wrow, int g,
                                                const signed char* aqb, float d8) {
    if (qtype == QT_IQ4_XS) return expert_dot_iq4xs_g_v(wrow, g, aqb, d8);
    if (qtype == QT_Q4_0)   return expert_dot_q4_0_g_v(wrow, g, aqb, d8);
    return expert_dot_g(qtype, wrow, g, aqb, d8);
}

// ---- DECODE-ONCE weight-group extractors (the MMQ tile-decode, split from the dp4a) ----
// The em/dot bodies above re-dequant the weight group on every (group,token) call; the compiler
// can't hoist that across an unrolled token loop (proven NEUTRAL, rung 2). These split the WEIGHT
// decode from the activation dp4a: decode fills wq[8] (32 int8 weight quants packed as 8 int32,
// EXACTLY the values dp4a'd inside expert_dot_*) + a per-group (fscale, iscale). The reuse kernel
// then dp4a's each pre-decoded group against MANY tokens. FP-ORDER: contrib is computed as
// `fscale * (float)(iscale * sumi) * d8` — byte-identical to expert_dot_iq3s_g (iscale=1 =>
// fscale=db) and expert_dot_iq4xs_g (iscale=scale => fscale=d_sb). Per-group accumulate order is
// unchanged, so BW24_MOE_GATE byte-identity holds vs the pair-major/sequential paths.
__device__ __forceinline__ void expert_decode_iq3s_g(const unsigned char* wrow, int g,
                                                     int wq[8], int* iscale, float* fscale) {
    int sblk = g >> 3, ib32 = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    float d = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs    = b + 2  + ib32 * 8;
    unsigned char qh           = b[66 + ib32];
    const unsigned char* signs = b + 74 + ib32 * 4;
    const unsigned char* scales= b + 106;
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    *fscale = d * (1.0f + 2.0f * (float)sc_nib);
    *iscale = 1;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
        unsigned char sb = signs[l0 / 2];
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        wq[l0 + 0] = __vsub4(gl ^ signs0, signs0);
        wq[l0 + 1] = __vsub4(gh ^ signs1, signs1);
    }
}
__device__ __forceinline__ void expert_decode_iq4xs_g(const unsigned char* wrow, int g,
                                                      int wq[8], int* iscale, float* fscale) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    *fscale = half_to_float(*(const unsigned short*)b);
    unsigned short sh = *(const unsigned short*)(b + 2);
    const unsigned char* sl = b + 4;
    const unsigned char* qs = b + 8 + ib * 16;
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    *iscale = ls - 32;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        wq[k]   = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
        wq[k+4] = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
    }
}
// NOTE the int-lane pairing: IQ3_S packs [grid_l,grid_h] interleaved (wq[0..7] = l0=0,0,2,2,4,4,6,6)
// and the activation int order in aq matches (aq4[l0], aq4[l0+1]); IQ4_XS packs [lo x4, hi x4] and
// the activation is aLo[0..3] then aHi[0..3]. So the dp4a token loop must feed activation ints in
// the SAME split for each qtype. We store the activation-int layout choice per qtype via a flag.
// Q4_0 decode-once: fold the -8 offset INTO the int weights ((nib-8) in [-8,7], __vsub4) —
// the group int sum then equals the _em chain's (sumi - 8*sums) EXACTLY (integer identity),
// and the float chain fscale*(iscale*sumi)*d8 == d4*(sumi-8*sums)*d8 bit-for-bit.
__device__ __forceinline__ void expert_decode_q4_0_g(const unsigned char* wrow, int g,
                                                     int wq[8], int* iscale, float* fscale) {
    const unsigned char* b = wrow + (long)g * 18;
    *fscale = half_to_float(*(const unsigned short*)b);
    *iscale = 1;
    const unsigned char* qs = b + 2;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
        wq[k]     = __vsub4((int)(raw & 0x0F0F0F0Fu), 0x08080808);
        wq[4 + k] = __vsub4((int)((raw >> 4) & 0x0F0F0F0Fu), 0x08080808);
    }
}

__device__ __forceinline__ void expert_decode_g(int qtype, const unsigned char* wrow, int g,
                                               int wq[8], int* iscale, float* fscale) {
    if (qtype == QT_IQ3_S)  { expert_decode_iq3s_g(wrow, g, wq, iscale, fscale); return; }
    if (qtype == QT_Q4_0)   { expert_decode_q4_0_g(wrow, g, wq, iscale, fscale); return; }
    expert_decode_iq4xs_g(wrow, g, wq, iscale, fscale);
}
// dp4a a pre-decoded weight group (wq[8]) against one token's 32 activation int8 (aqb) with the
// qtype's int pairing. IQ3_S: sequential aq4[0..7]; IQ4_XS: aLo[0..3]=aqb[0..15], aHi[0..3]=aqb[16..31]
// interleaved as (wq[k]*aLo[k], wq[k+4]*aHi[k]) — matches expert_dot_iq4xs_g's dp4a issue order.
__device__ __forceinline__ int expert_dp4a_group(int qtype, const int wq[8], const signed char* aqb) {
    const int* a = (const int*)aqb;
    int sumi = 0;
    if (qtype == QT_IQ3_S) {
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wq[k], a[k], sumi);
    } else { // IQ4_XS: lo half then hi half
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            sumi = dp4a(wq[k],   a[k],     sumi);
            sumi = dp4a(wq[k+4], a[k + 4], sumi);
        }
    }
    return sumi;
}

// q8_1-activation MoE expert matvec (warp-per-row like mmvq): the staged/sequential expert path
// upgrade — replaces the 256-thread f32-dequant qmatvec_f32 (Stage-A) for IQ3_S/IQ4_XS experts.
// FP-ORDER NOTE: different reduction than qmatvec_f32 (int dp4a + per-group f32 accumulate,
// 32-lane warp tree) — logits SHIFT; argmax/run-gen/stream-identity gates arbitrate, and the
// fused _q8 twins below MUST ship in the same commit (BW24_MOE_GATE byte-identity pair contract).
extern "C" __global__ void qmatvec_expert_q8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, int qtype, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g(qtype, wrow, g, arow + (size_t)g * 32, adrow[g]);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ---- MoE PREFILL PAIR-BATCH kernels (2026-07-06, the 16x pp hole) ----
// ONE launch per (proj, layer) covers ALL (token, expert) routed pairs: grid.y = pair index,
// grid.x tiles the expert-FFN rows (BW24_MMVQ_ROWS warps/block, warp-per-row). Per pair the body
// is qmatvec_expert_q8 verbatim (same expert_dot_g order per (pair,row) — bit-identity class).
// Replaces the per-expert loop (256 experts x 3-4 launches x tiny m_e = the 1000+ launch/layer
// prefill wall; llama's fused MoE MMQ analog). Inputs: pair_tok[p] (activation row), pair_ex[p]
// (expert id -> device slab ptr table like _dev), q8_1 activations for ALL T tokens.
extern "C" __global__ void moe_pairs_matvec_q8(
        const unsigned long long* __restrict__ table,   // [3, n_expert] slab base ptrs
        int proj,                                        // 0=gate 1=up 2=down (table row)
        const int* __restrict__ pair_tok,                // [n_pairs]
        const int* __restrict__ pair_ex,                 // [n_pairs]
        const signed char* __restrict__ aq,              // [T, in_f] q8_1 (token-major)
        const float* __restrict__ ad,                    // [T, in_f/32]
        float* __restrict__ y,                           // [n_pairs, out_f]
        int in_f, int out_f, int n_expert, int n_pairs, int qtype, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int pr = blockIdx.y;
    if (o >= out_f || pr >= n_pairs) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int tok = pair_tok[pr];
    int ex  = pair_ex[pr];
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float*       adrow = ad + (size_t)tok * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g(qtype, wrow, g, arow + (size_t)g * 32, adrow[g]);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)pr * out_f + o] = acc;
}
// silu(gate)*up over the pair-major activation buffers (gate/up both [n_pairs, n_ff]).
// EXPERT-MAJOR variant (rung 2): CSR over experts — block = (expert-segment e, row-tile); the
// warp loads each weight GROUP once into registers and dp4a's it against ALL of expert e's
// tokens (weight reuse across the token group = llama-MMQ's core win; the pair-major kernel
// re-read the weight per pair). Same expert_dot_g per (pair,row) — bit-identical output order.
// ex_off: [n_active+1] CSR into ex_pairs (pair ids grouped by expert); ex_ids: [n_active].
extern "C" __global__ void moe_pairs_matvec_q8_em(
        const unsigned long long* __restrict__ table, int proj,
        const int* __restrict__ ex_ids, const int* __restrict__ ex_off,
        const int* __restrict__ ex_pairs, const int* __restrict__ pair_tok,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y,
        int in_f, int out_f, int n_expert, int n_active, int qtype, long row_bytes) {
    int seg = blockIdx.y;                 // active-expert segment
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (seg >= n_active || o >= out_f) return;
    int lane = threadIdx.x;
    int ex = ex_ids[seg];
    int lo = ex_off[seg], hi = ex_off[seg + 1];
    int nsb = in_f >> 5;
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    // accumulators for up to 16 tokens per pass (register cap); loop passes if more.
    for (int base = lo; base < hi; base += 16) {
        int cnt = min(16, hi - base);
        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            // weight group decoded ONCE (expert_dot_g re-dequants per call — acceptable: the
            // dp4a-int weight ints stay in L1/registers via the compiler across the token loop
            // when cnt is unrolled; the HBM read happens once per g per row-tile pass).
            #pragma unroll 4
            for (int i = 0; i < cnt; i++) {
                int pr = ex_pairs[base + i];
                int tok = pair_tok[pr];
                acc[i] += expert_dot_g(qtype, wrow, g,
                                       aq + (size_t)tok * in_f + (size_t)g * 32,
                                       ad[(size_t)tok * nsb + g]);
            }
        }
        #pragma unroll
        for (int i = 0; i < cnt; i++) {
            float v = warp_reduce_sum(acc[i]);
            if (lane == 0) y[(size_t)ex_pairs[base + i] * out_f + o] = v;
        }
    }
}

// DECODE-ONCE expert-major MMQ (rung 3): same CSR shape as _em, but the weight group is dequanted
// ONCE per (row, group) via expert_decode_g, then dp4a'd against every token of the expert segment.
// This is the actual MMQ win the _em kernel's comment CLAIMED but did not deliver (expert_dot_g
// re-decoded per token — proven NEUTRAL). Here the decode cost amortizes over the token group.
// FP-ORDER: per-group accumulate `acc[i] += fscale*(float)(iscale*sumi)*d8` in the SAME g-strided
// order as _em/pair-major -> byte-identical logits (BW24_MOE_GATE pair contract holds).
extern "C" __global__ void moe_pairs_matvec_q8_dec(
        const unsigned long long* __restrict__ table, int proj,
        const int* __restrict__ ex_ids, const int* __restrict__ ex_off,
        const int* __restrict__ ex_pairs, const int* __restrict__ pair_tok,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y,
        int in_f, int out_f, int n_expert, int n_active, int qtype, long row_bytes) {
    int seg = blockIdx.y;
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    if (seg >= n_active || o >= out_f) return;
    int lane = threadIdx.x;
    int ex = ex_ids[seg];
    int lo = ex_off[seg], hi = ex_off[seg + 1];
    int nsb = in_f >> 5;
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    for (int base = lo; base < hi; base += 32) {
        int cnt = min(32, hi - base);
        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            int wq[8]; int iscale; float fscale;
            expert_decode_g(qtype, wrow, g, wq, &iscale, &fscale);  // ONCE per (row, group)
            #pragma unroll 4
            for (int i = 0; i < cnt; i++) {
                int tok = pair_tok[ex_pairs[base + i]];
                int sumi = expert_dp4a_group(qtype, wq, aq + (size_t)tok * in_f + (size_t)g * 32);
                acc[i] += fscale * (float)(iscale * sumi) * ad[(size_t)tok * nsb + g];
            }
        }
        #pragma unroll
        for (int i = 0; i < cnt; i++) {
            float v = warp_reduce_sum(acc[i]);
            if (lane == 0) y[(size_t)ex_pairs[base + i] * out_f + o] = v;
        }
    }
}

extern "C" __global__ void moe_pairs_silu_mul(
        const float* __restrict__ gate, const float* __restrict__ up,
        float* __restrict__ act, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; act[i] = (g / (1.0f + expf(-g))) * up[i]; }
}
// gemma4 GELU twin of moe_pairs_silu_mul (gelu_tanh_mul_f32 expression).
extern "C" __global__ void moe_pairs_gelu_mul(
        const float* __restrict__ gate, const float* __restrict__ up,
        float* __restrict__ act, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[i] = 0.5f * x * (1.0f + th) * up[i];
    }
}
// scatter: moe_out[tok] += w[pr] * y_down[pr] — slot-ORDERED per token for bit-identity with the
// sequential axpy chain: one block per (token, col-tile); walks the token's pairs in SLOT order
// via the per-token pair list (tok_pairs CSR built on host: for each token its n_used pair ids
// in slot order).
extern "C" __global__ void moe_pairs_scatter(
        const float* __restrict__ y_down,               // [n_pairs, n_embd]
        const float* __restrict__ pair_w,               // [n_pairs]
        const int* __restrict__ tok_pair_off,            // [T+1] CSR offsets
        const int* __restrict__ tok_pair_ids,            // [n_pairs] pair ids, slot-ordered per token
        float* __restrict__ moe_out,                     // [T, n_embd]
        int n_embd) {
    int tok = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) return;
    int lo = tok_pair_off[tok], hi = tok_pair_off[tok + 1];
    float acc = 0.0f;
    for (int i = lo; i < hi; i++) {
        int pr = tok_pair_ids[i];
        acc = __fmaf_rn(pair_w[pr], y_down[(size_t)pr * n_embd + c], acc);
    }
    moe_out[(size_t)tok * n_embd + c] = acc;
}

extern "C" __global__ void qmatvec_iq4_XS_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, ib = g & 7;
        const unsigned char* b = wrow + (long)sblk * 136;
        float d_sb = half_to_float(*(const unsigned short*)b);
        unsigned short sh = *(const unsigned short*)(b + 2);
        const unsigned char* sl = b + 4;
        const unsigned char* qs = b + 8 + ib * 16;
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        int scale = ls - 32;
        const signed char* aqb = arow + (size_t)g * 32;
        const int* aLo = (const int*)(aqb);        // elems 0..15
        const int* aHi = (const int*)(aqb + 16);   // elems 16..31
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
        acc += d_sb * (float)(scale * sumi) * adrow[g];
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// y[m,out] = x[m,in] @ W[out,in]^T. W quantized rows of `row_bytes` each.
// grid: (out, m); block: 256 threads reduce over `in`.
extern "C" __global__ void qmatvec_f32(
        const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, int m, int qtype, long row_bytes) {
    int o = blockIdx.x;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    const uint8_t* wrow = W + (long)o * row_bytes;
    const float* xrow = x + (long)t * in_f;
    float acc = 0.0f;
    if (qtype == QT_NVFP4_RP) {
        // split-plane NVFP4: same per-element value/product order as deq_nvfp4 -> bit-identical.
        int nsb64 = in_f >> 6;
        const uint8_t* qrow = W + (size_t)o * nsb64 * 32;
        const uint8_t* srow = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
        for (int i = tid; i < in_f; i += blockDim.x) {
            int blk = i >> 6, jj = i & 63;
            int s = jj >> 4, within = jj & 15;
            int byte = qrow[blk * 32 + s * 8 + (within & 7)];
            int code = (within < 8) ? (byte & 0xF) : (byte >> 4);
            acc += (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(srow[blk * 4 + s]) * xrow[i];
        }
    } else
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qtype, wrow, i) * xrow[i];
    // block reduce
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(long)t * out_f + o] = v;
    }
}

// ================================================================================================
// STAGE-2 GROUPED DECODE (2026-07-04): single-launch 8-expert MoE matvecs for m=1.
//
// The sequential decode path launches 8 experts x (gate,up,silu,down,axpy) = 40 kernels per MoE
// layer per token, each a tiny m=1 matvec (~5 us) — 2533 launches/token total on the 35B, host
// launch time ~7.9 ms/tok vs 11.7 ms/tok GPU time (nsys 2026-07-04). These two kernels fold one
// layer's routed-expert FFN into TWO launches via expert-pointer indirection (the SLRU cache slots
// are fixed-address, so the 8 weight pointers are stable for the whole launch).
//
// BIT-IDENTITY CONTRACT (vs the sequential qmatvec_f32 + silu_mul_f32 + axpy_f32 chain):
//  - each dot reproduces qmatvec_f32's EXACT reduction: same 256-thread striding over in_f, same
//    warp shuffle tree, same s[32] two-level reduce. Identical partial-sum order => identical f32.
//  - the SiLU epilogue is silu_mul_f32's expression on the SAME dot values (f32 store/load of the
//    intermediates is exact, so register-passing them is bit-identical).
//  - the down epilogue reproduces the 8 sequential axpy_f32 accumulations: acc starts 0.0 (the
//    e.zeros moe_out) and chains __fmaf_rn(w[j], y_j, acc) in slot order j=0..7 — the same FMA
//    axpy_f32 compiles to (the A2 slot-scheme argument, byte-identity-gated there).
// ================================================================================================

typedef struct { const unsigned char* p[8]; } wptr8_t;
typedef struct { float v[8]; } f32x8_t;

// One MoE layer's gate+up+SiLU for all 8 routed experts of ONE token in ONE launch.
// act[j*n_ff + o] = silu(gate_j[o] . x) * (up_j[o] . x). grid: (n_ff, n_used); block: 256.
extern "C" __global__ void moe_gate_up_silu8_f32(
        wptr8_t gp, wptr8_t up, const float* __restrict__ x, float* __restrict__ act,
        int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row 0..n_ff-1
    int j = blockIdx.y;              // routed-expert slot 0..n_used-1
    int tid = threadIdx.x;
    __shared__ float s[32];
    __shared__ float g_final;
    // ---- gate dot: EXACT qmatvec_f32 structure ----
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    float acc = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt_g, grow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) g_final = v;
    }
    __syncthreads();                 // s + g_final ready; s reused below
    // ---- up dot: same structure ----
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float acc2 = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc2 += deq(qt_u, urow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc2;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) {
            float g = g_final;
            // silu_mul_f32's exact expression on the exact dot values.
            act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * v;
        }
    }
}

// ---- dp4a _q8 TWINS of the fused MoE kernels (matched pair with qmatvec_expert_q8) ----
// Same grid/block/slot-order/silu expression as the _f32 versions; ONLY the dot changes:
// warp-per-row int dp4a vs the q8_1 activation (aq/ad), block=(32, ROWS) covering n_ff rows
// like the f32 version's grid. Reduction = 32-lane warp tree per row (matches expert_q8).
extern "C" __global__ void moe_gate_up_silu8_q8(
        wptr8_t gp, wptr8_t up, const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act, int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row
    int j = blockIdx.y;              // routed slot
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,j)
    int nsb = in_f >> 5;
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
// act is quantized per-slot by the caller (aq2/ad2 hold n_used rows of q8_1).
extern "C" __global__ void moe_down8_fma_q8(
        wptr8_t dp, f32x8_t w, const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g]);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(w.v[j], acc, chain);
    }
    if (lane == 0) dst[o] = chain;
}

// One MoE layer's down-proj + weighted accumulation for all 8 routed experts in ONE launch.
// dst[o] = fma(w[7], y_7[o], ... fma(w[0], y_0[o], 0.0f)) where y_j = W_down_j @ act_j.
// Reproduces zeros(moe_out) + 8 sequential axpy_f32 in slot order. grid: (out_f); block: 256.
extern "C" __global__ void moe_down8_fma_f32(
        wptr8_t dp, f32x8_t w, const float* __restrict__ act, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s[32];
    float chain = 0.0f;              // tid 0's slot-ordered accumulator (other threads' unused)
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const float* xrow = act + (size_t)j * in_f;
        float acc = 0.0f;
        for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt, wrow, i) * xrow[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            // slot-ordered FMA chain == the sequential axpy_f32 accumulation (see header).
            if (tid == 0) chain = __fmaf_rn(w.v[j], v, chain);
        }
        __syncthreads();             // s[] reused next iteration
    }
    if (tid == 0) dst[o] = chain;
}

// ================================================================================================
// LAUNCH-STRUCTURE STAGE 3 (2026-07-05): DEVICE-SIDE ROUTED DISPATCH for fully-resident layers.
//
// The stage-1 router still pays ONE DtoH + stream sync per MoE layer per token (~36us of host
// stall x 40 layers = the largest non-kernel slice of the 35B decode wall after stage 2). When
// EVERY block of a layer is SLRU-resident (prewarmed or organically), the host does not need
// sel/w at all: these twins read the router's device sel/w output directly and fetch the 8
// expert weight pointers from a per-layer device table [3, n_expert] of slot base addresses
// (gate row, up row, down row — fixed addresses for the cache's lifetime).
//
// BIT-IDENTITY vs moe_gate_up_silu8_f32/moe_down8_fma_f32: the ONLY change is where the weight
// pointer and the w scalar come from (device loads instead of kernel params). Same grid/block,
// same dot reduction order, same SiLU expression, same slot-ordered __fmaf_rn chain. The sel/w
// VALUES are the same bits either way (both paths consume moe_router_topk_f32's output).
// ================================================================================================
// q8 dp4a twins of the _dev pair (resident-experts arc, 2026-07-06): device sel/w + pointer
// table (like _dev) + int dp4a dots vs a q8_1 activation (like the _q8 pair). One warp per
// (row, slot); same silu expression / slot-ordered FMA chain as every twin in this family.
extern "C" __global__ void moe_gate_up_silu8_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
// gemma4 GELU twin of moe_gate_up_silu8_dev_q8: identical dots/reduce, gelu_tanh epilogue
// (the gelu_tanh_mul_f32 expression exactly).
extern "C" __global__ void moe_gate_up_gelu8_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float x = accg;
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[(size_t)j * n_ff + o] = 0.5f * x * (1.0f + th) * accu;
    }
}

// gemma4 R3: fold the per-expert OUTPUT scale into the routing weights on device:
// w[i] *= s[sel[i]] (post-renorm — associative with the down accumulate's w*dot).
extern "C" __global__ void moe_w_exscale(float* __restrict__ w, const int* __restrict__ sel,
                                         const float* __restrict__ s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= s[sel[i]];
}

extern "C" __global__ void moe_down8_fma_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int ex = sel[j];
        const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g]);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(w[j], acc, chain);
    }
    if (lane == 0) dst[o] = chain;
}

// ---- dev_q8 LAUNCH-GEOMETRY VARIANTS (multirow/occupancy arc 2026-07-05, g7e lane) ----
// Baseline geometry is warp-starved on 188 SMs: gate_up = 4096 one-warp blocks (n_ff x n_used),
// down = 2048 one-warp blocks with an n_used=8 SERIAL slot loop AND nsb=16 (in_f=512) leaving
// lanes 16..31 idle in every dot. These variants change ONLY launch geometry; the per-(row,slot)
// accumulation is expert_dot_g in the SAME g order + the SAME warp_reduce_sum tree, and the down
// FMA chain stays slot-ordered serial -> outputs BIT-IDENTICAL to the base pair.
//
//   gu_geom<RPW>: each warp computes RPW consecutive rows of ONE slot; the activation group
//   (aqb/d8) is read once per g and reused across the RPW gate+up dots (RPW weight streams in
//   flight hide load latency — the q4k/q5k mmvq multirow recipe). blockDim.y packs several
//   row-tiles per block for scheduler occupancy; grid = (ceil(n_ff/(RPW*wpb)), n_used).
//
//   down_w8<RPW>: block = (32, n_used) — warp j computes slot j's dot for RPW consecutive rows
//   (identical 32-lane tree per (row,slot)), partials land in smem, then warp 0 lane 0 replays
//   the slot-ordered __fmaf_rn chain per row (8 sequential FMAs — cheap). The n_used loop
//   parallelizes; ONLY the chain stays serial (bit-identity contract).
template<int RPW>
__device__ __forceinline__ void moe_gu_dev_q8_geom(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o0 = ((int)blockIdx.x * (int)blockDim.y + (int)threadIdx.y) * RPW;
    int j = blockIdx.y;
    if (o0 >= n_ff) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* gbase = (const unsigned char*)table[ex];
    const unsigned char* ubase = (const unsigned char*)table[n_expert + ex];
    float accg[RPW], accu[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { accg[r] = 0.0f; accu[r] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= n_ff) break;
            accg[r] += expert_dot_g(qt_g, gbase + (long)o * rb_g, g, aqb, d8);
            accu[r] += expert_dot_g(qt_u, ubase + (long)o * rb_u, g, aqb, d8);
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= n_ff) break;
        float ag = warp_reduce_sum(accg[r]);
        float au = warp_reduce_sum(accu[r]);
        if (lane == 0) act[(size_t)j * n_ff + o] = (ag / (1.0f + expf(-ag))) * au;
    }
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r1(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    moe_gu_dev_q8_geom<1>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u);
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    moe_gu_dev_q8_geom<2>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u);
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    moe_gu_dev_q8_geom<4>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u);
}
template<int RPW>
__device__ __forceinline__ void moe_down8_dev_q8_w8_geom(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * RPW;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    int nsb = in_f >> 5;
    __shared__ float s[RPW][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            const signed char* aqb = arow + (size_t)g * 32;
            float d8 = adrow[g];
            #pragma unroll
            for (int r = 0; r < RPW; r++) {
                int o = o0 + r;
                if (o >= out_f) break;
                acc[r] += expert_dot_g(qt, wbase + (long)o * rb, g, aqb, d8);
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            float a = warp_reduce_sum(acc[r]);
            if (lane == 0 && o0 + r < out_f) s[r][j] = a;
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;          // slot-ordered serial chain == base kernel's exact FP order
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r1(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<1>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<2>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<4>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}

// ---- ROUND 2 geometry variants (same arc): idle-lane fix + warp-split ----
//
// down HALF-WARP DUAL-ROW (nsb==16 ONLY, i.e. in_f==512 — the 35B expert down shape): the base
// dot loop `for (g = lane; g < nsb; g += 32)` leaves lanes 16..31 IDLE when nsb=16. Here lanes
// 0..15 compute row o0 and lanes 16..31 compute row o0+1 (same g = lane&15 per half — exactly
// the base kernel's per-lane group assignment, single iteration so no accumulation-order change).
// BIT-IDENTITY of the reduce: the base 32-lane tree runs with lanes 16..31 holding 0.0f; row A
// reproduces that exactly by masking the upper half to 0.0f; row B's partials are shifted down
// 16 lanes first (so group g sits at lane g, like base) then upper half masked — SAME tree, SAME
// bits. The FMA chain stays slot-ordered serial on warp 0.
//   _h2:   block (32,1)  grid (out_f/2)      — serial n_used loop, 2 rows/warp
//   _w8h2: block (32,8)  grid (out_f/2)      — warp j = slot j, 2 rows/warp, smem chain replay
__device__ __forceinline__ float2 down_h2_dot(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, int qt, long rb, int lane) {
    int nsb = in_f >> 5;                       // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                              + (long)(o0 + half) * rb;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    // one group per lane (nsb==16): identical expert_dot_g call to the base kernel's lane l16.
    float acc = expert_dot_g_v(qt, wrow, l16, arow + (size_t)l16 * 32, adrow[l16]);
    // row A (o0): lanes 0..15 partials, upper half 0 — the base tree layout verbatim.
    float accA = (half == 0) ? acc : 0.0f;
    float a0 = warp_reduce_sum(accA);
    // row B (o0+1): shift partials down 16 so group g sits at lane g, mask upper half.
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f;
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void moe_down8_fma_dev_q8_h2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    float chain0 = 0.0f, chain1 = 0.0f;
    for (int j = 0; j < n_used; j++) {
        float2 a = down_h2_dot(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) {
            chain0 = __fmaf_rn(w[j], a.x, chain0);
            chain1 = __fmaf_rn(w[j], a.y, chain1);
        }
    }
    if (lane == 0) {
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}

// w8h2 x mr2: each half-warp computes TWO serial rows (activation group regs reused across the
// row pair — the mr2 recipe stacked on h2). 4 rows/block, block (32,8), grid (out_f/4).
// BIT-IDENTITY per row: same single-group expert_dot_g call, same masked 32-lane tree as h2.
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 4;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    int nsb = in_f >> 5;                 // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    __shared__ float s[4][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* aqb = aq2 + (size_t)j * in_f + (size_t)l16 * 32;
        float d8 = ad2[(size_t)j * nsb + l16];
        #pragma unroll
        for (int r = 0; r < 2; r++) {    // two row-pairs, activation regs (aqb/d8) reused
            int o = o0 + 2 * r + half;
            float acc = (o < out_f)
                ? expert_dot_g(qt, wbase + (long)o * rb, l16, aqb, d8) : 0.0f;
            float accA = (half == 0) ? acc : 0.0f;
            float a0 = warp_reduce_sum(accA);
            float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
            float accB = (lane < 16) ? shifted : 0.0f;
            float a1 = warp_reduce_sum(accB);
            if (lane == 0) { s[2 * r][j] = a0; s[2 * r + 1][j] = a1; }
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}

// ---- WIDE-LOAD (_v) twins (down8 lane 2026-07-08) ----
// The w8h2/w8h2r2/base-gate_up bodies VERBATIM with expert_dot_g swapped for expert_dot_g_v:
// same geometry, same g order, same masked 32-lane tree, same slot-ordered __fmaf_rn chain,
// same SiLU expression. Only the IQ4_XS group-dot internals change (value-identical wide loads,
// see expert_dot_iq4xs_g_v) -> outputs BIT-IDENTICAL to their scalar twins.
//   _w8h2v:   BW24_MOE_DEVQ8_DOWN=w8h2v   (w8h2 geometry — the current 35B auto winner)
//   _w8h2r2v: BW24_MOE_DEVQ8_DOWN=w8h2r2v (r2 re-test: activation-reg reuse may pay once the
//             decode is cheap — the tradeoff that lost by 1% at scalar decode cost)
//   gate_up _v: BW24_MOE_DEVQ8_GU=v (same dot body feeds the 69%-eff gate_up twin, 15.1us x
//             40/tok — bigger absolute slice than down; base geometry)
__device__ __forceinline__ float2 down_h2_dot_v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, int qt, long rb, int lane) {
    int nsb = in_f >> 5;                       // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                              + (long)(o0 + half) * rb;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    float acc = expert_dot_g_v(qt, wrow, l16, arow + (size_t)l16 * 32, adrow[l16]);
    float accA = (half == 0) ? acc : 0.0f;     // row A: base tree layout verbatim
    float a0 = warp_reduce_sum(accA);
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f; // row B: shift-down-16 then mask, same tree
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_v(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2r2v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 4;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    int nsb = in_f >> 5;                 // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    __shared__ float s[4][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* aqb = aq2 + (size_t)j * in_f + (size_t)l16 * 32;
        float d8 = ad2[(size_t)j * nsb + l16];
        #pragma unroll
        for (int r = 0; r < 2; r++) {    // two row-pairs, activation regs (aqb/d8) reused
            int o = o0 + 2 * r + half;
            float acc = (o < out_f)
                ? expert_dot_g_v(qt, wbase + (long)o * rb, l16, aqb, d8) : 0.0f;
            float accA = (half == 0) ? acc : 0.0f;
            float a0 = warp_reduce_sum(accA);
            float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
            float accB = (lane < 16) ? shifted : 0.0f;
            float a1 = warp_reduce_sum(accB);
            if (lane == 0) { s[2 * r][j] = a0; s[2 * r + 1][j] = a1; }
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// ---- WALL-GAP ARC (2026-07-10, owner: "94% of wall is not 100%"): cp.async ROW-STAGED
// gate_up twin. The _v dot issues ~24 scattered synchronous byte-loads per lane per iteration
// (IQ3_S superblock: qs/qh/signs/scales all separate) — measured 482GB/s = 56% of wall, the
// b4-tier long_scoreboard signature. This twin bulk-stages BOTH expert rows to shared memory
// with cp.async 16B chunks (one commit/wait, no ring), then runs the dot bodies VERBATIM from
// smem — same bytes, same order, byte-identical outputs. Rows are 16B-aligned by construction
// (IQ3_S rb = in_f/256*110: 880B at in_f 2048; slab bases 256B-aligned).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_vsm(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    extern __shared__ unsigned char srow_vsm[];          // [rb_g + rb_u]
    for (int off = lane * 16; off < (int)rb_g; off += 32 * 16)
        cp_async16_g(srow_vsm + off, grow + off);
    for (int off = lane * 16; off < (int)rb_u; off += 32 * 16)
        cp_async16_g(srow_vsm + rb_g + off, urow + off);
    cp_async_commit();
    cp_async_wait<0>();
    __syncwarp();
    const unsigned char* gsm = srow_vsm;
    const unsigned char* usm = srow_vsm + rb_g;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// vsm2: 2-stage pipelined variant — rows split in half along superblocks; half h+1's cp.async
// is in flight while half h computes. Same dot bodies from smem (byte-identical values); the
// per-lane group order is UNCHANGED?? NO — it is: lane processes g = lane, lane+32 which spans
// both halves (g=lane in half0 for lane<32 when nsb=64: g=lane -> superblock g/8 -> halves by
// g<nsb/2). Loop restructured to walk halves outer, g-within-half inner: per lane the two
// g-values (lane, lane+32) land one in EACH half at nsb=64 -> same two groups, same ORDER
// (lane < lane+32 == half0 then half1). accg accumulation order preserved -> bit-identical.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_vsm2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    extern __shared__ unsigned char srow2[];             // [rb_g + rb_u]
    int hg = (int)rb_g / 2, hu = (int)rb_u / 2;          // halves are 16B-aligned (880/2=440.. NOT 16-aligned!)
    // 440 % 16 != 0 -> split at superblock granularity instead: half0 = first (nsb/2) groups'
    // superblocks. IQ3_S: 8 groups per 110B superblock; nsb=64 -> 8 superblocks -> half = 4
    // superblocks = 440B. cp.async 16B needs 16B alignment: 440 % 16 = 8 -> VIOLATION.
    // Fallback: stage half0 = ceil-to-16B prefix; the boundary superblock loads land in stage 0.
    int h0g = (hg + 15) & ~15;
    int h0u = (hu + 15) & ~15;
    if (h0g > (int)rb_g) h0g = (int)rb_g;
    if (h0u > (int)rb_u) h0u = (int)rb_u;
    // stage 0: first halves of both rows
    for (int off = lane * 16; off < h0g; off += 512) cp_async16_g(srow2 + off, grow + off);
    for (int off = lane * 16; off < h0u; off += 512) cp_async16_g(srow2 + rb_g + off, urow + off);
    cp_async_commit();
    // stage 1: second halves (issued now, awaited after half-0 compute)
    for (int off = h0g + lane * 16; off < (int)rb_g; off += 512) cp_async16_g(srow2 + off, grow + off);
    for (int off = h0u + lane * 16; off < (int)rb_u; off += 512) cp_async16_g(srow2 + rb_g + off, urow + off);
    cp_async_commit();
    const unsigned char* gsm = srow2;
    const unsigned char* usm = srow2 + rb_g;
    float accg = 0.0f, accu = 0.0f;
    int half_nsb = nsb / 2;
    cp_async_wait<1>();                                   // half 0 resident
    __syncwarp();
    for (int g = lane; g < half_nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8);
    }
    cp_async_wait<0>();                                   // half 1 resident
    __syncwarp();
    for (int g = half_nsb + lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// ---- SMALL-M VERIFY ROWS TWINS (BW24_SPEC_M2, lane/spec-m2 2026-07-08) ----
// grid.z = token: the spec verify's MoE dev token loop (t = 2..K+2) ran one launch-pair per
// token (plus per-token quantizes: 4t launches/layer). These twins run the serial loop's
// per-token program on a z axis of tokens — every pointer is offset by tok exactly as the host
// loop sliced it (sel/w + tok*n_used; aq/ad + token activation rows; act/dst + token output
// rows). Per (token, row, slot) the body is the _v / w8h2v kernel VERBATIM: same dot order,
// same warp tree, same slot-ordered __fmaf_rn chain -> outputs BIT-IDENTICAL to the serial
// loop. n_used rides a kernel param here (the non-rows gate_up encodes it as gridDim.y, which
// the z-twin keeps; down needs it for the activation-row stride).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_v_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[tok * n_used + j];
    const signed char* aqt = aq + (size_t)tok * in_f;
    const float* adt = ad + (size_t)tok * nsb;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aqt + (size_t)g * 32;
        float d8 = adt[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[((size_t)tok * n_used + j) * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
// gemma4 GELU rows twin (verify t=2..K+2, one launch for all tokens): per (token, row, slot)
// the body is moe_gate_up_gelu8_dev_q8 VERBATIM (expert_dot_g order, warp tree, gelu epilogue).
extern "C" __global__ void moe_gate_up_gelu8_dev_q8_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[tok * n_used + j];
    const signed char* aqt = aq + (size_t)tok * in_f;
    const float* adt = ad + (size_t)tok * nsb;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aqt + (size_t)g * 32;
        float d8 = adt[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float x = accg;
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[((size_t)tok * n_used + j) * n_ff + o] = 0.5f * x * (1.0f + th) * accu;
    }
}

// gemma4 generic down rows twin: grid.z = token; per row the base moe_down8_fma_dev_q8 body
// VERBATIM (serial slot-ordered __fmaf_rn chain).
extern "C" __global__ void moe_down8_fma_dev_q8_rows_g(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const int* selt = sel + tok * n_used;
    const float* wt = w + tok * n_used;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int ex = selt[j];
        const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex] + (long)o * rb;
        const signed char* arow = aq2 + ((size_t)tok * n_used + j) * in_f;
        const float* adrow = ad2 + ((size_t)tok * n_used + j) * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g_v(qt, wrow, g, arow + (size_t)g * 32, adrow[g]);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(wt[j], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}

// down rows twin of w8h2v (the AUTO winner for the 35B shape, in_f==512 && n_used<=8 —
// dispatch-gated by the host). Same down_h2_dot_v body per (token, row-pair, slot).
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2v_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int tok = blockIdx.z;
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    const int* selt = sel + tok * n_used;
    const float* wt = w + tok * n_used;
    const signed char* aq2t = aq2 + (size_t)tok * n_used * in_f;
    const float* ad2t = ad2 + (size_t)tok * n_used * (in_f >> 5);
    float* dstt = dst + (size_t)tok * out_f;
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_v(table, selt, aq2t, ad2t, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(wt[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(wt[jj], s[1][jj], chain1);
        }
        dstt[o0] = chain0;
        if (o0 + 1 < out_f) dstt[o0 + 1] = chain1;
    }
}

// gate_up SLOT-PACKED blocks: block (32, n_used), warp j = slot j for the SAME row o — one block
// per row, 8x fewer blocks, same warp count; the 8 warps share the row's activation groups via
// L1. Each warp's body is the base kernel VERBATIM (same loop, same tree) -> bit-identical.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_j8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = threadIdx.y;                 // slot from block y-dim; blockDim.y == n_used
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// ---- IQ3_S SMEM-GRID twins (2026-07-06 g7e lane) ----
// The iq3s_grid LUT moved __constant__ -> __device__ (+11.8% decode: constant-cache divergent-read
// serialization). It still rides L1 though, CONTENDING with the weight stream (each gate_up warp
// does ~32 divergent grid lookups per lane per launch on the 35B shape). These twins copy the 2KB
// grid (512 u32) into SHARED memory once per block and look up from smem: banked, divergent-
// friendly, zero L1 contention. VALUES are the same table bytes -> outputs BIT-IDENTICAL (same
// expert_dot_iq3s_g expression, same g order, same warp tree).
__device__ __forceinline__ float expert_dot_iq3s_g_sm(const unsigned char* wrow, int g,
                                                      const signed char* aqb, float d8,
                                                      const unsigned int* gsm) {
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
        int gl = gsm[qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100)];
        int gh = gsm[qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100)];
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
// smem-or-L1 dot: IQ3_S goes through the smem grid; every other qtype = expert_dot_g verbatim.
__device__ __forceinline__ float expert_dot_g_sm(int qtype, const unsigned char* wrow, int g,
                                                 const signed char* aqb, float d8,
                                                 const unsigned int* gsm) {
    if (qtype == QT_IQ3_S) return expert_dot_iq3s_g_sm(wrow, g, aqb, d8, gsm);
    return expert_dot_g(qtype, wrow, g, aqb, d8);
}
// base-geometry twin: grid (n_ff, n_used), block (32,1) — ONE warp both copies the 2KB grid
// (16 coalesced u32 loads/lane) and runs the base dot loop.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_sg(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    __shared__ unsigned int gsm[512];
    int lane = threadIdx.x;
    #pragma unroll
    for (int i = lane; i < 512; i += 32) gsm[i] = iq3s_grid_const[i];
    __syncwarp();
    int o = blockIdx.x;
    int j = blockIdx.y;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_sm(qt_g, grow, g, aqb, d8, gsm);
        accu += expert_dot_g_sm(qt_u, urow, g, aqb, d8, gsm);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
// j8-geometry twin: block (32, n_used) — ONE 2KB copy (spread over all 32*n_used threads)
// serves n_used warps; 8x fewer blocks = 8x less copy traffic than _sg.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_j8sg(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    __shared__ unsigned int gsm[512];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int nth = blockDim.y * 32;
    for (int i = tid; i < 512; i += nth) gsm[i] = iq3s_grid_const[i];
    __syncthreads();
    int o = blockIdx.x;
    int j = threadIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_sm(qt_g, grow, g, aqb, d8, gsm);
        accu += expert_dot_g_sm(qt_u, urow, g, aqb, d8, gsm);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// gate_up 2-WARP SPLIT: the RPW multirow direction REDUCED warp count and lost; this DOUBLES it.
// block (32,2): warp 0 computes the gate dot, warp 1 the up dot — each with the base kernel's
// exact per-warp g order + 32-lane tree (bit-identical partials); warp 0 lane 0 applies the same
// silu expression after the smem exchange. grid unchanged (n_ff, n_used) -> 2x warps in flight
// on the same latency-bound weight streams, zero extra launches, zero numeric change.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_s2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int which = threadIdx.y;             // 0 = gate, 1 = up
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* wrow = (which == 0)
        ? (const unsigned char*)table[ex] + (long)o * rb_g
        : (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    long qt = (which == 0) ? qt_g : qt_u;
    __shared__ float sg, su;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g((int)qt, wrow, g, aq + (size_t)g * 32, ad[g]);
    acc = warp_reduce_sum(acc);
    if (lane == 0) { if (which == 0) sg = acc; else su = acc; }
    __syncthreads();
    if (which == 0 && lane == 0) {
        float g = sg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * su;
    }
}
// gate_up 4-WARP G-SPLIT (nsb==64 ONLY, i.e. in_f==2048 — the 35B expert gate/up shape): block
// (32,4), warp y: 0=gate-low 1=gate-high 2=up-low 3=up-high. "low" computes group g=lane, "high"
// g=lane+32 — TOGETHER exactly the base warp's two serial iterations. BIT-IDENTITY: base per-lane
// acc = (0 + d(g=l)) + d(g=l+32); here low's d(l) (0+x==x) merges with high's d(l+32) via smem in
// that same order, then the SAME 32-lane tree runs in the low warp. Halves each warp's serial
// group count AND 4x the warps in flight; the up-warp's silu operand crosses via smem.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_gs4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int wy = threadIdx.y;                // 0..3: 0=gate-low 1=gate-high 2=up-low 3=up-high
    int is_up = wy >> 1, is_hi = wy & 1;
    int ex = sel[j];
    const unsigned char* wrow = is_up
        ? (const unsigned char*)table[n_expert + ex] + (long)o * rb_u
        : (const unsigned char*)table[ex] + (long)o * rb_g;
    int qt = is_up ? qt_u : qt_g;
    int g = lane + (is_hi << 5);
    float d = expert_dot_g(qt, wrow, g, aq + (size_t)g * 32, ad[g]);
    __shared__ float hi[2][32];
    __shared__ float gu[2];
    if (is_hi) hi[is_up][lane] = d;
    __syncthreads();
    if (!is_hi) {
        float acc = d + hi[is_up][lane]; // base per-lane serial order verbatim
        acc = warp_reduce_sum(acc);      // base 32-lane tree
        if (lane == 0) gu[is_up] = acc;
    }
    __syncthreads();
    if (wy == 0 && lane == 0) {
        float gg = gu[0];
        act[(size_t)j * n_ff + o] = (gg / (1.0f + expf(-gg))) * gu[1];
    }
}
// gate_up nsb==64 UNROLLED twin (in_f==2048 — the 35B expert gate/up shape): the base loop's two
// g-iterations (g=lane, g=lane+32) are issued as INDEPENDENT expressions so all 4 dot bodies'
// loads pipeline (base: accg/accu serialize each warp's second iteration behind the first).
// BIT-IDENTITY: accg = (0 + dg(l)) + dg(l+32) — the base loop's exact accumulation order — then
// the same warp tree + silu expression. Geometry unchanged (one warp per (row,slot)).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_u64(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    int g0 = lane, g1 = lane + 32;       // nsb==64 exactly (dispatch-gated)
    const signed char* a0 = aq + (size_t)g0 * 32;
    const signed char* a1 = aq + (size_t)g1 * 32;
    float d80 = ad[g0], d81 = ad[g1];
    float g_lo = expert_dot_g(qt_g, grow, g0, a0, d80);
    float g_hi = expert_dot_g(qt_g, grow, g1, a1, d81);
    float u_lo = expert_dot_g(qt_u, urow, g0, a0, d80);
    float u_hi = expert_dot_g(qt_u, urow, g1, a1, d81);
    float accg = (0.0f + g_lo) + g_hi;   // base loop's accumulation order verbatim
    float accu = (0.0f + u_lo) + u_hi;
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}
// s2 with ROWS packed per block for scheduler density: block (32,2,rz), grid (n_ff/rz, n_used).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_s2z(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = (int)blockIdx.x * (int)blockDim.z + (int)threadIdx.z;
    if (o >= n_ff) return;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int which = threadIdx.y;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* wrow = (which == 0)
        ? (const unsigned char*)table[ex] + (long)o * rb_g
        : (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    long qt = (which == 0) ? qt_g : qt_u;
    __shared__ float sgu[16][2];         // [z][which]
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g((int)qt, wrow, g, aq + (size_t)g * 32, ad[g]);
    acc = warp_reduce_sum(acc);
    if (lane == 0) sgu[threadIdx.z][which] = acc;
    __syncthreads();
    if (which == 0 && lane == 0) {
        float g = sgu[threadIdx.z][0];
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * sgu[threadIdx.z][1];
    }
}

extern "C" __global__ void moe_gate_up_silu8_dev(
        const unsigned long long* __restrict__ table,  // [3, n_expert] slot base addresses
        const int* __restrict__ sel,                   // [n_used] this token's expert ids (device)
        const float* __restrict__ x, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row 0..n_ff-1
    int j = blockIdx.y;              // routed-expert slot 0..n_used-1
    int tid = threadIdx.x;
    __shared__ float s[32];
    __shared__ float g_final;
    const int ex = sel[j];           // broadcast load
    // ---- gate dot: EXACT qmatvec_f32 structure ----
    const unsigned char* grow = (const unsigned char*)(uintptr_t)table[ex] + (long)o * rb_g;
    float acc = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt_g, grow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) g_final = v;
    }
    __syncthreads();                 // s + g_final ready; s reused below
    // ---- up dot: same structure ----
    const unsigned char* urow = (const unsigned char*)(uintptr_t)table[n_expert + ex] + (long)o * rb_u;
    float acc2 = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc2 += deq(qt_u, urow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc2;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) {
            float g = g_final;
            // silu_mul_f32's exact expression on the exact dot values.
            act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * v;
        }
    }
}

extern "C" __global__ void moe_down8_fma_dev(
        const unsigned long long* __restrict__ table,  // [3, n_expert]; down row at 2*n_expert
        const int* __restrict__ sel,                   // [n_used] (device)
        const float* __restrict__ w,                   // [n_used] renormalized weights (device)
        const float* __restrict__ act, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s[32];
    float chain = 0.0f;              // tid 0's slot-ordered accumulator (other threads' unused)
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow =
            (const unsigned char*)(uintptr_t)table[2 * n_expert + sel[j]] + (long)o * rb;
        const float* xrow = act + (size_t)j * in_f;
        float acc = 0.0f;
        for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt, wrow, i) * xrow[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            // slot-ordered FMA chain == the sequential axpy_f32 accumulation (see header).
            if (tid == 0) chain = __fmaf_rn(w[j], v, chain);
        }
        __syncthreads();             // s[] reused next iteration
    }
    if (tid == 0) dst[o] = chain;
}

// ---- CSR EXPERT-DEDUP VERIFY TWINS (verify-cost target #1, 2026-07-10) ----
// The _rows twins re-stream + re-decode each selected expert's full gate/up/down rows once per
// (token, slot) pair. Measured cross-token overlap at verify (BW24_MOE_OVERLAP, 35B K=3 p2,
// t=4): unique/pairs = 0.60-0.62 — 38-40% of the expert weight traffic AND nibble decode is
// duplicated. These twins group pairs by expert (CSR built ON DEVICE — ZERO-DtoH preserved),
// hoist the IQ4_XS/IQ3_S group decode into registers once per (expert, row, group), and replay only
// the per-pair dp4a chain per token.
// BIT-IDENTITY CONTRACT: exp_decode_g_cached + exp_dot_cached replay expert_dot_iq4xs_g /
// expert_dot_iq3s_g VERBATIM (same packing, same dp4a order, same scalar expression);
// per pair the lane-strided g order and the warp tree match the _v_rows / w8h2v bodies; the
// down combine replays the slot-ordered __fmaf_rn chain. Outputs bit-identical to the _rows
// twins. Host dispatch-gates each projection qtype to {IQ4_XS, IQ3_S} (the k-quant tail
// layers keep the _rows twins).
// Cached group decode: 8 dp4a weight words + (d, scale) such that the dot below replays the
// expert_dot_*_g expression bit-for-bit. IQ4_XS: w[k]=wlo[k], w[4+k]=whi[k], expression
// d_sb*(float)(scale*sumi)*d8. IQ3_S: w[k] = signed grid ints in linear aq4 order, scale=1
// (so (float)(1*sumi) == (float)sumi), expression db*(float)sumi*d8. The 35B UD mix runs
// gate/up = IQ3_S, down = IQ4_XS.
struct expg { int w[8]; float d; int scale; };
__device__ __forceinline__ expg exp_decode_g_cached(int qt, const unsigned char* wrow, int g) {
    expg r;
    if (qt == 5) {                              // QT_IQ4_XS
        int sblk = g >> 3, ib = g & 7;
        const unsigned char* b = wrow + (long)sblk * 136;
        r.d = half_to_float(*(const unsigned short*)b);
        unsigned short sh = *(const unsigned short*)(b + 2);
        const unsigned char* sl = b + 4;
        const unsigned char* qs = b + 8 + ib * 16;
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        r.scale = ls - 32;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            r.w[k]   = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                     | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
            r.w[4+k] = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                     | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
        }
    } else if (qt == 12) {                      // QT_Q4_0: -8 folded into the ints (gemma)
        const unsigned char* b = wrow + (long)g * 18;
        r.d = half_to_float(*(const unsigned short*)b);
        r.scale = 1;
        const unsigned char* qs = b + 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
            r.w[k]     = __vsub4((int)(raw & 0x0F0F0F0Fu), 0x08080808);
            r.w[4 + k] = __vsub4((int)((raw >> 4) & 0x0F0F0F0Fu), 0x08080808);
        }
    } else {                                    // QT_IQ3_S (host-gated to {5, 6, 12})
        int sblk = g >> 3, ib32 = g & 7;
        const unsigned char* b = wrow + (long)sblk * 110;
        float d = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs    = b + 2  + ib32 * 8;
        unsigned char qh           = b[66 + ib32];
        const unsigned char* signs = b + 74 + ib32 * 4;
        const unsigned char* scales= b + 106;
        int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
        r.d = d * (1.0f + 2.0f * (float)sc_nib);
        r.scale = 1;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
            int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
            unsigned char sb = signs[l0 / 2];
            int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
            int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
            r.w[l0 + 0] = __vsub4(gl ^ signs0, signs0);
            r.w[l0 + 1] = __vsub4(gh ^ signs1, signs1);
        }
    }
    return r;
}
// dp4a is exact integer math — dot ORDER is bit-irrelevant; only the closing FLOAT ops care.
// CODEGEN CONTRACT (ULP lesson, 2026-07-10): the _rows twins compile `acc += d*(float)(s*sumi)*d8`
// with nvcc's default fmad contraction — the final x*d8 fuses into the accumulate as
// fma(d*(float)(s*sumi), d8, acc). A structurally different kernel contracts DIFFERENTLY and
// drifts last-ULP (measured: 35% of ACT elements). So the accumulate is written as EXPLICIT
// intrinsics here — __fmaf_rn(__fmul_rn(d,(float)(s*sumi)), d8, acc) — pinning the exact
// rounding sequence instead of trusting the optimizer to match.
__device__ __forceinline__ int exp_sumi_cached(int qt, const expg& e, const signed char* aqb) {
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    if (qt == 5 || qt == 12) {                   // IQ4_XS / Q4_0: (w[k],a[k]),(w[4+k],a[4+k])
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            sumi = dp4a(e.w[k],     aq4[k],     sumi);
            sumi = dp4a(e.w[4 + k], aq4[4 + k], sumi);
        }
    } else {                                     // IQ3_S linear order
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(e.w[k], aq4[k], sumi);
    }
    return sumi;
}
__device__ __forceinline__ float exp_dot_acc_cached(int qt, const expg& e,
                                                    const signed char* aqb, float d8, float acc) {
    int sumi = exp_sumi_cached(qt, e, aqb);
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(e.d, (float)(e.scale * sumi)), d8));
}
// single-group form (down: nsb==16, one group per lane, NO accumulate) — pure rounded muls.
__device__ __forceinline__ float exp_dot_cached(int qt, const expg& e, const signed char* aqb,
                                                float d8) {
    int sumi = exp_sumi_cached(qt, e, aqb);
    return __fmul_rn(__fmul_rn(e.d, (float)(e.scale * sumi)), d8);
}

#define CSR_MAXP 10   // pairs per expert <= t <= 10 (verify t = 2..K+2, K <= 8); host-gated
// OWNER-SCAN dedup (v3): no separate CSR build — grid.y = pair index; the block whose pair is
// the FIRST occurrence of its expert OWNS the expert and serves every pair that selected it;
// duplicate blocks exit after an n_pairs-long L1 scan (~24-80 loads). v2's one-thread build
// kernel measured 18.2us/launch (5.5% of the round loop) and its parallel fix still cost a
// launch + 4 allocs per layer — inlining the scan makes the dedup's fixed cost ~0.
// gemma4 GELU CSR twin (verify dedup): owner-scan body of _csr_iq4 with the gelu epilogue.
extern "C" __global__ void moe_gate_up_gelu8_dev_q8_csr(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used, int n_pairs) {
    int pself = blockIdx.y;
    int ex = sel[pself];
    for (int q = 0; q < pself; q++) if (sel[q] == ex) return;
    int plist[CSR_MAXP];
    int np = 0;
    for (int q = pself; q < n_pairs; q++) if (sel[q] == ex && np < CSR_MAXP) plist[np++] = q;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg[CSR_MAXP], accu[CSR_MAXP];
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        expg wg = exp_decode_g_cached(qt_g, grow, g);
        expg wu = exp_decode_g_cached(qt_u, urow, g);
        #pragma unroll
        for (int i = 0; i < CSR_MAXP; i++) {
            if (i < np) {
                int tok = plist[i] / n_used;
                const signed char* aqb = aq + (size_t)tok * in_f + (size_t)g * 32;
                float d8 = ad[(size_t)tok * nsb + g];
                accg[i] = exp_dot_acc_cached(qt_g, wg, aqb, d8, accg[i]);
                accu[i] = exp_dot_acc_cached(qt_u, wu, aqb, d8, accu[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) {
        if (i < np) {
            float sg = warp_reduce_sum(accg[i]);
            float su = warp_reduce_sum(accu[i]);
            if (lane == 0) {
                float x = sg;
                float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
                act[(size_t)plist[i] * n_ff + o] = 0.5f * x * (1.0f + th) * su;
            }
        }
    }
}

extern "C" __global__ void moe_gate_up_silu8_dev_q8_csr_iq4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used, int n_pairs) {
    int pself = blockIdx.y;
    int ex = sel[pself];
    for (int q = 0; q < pself; q++) if (sel[q] == ex) return;   // duplicate: owner is earlier
    int plist[CSR_MAXP];
    int np = 0;
    for (int q = pself; q < n_pairs; q++) if (sel[q] == ex && np < CSR_MAXP) plist[np++] = q;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg[CSR_MAXP], accu[CSR_MAXP];
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        expg wg = exp_decode_g_cached(qt_g, grow, g);
        expg wu = exp_decode_g_cached(qt_u, urow, g);
        #pragma unroll
        for (int i = 0; i < CSR_MAXP; i++) {
            if (i < np) {
                int tok = plist[i] / n_used;
                const signed char* aqb = aq + (size_t)tok * in_f + (size_t)g * 32;
                float d8 = ad[(size_t)tok * nsb + g];
                accg[i] = exp_dot_acc_cached(qt_g, wg, aqb, d8, accg[i]);
                accu[i] = exp_dot_acc_cached(qt_u, wu, aqb, d8, accu[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) {
        if (i < np) {
            float sg = warp_reduce_sum(accg[i]);
            float su = warp_reduce_sum(accu[i]);
            if (lane == 0)
                act[(size_t)plist[i] * n_ff + o] = (sg / (1.0f + expf(-sg))) * su;
        }
    }
}


// ===== Q4_0 decode MMVQ (gemma-4 QAT GGUF, 2026-07-10). Block = 18B per 32 elems: fp16 d +
// 16B nibbles (elem i = low nibble of byte i for i<16, high nibble of byte i-16 for i>=16).
// value = d*(q-8); with per-32 q8_1 activations (aq int8 + ad group scale):
//   dot_g = d * (sumi - 8*sums) * d8, sumi = dp4a(q, a), sums = dp4a(1, a) — exact ints,
// one float expression per group (the q4_K vendoring pattern; llama vec_dot_q4_0_q8_1 math).
// Q4_0 mr2 (gemma trunk lane): 2 rows/warp — the activation int4 loads AND the row-independent
// ones-sum (sums) are computed ONCE per group and reused across both rows' dp4a chains.
// Per-row accumulation chain identical to qmatvec_q4_0_mmvq (bit-identical per row).
__device__ __forceinline__ void q4_0_mmvq_row2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f, acc1 = 0.0f;
    bool two = (o0 + 1) < out_f;
    const unsigned char* w0 = W + (long)o0 * row_bytes;
    const unsigned char* w1 = W + (long)(o0 + 1) * row_bytes;
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            const unsigned char* b = w0 + (long)g * 18;
            float d4 = half_to_float(*(const unsigned short*)b);
            const unsigned char* qs = b + 2;
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                sumi = dp4a((int)(raw & 0x0F0F0F0Fu), aq4[k], sumi);
                sumi = dp4a((int)((raw >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
        if (two) {
            const unsigned char* b = w1 + (long)g * 18;
            float d4 = half_to_float(*(const unsigned short*)b);
            const unsigned char* qs = b + 2;
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                sumi = dp4a((int)(raw & 0x0F0F0F0Fu), aq4[k], sumi);
                sumi = dp4a((int)((raw >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc1 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (two) acc1 = warp_reduce_sum(acc1);
    if (lane == 0) {
        y[(size_t)t * out_f + o0] = acc0;
        if (two) y[(size_t)t * out_f + o0 + 1] = acc1;
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_row2(W, aq, ad, y, in_f, out_f, m, row_bytes,
                   (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2, blockIdx.y);
}
// ----- FUSED Q4_0 m=1 PAIR/TRIPLE (gemma: gate+up / wq+wk+wv share the quantized input).
// Block-offset partition over the mr2 row pairs; per (tensor,row) chain = mr2 VERBATIM. -----
extern "C" __global__ void qmatvec_q4_0_mmvq_fused2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long rb0, long rb1) {
    int pairs0 = (out0 + 1) / 2;
    int nb0 = (pairs0 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    if (b < nb0) {
        q4_0_mmvq_row2(W0, aq, ad, y0, in_f, out0, 1, rb0,
                       (b * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
    } else {
        b -= nb0;
        q4_0_mmvq_row2(W1, aq, ad, y1, in_f, out1, 1, rb1,
                       (b * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused3(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long rb0, long rb1, long rb2) {
    int nb0 = ((out0 + 1) / 2 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int nb1 = ((out1 + 1) / 2 + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f; long rb;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; rb = rb0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; rb = rb1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; rb = rb2; b -= nb0 + nb1; }
    q4_0_mmvq_row2(W, aq, ad, y, in_f, out_f, 1, rb,
                   (b * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
}

// ----- Q4_0 SPLIT-PLANE (rp) MIRROR twins (2026-07-10, the verify-trunk/decode-trunk 18B-
// straggle cure — A6's NVFP4 layout applied to Q4_0). Mirror layout in ONE buffer:
// qs plane [out_f x nblk x 16B] (16B-aligned -> ONE LDG.128/block) then d plane
// [out_f x nblk x 2B] (dense u16). Raw GGUF bytes stay resident for prefill/gemm/Stage-A;
// decode-class kernels read the mirror. Every twin's per (token,row) float chain
// (d4*(sumi-8*sums)*d8 in ascending-g order) is VERBATIM its block-layout source kernel —
// the standing batched==mr2 bit-identity contract extends to the rp family (kernel gates +
// VERIFY-GATE + run-spec battery arbitrate). Microprobe: m=1 1.34x, m=3 1.17x, m=4 1.13x,
// bitwise-exact (rp_q4_probe). -----
__device__ __forceinline__ void q4_0_rp_planes(const unsigned char* W, int out_f,
                                               int o, int nblk,
                                               const unsigned char** wq,
                                               const unsigned short** wd) {
    // planes derived from shape (the NVFP4 rp convention): qs plane is out_f*nblk*16 bytes.
    *wq = W + ((size_t)o * nblk) * 16;
    *wd = (const unsigned short*)(W + (size_t)out_f * nblk * 16) + (size_t)o * nblk;
}
// device-side mirror build: one thread per q4_0 block, pure byte permutation.
extern "C" __global__ void q4_0_split_rp_build(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int out_f, int nblk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_f * nblk) return;
    const unsigned char* b = src + (size_t)i * 18;
    long qplane = (long)out_f * nblk * 16;
    unsigned char* q = dst + (size_t)i * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) q[k] = b[2 + k];
    dst[qplane + (size_t)i * 2 + 0] = b[0];
    dst[qplane + (size_t)i * 2 + 1] = b[1];
}
// m=1 two-rows-per-warp twin (the mr2 body with split loads).
__device__ __forceinline__ void q4_0_mmvq_row2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    (void)row_bytes;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f, acc1 = 0.0f;
    bool two = (o0 + 1) < out_f;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nsb, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nsb, &wq1, &wd1);
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            int4 qv = *(const int4*)(wq0 + (size_t)g * 16);
            float d4 = half_to_float(wd0[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
        if (two) {
            int4 qv = *(const int4*)(wq1 + (size_t)g * 16);
            float d4 = half_to_float(wd1[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc1 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (two) acc1 = warp_reduce_sum(acc1);
    if (lane == 0) {
        y[(size_t)t * out_f + o0] = acc0;
        if (two) y[(size_t)t * out_f + o0 + 1] = acc1;
    }
}
// mr1 split-plane twin (E4B mr2-efficiency probe, 2026-07-13): ONE row per warp — 2x the
// blocks of mr2 for tall-input/short-output shapes (ffn_down 10240->2560 runs 69% of the
// byte floor under mr2's 4-wave grid; more blocks = more latency hiding + finer tail).
// Per-row dot = q4_0_mmvq_row2_rp's acc0 path VERBATIM (bit-identical per row).
__device__ __forceinline__ void q4_0_mmvq_row1_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    (void)row_bytes;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f;
    const unsigned char* wq0; const unsigned short* wd0;
    q4_0_rp_planes(W, out_f, o0, nsb, &wq0, &wd0);
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            int4 qv = *(const int4*)(wq0 + (size_t)g * 16);
            float d4 = half_to_float(wd0[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (lane == 0) y[(size_t)t * out_f + o0] = acc0;
}
extern "C" __global__ void qmatvec_q4_0_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_row1_rp(W, aq, ad, y, in_f, out_f, m, row_bytes,
                      blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y, blockIdx.y);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_row2_rp(W, aq, ad, y, in_f, out_f, m, row_bytes,
                      (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2, blockIdx.y);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long rb0, long rb1) {
    // GRID-STRIDE (2026-07-12, 31B wave-quantization fix): the flat launch ran 5.46 waves/SM
    // — the 0.46 tail wave idled ~54% of the card for ~8% of the duration (ncu, DRAM capped
    // at 92.9%). Striding distributes the tail one iteration wide across every SM. Per-row
    // math and each warp's row assignment order are untouched — bit-identical outputs.
    const int rpb = (int)blockDim.y;   // host BW24_FUSED_RPB knob (tail-wave granularity)
    int pairs0 = (out0 + 1) / 2;
    int nb0 = (pairs0 + rpb - 1) / rpb;
    int nb1 = ((out1 + 1) / 2 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1; vb += gridDim.x) {
        int b = vb;
        if (b < nb0) {
            q4_0_mmvq_row2_rp(W0, aq, ad, y0, in_f, out0, 1, rb0,
                              (b * rpb + (int)threadIdx.y) * 2, 0);
        } else {
            b -= nb0;
            q4_0_mmvq_row2_rp(W1, aq, ad, y1, in_f, out1, 1, rb1,
                              (b * rpb + (int)threadIdx.y) * 2, 0);
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long rb0, long rb1, long rb2) {
    const int rpb = (int)blockDim.y;   // host BW24_FUSED_RPB knob
    int nb0 = ((out0 + 1) / 2 + rpb - 1) / rpb;
    int nb1 = ((out1 + 1) / 2 + rpb - 1) / rpb;
    int nb2 = ((out2 + 1) / 2 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1 + nb2; vb += gridDim.x) {
        int b = vb;
        const unsigned char* W; float* y; int out_f; long rb;
        if (b < nb0)            { W = W0; y = y0; out_f = out0; rb = rb0; }
        else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; rb = rb1; b -= nb0; }
        else                    { W = W2; y = y2; out_f = out2; rb = rb2; b -= nb0 + nb1; }
        q4_0_mmvq_row2_rp(W, aq, ad, y, in_f, out_f, 1, rb,
                          (b * rpb + (int)threadIdx.y) * 2, 0);
    }
}
// batched (weight-read-once, m<=MCOLS) twin — body mirrors q4_0_mmvq_batched.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q4_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 qv = *(const int4*)(wq + (size_t)blk * 16);
        float d4 = half_to_float(wd[blk]);
        int lo[4], hi[4];
        const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            lo[k] = qk[k] & 0x0F0F0F0F;
            hi[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13, the L1TEX fix): same values, same dp4a order.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            const int al[4] = { a01.x, a01.y, a01.z, a01.w };
            const int ah[4] = { a23.x, a23.y, a23.z, a23.w };
            int sumi = 0, sums = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(lo[k], al[k], sumi);
                sumi = dp4a(hi[k], ah[k], sumi);
                sums = dp4a(0x01010101, al[k], sums);
                sums = dp4a(0x01010101, ah[k], sums);
            }
            acc[c] += d4 * (float)(sumi - 8 * sums) * ad[(size_t)c * nblk + blk];
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// batched 2-rows-per-warp twin — body mirrors q4_0_mmvq_batched_mr2 (row-shared activation
// int4 loads + ones-sums, per-row chains in the same order).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = *(const int4*)(wq0 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = *(const int4*)(wq1 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13): the 8 scalar int loads were 4x the L1TEX
            // transactions of the t=1 walk's two 16B loads — L1TEX measured 90% saturated
            // (the b-tier limiter). Same bytes, same order per k — bit-identical.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)c * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0 = dp4a(lo0[k], a[k], s0);
                s0 = dp4a(hi0[k], a[4 + k], s0);
            }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// ---- Q4_0 M-SPLIT r2 twin (2026-07-13, the 31B depth-verify occupancy fix): the b8_r2_rp
// kernel is REGISTER-CHOKED (ncu: 72 regs, occupancy capped at 7 blocks, warps 47-55%,
// DRAM 25-45% of wall) — the acc[2][MCOLS] array is the pressure. The rpms pattern (NVFP4,
// 2026-07-06) splits the M columns across a warp PAIR: both warps walk the FULL k-range of
// the SAME 2 rows, each owning half the columns — acc drops to [2][MCOLS/2], grid.x doubles
// (block (32,4) = 2 pairs x 2 rows), and every (token,row) dot keeps the reference per-lane
// serial chain + warp_reduce_sum -> BIT-IDENTICAL to _r2_rp (column partition, not k-order;
// the rpks k-order lesson). The twin warp re-reads the same weight int4s in near-lockstep ->
// L1 serves the second copy.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    constexpr int CH = MCOLS / 2;           // columns per warp
    int pair = (int)threadIdx.y >> 1;       // 0..1: which 2-row group of the block
    int kc   = (int)threadIdx.y & 1;        // 0..1: which column half
    int o0 = (blockIdx.x * 2 + pair) * 2;
    if (o0 >= out_f) return;
    int c0 = kc * CH;
    if (c0 >= m) return;                    // whole column half masked
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[CH], acc1[CH];
    #pragma unroll
    for (int c = 0; c < CH; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = *(const int4*)(wq0 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = *(const int4*)(wq1 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            int col = c0 + c;
            if (col >= m) break;
            const signed char* arow = aq + (size_t)col * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)col * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0 = dp4a(lo0[k], a[k], s0);
                s0 = dp4a(hi0[k], a[4 + k], s0);
            }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < CH; c++) {
        int col = c0 + c;
        if (col >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)col * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)col * out_f + o0 + 1] = a1;
        }
    }
}
// ---- Q4_0 LOAD-AHEAD r2 twin (2026-07-13, same target as the smem-slab probe): the
// c-loop's activation loads are serial 32B L2 dependency chains (long_scoreboard 42.5%).
// This twin double-buffers the per-column activation int4s in registers: column c+1's
// loads are ISSUED before column c's dp4a chain executes, so the load latency overlaps
// compute instead of stalling it. No smem, no syncs. Math order per (row,col) unchanged
// (bit-identical); +8 int registers.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = *(const int4*)(wq0 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = *(const int4*)(wq1 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        // prime column 0's loads
        int a_nxt[8]; float d8_nxt = 0.0f;
        {
            const int* aq4 = (const int*)(aq + (size_t)0 * in_f + (size_t)blk * 32);
            #pragma unroll
            for (int k = 0; k < 8; k++) a_nxt[k] = aq4[k];
            d8_nxt = ad[(size_t)0 * nblk + blk];
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            int a[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) a[k] = a_nxt[k];
            float d8 = d8_nxt;
            if (c + 1 < MCOLS && c + 1 < m) {   // issue c+1's loads BEFORE computing c
                const int* aq4 = (const int*)(aq + (size_t)(c + 1) * in_f + (size_t)blk * 32);
                #pragma unroll
                for (int k = 0; k < 8; k++) a_nxt[k] = aq4[k];
                d8_nxt = ad[(size_t)(c + 1) * nblk + blk];
            }
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            int s0i = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0i = dp4a(lo0[k], a[k], s0i);
                s0i = dp4a(hi0[k], a[4 + k], s0i);
            }
            acc0[c] += d40 * (float)(s0i - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// ---- Q4_0 SMEM-SLAB r2 twin (2026-07-13, the 31B depth-verify latency fix): the r2 b-tiers
// are LATENCY-bound (ncu: long_scoreboard 42.5% of stalls, DRAM 22-45%, L2 hit 39%) — the
// c-loop's per-column activation loads are 8 serial 32B L2 dependency chains per warp
// iteration. This twin stages a 32-k-block SLAB of ALL columns' activations (+ q8 scales)
// into shared memory cooperatively (one coalesced pass per block instead of per-warp chains),
// then each lane consumes ITS k-block of the slab from smem — the inner loop's only global
// stream left is the weight planes. BIT-IDENTITY: per-(row,col) dot chain, lane->k-block
// mapping, and warp_reduce order are the r2_rp body's verbatim; only the activation LOAD
// SOURCE changes (same bytes). smem = MCOLS*(1KB act + 128B scales) (+pad), single-buffered.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o0 = (blockIdx.x * BW24_MMVQ_ROWS + (int)threadIdx.y) * 2;
    bool row_ok = o0 < out_f;
    bool two = row_ok && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int tid = (int)threadIdx.y * 32 + lane;
    int nblk = in_f / 32;
    const unsigned char* wq0 = nullptr; const unsigned short* wd0 = nullptr;
    const unsigned char* wq1 = nullptr; const unsigned short* wd1 = nullptr;
    if (row_ok) {
        q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
        q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    }
    extern __shared__ int sm_slab[];                    // [MCOLS][32 blk][9 int] (8 + pad —
                                                        // stride 8 = 4-way bank conflicts)
    float* sm_d8 = (float*)(sm_slab + MCOLS * 32 * 9);  // [MCOLS][32] scales
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int s0 = 0; s0 < nblk; s0 += 32) {
        int slab = min(32, nblk - s0);
        __syncthreads();
        // cooperative stage: MCOLS*slab*8 ints + MCOLS*slab scales, 128 threads.
        for (int i = tid; i < MCOLS * slab * 8; i += 128) {
            int c = i / (slab * 8);
            int r = i - c * (slab * 8);        // blk_in_slab*8 + word
            if (c < m) {
                const int* arow = (const int*)(aq + (size_t)c * in_f + (size_t)s0 * 32);
                sm_slab[(c * 32 + r / 8) * 9 + (r & 7)] = arow[r];
            }
        }
        for (int i = tid; i < MCOLS * slab; i += 128) {
            int c = i / slab;
            int b = i - c * slab;
            if (c < m) sm_d8[c * 32 + b] = ad[(size_t)c * nblk + s0 + b];
        }
        __syncthreads();
        int blk = s0 + lane;
        if (!row_ok || lane >= slab) continue;
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = *(const int4*)(wq0 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = *(const int4*)(wq1 + (size_t)blk * 16);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const int* aq4 = &sm_slab[(c * 32 + lane) * 9];
            int a[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) a[k] = aq4[k];
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = sm_d8[c * 32 + lane];
            int s0i = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0i = dp4a(lo0[k], a[k], s0i);
                s0i = dp4a(hi0[k], a[4 + k], s0i);
            }
            acc0[c] += d40 * (float)(s0i - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ ad_q,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<8>(W, ad_q, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


extern "C" __global__ void qmatvec_q4_0_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * BW24_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const unsigned char* b = wrow + (long)g * 18;
        float d4 = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs = b + 2;
        const int* aq4 = (const int*)(arow + (size_t)g * 32);
        int sumi = 0, sums = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw;
            memcpy(&raw, qs + 4 * k, 4);
            int lo = (int)(raw & 0x0F0F0F0Fu);
            int hi = (int)((raw >> 4) & 0x0F0F0F0Fu);
            int a_lo = aq4[k];
            int a_hi = aq4[4 + k];
            sumi = dp4a(lo, a_lo, sumi);
            sumi = dp4a(hi, a_hi, sumi);
            sums = dp4a(0x01010101, a_lo, sums);
            sums = dp4a(0x01010101, a_hi, sums);
        }
        acc += d4 * (float)(sumi - 8 * sums) * adrow[g];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
