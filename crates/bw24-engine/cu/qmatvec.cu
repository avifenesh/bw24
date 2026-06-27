// Resident-quantized matmul: weights stay in GGUF block format in VRAM, dequantized in-register
// inside the kernel (never materialized as f32/f16). Fixes the OOM. Activations are f32 (Stage A:
// correctness-first; Stage B will quantize activations to q8_1 + int8 dp4a like llama.cpp MMVQ/MMQ).
//
// y[m, out] = x[m, in] @ W[out, in]^T,  W is quantized (ggml block layout), x/y are f32.
// Layout: x token-major [m, in] (x[t*in + i]); W row o = out-feature o, `in` elements quantized;
//         y token-major [m, out] (y[t*out + o]). One block per (token, out-row); threads reduce over `in`.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// IQ3_S grid: 512 u32 entries, each packs 4 unsigned bytes. Verbatim from ggml-common.h:1042.
__device__ __constant__ unsigned int iq3s_grid_const[512] = {
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

// device codebook tables
__device__ __constant__ signed char kvalues_iq4nl_d[16] =
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

enum QType { QT_Q8_0 = 0, QT_Q4_K = 1, QT_Q6_K = 2,
             QT_Q5_K = 3, QT_Q3_K = 4, QT_IQ4_XS = 5, QT_IQ3_S = 6, QT_NVFP4 = 7,
             QT_F32 = 8 };

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
        // Unquantized f32 weight row (safetensors MoE Path A: experts gathered + dequantized to f32
        // host-resident, staged verbatim). `row` is the start of one out-row of `in_f` contiguous f32s.
        case QT_F32:    return ((const float*)row)[j];
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
extern "C" __global__ void quantize_q8_1(const float* __restrict__ x, signed char* __restrict__ out_q,
                                         float* __restrict__ out_d, int in_f, int m) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;   // global block-of-32 index
    int nblk_row = in_f / 32;
    if (blk >= m * nblk_row) return;
    int t = blk / nblk_row;
    int b = blk % nblk_row;
    if (t >= m) return;
    const float* xr = x + (size_t)t * in_f + b * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(xr[j]));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    signed char* oq = out_q + (size_t)t * in_f + b * 32;
    for (int j = 0; j < 32; j++) oq[j] = (signed char)__float2int_rn(xr[j] * id);
    out_d[(size_t)t * nblk_row + b] = d;
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
        #pragma unroll
        for (int k = 0; k < 8; k++) {
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
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- NVFP4 warp-per-row MMVQ. Body lifted from qmatvec_nvfp4_dp4a (loop @ ~line 674). -----
extern "C" __global__ void qmatvec_nvfp4_mmvq(
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
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
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
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = k * 4 + e;
                int lowbits = hi ? (q[idx] >> 4) : (q[idx] & 0x0F);
                int h = (qh[idx] >> hbit) & 1;
                int w = lowbits | (h << 4);          // 0..31
                wpack |= (w & 0xff) << (e * 8);
            }
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
