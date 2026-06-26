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

enum QType { QT_Q8_0 = 0, QT_Q4_K = 1, QT_Q6_K = 2 };

__device__ __forceinline__ float deq(int qtype, const uint8_t* row, int j) {
    switch (qtype) {
        case QT_Q8_0: return deq_q8_0(row, j);
        case QT_Q4_K: return deq_q4_k(row, j);
        case QT_Q6_K: return deq_q6_k(row, j);
    }
    return 0.0f;
}

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

// Q8_0 weight x q8_1 activation, int8 dp4a. y[m,out] = sum_blocks d_w*d_a*dp4a(w_qs, a_qs).
// W: block_q8_0 rows (34 bytes/block). aq: int8 [m,in]; ad: f32 [m, in/32].
// grid (out, m); block 64 threads, each handles a stripe of the in/32 blocks.
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
        const signed char* wq = (const signed char*)(wb + 2);   // NOT 4-byte aligned -> no int* cast
        const signed char* aqb = arow + blk * 32;               // activation: 32-aligned, int* OK
        const int* aq4 = (const int*)aqb;
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // assemble 4 weight int8 into one int (byte-wise, alignment-safe)
            int wpack = (wq[k*4] & 0xff) | ((wq[k*4+1] & 0xff) << 8)
                      | ((wq[k*4+2] & 0xff) << 16) | ((wq[k*4+3] & 0xff) << 24);
            sumi = dp4a(wpack, aq4[k], sumi);
        }
        acc += dw * adrow[blk] * (float)sumi;
    }
    // block reduce
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
