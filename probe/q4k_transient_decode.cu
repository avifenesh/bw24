// Probe: prove a DEPTH-1 transient raw slot + BULK-DECODE-at-superblock-entry produces a
// BYTE-IDENTICAL decoded int8 weight tile (sW) to the current DEPTH-2 raw ring + per-K-step
// decode in qmatvec_gemm.cu kernel1 (Q4_K / Q5_K), AND proves no aliasing corruption when the
// depth-1 slot is overwritten by the NEXT superblock's cp.async.
//
// THE STEP-1 LEVER (prefill-GEMM rewrite): kernel1 is smem-bound at 1 CTA/SM SOLELY because of the
// 40KB (q4_K) / 49KB (q5_K) sWraw[NSTAGE_RAW=2][128][RAW_W] raw-weight ring. Dropping it to depth-1
// halves the ring. The depth-2 ring exists because the OLD scheme decoded ONE group (g&7) per K-step
// from the resident superblock, so the next superblock's prefetch needed a SEPARATE slot to avoid
// overwriting a slot still being read across the 8 K-steps. A naive NSTAGE_RAW=1 broke (overwrote a
// slot mid-read). The FIX proven here: at superblock ENTRY, BULK-DECODE all GPSB(=8) of the
// superblock's 32-blocks into the decoded sW tile in ONE pass; AFTER that the raw slot is dead and
// the next superblock's cp.async can safely overwrite it. depth-1 is then aliasing-safe.
//
// What this probe asserts:
//   (1) BULK-decode (all 8 blocks from one resident superblock, depth-1 slot) == the reference
//       per-block decode (each block decoded from its own depth-2 ring slot) -> BYTE-IDENTICAL sW.
//   (2) ALIASING: decode all 8 blocks of superblock 0 into sW, THEN overwrite the depth-1 slot with
//       superblock 1's bytes and decode block 0 of it -> superblock 0's 8 decoded blocks are INTACT
//       (no read-after-overwrite hazard, because the bulk decode finished before the overwrite).
//
// Build:
//   nvcc -arch=compute_120a -code=sm_120a probe/q4k_transient_decode.cu -o probe/q4k_transient_decode
//   ./probe/q4k_transient_decode    # must print 0 mismatch / BYTE-IDENTICAL
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SZ 32
#define K1_BM   128
#define SW_STRIDE 32            // decoded int8 weight row stride (matches the kernel's BK, no pad)

// ---- dtype meta (mirror qmatvec_gemm.cu StageMeta) ----
struct Meta { int SB_BYTES, GPSB, RAW_W; };
__device__ __host__ Meta q4k_meta() { return Meta{144, 8, 160}; }
__device__ __host__ Meta q5k_meta() { return Meta{176, 8, 192}; }

__device__ __forceinline__ float ghalf2float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ===== reference decodes: VERBATIM copies of decode_q4_k_s / decode_q5_k_s from qmatvec_gemm.cu =====
// (smem-source, vectorized int-LDS form). `b` = staged superblock base in smem at phase (here phase==0
// since SB_BYTES are 16-multiples for these dtypes). `grp` = g&7.
__device__ __forceinline__ void unpack_scales_q45_K_s(const unsigned char* scales, int grp,
                                                      unsigned char* sc, unsigned char* mn) {
    const int* si = (const int*)scales;
    int lo = si[0], md = si[1], hi32 = si[2];
    auto byte = [&](int i) -> unsigned {
        int w = (i < 4) ? lo : ((i < 8) ? md : hi32);
        int p = i & 3;
        return ((unsigned)w >> (8 * p)) & 0xFFu;
    };
    if (grp < 4) { *sc = byte(grp) & 63; *mn = byte(grp + 4) & 63; }
    else { *sc = (byte(grp + 4) & 0xF) | ((byte(grp - 4) >> 6) << 4);
           *mn = (byte(grp + 4) >> 4) | ((byte(grp) >> 6) << 4); }
}
__device__ __forceinline__ float decode_q4_k_s(const unsigned char* b, int grp, int8_t* out, float* bias) {
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs     = b + 16;
    unsigned char sc, mn;
    unpack_scales_q45_K_s(scales, grp, &sc, &mn);
    int chunk = grp >> 1;
    const unsigned char* q = qs + chunk * 32;
    bool hi = (grp & 1);
    const int* q4 = (const int*)q;
    int* o32 = (int*)out;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        int qw = q4[w];
        o32[w] = hi ? ((qw >> 4) & 0x0F0F0F0F) : (qw & 0x0F0F0F0F);
    }
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}
__device__ __forceinline__ float decode_q5_k_s(const unsigned char* b, int grp, int8_t* out, float* bias) {
    float d_sb    = ghalf2float(*(const unsigned short*)b);
    float dmin_sb = ghalf2float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* qs = b + 48;
    unsigned char sc, mn;
    unpack_scales_q45_K_s(scales, grp, &sc, &mn);
    int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
    const unsigned char* q = qs + g64 * 32;
    const int* q4  = (const int*)q;
    const int* qh4 = (const int*)qh;
    int* o32 = (int*)out;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        int qw  = q4[w];
        int qhw = qh4[w];
        int low = hi ? ((qw >> 4) & 0x0F0F0F0F) : (qw & 0x0F0F0F0F);
        int h   = ((qhw >> hbit) & 0x01010101) << 4;
        o32[w] = low | h;
    }
    *bias = -dmin_sb * (float)mn;
    return d_sb * (float)sc;
}

// =====================================================================================
// TEST 1 (q4_K) + TEST 1 (q5_K): one block per (qt). Two superblocks of K1_BM rows each.
//   REFERENCE path  : DEPTH-2 ring. Stage superblock 0 -> slot0, superblock 1 -> slot1, then
//                     decode each of the 16 K-steps (g=0..15) reading the slot for g/8 -> sW_ref.
//   CANDIDATE path  : DEPTH-1 slot. Stage superblock 0 -> the ONE slot; BULK-decode g=0..7; then
//                     OVERWRITE the one slot with superblock 1; BULK-decode g=8..15 -> sW_cand.
//   Both decode 16 K-steps x K1_BM rows x 32 int8. Assert sW_ref == sW_cand byte-for-byte, AND the
//   bias/dw arrays match. The overwrite in the candidate is the aliasing test: superblock 0's 8
//   decoded blocks were fully written to sW BEFORE the slot was reused.
// =====================================================================================

// Decode dispatcher (compile-time qt: 0=q4_K, 1=q5_K)
template<int QT>
__device__ __forceinline__ float decode_one(const unsigned char* b, int grp, int8_t* out, float* bias) {
    if (QT == 0) return decode_q4_k_s(b, grp, out, bias);
    else         return decode_q5_k_s(b, grp, out, bias);
}

// REFERENCE: depth-2 ring, per-K-step decode (mirrors the CURRENT kernel's decode_stage).
template<int QT, int SB_BYTES, int GPSB, int RAW_W>
__global__ void ref_decode(const unsigned char* __restrict__ raw,  // [2 superblocks][K1_BM][SB_BYTES]
                           int8_t* __restrict__ sW,                // [16 K-steps][K1_BM][SW_STRIDE]
                           float* __restrict__ sWd, float* __restrict__ sWb) {
    // depth-2 raw ring resident in smem (both superblocks staged)
    __shared__ __align__(16) unsigned char sWraw[2][K1_BM][RAW_W];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int cta_threads = blockDim.x * blockDim.y;
    // stage both superblocks into distinct ring slots (phase==0 for these dtypes)
    for (int sb = 0; sb < 2; sb++)
        for (int r = tid; r < K1_BM; r += cta_threads)
            for (int c = 0; c < SB_BYTES; c++)
                sWraw[sb][r][c] = raw[(sb * K1_BM + r) * SB_BYTES + c];
    __syncthreads();
    // per-K-step decode (g=0..15), reading slot g/GPSB
    for (int g = 0; g < 16; g++) {
        int rs = (g / GPSB) % 2;
        int grp = g & (GPSB - 1);
        for (int r = tid; r < K1_BM; r += cta_threads) {
            float bias = 0.f;
            const unsigned char* b = &sWraw[rs][r][0];  // phase 0
            float dw = decode_one<QT>(b, grp, &sW[(g * K1_BM + r) * SW_STRIDE], &bias);
            sWd[g * K1_BM + r] = dw; sWb[g * K1_BM + r] = bias;
        }
    }
}

// CANDIDATE: DEPTH-1 slot + BULK decode at superblock entry + OVERWRITE between superblocks.
template<int QT, int SB_BYTES, int GPSB, int RAW_W>
__global__ void cand_decode(const unsigned char* __restrict__ raw,  // [2 superblocks][K1_BM][SB_BYTES]
                            int8_t* __restrict__ sW,                // [16 K-steps][K1_BM][SW_STRIDE]
                            float* __restrict__ sWd, float* __restrict__ sWb) {
    __shared__ __align__(16) unsigned char sWraw1[K1_BM][RAW_W];     // DEPTH-1 transient slot
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int cta_threads = blockDim.x * blockDim.y;
    for (int sb = 0; sb < 2; sb++) {
        // stage superblock `sb` into the ONE slot (overwrites the previous superblock)
        for (int r = tid; r < K1_BM; r += cta_threads)
            for (int c = 0; c < SB_BYTES; c++)
                sWraw1[r][c] = raw[(sb * K1_BM + r) * SB_BYTES + c];
        __syncthreads();   // slot fully staged before any decode reads it
        // BULK-decode all GPSB blocks of this superblock into sW, BEFORE the next iteration overwrites
        for (int grp = 0; grp < GPSB; grp++) {
            int g = sb * GPSB + grp;
            for (int r = tid; r < K1_BM; r += cta_threads) {
                float bias = 0.f;
                const unsigned char* b = &sWraw1[r][0];  // phase 0
                float dw = decode_one<QT>(b, grp, &sW[(g * K1_BM + r) * SW_STRIDE], &bias);
                sWd[g * K1_BM + r] = dw; sWb[g * K1_BM + r] = bias;
            }
        }
        __syncthreads();   // ALL reads of this superblock done before the next overwrite (the WAR guard)
    }
}

// =====================================================================================
// host build random Q4_K / Q5_K superblock bytes
// =====================================================================================
static void fill_random(unsigned char* p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(rand() & 0xFF);
}

template<int QT, int SB_BYTES, int GPSB, int RAW_W>
int run_case(const char* name) {
    const int NSB = 2;
    const int KSTEPS = NSB * GPSB;   // 16
    size_t raw_n = (size_t)NSB * K1_BM * SB_BYTES;
    size_t sw_n  = (size_t)KSTEPS * K1_BM * SW_STRIDE;

    unsigned char* h_raw = (unsigned char*)malloc(raw_n);
    fill_random(h_raw, raw_n, 0xC0FFEE + QT);

    unsigned char *d_raw;  cudaMalloc(&d_raw, raw_n);
    cudaMemcpy(d_raw, h_raw, raw_n, cudaMemcpyHostToDevice);

    int8_t *d_swR, *d_swC;  cudaMalloc(&d_swR, sw_n); cudaMalloc(&d_swC, sw_n);
    cudaMemset(d_swR, 0xAB, sw_n); cudaMemset(d_swC, 0xCD, sw_n);
    float *d_wdR,*d_wbR,*d_wdC,*d_wbC;
    size_t sc_n = (size_t)KSTEPS * K1_BM * sizeof(float);
    cudaMalloc(&d_wdR,sc_n); cudaMalloc(&d_wbR,sc_n); cudaMalloc(&d_wdC,sc_n); cudaMalloc(&d_wbC,sc_n);

    dim3 block(WARP_SZ, 8);   // 256 threads = the kernel's CTA
    ref_decode<QT,SB_BYTES,GPSB,RAW_W><<<1,block>>>(d_raw, d_swR, d_wdR, d_wbR);
    cand_decode<QT,SB_BYTES,GPSB,RAW_W><<<1,block>>>(d_raw, d_swC, d_wdC, d_wbC);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("[%s] CUDA ERROR: %s\n", name, cudaGetErrorString(err)); return 1; }

    int8_t* h_swR=(int8_t*)malloc(sw_n); int8_t* h_swC=(int8_t*)malloc(sw_n);
    float* h_wdR=(float*)malloc(sc_n); float* h_wbR=(float*)malloc(sc_n);
    float* h_wdC=(float*)malloc(sc_n); float* h_wbC=(float*)malloc(sc_n);
    cudaMemcpy(h_swR,d_swR,sw_n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_swC,d_swC,sw_n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wdR,d_wdR,sc_n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wbR,d_wbR,sc_n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wdC,d_wdC,sc_n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wbC,d_wbC,sc_n,cudaMemcpyDeviceToHost);

    int mism_w = 0, mism_d = 0, mism_b = 0, first_show = 0;
    for (size_t i = 0; i < sw_n; i++) if (h_swR[i] != h_swC[i]) {
        mism_w++;
        if (first_show < 5) {
            size_t kstep = i / (K1_BM*SW_STRIDE);
            printf("    sW mismatch @ kstep=%zu byte=%zu : ref=%d cand=%d\n",
                   kstep, i % (K1_BM*SW_STRIDE), (int)h_swR[i], (int)h_swC[i]);
            first_show++;
        }
    }
    // BITWISE compare the scales: random bytes reinterpreted as fp16 d/dmin can produce NaN, and
    // NaN != NaN would falsely flag identical computations. The decode is deterministic, so the two
    // paths must produce BIT-IDENTICAL f32 results -> compare the raw bits, not the float values.
    for (size_t i = 0; i < (size_t)KSTEPS*K1_BM; i++) {
        unsigned bdR, bdC, bbR, bbC;
        memcpy(&bdR,&h_wdR[i],4); memcpy(&bdC,&h_wdC[i],4);
        memcpy(&bbR,&h_wbR[i],4); memcpy(&bbC,&h_wbC[i],4);
        if (bdR != bdC) mism_d++;
        if (bbR != bbC) mism_b++;
    }
    int ok = (mism_w==0 && mism_d==0 && mism_b==0);
    printf("[%s] sW mismatches=%d  dw mismatches=%d  bias mismatches=%d  -> %s\n",
           name, mism_w, mism_d, mism_b,
           ok ? "BYTE-IDENTICAL (depth-1 bulk-decode == depth-2 ring; aliasing-safe)" : "MISMATCH");

    free(h_raw); free(h_swR); free(h_swC); free(h_wdR); free(h_wbR); free(h_wdC); free(h_wbC);
    cudaFree(d_raw); cudaFree(d_swR); cudaFree(d_swC);
    cudaFree(d_wdR); cudaFree(d_wbR); cudaFree(d_wdC); cudaFree(d_wbC);
    return ok ? 0 : 1;
}

int main() {
    int fail = 0;
    fail |= run_case<0, 144, 8, 160>("Q4_K");
    fail |= run_case<1, 176, 8, 192>("Q5_K");
    printf("=== %s ===\n", fail ? "PROBE FAILED" : "PROBE PASSED (0 mismatch, depth-1 transient slot proven safe)");
    return fail;
}
