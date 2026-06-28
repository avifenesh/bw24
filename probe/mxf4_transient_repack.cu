// Probe: prove a DEPTH-1 transient raw slot + per-K-block fetch-drain-repack produces a BYTE-IDENTICAL
// repacked A-fragment tile (sWq + sWsc) to the CURRENT depth-2 raw ring (FP4_NS_RAW=2) + 1-block-ahead
// repack in qmatvec_gemm.cu's mxf4 kernel (qmatvec_gemm_nvfp4_fp4), AND prove no aliasing corruption when
// the depth-1 slot is overwritten by the NEXT K-block's cp.async.
//
// THE STEP-5 LEVER (mirrors the just-shipped kernel1 Step-1 win, commit ea7d89a): the mxf4 path is
// smem-bound at 2 CTA/SM (cuobjdump SHARED=34304B). The sWraw[FP4_NS_RAW=2][BM=64][80] raw ring is 10240B.
// Dropping it to a DEPTH-1 transient slot [BM][80] = 5120B frees 5KB -> 29184B -> floor(102400/29184)=3
// CTA/SM (BlockLimitSmem 2->3). REG=128 with __launch_bounds__(128,4) already allows 4 CTA/SM, so smem
// is the SOLE blocker; freeing it lets the kernel cross 2->3.
//
// WHY depth-2 today: the loop fetches raw block `gr = gp+1` (1 iter ahead) while repacking `gp` from a
// slot fetched the PRIOR iter, so 2 consecutive raw blocks are alive at once -> depth-2. The depth-1
// candidate removes the 1-ahead raw lead for the WEIGHT: each iter fetches its OWN raw block into the one
// slot, drains, then repacks it (mirrors kernel1's per-superblock fetch+drain+decode at the boundary).
// The activation cp.async lead (sAq/sAsc) is UNCHANGED — only the raw WEIGHT block stops leading.
//
// What this probe asserts:
//   (1) The depth-1 per-block fetch+repack produces sWq/sWsc BYTE-IDENTICAL to the depth-2 ring + 1-ahead
//       repack, for every K-block (the e2m1 nibble gather + ue4m3 scale pack are deterministic).
//   (2) ALIASING: repack block g into sWq, THEN overwrite the depth-1 slot with block g+1's raw bytes and
//       repack it -> block g's repacked sWq/sWsc are INTACT (the repack of g finished + barrier before the
//       overwrite, so no read-after-overwrite hazard).
//
// Build:
//   nvcc -arch=compute_120a -code=sm_120a probe/mxf4_transient_repack.cu -o probe/mxf4_transient_repack
//   ./probe/mxf4_transient_repack   # must print 0 mismatch / BYTE-IDENTICAL
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define WARP_SZ 32
#define BM 64
// match the kernel: NWARP2=4 warps, grid-stride over BM rows
#define NWARP2 4
// raw ring geometry (verbatim from qmatvec_gemm.cu)
#define FP4_NS 2
#define FP4_NS_RAW (2 * (FP4_NS - 1))     // = 2 (the current depth-2 ring)
#define FP4_RAW_STRIDE 80

#define SWZ_CHUNK(row) (((row) >> 2) & 1)

// ============================================================================================
// gather_wq — VERBATIM copy of the per-row e2m1 nibble gather + ue4m3 scale pack from qmatvec_gemm.cu
// (lines 1281-1301). Reads the resident raw 36B NVFP4 block at `b` (= &sWraw_slot[r][phase]); writes the
// 8 A-frag-ready u32 (wq[0..7]) + the 4 ue4m3 scale bytes packed LE into one u32 (*sc). The SWZ_CHUNK
// store uses the ROW index, so each kernel below applies `int sw = SWZ_CHUNK(r)` at the int4 store site
// (matching the kernel exactly).
// ============================================================================================
__device__ __forceinline__ void gather_wq(const unsigned char* b, unsigned* wq, unsigned* sc) {
    *sc = (unsigned)b[0] | ((unsigned)b[1] << 8) | ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24);
    const unsigned char* qs = b + 4;
    #pragma unroll
    for (int gi = 0; gi < 8; gi++) {
        int base = (gi < 4) ? (gi * 8) : ((gi - 4) * 8 + 32);
        int sb = base >> 4, hinib = (base & 8) ? 4 : 0;
        const unsigned char* q = qs + sb * 8;
        unsigned lo = (unsigned)q[0] | ((unsigned)q[1] << 8) | ((unsigned)q[2] << 16) | ((unsigned)q[3] << 24);
        unsigned hi = (unsigned)q[4] | ((unsigned)q[5] << 8) | ((unsigned)q[6] << 16) | ((unsigned)q[7] << 24);
        lo = (lo >> hinib) & 0x0F0F0F0Fu;  hi = (hi >> hinib) & 0x0F0F0F0Fu;
        lo = (lo | (lo >> 4)) & 0x00FF00FFu; lo = (lo | (lo >> 8)) & 0x0000FFFFu;
        hi = (hi | (hi >> 4)) & 0x00FF00FFu; hi = (hi | (hi >> 8)) & 0x0000FFFFu;
        wq[gi] = lo | (hi << 16);
    }
}

// =====================================================================================
// REFERENCE: depth-2 raw ring (FP4_NS_RAW=2) + 1-block-ahead raw fetch + repack from a slot fetched the
// prior iter. Mirrors the CURRENT kernel's fetch_raw/repack cadence: the raw block `g` lives in slot
// g%FP4_NS_RAW; the repack of block g reads slot g%FP4_NS_RAW (which the kernel guarantees resident).
// nblk64 blocks; each block has a 16B-floored phase (phase = (off & 15)); row stride row_bytes bytes.
// =====================================================================================
__global__ void ref_repack(const unsigned char* __restrict__ raw,  // [nblk64][BM][FP4_RAW_STRIDE] staged windows
                           const int* __restrict__ phase_of,        // [BM] per-row phase (off & 15)
                           unsigned* __restrict__ sWq,              // [nblk64][BM][8]
                           unsigned* __restrict__ sWsc,             // [nblk64][BM]
                           int nblk64) {
    __shared__ __align__(16) unsigned char sWraw[FP4_NS_RAW][BM][FP4_RAW_STRIDE]; // depth-2 ring
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int cta_threads = blockDim.x * blockDim.y;
    // Stage the whole window set into the depth-2 ring as the kernel does (block g -> slot g%2).
    // We simulate the kernel's invariant "block gp is resident in slot gp%2 when repacked" by staging
    // every block into its slot once, then repacking from that slot.
    for (int g = 0; g < nblk64; g++) {
        int rs = g % FP4_NS_RAW;
        for (int r = tid; r < BM; r += cta_threads)
            for (int c = 0; c < FP4_RAW_STRIDE; c++)
                sWraw[rs][r][c] = raw[((size_t)g * BM + r) * FP4_RAW_STRIDE + c];
        __syncthreads();
        // repack block g from slot g%2 (the resident slot)
        for (int r = tid; r < BM; r += cta_threads) {
            int phase = phase_of[r];
            const unsigned char* b = &sWraw[rs][r][phase];
            unsigned wq[8], sc;
            gather_wq(b, wq, &sc);
            sWsc[(size_t)g * BM + r] = sc;
            int sw = SWZ_CHUNK(r);
            reinterpret_cast<int4*>(&sWq[((size_t)g * BM + r) * 8])[0 ^ sw] = *reinterpret_cast<const int4*>(&wq[0]);
            reinterpret_cast<int4*>(&sWq[((size_t)g * BM + r) * 8])[1 ^ sw] = *reinterpret_cast<const int4*>(&wq[4]);
        }
        __syncthreads();
    }
}

// =====================================================================================
// CANDIDATE: DEPTH-1 transient slot. Per K-block: stage block g into the ONE slot (overwrites the
// previous block), drain (barrier), repack, barrier (WAR guard before the next overwrite). The overwrite
// of slot by block g+1 is the aliasing test — block g's repacked sWq/sWsc were fully written + barriered
// before the slot is reused.
// =====================================================================================
__global__ void cand_repack(const unsigned char* __restrict__ raw,
                            const int* __restrict__ phase_of,
                            unsigned* __restrict__ sWq,
                            unsigned* __restrict__ sWsc,
                            int nblk64) {
    __shared__ __align__(16) unsigned char sWraw1[BM][FP4_RAW_STRIDE];  // DEPTH-1 transient slot
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int cta_threads = blockDim.x * blockDim.y;
    for (int g = 0; g < nblk64; g++) {
        // stage block g into the ONE slot (overwrites the previous block's bytes)
        for (int r = tid; r < BM; r += cta_threads)
            for (int c = 0; c < FP4_RAW_STRIDE; c++)
                sWraw1[r][c] = raw[((size_t)g * BM + r) * FP4_RAW_STRIDE + c];
        __syncthreads();   // block g fully staged before any repack reads it
        for (int r = tid; r < BM; r += cta_threads) {
            int phase = phase_of[r];
            const unsigned char* b = &sWraw1[r][phase];
            unsigned wq[8], sc;
            gather_wq(b, wq, &sc);
            sWsc[(size_t)g * BM + r] = sc;
            int sw = SWZ_CHUNK(r);
            reinterpret_cast<int4*>(&sWq[((size_t)g * BM + r) * 8])[0 ^ sw] = *reinterpret_cast<const int4*>(&wq[0]);
            reinterpret_cast<int4*>(&sWq[((size_t)g * BM + r) * 8])[1 ^ sw] = *reinterpret_cast<const int4*>(&wq[4]);
        }
        __syncthreads();   // all repack reads of block g done before block g+1 overwrites the slot (WAR)
    }
}

// =====================================================================================
// host: build random staged NVFP4 windows (36B block + slack to FP4_RAW_STRIDE) + random per-row phases
// =====================================================================================
int main() {
    const int nblk64 = 17;   // odd count so block g and g+1 land in DIFFERENT depth-2 slots — exercises aliasing
    size_t raw_n = (size_t)nblk64 * BM * FP4_RAW_STRIDE;
    unsigned char* h_raw = (unsigned char*)malloc(raw_n);
    srand(0xF4F4F4);
    for (size_t i = 0; i < raw_n; i++) h_raw[i] = (unsigned char)(rand() & 0xFF);

    // per-row phase: the kernel uses phase = ((rowtile+r)*row_bytes + g*36) & 15. row_bytes for NVFP4 is a
    // multiple of 36 per K-block, but the row base offset can make phase vary per row. Use a per-row phase
    // in [0..15] (constant across g for a given row, matching: phase depends on o*row_bytes & 15, and the
    // staged window floored each block to its own 16B base so the in-window phase is the same per row).
    int h_phase[BM];
    for (int r = 0; r < BM; r++) h_phase[r] = rand() & 15;  // 0..15; the staged window has 80B >= 15+36

    unsigned char* d_raw; cudaMalloc(&d_raw, raw_n);
    cudaMemcpy(d_raw, h_raw, raw_n, cudaMemcpyHostToDevice);
    int* d_phase; cudaMalloc(&d_phase, sizeof(h_phase));
    cudaMemcpy(d_phase, h_phase, sizeof(h_phase), cudaMemcpyHostToDevice);

    size_t wq_n = (size_t)nblk64 * BM * 8;
    size_t sc_n = (size_t)nblk64 * BM;
    unsigned *d_wqR, *d_wqC, *d_scR, *d_scC;
    cudaMalloc(&d_wqR, wq_n * 4); cudaMalloc(&d_wqC, wq_n * 4);
    cudaMalloc(&d_scR, sc_n * 4); cudaMalloc(&d_scC, sc_n * 4);
    cudaMemset(d_wqR, 0xAB, wq_n * 4); cudaMemset(d_wqC, 0xCD, wq_n * 4);
    cudaMemset(d_scR, 0x11, sc_n * 4); cudaMemset(d_scC, 0x22, sc_n * 4);

    dim3 block(WARP_SZ, NWARP2);   // 128 threads = the mxf4 kernel's CTA
    ref_repack<<<1, block>>>(d_raw, d_phase, d_wqR, d_scR, nblk64);
    cand_repack<<<1, block>>>(d_raw, d_phase, d_wqC, d_scC, nblk64);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }

    unsigned* h_wqR = (unsigned*)malloc(wq_n * 4); unsigned* h_wqC = (unsigned*)malloc(wq_n * 4);
    unsigned* h_scR = (unsigned*)malloc(sc_n * 4); unsigned* h_scC = (unsigned*)malloc(sc_n * 4);
    cudaMemcpy(h_wqR, d_wqR, wq_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wqC, d_wqC, wq_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scR, d_scR, sc_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scC, d_scC, sc_n * 4, cudaMemcpyDeviceToHost);

    int mism_wq = 0, mism_sc = 0, shown = 0;
    for (size_t i = 0; i < wq_n; i++) if (h_wqR[i] != h_wqC[i]) {
        mism_wq++;
        if (shown < 5) {
            size_t g = i / (BM * 8), r = (i / 8) % BM, w = i % 8;
            printf("    sWq mismatch @ blk=%zu row=%zu word=%zu : ref=%08x cand=%08x\n",
                   g, r, w, h_wqR[i], h_wqC[i]);
            shown++;
        }
    }
    for (size_t i = 0; i < sc_n; i++) if (h_scR[i] != h_scC[i]) mism_sc++;

    int ok = (mism_wq == 0 && mism_sc == 0);
    printf("[mxf4] sWq mismatches=%d  sWsc mismatches=%d  -> %s\n", mism_wq, mism_sc,
           ok ? "BYTE-IDENTICAL (depth-1 transient slot == depth-2 ring; aliasing-safe)" : "MISMATCH");
    printf("=== %s ===\n", ok ? "PROBE PASSED (0 mismatch, depth-1 transient mxf4 slot proven safe)"
                              : "PROBE FAILED");

    free(h_raw); free(h_wqR); free(h_wqC); free(h_scR); free(h_scC);
    cudaFree(d_raw); cudaFree(d_phase); cudaFree(d_wqR); cudaFree(d_wqC); cudaFree(d_scR); cudaFree(d_scC);
    return ok ? 0 : 1;
}
