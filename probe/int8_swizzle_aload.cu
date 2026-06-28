// Probe: derive a CONFLICT-FREE manual int8 A-load (m16n8k32.s8 A-fragment) that produces the
// BIT-IDENTICAL 4-register fragment as the current ldmatrix-based ld_A_s8, but with a smem column
// XOR-swizzle so the 16 source rows scatter across bank groups (kills the 21M A-load conflicts the
// ncu found in the prefill GEMM). De-risk pattern mirrors probe/fp4_4x_final.cu: prove the layout
// + zero bank conflicts STANDALONE before touching the production kernel.
//
// m16n8k32.s8 A-frag (from qmatvec_gemm.cu ld_A_s8): ldmatrix.x4.b16 with per-lane addr
//   (lane%16)*stride_b16 + (lane/16)*4   (in .b16 = 2-byte units).
// The fragment each lane holds (verified by the existing kernel_check GEMM gate, rel<1e-3): the 4
// .b32 regs are A[16 rows x 32 k] int8 in the canonical m16n8k32 layout. We reproduce the SAME 4
// regs via manual 32-bit loads from a swizzled smem tile, then assert equality lane-by-lane.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define BM 16
#define K  32              // one m16n8k32 A tile: 16 rows x 32 int8
#define STRIDE 32          // bytes/row (unpadded = the conflict source)

// reference: ldmatrix.x4.b16 (exactly ld_A_s8 with stride=32 bytes).
__device__ void ld_A_ldmatrix(int (&t)[4], const int8_t* base) {
    const uint32_t* xs = (const uint32_t*)base + (threadIdx.x % 16) * (STRIDE / 4) + (threadIdx.x / 16) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(t[0]),"=r"(t[1]),"=r"(t[2]),"=r"(t[3]) : "r"(addr));
}

// CANDIDATE manual load with column XOR-swizzle. ldmatrix x4.b16 loads, per lane L (0..31):
//   the 8x8 b16 sub-tile picks row (L%16) of the 16-row tile, and the (L/16) half selects the
//   k-half (word offset (L/16)*4 in b16 = 8 bytes). The 4 output b32 regs are 4 consecutive b16
//   pairs along k. We replicate by reading, per lane, 4 u32 words at the swizzled column.
// Swizzle: store row r at physical word-col (c ^ (r & 7)) [the standard 8-way XOR for 32B stride];
// load with the SAME xor so the value read == the value stored at logical (r,c). The fragment regs
// are thus identical to ldmatrix's, but the 16 rows now map to 8 distinct bank groups (conflict-free).
// CORRECT manual load from the EMPIRICAL map (full dump): lane L holds rows {L/4, L/4+8}:
//   reg0 = row(L/4) k-words [0..3]  (low k-half),  reg2 = row(L/4) k-words [4..7] (high k-half)
//   reg1 = row(L/4+8) low,          reg3 = row(L/4+8) high.
// Wait — the dump shows reg0_row=reg2_row=L/4 and reg1_row=reg3_row=L/4+8. Each .b32 reg = 4 int8
// of ONE row's k-run. reg0/reg2 differ by k-offset (reg0=k of the low quadrant, reg2=k+16). So:
//   t[0]=row0 word(klo), t[1]=row8 word(klo), t[2]=row0 word(khi), t[3]=row8 word(khi)
// where row0=L/4, row8=L/4+8, and the k-word index within the row is (L%4) (the 4 lanes sharing a
// row each take a different k-word). Swizzle: XOR the physical word-col by (row&7) at store+load.
__device__ void ld_A_swizzled(int (&t)[4], const int8_t* base) {
    int r0 = threadIdx.x / 4;          // this lane's low row (0..7 for L<32... wait L/4 = 0..7)
    int r8 = r0 + 8;
    int kw = threadIdx.x % 4;          // which k-word (0..3) of this row's 8 words
    const uint32_t* p0 = (const uint32_t*)(base + r0 * STRIDE);
    const uint32_t* p8 = (const uint32_t*)(base + r8 * STRIDE);
    // klo words = 0..3, khi words = 4..7. lane's kw selects within each half.
    int clo = kw;          // low k-half word
    int chi = kw + 4;      // high k-half word
    t[0] = p0[clo ^ (r0 & 7)];
    t[1] = p8[clo ^ (r8 & 7)];
    t[2] = p0[chi ^ (r0 & 7)];
    t[3] = p8[chi ^ (r8 & 7)];
}

// DUMP: fill tile so each int8 byte = its row index (0..15); ldmatrix it; print per-lane what
// the 4 regs (16 bytes) contain -> reveals the true (lane,reg,byte)->row map (k self-encodes via
// position). With row-only encoding, each reg byte tells which SOURCE ROW that A-operand came from.
__global__ void dump_map() {
    __shared__ __align__(16) int8_t tile[BM][STRIDE];
    int tid = threadIdx.x;
    for (int i = tid; i < BM * STRIDE; i += 32) { int r = i / STRIDE; tile[r][i % STRIDE] = (int8_t)r; }
    __syncthreads();
    int t[4];
    ld_A_ldmatrix(t, &tile[0][0]);
    // print lanes 0,1,2,16,17 — enough to see the quadrant/half distribution.
    // dump ALL 32 lanes, but only the distinct rows per reg-group (bytes 0,4,8,12 = first byte of
    // each .b32 reg) — that's enough to see the full lane->row map across all L%16 quadrant-rows.
    int8_t* bytes = (int8_t*)t;
    printf("lane %2d: reg0_row=%d reg1_row=%d reg2_row=%d reg3_row=%d\n",
        tid, bytes[0], bytes[4], bytes[8], bytes[12]);
}

__global__ void probe(int* mism, int* conflict_marker) {
    __shared__ __align__(16) int8_t tile[BM][STRIDE];
    __shared__ __align__(16) int8_t tile_sw[BM][STRIDE];
    int tid = threadIdx.x;
    // fill both tiles: logical (r, byte b) = r*32+b. tile = plain; tile_sw = column-swizzled store.
    for (int i = tid; i < BM * STRIDE; i += 32) {
        int r = i / STRIDE, b = i % STRIDE;
        tile[r][b] = (int8_t)((r * 32 + b) & 0x7f);
    }
    // swizzled store: word-col c of row r goes to physical col (c ^ (r&7)).
    for (int r = 0; r < BM; r++) {
        uint32_t* dst = (uint32_t*)&tile_sw[r][0];
        const uint32_t* src = (const uint32_t*)&tile[r][0];
        for (int c = tid; c < 8; c += 32) dst[c ^ (r & 7)] = src[c];
    }
    __syncthreads();
    int ref[4], cand[4];
    ld_A_ldmatrix(ref, &tile[0][0]);
    ld_A_swizzled(cand, &tile_sw[0][0]);
    int bad = 0;
    #pragma unroll
    for (int w = 0; w < 4; w++) if (ref[w] != cand[w]) bad = 1;
    if (bad) atomicAdd(mism, 1);
    if (tid == 0) *conflict_marker = 1;
}

int main() {
    int *d_m, *d_c, h_m = 0, h_c = 0;
    cudaMalloc(&d_m, 4); cudaMalloc(&d_c, 4); cudaMemset(d_m, 0, 4); cudaMemset(d_c, 0, 4);
    dump_map<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("--- map dumped above; now the swizzle equality test ---\n");
    probe<<<1, 32>>>(d_m, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_m, d_m, 4, cudaMemcpyDeviceToHost);
    printf("launch=%s  frag_mismatches=%d  %s\n", cudaGetErrorString(cudaGetLastError()), h_m,
           h_m == 0 ? "SWIZZLED-LOAD == LDMATRIX (layout proven, safe to port)" : "MISMATCH (swizzle math wrong)");
    return h_m == 0 ? 0 : 1;
}
