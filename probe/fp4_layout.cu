// fp4_layout.cu — empirically nail the m16n8k64 mxf4 block_scale A/B/scale fragment layout
// on sm_120a, and confirm which instruction forms assemble. Single warp, single 16x8x64 mma.
//
// Strategy: e2m1 nibble values are {0:+0, 1:+0.5, 2:+1, 3:+1.5, 4:+2, 5:+3, 6:+4, 7:+6, 8:-0 ...}.
// Set exactly ONE A nibble = code 2 (=+1.0) and ONE B nibble = code 2 (=+1.0), scales = 1.0
// (ue8m0 exponent 127 = byte 127; or ue4m3 = 1.0). The output D[m,n] f32 tells us which (m,n)
// the (a_nibble_pos, b_nibble_pos) contracted into -> reveals the A row / B col / K alignment.
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>

// ue8m0: 8-bit exponent-only, value = 2^(byte-127). byte 127 => 1.0.
// ue4m3 (per NVFP4): handled by separate form below.
#define MMA_2X_UE8M0(D,A,B,SA,SB) \
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]) \
    : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),"r"(SA),"r"(SB))

// candidate NVF4 form: 4 scales/64 (one per 16), ue4m3 scales. May or may not assemble on sm_120a.
#define MMA_4X_UE4M3(D,A,B,SA,SB) \
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]) \
    : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),"r"(SA),"r"(SB))

// Place a single e2m1 code into A: row r (0..15), kcol kc (0..63). Returns which lane/reg/shift.
// A frag (Colfax): lane t owns rows (t/4) and (t/4)+8; K-cols [(t%4)*8 .. +7] (reg pair) and +32.
// reg index: a0=row(t/4) Klow, a1=row(t/4) Khigh(+32), a2=row(t/4+8) Klow, a3=row(t/4+8) Khigh.
// nibble within a reg: K offset within the 8-wide group -> nibble (kc%8); low/high nibble of byte = ?
// We DON'T assume the within-reg nibble order; we sweep it in the experiment.

__global__ void probe_A(float* out, int testRow, int testKc, int code) {
    int lane = threadIdx.x;
    unsigned A[4] = {0,0,0,0}, B[2] = {0,0};
    float D[4] = {0,0,0,0};
    // B: put code at (k=0, n=0). B frag: lane t owns cols (t/4); rows K [(t%4)*8..]. col0 K0 -> lane0 reg0 nibble0.
    if (lane == 0) B[0] = (unsigned)code;             // b: K0..7 of col0 in reg0; nibble0 = K0
    // A: set the requested element. Compute owning lane + reg + nibble.
    int owner = (testRow % 8) * 4 + (testKc % 32) / 8;  // lane = (row%8)*4 + kgroup(0..3)
    int reg = ((testRow >= 8) ? 2 : 0) + ((testKc >= 32) ? 1 : 0);
    int nib = testKc % 8;                               // 0..7 within the 8-wide K group
    if (lane == owner) A[reg] = ((unsigned)code) << (4 * nib);
    unsigned SA = 127, SB = 127;                        // ue8m0 1.0
    MMA_2X_UE8M0(D, A, B, SA, SB);
    // D layout: reg ci -> row = lane/4 + (ci>=2)*8 ; col = (lane%4)*2 + (ci&1).
    for (int ci = 0; ci < 4; ci++) {
        int r = lane/4 + (ci/2)*8, c = (lane%4)*2 + (ci&1);
        if (D[ci] != 0.0f) out[r*8 + c] = D[ci];
    }
}

int main() {
    float* d; cudaMalloc(&d, 16*8*4);
    auto run = [&](int row, int kc, int code)->void{
        cudaMemset(d, 0, 16*8*4);
        probe_A<<<1,32>>>(d, row, kc, code);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("LAUNCH ERR: %s\n", cudaGetErrorString(e)); return; }
        float h[128]; cudaMemcpy(h, d, 512, cudaMemcpyDeviceToHost);
        printf("A(row=%2d,kc=%2d,code=%d) -> D nonzero at:", row, kc, code);
        for (int r=0;r<16;r++) for (int c=0;c<8;c++) if (h[r*8+c]!=0.f) printf(" D[%d,%d]=%.3f", r, c, h[r*8+c]);
        printf("\n");
    };
    // code 2 = e2m1 +1.0 ; with B(k0,n0)=+1.0 the dot = a*b summed over k.
    // Expect: A element at (row R, kc 0) * B(k0,n0) -> D[R,0] = 1.0  (only kc==0 contracts with B's k0).
    printf("=== A-row/col discovery (B fixed at k0,n0=+1.0) ===\n");
    for (int r=0;r<16;r++) run(r, 0, 2);     // sweep A row at kc=0
    printf("=== A K-column discovery (row 0) ===\n");
    for (int kc=0;kc<64;kc++) run(0, kc, 2); // which kc contracts with B's k0 -> only kc 0
    cudaFree(d);
    return 0;
}
