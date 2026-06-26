// mma_validate.cu — isolated validation of the m16n8k16 bf16 MMA primitives for bw24 FA (sm_120).
//
// PORTS (concretized to C, no ggml templates) from
//   /home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/mma.cuh
//     - load_ldmatrix       m16n8 x4      (mma.cuh:829-859 — per-LANE address; fixes review C1)
//     - load_ldmatrix_trans m16x8 x4.trans(mma.cuh:884-918 — the {r0,r2,r1,r3} reorder; fixes C3)
//     - mma m16n8k16 f32.bf16.bf16.f32    (mma.cuh:1181-1194)
//     - tile<16,8,float> accumulator map  (mma.cuh:244-262)
//   and the OPERAND ROLES from fattn-mma-f16.cuh (consumer-GPU `#else` path):
//     - mma_tile_sizes primary template   (fattn-mma-f16.cuh:1028-1035): KQ A&B are tile<16,8>,
//       C is tile<16,16>; VKQ V loaded via load_ldmatrix_trans. Both A and B come from the SAME
//       16x8 x4 ldmatrix loader (16B-aligned per lane); the 4 B-registers split into two
//       m16n8k16 calls: n-block 0 = {B[0],B[2]}, n-block 1 = {B[1],B[3]} (mma.cuh:1204-1209).
//
// CRITICAL fixes proven on-box vs the original v1 design:
//   * ldmatrix address MUST be a 32-bit .shared address: (uint32_t)__cvta_generic_to_shared(ptr)
//     passed via "r" — NOT a 64-bit generic pointer via "l" (that yields "misaligned address").
//   * smem tiles MUST be __align__(16) (ldmatrix.aligned requirement).
//   * The 8x8 x2 B-loader from mma.cuh is fragile here (its (lane/8)*2 offset is 8B-misaligned for
//     16-wide rows). Use the 16x8 x4 loader for BOTH A and B and split B into two n-blocks instead.
//
// Two standalone kernels at the real FA dimensions (head_dim=256):
//   qk_test : S[16, NK] = Q[16,256] @ K[NK,256]^T   (16 k-steps over head_dim; NK/8 n-blocks)
//   pv_test : O[16,256] = P[16,NK] @ V[NK,256]       (NK/16 k-steps; 256/8 = 32 d n-blocks)
// Each diffed vs an f64 CPU reference over bf16-rounded operands; bf16 tolerance: see thresholds.
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o mma_validate mma_validate.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define HEAD_DIM  256
#define MTILE     16                 // q rows  (m of m16n8k16)
#define NK        32                 // keys handled (multiple of 16 to exercise PV k-loop)
#define KSTEP     16                 // k of m16n8k16
#define QK_KSTEPS (HEAD_DIM / KSTEP) // 16 contraction steps over head_dim for QK
#define QK_NB8    (NK / 8)           // n-blocks of 8 keys for the QK output (4)
#define PV_KSTEPS (NK / KSTEP)       // contraction steps over keys for PV (2)
#define PV_ND8    (HEAD_DIM / 8)      // 8-wide output-d n-blocks (32)

// --- C/D accumulator tile<16,8,float> lane map (mma.cuh:244-262, I==16 J==8) ----------------
// lane l in [0,4): get_i(l)=((l/2)*8)+(lane/4)  ; get_j(l)=((lane%4)*2)+(l%2)
__device__ __forceinline__ int C_get_i(int l, int lane) { return ((l / 2) * 8) + (lane / 4); }
__device__ __forceinline__ int C_get_j(int l, int lane) { return ((lane % 4) * 2) + (l % 2); }

// --- load_ldmatrix m16n8 x4 (mma.cuh:833-837). 4 u32/lane = 8 bf16 spanning k0..15. ----------
// stride_pairs = smem row stride measured in nv_bfloat162 (u32) units.
__device__ __forceinline__ void ldmatrix_16x8(uint32_t (&r)[4], const __nv_bfloat16* xs0, int stride_pairs) {
    const int lane = threadIdx.x % WARP_SIZE;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(xs0);
    const uint32_t* xs   = base + (lane % 16) * stride_pairs + (lane / 16) * 4; // (J/2)=4
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// --- load_ldmatrix_trans m16x8 x4.trans (mma.cuh:890-894). NOTE the {r0,r2,r1,r3} reorder. ----
__device__ __forceinline__ void ldmatrix_trans_16x8(uint32_t (&r)[4], const __nv_bfloat16* xs0, int stride_pairs) {
    const int lane = threadIdx.x % WARP_SIZE;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(xs0);
    const uint32_t* xs   = base + (lane % 16) * stride_pairs + (lane / 16) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[2]), "=r"(r[1]), "=r"(r[3]) : "r"(addr));
}

// --- mma m16n8k16 f32.bf16.bf16.f32 (mma.cuh:1187). A=4u32, B=2u32, D=4f32 (+=). --------------
__device__ __forceinline__ void mma_m16n8k16(float (&D)[4], const uint32_t (&A)[4], uint32_t b0, uint32_t b1) {
    int* Dxi = reinterpret_cast<int*>(D);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(b0), "r"(b1));
}

// ===========================================================================
// qk_test : S[16, NK] = Q[16,256] @ K[NK,256]^T.  One warp.
//   A = Q tile (16 q x k16), loaded with ldmatrix_16x8 from Qs (row stride HEAD_DIM bf16).
//   B = K tile (16 keys x k16), loaded with ldmatrix_16x8 from Ks; its 4 regs hold two 8-key
//       n-blocks: n0={B[0],B[2]} (keys base+0..7), n1={B[1],B[3]} (keys base+8..15).
//   Output C is tile<16,8,float> per n-block -> S[q][key] via C_get_i/C_get_j.
// ===========================================================================
__global__ void qk_test(const __nv_bfloat16* __restrict__ Qg,
                         const __nv_bfloat16* __restrict__ Kg,
                         float* __restrict__ Sg /* [16][NK] row-major */) {
    __shared__ __align__(16) __nv_bfloat16 Qs[MTILE * HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 Ks[NK    * HEAD_DIM];
    const int t = threadIdx.x;
    for (int i = t; i < MTILE * HEAD_DIM; i += blockDim.x) Qs[i] = Qg[i];
    for (int i = t; i < NK    * HEAD_DIM; i += blockDim.x) Ks[i] = Kg[i];
    __syncthreads();

    const int lane = t % WARP_SIZE;
    // One C accumulator per 16-key group (each group = 2 n-blocks of 8). NK/16 groups.
    const int NGROUP = NK / 16;
    float Cacc[NK / 16][2][4];                       // [keygroup][nblock-in-group][4]
    for (int g = 0; g < NGROUP; g++)
        for (int nb = 0; nb < 2; nb++)
            for (int x = 0; x < 4; x++) Cacc[g][nb][x] = 0.0f;

    for (int ks = 0; ks < QK_KSTEPS; ks++) {
        const int d0 = ks * KSTEP;
        uint32_t A[4];
        ldmatrix_16x8(A, Qs + d0, HEAD_DIM / 2);               // Q rows 0..15, d = d0..d0+15
        for (int g = 0; g < NGROUP; g++) {
            uint32_t B[4];
            ldmatrix_16x8(B, Ks + (g * 16) * HEAD_DIM + d0, HEAD_DIM / 2); // keys g*16..g*16+15
            mma_m16n8k16(Cacc[g][0], A, B[0], B[2]);            // keys g*16 + 0..7
            mma_m16n8k16(Cacc[g][1], A, B[1], B[3]);            // keys g*16 + 8..15
        }
    }
    for (int g = 0; g < NGROUP; g++)
        for (int nb = 0; nb < 2; nb++)
            for (int x = 0; x < 4; x++) {
                int q   = C_get_i(x, lane);
                int c8  = C_get_j(x, lane);
                int key = g * 16 + nb * 8 + c8;
                Sg[q * NK + key] = Cacc[g][nb][x];
            }
}

// ===========================================================================
// pv_test : O[16,256] = P[16,NK] @ V[NK,256].  One warp.
//   A = P tile (16 q x k16 keys), loaded with ldmatrix_16x8 from Ps (row stride NK bf16).
//   B = V^T (n=d x k=keys). V is [key][d] row-major; ldmatrix_trans loads a 16-key x 8-d source
//       and transposes -> col-major (d x keys). Its 4 regs hold two 8-d n-blocks:
//       n0(d 0..7)={B[0],B[2]}, n1(d 8..15)={B[1],B[3]} for the 16-key k-chunk.
//   Contract over keys in k16 chunks (PV_KSTEPS). Output O[q][d] via C_get_i/C_get_j.
// ===========================================================================
__global__ void pv_test(const __nv_bfloat16* __restrict__ Pg,
                         const __nv_bfloat16* __restrict__ Vg,
                         float* __restrict__ Og /* [16][256] row-major */) {
    __shared__ __align__(16) __nv_bfloat16 Ps[MTILE * NK];
    __shared__ __align__(16) __nv_bfloat16 Vs[NK    * HEAD_DIM];
    const int t = threadIdx.x;
    for (int i = t; i < MTILE * NK;       i += blockDim.x) Ps[i] = Pg[i];
    for (int i = t; i < NK    * HEAD_DIM; i += blockDim.x) Vs[i] = Vg[i];
    __syncthreads();

    const int lane = t % WARP_SIZE;
    // Output d processed two 8-blocks at a time (16 d per ldmatrix_trans). PV_ND8/2 = 16 groups.
    for (int dg = 0; dg < PV_ND8 / 2; dg++) {
        const int d0 = dg * 16;
        float C0[4] = {0,0,0,0};   // d = d0+0..7
        float C1[4] = {0,0,0,0};   // d = d0+8..15
        for (int kc = 0; kc < PV_KSTEPS; kc++) {
            const int k0 = kc * KSTEP;
            uint32_t A[4];
            ldmatrix_16x8(A, Ps + k0, NK / 2);                                 // P q0..15, keys k0..k0+15
            uint32_t B[4];
            ldmatrix_trans_16x8(B, Vs + k0 * HEAD_DIM + d0, HEAD_DIM / 2);      // V keys k0..15, d d0..d0+15 -> V^T
            mma_m16n8k16(C0, A, B[0], B[2]);
            mma_m16n8k16(C1, A, B[1], B[3]);
        }
        for (int x = 0; x < 4; x++) {
            int q  = C_get_i(x, lane);
            int c8 = C_get_j(x, lane);
            Og[q * HEAD_DIM + d0 + 0 + c8] = C0[x];
            Og[q * HEAD_DIM + d0 + 8 + c8] = C1[x];
        }
    }
}

// ===========================================================================
// Host: random inputs, f64 CPU reference over bf16-rounded operands, tolerance check.
// ===========================================================================
static float frand() { return (float)((rand() % 2001) - 1000) / 1000.0f; }   // [-1,1]

int main() {
    srand(1234);
    int fails = 0;

    // -------- QK --------  S[16,NK] = Q[16,256] @ K[NK,256]^T
    {
        const int M = MTILE, D = HEAD_DIM, N = NK;
        float *Qf=(float*)malloc(M*D*4), *Kf=(float*)malloc(N*D*4);
        for (int i=0;i<M*D;i++) Qf[i]=frand();
        for (int i=0;i<N*D;i++) Kf[i]=frand();
        __nv_bfloat16 *Qb=(__nv_bfloat16*)malloc(M*D*2), *Kb=(__nv_bfloat16*)malloc(N*D*2);
        for (int i=0;i<M*D;i++) Qb[i]=__float2bfloat16(Qf[i]);
        for (int i=0;i<N*D;i++) Kb[i]=__float2bfloat16(Kf[i]);
        double *Sref=(double*)malloc(M*N*8);
        for (int r=0;r<M;r++) for (int c=0;c<N;c++){
            double acc=0; for(int d=0;d<D;d++) acc += (double)__bfloat162float(Qb[r*D+d]) * (double)__bfloat162float(Kb[c*D+d]);
            Sref[r*N+c]=acc;
        }
        __nv_bfloat16 *dQ,*dK; float *dS;
        cudaMalloc(&dQ,M*D*2); cudaMalloc(&dK,N*D*2); cudaMalloc(&dS,M*N*4);
        cudaMemcpy(dQ,Qb,M*D*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dK,Kb,N*D*2,cudaMemcpyHostToDevice);
        qk_test<<<1,32>>>(dQ,dK,dS);
        cudaError_t e=cudaDeviceSynchronize();
        printf("qk_test launch=%s\n", cudaGetErrorString(e));
        float *S=(float*)malloc(M*N*4);
        cudaMemcpy(S,dS,M*N*4,cudaMemcpyDeviceToHost);
        double maxabs=0, maxrel=0; int bad=0;
        // QK accumulates 256 bf16 products; scale ~ sqrt(256)=16. abs tol 0.5, rel 2e-2.
        for(int r=0;r<M;r++) for(int c=0;c<N;c++){
            double ref=Sref[r*N+c], got=S[r*N+c];
            double a=fabs(ref-got), rel=a/(fabs(ref)+1e-6);
            if(a>maxabs)maxabs=a; if(rel>maxrel)maxrel=rel;
            if(a>5e-1 && rel>2e-2){ if(bad<6) printf("  QK[%d][%d] ref=%.4f got=%.4f abs=%.4f rel=%.4f\n",r,c,ref,got,a,rel); bad++; }
        }
        printf("QK  maxabs=%.4e maxrel=%.4e  bad=%d  -> %s\n", maxabs, maxrel, bad, bad==0?"PASS":"FAIL");
        if(bad) fails++;
        free(Qf);free(Kf);free(Qb);free(Kb);free(Sref);free(S);
        cudaFree(dQ);cudaFree(dK);cudaFree(dS);
    }

    // -------- PV --------  O[16,256] = P[16,NK] @ V[NK,256]
    {
        const int M = MTILE, N = NK, D = HEAD_DIM;
        float *Pf=(float*)malloc(M*N*4), *Vf=(float*)malloc(N*D*4);
        for (int i=0;i<M*N;i++) Pf[i]=(float)(rand()%1000)/1000.0f;   // softmax-like nonneg weights
        for (int i=0;i<N*D;i++) Vf[i]=frand();
        __nv_bfloat16 *Pb=(__nv_bfloat16*)malloc(M*N*2), *Vb=(__nv_bfloat16*)malloc(N*D*2);
        for (int i=0;i<M*N;i++) Pb[i]=__float2bfloat16(Pf[i]);
        for (int i=0;i<N*D;i++) Vb[i]=__float2bfloat16(Vf[i]);
        double *Oref=(double*)malloc(M*D*8);
        for (int r=0;r<M;r++) for (int d=0;d<D;d++){
            double acc=0; for(int k=0;k<N;k++) acc += (double)__bfloat162float(Pb[r*N+k]) * (double)__bfloat162float(Vb[k*D+d]);
            Oref[r*D+d]=acc;
        }
        __nv_bfloat16 *dP,*dV; float *dO;
        cudaMalloc(&dP,M*N*2); cudaMalloc(&dV,N*D*2); cudaMalloc(&dO,M*D*4);
        cudaMemcpy(dP,Pb,M*N*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dV,Vb,N*D*2,cudaMemcpyHostToDevice);
        pv_test<<<1,32>>>(dP,dV,dO);
        cudaError_t e=cudaDeviceSynchronize();
        printf("pv_test launch=%s\n", cudaGetErrorString(e));
        float *O=(float*)malloc(M*D*4);
        cudaMemcpy(O,dO,M*D*4,cudaMemcpyDeviceToHost);
        double maxabs=0, maxrel=0; int bad=0;
        // PV accumulates NK bf16 products (NK=32); magnitudes ~O(1). abs tol 0.1, rel 2e-2.
        for(int r=0;r<M;r++) for(int d=0;d<D;d++){
            double ref=Oref[r*D+d], got=O[r*D+d];
            double a=fabs(ref-got), rel=a/(fabs(ref)+1e-6);
            if(a>maxabs)maxabs=a; if(rel>maxrel)maxrel=rel;
            if(a>1e-1 && rel>2e-2){ if(bad<6) printf("  PV[%d][%d] ref=%.4f got=%.4f abs=%.4f rel=%.4f\n",r,d,ref,got,a,rel); bad++; }
        }
        printf("PV  maxabs=%.4e maxrel=%.4e  bad=%d  -> %s\n", maxabs, maxrel, bad, bad==0?"PASS":"FAIL");
        if(bad) fails++;
        free(Pf);free(Vf);free(Pb);free(Vb);free(Oref);free(O);
        cudaFree(dP);cudaFree(dV);cudaFree(dO);
    }

    printf("\n=== %s ===\n", fails==0 ? "ALL PASS" : "FAILURES");
    return fails;
}
