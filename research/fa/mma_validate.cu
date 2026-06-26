// mma_validate.cu — isolated validation of the m16n8k16 bf16 MMA primitives for bw24 FA.
// PORTS (verbatim, concretized to C, no ggml templates) from
//   /home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/mma.cuh
//     - tile<16,8,nv_bfloat162> lane map     (lines 469-503, non-AMD/non-Volta branch)
//     - tile<8,8,nv_bfloat162>  lane map     (lines 469-503)
//     - tile<16,8,float> accumulator lane map(lines 244-262)
//     - load_ldmatrix      m16n8 x4          (lines 829-859, the per-LANE address math, fixes C1)
//     - load_ldmatrix_trans m16x8 x4.trans   (lines 884-918, the xi[0],xi[2],xi[1],xi[3] reorder, fixes C3/C5)
//     - mma m16n8k16 f32.bf16.bf16.f32       (lines 1181-1194)
//
// Two standalone kernels:
//   qk_test : S[16,NK] = Q[16,256] @ K[NK,256]^T     -> isolates C1/C5 (A=Q row, B=K col)
//   pv_test : O[16,256] = P[16,NK] @ V[NK,256]       -> isolates C3/C4 (V via ldmatrix.trans)
// Each diffed vs a CPU reference computed in f64; bf16 tolerance ~1e-2 (relative) / abs as printed.
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o mma_validate mma_validate.cu
// (TURING_MMA_AVAILABLE / AMPERE_MMA_AVAILABLE are >= sm_75 / sm_80; sm_120 satisfies both.)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define HEAD_DIM  256
#define MTILE     16     // q rows  (m of m16n8k16)
#define NK        16     // number of keys handled (multiple of 8). 2 n-blocks of 8.
#define KSTEP     16     // k of m16n8k16
#define NKSTEPS   (HEAD_DIM / KSTEP)   // 16 contraction steps over head_dim
#define NB8       (NK / 8)             // n-blocks of 8 keys = 2

// ---------------------------------------------------------------------------
// Lane maps (ported verbatim from mma.cuh, concretized).
// All return PHYSICAL 32-bit element index; for bf16 tiles the "element" is an
// nv_bfloat162 (2 logical bf16), so logical column k = 2*get_j_bf16(l).
// ---------------------------------------------------------------------------

// tile<16,8,nv_bfloat162> A-operand: ne=4 (4 nv_bfloat162 = 8 bf16 per lane).  mma.cuh:479-503
__device__ __forceinline__ int A16x8_get_i(int l, int lane) { return ((l % 2) * 8) + (lane / 4); }
__device__ __forceinline__ int A16x8_get_j(int l, int lane) { return ((l / 2) * 4) + (lane % 4); }

// tile<8,8,nv_bfloat162> B-operand: ne=2 (2 nv_bfloat162 = 4 bf16 per lane).   mma.cuh:480-498 (I==8,J==8)
__device__ __forceinline__ int B8x8_get_i(int l, int lane)  { return lane / 4; }
__device__ __forceinline__ int B8x8_get_j(int l, int lane)  { return (l * 4) + (lane % 4); }

// tile<16,8,float> C/D accumulator: ne=4 (4 f32 per lane).                     mma.cuh:244-262 (I==16,J==8)
__device__ __forceinline__ int C16x8_get_i(int l, int lane) { return ((l / 2) * 8) + (lane / 4); }
__device__ __forceinline__ int C16x8_get_j(int l, int lane) { return ((lane % 4) * 2) + (l % 2); }

// ---------------------------------------------------------------------------
// load_ldmatrix : m16n8 x4. PER-LANE address (mma.cuh:833-837). FIXES C1.
//   const int* xs = (const int*)xs0 + (lane % I)*stride + (lane / I)*(J/2);
// xs0 points at the bf16 tile origin in smem (row stride = `stride` 32-bit words,
//   i.e. stride bf16-pairs). I=16, J=8 -> (J/2)=4.  Loads 4 u32 = 8 bf16/lane.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_ldmatrix_16x8(uint32_t (&r)[4], const __nv_bfloat16* xs0_bf16, int stride_pairs) {
    const int lane = threadIdx.x % WARP_SIZE;
    // smem holds bf16; ldmatrix consumes 16-bit elements via 32-bit (nv_bfloat162) addressing.
    const uint32_t* xs0 = reinterpret_cast<const uint32_t*>(xs0_bf16);
    const uint32_t* xs  = xs0 + (lane % 16) * stride_pairs + (lane / 16) * 4; // (J/2)=4
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "l"(xs));
}

// load_ldmatrix B-operand (8x8 x2). PER-LANE address.  mma.cuh:786-798 (m8n8 x2).
//   const int* xs = (const int*)xs0 + (lane % I)*stride + ((lane / I)*(J/2)) % J;
//   I=8, J=4 (physical 32b cols for 8x8 bf16 tile)? B tile is 8x8 logical bf16 -> 4 nv_bfloat162 cols.
//   ne=2 (x2): J(physical)=4, J/2=2. (lane/8)*2 % 4.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_ldmatrix_8x8(uint32_t (&r)[2], const __nv_bfloat16* xs0_bf16, int stride_pairs) {
    const int lane = threadIdx.x % WARP_SIZE;
    const uint32_t* xs0 = reinterpret_cast<const uint32_t*>(xs0_bf16);
    const uint32_t* xs  = xs0 + (lane % 8) * stride_pairs + ((lane / 8) * 2) % 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1])
        : "l"(xs));
}

// load_ldmatrix_trans : m16x8 x4.trans. NOTE the register reorder xi[0],xi[2],xi[1],xi[3]
//   (mma.cuh:890-894). PER-LANE address identical to load_ldmatrix_16x8.  FIXES C3.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_ldmatrix_trans_16x8(uint32_t (&r)[4], const __nv_bfloat16* xs0_bf16, int stride_pairs) {
    const int lane = threadIdx.x % WARP_SIZE;
    const uint32_t* xs0 = reinterpret_cast<const uint32_t*>(xs0_bf16);
    const uint32_t* xs  = xs0 + (lane % 16) * stride_pairs + (lane / 16) * 4;
    // The reorder is in the OUTPUT operand binding: dest regs are r[0],r[2],r[1],r[3].
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[2]), "=r"(r[1]), "=r"(r[3])
        : "l"(xs));
}

// mma m16n8k16 f32.bf16.bf16.f32 (mma.cuh:1181-1194). A=4u32, B=2u32, D=4f32 (+=).
__device__ __forceinline__ void mma_m16n8k16(float (&D)[4], const uint32_t (&A)[4], const uint32_t (&B)[2]) {
    int* Dxi = reinterpret_cast<int*>(D);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// ===========================================================================
// QK kernel : S[16, NK] = Q[16,256] @ K[NK,256]^T.   one warp.
//   Q row-major [16][256] bf16 in smem; K row-major [NK][256] bf16 in smem.
//   A = Q tile (m16 x k16), B = K tile (n8 keys x k16) read as col-major (natural).
// ===========================================================================
__global__ void qk_test(const __nv_bfloat16* __restrict__ Qg,
                         const __nv_bfloat16* __restrict__ Kg,
                         float* __restrict__ Sg /* [16][NK] row-major */) {
    __shared__ __align__(16) __nv_bfloat16 Qs[MTILE * HEAD_DIM];  // [16][256]
    __shared__ __align__(16) __nv_bfloat16 Ks[NK * HEAD_DIM];     // [NK][256]
    const int t = threadIdx.x;
    for (int i = t; i < MTILE * HEAD_DIM; i += blockDim.x) Qs[i] = Qg[i];
    for (int i = t; i < NK * HEAD_DIM;    i += blockDim.x) Ks[i] = Kg[i];
    __syncthreads();

    const int lane = t % WARP_SIZE;
    float Sacc[NB8][4];                       // one C16x8 accumulator per 8-key block
    for (int nb = 0; nb < NB8; nb++)
        for (int x = 0; x < 4; x++) Sacc[nb][x] = 0.0f;

    for (int ks = 0; ks < NKSTEPS; ks++) {    // 16 contraction chunks over head_dim
        const int d0 = ks * KSTEP;            // 16 logical d per step
        uint32_t A[4];
        // Q smem row-major: row stride = HEAD_DIM bf16 = HEAD_DIM/2 nv_bfloat162 pairs.
        load_ldmatrix_16x8(A, Qs + 0 * HEAD_DIM + d0, HEAD_DIM / 2);
        for (int nb = 0; nb < NB8; nb++) {
            uint32_t B[2];
            // K smem row-major [key][d]: B wants 8 keys x k16 col-major == natural row-major K.
            load_ldmatrix_8x8(B, Ks + (nb * 8) * HEAD_DIM + d0, HEAD_DIM / 2);
            mma_m16n8k16(Sacc[nb], A, B);
        }
    }
    // Write S using the C16x8 accumulator lane map.
    for (int nb = 0; nb < NB8; nb++) {
        for (int x = 0; x < 4; x++) {
            int r  = C16x8_get_i(x, lane);          // q-row 0..15
            int c8 = C16x8_get_j(x, lane);          // key within 8-block 0..7
            int key = nb * 8 + c8;
            Sg[r * NK + key] = Sacc[nb][x];
        }
    }
}

// ===========================================================================
// PV kernel : O[16, 256] = P[16, NK] @ V[NK, 256].   one warp.
//   P row-major [16][NK] bf16; V row-major [NK][256] bf16.
//   A = P tile (m16 x k=NK), B = V^T (n=d x k=keys). V is [key][d] row-major; we need
//   V^T = [d][key]; ldmatrix.trans loads a [keys x d]->transposed fragment giving the
//   col-major (n=d) operand the mma expects. We tile output d in 8-wide n-blocks (32 of them),
//   and contract over keys in k16 steps (NK/16 steps; NK=16 -> 1 step).
// ===========================================================================
#define DKSTEPS (NK / KSTEP)    // contraction steps over keys (NK=16 -> 1)
#define ND8     (HEAD_DIM / 8)  // output-d n-blocks of 8 = 32

__global__ void pv_test(const __nv_bfloat16* __restrict__ Pg,
                         const __nv_bfloat16* __restrict__ Vg,
                         float* __restrict__ Og /* [16][256] row-major */) {
    __shared__ __align__(16) __nv_bfloat16 Ps[MTILE * NK];     // [16][NK]
    __shared__ __align__(16) __nv_bfloat16 Vs[NK * HEAD_DIM];  // [NK][256]
    const int t = threadIdx.x;
    for (int i = t; i < MTILE * NK;    i += blockDim.x) Ps[i] = Pg[i];
    for (int i = t; i < NK * HEAD_DIM; i += blockDim.x) Vs[i] = Vg[i];
    __syncthreads();

    const int lane = t % WARP_SIZE;
    for (int nd = 0; nd < ND8; nd++) {        // each 8-wide output-d block
        float Oacc[4] = {0,0,0,0};
        for (int kk = 0; kk < DKSTEPS; kk++) {
            const int k0 = kk * KSTEP;        // key offset
            uint32_t A[4];
            // P smem row-major [q][key]: A = P tile (m16 x k16). row stride = NK bf16 = NK/2 pairs.
            load_ldmatrix_16x8(A, Ps + 0 * NK + k0, NK / 2);
            uint32_t B[4];
            // V smem row-major [key][d]; want B = V^T (n=d x k=keys). Load a 16(keys)x8(d) tile
            // at [k0][nd*8] with ldmatrix.trans -> transposed to (d x keys) col-major operand.
            load_ldmatrix_trans_16x8(B, Vs + k0 * HEAD_DIM + (nd * 8), HEAD_DIM / 2);
            // B is 4 u32 (16 keys x 8 d). m16n8k16 B-operand wants 2 u32 (k16 x n8). The .trans x4
            // gives the two k-halves: regs {0,1} = k0..7 half, {2,3}=k8..15 half (after reorder).
            uint32_t Bk[2] = { B[0], B[1] };  // keys 0..15 packed: B[0]=k lo pair, B[1]=k hi pair
            mma_m16n8k16(Oacc, A, Bk);
        }
        for (int x = 0; x < 4; x++) {
            int r  = C16x8_get_i(x, lane);    // q-row 0..15
            int c8 = C16x8_get_j(x, lane);    // d within 8-block 0..7
            int d  = nd * 8 + c8;
            Og[r * HEAD_DIM + d] = Oacc[x];
        }
    }
}

// ===========================================================================
// Host: random inputs, CPU reference (f64), tolerance check.
// ===========================================================================
static float frand() { return (float)((rand() % 2001) - 1000) / 1000.0f; } // [-1,1]

int main() {
    srand(1234);
    cudaError_t e;

    // -------- QK --------
    {
        const int M = MTILE, D = HEAD_DIM, N = NK;
        float *Qf = (float*)malloc(M*D*sizeof(float));
        float *Kf = (float*)malloc(N*D*sizeof(float));
        for (int i=0;i<M*D;i++) Qf[i]=frand();
        for (int i=0;i<N*D;i++) Kf[i]=frand();
        // bf16 host copies (round to bf16 so CPU ref uses same precision as MMA inputs).
        __nv_bfloat16 *Qb=(__nv_bfloat16*)malloc(M*D*2), *Kb=(__nv_bfloat16*)malloc(N*D*2);
        for (int i=0;i<M*D;i++) Qb[i]=__float2bfloat16(Qf[i]);
        for (int i=0;i<N*D;i++) Kb[i]=__float2bfloat16(Kf[i]);
        // CPU ref in f64 using bf16-rounded operands.
        double *Sref=(double*)malloc(M*N*sizeof(double));
        for (int r=0;r<M;r++) for (int c=0;c<N;c++){
            double acc=0; for(int d=0;d<D;d++) acc += (double)__bfloat162float(Qb[r*D+d]) * (double)__bfloat162float(Kb[c*D+d]);
            Sref[r*N+c]=acc;
        }
        __nv_bfloat16 *dQ,*dK; float *dS;
        cudaMalloc(&dQ,M*D*2); cudaMalloc(&dK,N*D*2); cudaMalloc(&dS,M*N*sizeof(float));
        cudaMemcpy(dQ,Qb,M*D*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dK,Kb,N*D*2,cudaMemcpyHostToDevice);
        qk_test<<<1,32>>>(dQ,dK,dS);
        e=cudaDeviceSynchronize();
        printf("qk_test launch=%s\n", cudaGetErrorString(e));
        float *S=(float*)malloc(M*N*sizeof(float));
        cudaMemcpy(S,dS,M*N*sizeof(float),cudaMemcpyDeviceToHost);
        double maxabs=0, maxrel=0; int bad=0;
        for(int r=0;r<M;r++) for(int c=0;c<N;c++){
            double ref=Sref[r*N+c], got=S[r*N+c];
            double a=fabs(ref-got); double rel=a/(fabs(ref)+1e-6);
            if(a>maxabs)maxabs=a; if(rel>maxrel)maxrel=rel;
            if(a>1e-1 && rel>2e-2){ if(bad<6) printf("  QK[%d][%d] ref=%.4f got=%.4f abs=%.4f rel=%.4f\n",r,c,ref,got,a,rel); bad++; }
        }
        printf("QK  maxabs=%.4e maxrel=%.4e  bad=%d  -> %s\n", maxabs, maxrel, bad, bad==0?"PASS":"FAIL");
        free(Qf);free(Kf);free(Qb);free(Kb);free(Sref);free(S);
        cudaFree(dQ);cudaFree(dK);cudaFree(dS);
    }

    // -------- PV --------
    {
        const int M = MTILE, N = NK, D = HEAD_DIM;
        float *Pf=(float*)malloc(M*N*sizeof(float));
        float *Vf=(float*)malloc(N*D*sizeof(float));
        // P like softmax weights: nonneg, rows roughly summing to 1 (but test is just matmul).
        for (int i=0;i<M*N;i++) Pf[i]=(float)(rand()%1000)/1000.0f;
        for (int i=0;i<N*D;i++) Vf[i]=frand();
        __nv_bfloat16 *Pb=(__nv_bfloat16*)malloc(M*N*2), *Vb=(__nv_bfloat16*)malloc(N*D*2);
        for (int i=0;i<M*N;i++) Pb[i]=__float2bfloat16(Pf[i]);
        for (int i=0;i<N*D;i++) Vb[i]=__float2bfloat16(Vf[i]);
        double *Oref=(double*)malloc(M*D*sizeof(double));
        for (int r=0;r<M;r++) for (int d=0;d<D;d++){
            double acc=0; for(int k=0;k<N;k++) acc += (double)__bfloat162float(Pb[r*N+k]) * (double)__bfloat162float(Vb[k*D+d]);
            Oref[r*D+d]=acc;
        }
        __nv_bfloat16 *dP,*dV; float *dO;
        cudaMalloc(&dP,M*N*2); cudaMalloc(&dV,N*D*2); cudaMalloc(&dO,M*D*sizeof(float));
        cudaMemcpy(dP,Pb,M*N*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dV,Vb,N*D*2,cudaMemcpyHostToDevice);
        pv_test<<<1,32>>>(dP,dV,dO);
        e=cudaDeviceSynchronize();
        printf("pv_test launch=%s\n", cudaGetErrorString(e));
        float *O=(float*)malloc(M*D*sizeof(float));
        cudaMemcpy(O,dO,M*D*sizeof(float),cudaMemcpyDeviceToHost);
        double maxabs=0, maxrel=0; int bad=0;
        for(int r=0;r<M;r++) for(int d=0;d<D;d++){
            double ref=Oref[r*D+d], got=O[r*D+d];
            double a=fabs(ref-got); double rel=a/(fabs(ref)+1e-6);
            if(a>maxabs)maxabs=a; if(rel>maxrel)maxrel=rel;
            if(a>1e-1 && rel>2e-2){ if(bad<6) printf("  PV[%d][%d] ref=%.4f got=%.4f abs=%.4f rel=%.4f\n",r,d,ref,got,a,rel); bad++; }
        }
        printf("PV  maxabs=%.4e maxrel=%.4e  bad=%d  -> %s\n", maxabs, maxrel, bad, bad==0?"PASS":"FAIL");
        free(Pf);free(Vf);free(Pb);free(Vb);free(Oref);free(O);
        cudaFree(dP);cudaFree(dV);cudaFree(dO);
    }
    return 0;
}
