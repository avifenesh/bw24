// TASK #81 slice 3 — what the k16->k32 int8 depth lift is WORTH at the tile level.
//
// The 12-form table (rate_audit.cu) established the instruction bound: m16n8k16.s8 and
// m16n8k32.s8 BOTH cost 16.06 cyc/warp-MMA, so k32 delivers 1.997x the MACs per issue slot.
// That bound is an upper limit on a swap's value: a real MMQ tile also pays the per-k01 scale
// FOLD, and the k16 form folds TWICE as often (two C tiles, two dA loads, two FMAs per element)
// as the k32 form. So the tile-level ratio is NOT necessarily 1.997x — it is whatever
//    (2 x k16 MMA + 2-term fold)  vs  (1 x k32 MMA + 1-term fold)
// measures at the register/ILP shape the shipped tiles actually use.
//
// Both inner loops are replicated verbatim from the repo:
//   K16 arm = mmq_iq_experts.cu:322-331 / mmq_nvfp4_w4a8.cu:512-521
//       mma(C[0], A[n][k01/4+0], B[0]);
//       mma(C[1], A[n][k01/4+1], B[1]);
//       sum[..] += dB[l%2] * (C[0].x[l]*dA[n][l/2][k01/4+0] + C[1].x[l]*dA[n][l/2][k01/4+1]);
//   K32 arm = mmq_q8_0.cu:265-270
//       mma(C, A[n][k01/QI8_0], B);
//       sum[..] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
// Same K-values covered per unit (32), same tile_C::ne=4 epilogue element count, fresh C per
// step (as in the real tiles), sum accumulating across steps (as in the real tiles).
//
// ---- ANTI-CSE DISCIPLINE (v1 of this probe was WRONG and the SASS census caught it) ----
// v1 gave every accumulator the SAME A/B operands. ptxas collapsed all NACC copies AND hoisted
// them out of the iteration loop: `cuobjdump -sass` showed IMMA=2 (k16) and IMMA=1 (k32) for the
// WHOLE kernel at every NACC, so v1's "1.2148x" measured 2-vs-1 hoisted MMAs plus N folds —
// meaningless. Fixes, both verified in SASS before any number was believed:
//   (a) per-accumulator DISTINCT A operands => no cross-NACC CSE. v2 rotated the load index by
//       4*i, which ALIASES mod 32 (i and i+8 read the same slot) and CSE'd the NACC=16 row back
//       to 16 IMMA where 32 were required — caught in SASS again. v3 mixes a per-i golden-ratio
//       constant into the loaded value instead, so all 4*NACC A registers are distinct for any
//       NACC, with the mixing done BEFORE the timed region;
//   (b) a per-iteration `tick` XOR'd into the shared B operand => not loop-invariant, no hoisting;
//   (c) scales sourced from the input array (runtime-unknown) => the fold's FFMAs cannot be
//       constant-folded, and dA*dB cannot be pre-multiplied the way two literals could.
// The kernel prints its own SASS-verified MMA count expectation; verify with:
//   nvcc -cubin ... && cuobjdump -sass | grep -c IMMA
//
// READING THE SASS COUNT (the v3 census, so the next reader does not re-derive it): ptxas ALSO
// unrolls the outer `it` loop 8x, so the per-instantiation IMMA count is
//   NACC * mma_per_step * it_unroll, floored at 8 because ptxas keeps at least an 8-wide body.
// Measured: k32 NACC={1,2,4}=8, NACC=8 ->8, NACC=16 ->16; k16 NACC={1,2}=8, 4->8, 8->16, 16->32.
// The NACC<8 rows therefore have FEWER MMAs than NACC*expected -- that is the it-unroll folding
// several iterations into one body, NOT the operand CSE that killed v1/v2. Only NACC>=8 gives an
// exact per-step IMMA count (k16: 2*NACC, k32: 1*NACC), so THE VERDICT IS READ OFF NACC=8/16 and
// the low-NACC columns are ILP context only. (v1's defect was different and fatal: IMMA=2 and 1
// for the WHOLE kernel at every NACC.)
//
// NOT a correctness probe (operands are junk bit patterns; equal-math legality is argued from the
// scale-grid analysis in AUDIT.md and any real swap needs the exactness battery). COST only.
//
// Feasibility spike. Never linked into the engine.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ITERS 2048
#define NE 4          // tile_C::ne for tile<16,8,int>

// ---- K16 arm: two m16n8k16 MMAs + the 2-term fold, per 32 K-values ----
template<int NACC> __global__ void __launch_bounds__(256)
k_tile_k16(const unsigned* s, float* o, long long* c){
    // Per-accumulator DISTINCT A fragments (2 regs per k16 MMA, two MMAs => 4 regs each).
    unsigned a00[NACC], a01[NACC], a10[NACC], a11[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++){
        const unsigned mix = 0x9E3779B9u * (unsigned)(i + 1);
        a00[i]=s[(threadIdx.x+i)&31]^mix;        a01[i]=s[(threadIdx.x+i+1)&31]^(mix*3u);
        a10[i]=s[(threadIdx.x+i+2)&31]^(mix*5u); a11[i]=s[(threadIdx.x+i+3)&31]^(mix*7u);
    }
    unsigned b0=s[(threadIdx.x+17)&31], b1=s[(threadIdx.x+19)&31];
    // Runtime scales: TWO per-16 dA slots + one per-32 dB (the shipped k16 shapes).
    float dA0[NE/2], dA1[NE/2], dB[NE/2];
    #pragma unroll
    for(int l=0;l<NE/2;l++){
        dA0[l]=(float)s[(threadIdx.x+23+l)&31]*1e-9f;
        dA1[l]=(float)s[(threadIdx.x+25+l)&31]*1e-9f;
        dB [l]=(float)s[(threadIdx.x+27+l)&31]*1e-9f;
    }
    float sum[NACC][NE];
    #pragma unroll
    for(int i=0;i<NACC;i++){ _Pragma("unroll") for(int l=0;l<NE;l++) sum[i][l]=0.f; }
    unsigned tick=0;
    __syncthreads(); long long t0=clock64();
    for(int it=0;it<ITERS;++it){
        ++tick;                       // defeats loop-invariant hoisting of the MMAs
        const unsigned bb0=b0^tick, bb1=b1^tick;
        #pragma unroll
        for(int i=0;i<NACC;i++){
            int C0[4]={0,0,0,0}, C1[4]={0,0,0,0};
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                : "+r"(C0[0]),"+r"(C0[1]),"+r"(C0[2]),"+r"(C0[3]) : "r"(a00[i]),"r"(a01[i]),"r"(bb0));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                : "+r"(C1[0]),"+r"(C1[1]),"+r"(C1[2]),"+r"(C1[3]) : "r"(a10[i]),"r"(a11[i]),"r"(bb1));
            #pragma unroll
            for(int l=0;l<NE;l++)
                sum[i][l] += dB[l%2] * ((float)C0[l]*dA0[l/2] + (float)C1[l]*dA1[l/2]);
        }
    }
    long long t1=clock64(); float r=0.f;
    #pragma unroll
    for(int i=0;i<NACC;i++){ _Pragma("unroll") for(int l=0;l<NE;l++) r+=sum[i][l]; }
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}

// ---- K32 arm: one m16n8k32 MMA + the 1-term fold, per 32 K-values ----
template<int NACC> __global__ void __launch_bounds__(256)
k_tile_k32(const unsigned* s, float* o, long long* c){
    unsigned a0[NACC],a1[NACC],a2[NACC],a3[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++){
        const unsigned mix = 0x9E3779B9u * (unsigned)(i + 1);
        a0[i]=s[(threadIdx.x+i)&31]^mix;        a1[i]=s[(threadIdx.x+i+1)&31]^(mix*3u);
        a2[i]=s[(threadIdx.x+i+2)&31]^(mix*5u); a3[i]=s[(threadIdx.x+i+3)&31]^(mix*7u);
    }
    unsigned b0=s[(threadIdx.x+17)&31], b1=s[(threadIdx.x+19)&31];
    // ONE per-32 dA slot (the merged scale) + the same dB.
    float dA[NE/2], dB[NE/2];
    #pragma unroll
    for(int l=0;l<NE/2;l++){
        dA[l]=(float)s[(threadIdx.x+23+l)&31]*1e-9f;
        dB[l]=(float)s[(threadIdx.x+27+l)&31]*1e-9f;
    }
    float sum[NACC][NE];
    #pragma unroll
    for(int i=0;i<NACC;i++){ _Pragma("unroll") for(int l=0;l<NE;l++) sum[i][l]=0.f; }
    unsigned tick=0;
    __syncthreads(); long long t0=clock64();
    for(int it=0;it<ITERS;++it){
        ++tick;
        const unsigned bb0=b0^tick, bb1=b1^tick;
        #pragma unroll
        for(int i=0;i<NACC;i++){
            int C[4]={0,0,0,0};
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+r"(C[0]),"+r"(C[1]),"+r"(C[2]),"+r"(C[3])
                : "r"(a0[i]),"r"(a1[i]),"r"(a2[i]),"r"(a3[i]),"r"(bb0),"r"(bb1));
            #pragma unroll
            for(int l=0;l<NE;l++)
                sum[i][l] += (float)C[l]*dA[l/2]*dB[l%2];
        }
    }
    long long t1=clock64(); float r=0.f;
    #pragma unroll
    for(int i=0;i<NACC;i++){ _Pragma("unroll") for(int l=0;l<NE;l++) r+=sum[i][l]; }
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    const int SMs=p.multiProcessorCount;
    unsigned hs[32]; for(int i=0;i<32;i++) hs[i]=0x11223344u+i*0x01010101u;
    unsigned* dsrc; float* doutf; long long* dcyc;
    CK(cudaMalloc(&dsrc,128)); CK(cudaMemcpy(dsrc,hs,128,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&doutf,256*256*4)); CK(cudaMalloc(&dcyc,8));
    printf("# device %s SMs=%d cc=%d.%d (clocks LOCKED 1860 via nvidia-smi -lgc)\n",
           p.name,SMs,p.major,p.minor);
    printf("# UNIT = one 32-K-value tile step. k16 arm = 2 MMA + 2-term fold; k32 arm = 1 MMA + 1-term fold.\n");
    printf("# SASS contract (verify with cuobjdump -sass): IMMA count per instantiation = NACC*2*unroll (k16), NACC*1*unroll (k32).\n");

    printf("\n## NACC CONTROL -- 1 CTA of 4 warps, clock64 cyc per 32-K tile step\n");
    printf("# %-26s %9s %9s %9s %9s %9s\n","arm","NACC=1","NACC=2","NACC=4","NACC=8","NACC=16");
    double iv[2][5];
#define RUNI(K,N,F,S) do{ K<N><<<1,128>>>(dsrc,doutf,dcyc); CK(cudaDeviceSynchronize()); \
      long long hc; CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost)); iv[F][S]=hc/((double)ITERS*N); }while(0)
#define SWEEP(K,F,NM) do{ RUNI(K,1,F,0); RUNI(K,2,F,1); RUNI(K,4,F,2); RUNI(K,8,F,3); RUNI(K,16,F,4); \
      printf("  %-26s %9.4f %9.4f %9.4f %9.4f %9.4f\n",NM,iv[F][0],iv[F][1],iv[F][2],iv[F][3],iv[F][4]); }while(0)
    SWEEP(k_tile_k16, 0, "K16 (2 MMA + 2-fold)");
    SWEEP(k_tile_k32, 1, "K32 (1 MMA + 1-fold)");
    for(int sN=0; sN<5; ++sN){
        static const int NN[5]={1,2,4,8,16};
        printf("  -> tile-step ratio NACC=%-2d : %.4fx\n", NN[sN], iv[0][sN]/iv[1][sN]);
    }
    printf("  (instruction-only bound, from rate_audit.cu 12-form table, is 1.997x)\n");

    printf("\n## FULL-GPU wall clock (grid=%d CTAs x 256 thr; NACC=8, best of 5)\n",SMs);
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double ms[2];
    for(int f=0;f<2;f++){
        double best=1e18;
        for(int r=0;r<6;r++){
            if(f==0) k_tile_k16<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            else     k_tile_k32<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaDeviceSynchronize());
            if(r==0) continue;
            CK(cudaEventRecord(e0));
            if(f==0) k_tile_k16<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            else     k_tile_k32<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float t; CK(cudaEventElapsedTime(&t,e0,e1)); if(t<best) best=t;
        }
        ms[f]=best;
        printf("  %-26s %9.3f ms\n", f==0?"K16 (2 MMA + 2-fold)":"K32 (1 MMA + 1-fold)", best);
    }
    printf("  -> full-GPU tile-step speedup of the k32 form: %.4fx\n", ms[0]/ms[1]);
    return 0;
}
