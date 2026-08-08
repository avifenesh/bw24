// TASK #81 — repo-wide PTX MMA mnemonic RATE audit, sm_120a.
//
// Method (inherited verbatim from research/w4a8-prefill-20260806/tools/f8f4_issue_rate.cu):
// clock64() around a tight loop of MUTUALLY INDEPENDENT MMAs, sweeping NACC (accumulator count) so
// the ILP control distinguishes a pipe ISSUE INTERVAL (flat across NACC) from a LATENCY artifact
// (halves as NACC doubles). Then a full-GPU cudaEvent pass at the shipped 82 CTA x 256 thr shape to
// convert the interval into a delivered (FL)OP/s and cross-check the per-warp number.
//
// Every form here is one that some crates/memra-engine/cu/*.cu file ACTUALLY ISSUES, plus the
// equal-math SIBLING of each, so "which form is fast" stops being folklore.
//
// Feasibility spike only. Never linked into the engine.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ITERS 4096

// ---------------- GROUP A: int8, s32 accumulate ----------------
// A1 = mmq_nvfp4_w4a8.cu:211, mmq_iq_experts.cu:146  (k16)
#define MMA_S8_K16(d,a0,a1,b0) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
    : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) : "r"(a0),"r"(a1),"r"(b0))
// A2 = mmq_q8_0.cu:148, mmq_q45k.cu:153, mmq_q4_0.cu:160, qmatvec_gemm.cu:164,
//      mmq_q8_0_f32acc.cu:153  (k32) — the k16 sites' equal-math sibling at 2x depth
#define MMA_S8_K32(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// ---------------- GROUP B: 16-bit float ----------------
// B1 = mma_tile.cuh:127, flash_attn.cu:150, hybrid.cu:1502  (bf16, f32 acc)
#define MMA_BF16_F32(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// B2 = moe_f16_grouped.cu:357, hybrid.cu:1510  (f16, f32 acc)
#define MMA_F16_F32(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// B4 = tf32 m16n8k8, f32 acc (the mma.sync sibling of wgmma_common.cuh:58's tf32 wgmma)
// (A = 4x .b32, B = 2x .b32 for tf32 m16n8k8 — NOT the s8-k16 2/1 shape.)
#define MMA_TF32_K8(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// B4b = tf32 m16n8k16 DOES NOT EXIST on sm_120: ptxas rejects with
//   "Illegal instruction types specified for '_mma' with shape '.m16n8k16'"
// (PTX ISA gives tf32 only .m16n8k4 / .m16n8k8). So B4 has NO deeper-K sibling — verified, not
// assumed. Arm removed from the sweep rather than left as an untested claim.

// B3 = flash_attn.cu:974  (f16 in, f16 OUT — the MEMRA_FA_F16PV door). 2 accum regs, not 4.
#define MMA_F16_F16ACC(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};" \
    : "+r"(d[0]),"+r"(d[1]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// ---------------- GROUP C: FP8 / FP6 / FP4-in-8b-container, k32 ----------------
// C1 = mmq_nvfp4_w4a8.cu:1081 (rollback arm), mmq_fp8_blk.cu:245 (rollback arm),
//      mmq_q8_0_f32acc.cu:185 (rollback arm), mmq_nvfp4_f8f4.cu:42 (dead door)  — PLAIN kind
#define MMA_F8F4_PLAIN(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// C2 = mmq_nvfp4_w4a8.cu:1086, mmq_fp8_blk.cu:250, mmq_q8_0_f32acc.cu:190 — the SHIPPED fast form
#define MMA_MXF8_BLKSC_E4M3(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))
// C3 = plain kind, e2m1 A operand (the R-A variant the design doc considered)
#define MMA_F8F4_E2M1_A(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// C4 = block_scale kind, e2m1 A operand — what prefill-ilp slice 2b actually benched
#define MMA_MXF8_BLKSC_E2M1(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))

// ---------------- GROUP D: FP4, k64 ----------------
// D1 = mmq_fp4.cu:190, qmatvec_gemm.cu:1235 — NVFP4 unified kind, UE4M3 per-16 scales
#define MMA_MXF4NVF4_4X(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))
// D2 = the MXFP4 sibling: kind::mxf4, scale_vec::2X, UE8M0 per-32 scales (equal math ONLY when the
//      scale grid matches; measured here purely as a RATE datum for the k64 pipe)
#define MMA_MXF4_2X(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))

// ============================ kernel templates ============================
// 4x s32/f32 accumulator, 3-source (k16-class) and 6-source (k32/k64-class) bodies both fit.
#define KERN_I32(NAME, BODY) \
template<int NACC> __global__ void __launch_bounds__(256) \
NAME(const unsigned* s, int* o, long long* c){ \
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31]; \
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31]; (void)a2;(void)a3;(void)b1; \
    int d[NACC][4]; \
    _Pragma("unroll") for(int i=0;i<NACC;i++){d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0;} \
    __syncthreads(); long long t0=clock64(); \
    for(int it=0;it<ITERS;++it){ _Pragma("unroll") for(int i=0;i<NACC;i++){ BODY } } \
    long long t1=clock64(); int r=0; \
    _Pragma("unroll") for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3]; \
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0; \
    o[blockIdx.x*256+threadIdx.x]=r; \
}

#define KERN_F32(NAME, BODY) \
template<int NACC> __global__ void __launch_bounds__(256) \
NAME(const unsigned* s, float* o, long long* c){ \
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31]; \
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31]; \
    unsigned sa=s[(threadIdx.x+6)&31], sb=s[(threadIdx.x+7)&31]; (void)sa;(void)sb;(void)a2;(void)a3;(void)b1; \
    float d[NACC][4]; \
    _Pragma("unroll") for(int i=0;i<NACC;i++){d[i][0]=0.f;d[i][1]=0.f;d[i][2]=0.f;d[i][3]=0.f;} \
    __syncthreads(); long long t0=clock64(); \
    for(int it=0;it<ITERS;++it){ _Pragma("unroll") for(int i=0;i<NACC;i++){ BODY } } \
    long long t1=clock64(); float r=0.f; \
    _Pragma("unroll") for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3]; \
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0; \
    o[blockIdx.x*256+threadIdx.x]=r; \
}

// f16-out accumulator: 2 regs, not 4.
#define KERN_F16ACC(NAME, BODY) \
template<int NACC> __global__ void __launch_bounds__(256) \
NAME(const unsigned* s, int* o, long long* c){ \
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31]; \
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31]; \
    unsigned d[NACC][2]; \
    _Pragma("unroll") for(int i=0;i<NACC;i++){d[i][0]=0u;d[i][1]=0u;} \
    __syncthreads(); long long t0=clock64(); \
    for(int it=0;it<ITERS;++it){ _Pragma("unroll") for(int i=0;i<NACC;i++){ BODY } } \
    long long t1=clock64(); int r=0; \
    _Pragma("unroll") for(int i=0;i<NACC;i++) r+=(int)(d[i][0]^d[i][1]); \
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0; \
    o[blockIdx.x*256+threadIdx.x]=r; \
}

KERN_I32(k_s8_k16,   MMA_S8_K16(d[i],a0,a1,b0);)
KERN_I32(k_s8_k32,   MMA_S8_K32(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_bf16,     MMA_BF16_F32(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_f16f32,   MMA_F16_F32(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F16ACC(k_f16f16,MMA_F16_F16ACC(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_tf32k8,   MMA_TF32_K8(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_f8f4_plain, MMA_F8F4_PLAIN(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_blksc_e4m3, MMA_MXF8_BLKSC_E4M3(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)
KERN_F32(k_f8f4_e2m1a, MMA_F8F4_E2M1_A(d[i],a0,a1,a2,a3,b0,b1);)
KERN_F32(k_blksc_e2m1, MMA_MXF8_BLKSC_E2M1(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)
KERN_F32(k_mxf4nvf4,   MMA_MXF4NVF4_4X(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)
KERN_F32(k_mxf4,       MMA_MXF4_2X(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

#define NFORM 12
static const char* FNAME[NFORM] = {
  "A1 m16n8k16.s8.s8.s32",
  "A2 m16n8k32.s8.s8.s32",
  "B1 m16n8k16.f32.bf16.bf16.f32",
  "B2 m16n8k16.f32.f16.f16.f32",
  "B3 m16n8k16.f16.f16.f16.f16",
  "B4 m16n8k8.f32.tf32.tf32.f32",
  "C1 kind::f8f6f4 PLAIN e4m3xe4m3",
  "C2 mxf8f6f4.blksc.1X e4m3xe4m3",
  "C3 kind::f8f6f4 PLAIN e2m1xe4m3",
  "C4 mxf8f6f4.blksc.1X e2m1xe4m3",
  "D1 mxf4nvf4.blksc.4X k64 ue4m3",
  "D2 mxf4.blksc.2X k64 ue8m0",
};
// MACs per warp-MMA = m*n*k = 16*8*k
static const double FMAC[NFORM] = {
  2048, 4096, 2048, 2048, 2048, 1024, 4096, 4096, 4096, 4096, 8192, 8192
};

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    const int SMs=p.multiProcessorCount;
    unsigned hs[32]; for(int i=0;i<32;i++) hs[i]=0x11223344u+i*0x01010101u;
    unsigned* dsrc; int* douti; float* doutf; long long* dcyc;
    CK(cudaMalloc(&dsrc,128)); CK(cudaMemcpy(dsrc,hs,128,cudaMemcpyHostToDevice));
    const int MAXC=256*256; CK(cudaMalloc(&douti,MAXC*4)); CK(cudaMalloc(&doutf,MAXC*4)); CK(cudaMalloc(&dcyc,8));
    printf("# device %s SMs=%d cc=%d.%d (clock LOCKED 1860 via nvidia-smi -lgc)\n",
           p.name,SMs,p.major,p.minor);

    printf("\n## NACC CONTROL -- 1 CTA of 4 warps (1 warp/scheduler), clock64 cyc/warp-MMA\n");
    printf("# flat across NACC => pipe ISSUE INTERVAL; halving as NACC doubles => latency/NACC\n");
    printf("# %-34s %9s %9s %9s %9s %9s\n","form","NACC=1","NACC=2","NACC=4","NACC=8","NACC=16");
    double iv[NFORM][5];
#define RUNI(K,O,N,F,S) do{ K<N><<<1,128>>>(dsrc,O,dcyc); CK(cudaDeviceSynchronize()); \
      long long hc; CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost)); iv[F][S]=hc/((double)ITERS*N); }while(0)
#define SWEEP(K,O,F) do{ RUNI(K,O,1,F,0); RUNI(K,O,2,F,1); RUNI(K,O,4,F,2); RUNI(K,O,8,F,3); RUNI(K,O,16,F,4); \
      printf("  %-34s %9.4f %9.4f %9.4f %9.4f %9.4f\n",FNAME[F],iv[F][0],iv[F][1],iv[F][2],iv[F][3],iv[F][4]); }while(0)
    SWEEP(k_s8_k16,   douti, 0);
    SWEEP(k_s8_k32,   douti, 1);
    SWEEP(k_bf16,     doutf, 2);
    SWEEP(k_f16f32,   doutf, 3);
    SWEEP(k_f16f16,   douti, 4);
    SWEEP(k_tf32k8,   doutf, 5);
    SWEEP(k_f8f4_plain, doutf, 6);
    SWEEP(k_blksc_e4m3, doutf, 7);
    SWEEP(k_f8f4_e2m1a, doutf, 8);
    SWEEP(k_blksc_e2m1, doutf, 9);
    SWEEP(k_mxf4nvf4,   doutf,10);
    SWEEP(k_mxf4,       doutf,11);

    printf("\n## FULL-GPU wall clock (grid=%d CTAs x 256 thr = 8 warps/CTA; NACC=8, best of 5)\n",SMs);
    printf("# %-34s %8s %9s %13s %13s %10s\n","form","MACs","ms","cyc/MMA@1860","T(FL)OP/s","vs A1");
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double base=0;
    for(int f=0;f<NFORM;f++){
        double best=1e18;
        for(int r=0;r<6;r++){
#define LAUNCH(F) do{ switch(F){ \
            case 0: k_s8_k16<8><<<SMs,256>>>(dsrc,douti,dcyc); break; \
            case 1: k_s8_k32<8><<<SMs,256>>>(dsrc,douti,dcyc); break; \
            case 2: k_bf16<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 3: k_f16f32<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 4: k_f16f16<8><<<SMs,256>>>(dsrc,douti,dcyc); break; \
            case 5: k_tf32k8<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 6: k_f8f4_plain<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 7: k_blksc_e4m3<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 8: k_f8f4_e2m1a<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 9: k_blksc_e2m1<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 10: k_mxf4nvf4<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; \
            case 11: k_mxf4<8><<<SMs,256>>>(dsrc,doutf,dcyc); break; } }while(0)
            LAUNCH(f); CK(cudaDeviceSynchronize());
            if(r==0) continue;   // warm
            CK(cudaEventRecord(e0)); LAUNCH(f);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best) best=ms;
        }
        double warp_mmas=(double)SMs*8.0*ITERS*8.0;
        double secs=best/1e3, ops=warp_mmas*FMAC[f]*2.0;
        double cyc=secs*1.86e9*SMs*4.0/warp_mmas;
        if(f==0) base=ops/secs;
        printf("  %-34s %8.0f %9.3f %13.4f %13.1f %9.3fx\n",
               FNAME[f],FMAC[f],best,cyc,ops/secs/1e12,(ops/secs)/base);
    }
    return 0;
}
