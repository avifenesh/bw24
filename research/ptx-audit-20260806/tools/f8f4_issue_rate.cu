// SLICE 3: does the PLAIN kind::f8f6f4 m16n8k32 e4m3 x e4m3 form -- the one the SHIPPED
// mmq_nvfp4_f8f4 tile actually issues (mmq_nvfp4_w4a8.cu:1058) -- run at the 16-cycle tensor-pipe
// issue interval, or does it cost more?
//
// prefill-ilp slice 2b measured 16.06 cyc/warp-MMA for three forms and concluded "1.993x for the
// k32 route". But the form it benched was kind::mxf8f6f4.block_scale.scale_vec::1X with e2m1 x e4m3
// operands and ue8m0 scales. The shipped tile issues the PLAIN kind (no block_scale, no scale regs,
// e4m3 x e4m3). ncu on the live kernel implies 36.17 cyc/warp-MMA for the plain form. This spike
// tests that directly, at the instruction level, independent of the tile.
//
// Same method as spike/mma_issue_rate.cu: clock64() around a tight loop of MUTUALLY INDEPENDENT
// MMAs, sweeping NACC so the ILP control distinguishes issue interval from latency.
// Feasibility spike only. Never linked into the engine.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ITERS 4096

// A: the CURRENT default -- m16n8k16.s8.s8.s32 (the 16.06-cyc reference)
#define MMA_S8_K16(d,a0,a1,b0) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
    : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) : "r"(a0),"r"(a1),"r"(b0))

// B: THE SHIPPED F8F4 TILE'S INSTRUCTION -- plain kind::f8f6f4, e4m3 x e4m3, f32 acc, no scale regs
#define MMA_F8F4_PLAIN(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// C: what prefill-ilp slice 2b ACTUALLY benched -- kind::mxf8f6f4.block_scale.scale_vec::1X,
//    e2m1 x e4m3, ue8m0 scales (the form whose 16.06 produced the "1.993x" claim)
#define MMA_MXF8_BLKSC(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))

// D: plain kind::f8f6f4 with an e2m1 A operand (4-bit weights in 8-bit containers, e4m3 acts) --
//    the R-A variant, to check whether the operand FORMAT or the KIND carries the cost
#define MMA_F8F4_E2M1_A(d,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// E: kind::mxf8f6f4.block_scale.scale_vec::1X with e4m3 x e4m3 -- the SHIPPED TILE'S OPERANDS on
//    the FAST (16-cyc) form. If this assembles and runs at 16.06, the shipped R-B tile can keep its
//    numerics exactly and only swap the MMA form, with the scale operand set to the ue8m0 identity.
#define MMA_MXF8_BLKSC_E4M3(d,a0,a1,a2,a3,b0,b1,sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))

template<int NACC> __global__ void __launch_bounds__(256)
k_s8(const unsigned* s, int* o, long long* c){
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],b0=s[(threadIdx.x+2)&31];
    int d[NACC][4];
#pragma unroll
    for(int i=0;i<NACC;i++){d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0;}
    __syncthreads(); long long t0=clock64();
    for(int it=0;it<ITERS;++it){
#pragma unroll
        for(int i=0;i<NACC;i++) MMA_S8_K16(d[i],a0,a1,b0);
    }
    long long t1=clock64(); int r=0;
#pragma unroll
    for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}

#define FORM_KERNEL(NAME, BODY) \
template<int NACC> __global__ void __launch_bounds__(256) \
NAME(const unsigned* s, float* o, long long* c){ \
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31]; \
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31]; \
    unsigned sa=s[(threadIdx.x+6)&31], sb=s[(threadIdx.x+7)&31]; (void)sa;(void)sb; \
    float d[NACC][4]; \
    _Pragma("unroll") for(int i=0;i<NACC;i++){d[i][0]=0.f;d[i][1]=0.f;d[i][2]=0.f;d[i][3]=0.f;} \
    __syncthreads(); long long t0=clock64(); \
    for(int it=0;it<ITERS;++it){ _Pragma("unroll") for(int i=0;i<NACC;i++){ BODY } } \
    long long t1=clock64(); float r=0.f; \
    _Pragma("unroll") for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3]; \
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0; \
    o[blockIdx.x*256+threadIdx.x]=r; \
}

FORM_KERNEL(k_f8f4_plain,  MMA_F8F4_PLAIN(d[i],a0,a1,a2,a3,b0,b1);)
FORM_KERNEL(k_mxf8_blksc,  MMA_MXF8_BLKSC(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)
FORM_KERNEL(k_f8f4_e2m1a,  MMA_F8F4_E2M1_A(d[i],a0,a1,a2,a3,b0,b1);)
FORM_KERNEL(k_blksc_e4m3,  MMA_MXF8_BLKSC_E4M3(d[i],a0,a1,a2,a3,b0,b1,sa,sb);)

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    const int SMs=p.multiProcessorCount;
    unsigned hs[32]; for(int i=0;i<32;i++) hs[i]=0x11223344u+i*0x01010101u;
    unsigned* dsrc; int* douti; float* doutf; long long* dcyc;
    CK(cudaMalloc(&dsrc,128)); CK(cudaMemcpy(dsrc,hs,128,cudaMemcpyHostToDevice));
    const int MAXC=82*256; CK(cudaMalloc(&douti,MAXC*4)); CK(cudaMalloc(&doutf,MAXC*4)); CK(cudaMalloc(&dcyc,8));
    printf("# device %s SMs=%d (clock LOCKED 1860 via nvidia-smi -lgc)\n",p.name,SMs);

    printf("\n## CONTROL -- NACC sweep, 1 CTA of 4 warps (1 warp/scheduler), clock64 cyc/warp-MMA\n");
    printf("# flat across NACC => pipe ISSUE INTERVAL; halving as NACC doubles => latency/NACC\n");
    printf("# %-5s %13s %15s %15s %15s %16s\n","NACC","s8.k16","f8f4plain.k32","mxf8blksc.k32","f8f4e2m1A.k32","blksc_e4m3.k32");
    long long hc; double v[5];
#define RUNI(K,O,N,S) do{ K<N><<<1,128>>>(dsrc,O,dcyc); CK(cudaDeviceSynchronize()); \
      CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost)); v[S]=hc/((double)ITERS*N); }while(0)
#define ROW(N) do{ RUNI(k_s8,douti,N,0); RUNI(k_f8f4_plain,doutf,N,1); RUNI(k_mxf8_blksc,doutf,N,2); \
      RUNI(k_f8f4_e2m1a,doutf,N,3); RUNI(k_blksc_e4m3,doutf,N,4); \
      printf("  %-5d %13.4f %15.4f %15.4f %15.4f %16.4f\n",N,v[0],v[1],v[2],v[3],v[4]); }while(0)
    ROW(1); ROW(2); ROW(4); ROW(8); ROW(16);

    printf("\n## FULL-GPU wall clock (grid=%d CTAs x 256 thr = 8 warps/CTA, the SHIPPED shape; NACC=8, best of 5)\n",SMs);
    printf("# MACs/warp-MMA: s8.k16=2048, all k32 forms=4096\n");
    printf("# %-30s %10s %14s %14s %10s\n","form","ms","cyc/MMA@1860","T(FL)OP/s","vs s8");
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double base=0;
    for(int form=0;form<5;form++){
        double best=1e18;
        for(int r=0;r<6;r++){
            if(form==0) k_s8<8><<<SMs,256>>>(dsrc,douti,dcyc);
            if(form==1) k_f8f4_plain<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==2) k_mxf8_blksc<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==3) k_f8f4_e2m1a<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==4) k_blksc_e4m3<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaDeviceSynchronize());
            if(r==0) continue;   // warm
            CK(cudaEventRecord(e0));
            if(form==0) k_s8<8><<<SMs,256>>>(dsrc,douti,dcyc);
            if(form==1) k_f8f4_plain<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==2) k_mxf8_blksc<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==3) k_f8f4_e2m1a<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==4) k_blksc_e4m3<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best) best=ms;
        }
        double macs = form==0?2048.0:4096.0;
        double warp_mmas=(double)SMs*8.0*ITERS*8.0;
        double secs=best/1e3, ops=warp_mmas*macs*2.0;
        double cyc=secs*1.86e9*SMs*4.0/warp_mmas;
        const char* nm = form==0?"m16n8k16.s8.s8.s32":(form==1?"f8f6f4 PLAIN e4m3xe4m3":(form==2?"mxf8f6f4 blkscale e2m1xe4m3":(form==3?"f8f6f4 PLAIN e2m1xe4m3":"mxf8f6f4 blkscale e4m3xe4m3")));
        if(form==0) base=ops/secs;
        printf("  %-30s %10.3f %14.4f %14.1f %10.3fx\n",nm,best,cyc,ops/secs/1e12,(ops/secs)/base);
    }
    return 0;
}
