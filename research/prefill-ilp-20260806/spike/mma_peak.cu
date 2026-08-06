// PROBE 2 (c) part 2: two controls the cycles/MMA number NEEDS before it can be called an
// "issue interval", plus the absolute per-form peak.
//
// CONTROL 1 (NACC sweep): a += chain on accumulator i is loop-carried across iterations. If the
//   measured cycles/MMA were LATENCY/NACC rather than the pipe's issue interval, doubling NACC
//   would halve it. Sweeping NACC = 1,2,4,8,16 separates the two.
// CONTROL 2 (full-GPU wall-clock throughput): clock64() on one CTA cannot see the SM-wide or
//   chip-wide rate. Launch enough CTAs to fill all 82 SMs and time with cudaEvent, then convert
//   to TOP/s / TFLOP/s so the interval can be checked against a physical peak.
//
// Feasibility spike only. Never linked into the engine.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ITERS 4096

#define MMA_S8_K16(D,A0,A1,B0) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
    : "+r"(D[0]),"+r"(D[1]),"+r"(D[2]),"+r"(D[3]) : "r"(A0),"r"(A1),"r"(B0))

#define MMA_MXF4_K64(D,A0,A1,A2,A3,B0,B1,SA,SB) \
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]) \
    : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(SA),"r"(SB))

#define MMA_MXF8_K32(D,A0,A1,A2,A3,B0,B1,SA,SB) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]) \
    : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(SA),"r"(SB))

template<int NACC>
__global__ void __launch_bounds__(256) k_s8(const unsigned* __restrict__ s, int* __restrict__ o, long long* __restrict__ c) {
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],b0=s[(threadIdx.x+2)&31];
    int d[NACC][4];
#pragma unroll
    for(int i=0;i<NACC;i++){d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0;}
    __syncthreads();
    long long t0=clock64();
    for(int it=0;it<ITERS;++it){
#pragma unroll
        for(int i=0;i<NACC;i++) MMA_S8_K16(d[i],a0,a1,b0);
    }
    long long t1=clock64();
    int r=0;
#pragma unroll
    for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}
template<int NACC>
__global__ void __launch_bounds__(256) k_mxf4(const unsigned* __restrict__ s, float* __restrict__ o, long long* __restrict__ c) {
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31];
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31],sa=s[(threadIdx.x+6)&31],sb=s[(threadIdx.x+7)&31];
    float d[NACC][4];
#pragma unroll
    for(int i=0;i<NACC;i++){d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0;}
    __syncthreads();
    long long t0=clock64();
    for(int it=0;it<ITERS;++it){
#pragma unroll
        for(int i=0;i<NACC;i++) MMA_MXF4_K64(d[i],a0,a1,a2,a3,b0,b1,sa,sb);
    }
    long long t1=clock64();
    float r=0;
#pragma unroll
    for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}
template<int NACC>
__global__ void __launch_bounds__(256) k_mxf8(const unsigned* __restrict__ s, float* __restrict__ o, long long* __restrict__ c) {
    unsigned a0=s[threadIdx.x&31],a1=s[(threadIdx.x+1)&31],a2=s[(threadIdx.x+2)&31],a3=s[(threadIdx.x+3)&31];
    unsigned b0=s[(threadIdx.x+4)&31],b1=s[(threadIdx.x+5)&31],sa=s[(threadIdx.x+6)&31],sb=s[(threadIdx.x+7)&31];
    float d[NACC][4];
#pragma unroll
    for(int i=0;i<NACC;i++){d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0;}
    __syncthreads();
    long long t0=clock64();
    for(int it=0;it<ITERS;++it){
#pragma unroll
        for(int i=0;i<NACC;i++) MMA_MXF8_K32(d[i],a0,a1,a2,a3,b0,b1,sa,sb);
    }
    long long t1=clock64();
    float r=0;
#pragma unroll
    for(int i=0;i<NACC;i++) r+=d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if(threadIdx.x==0&&blockIdx.x==0) c[0]=t1-t0;
    o[blockIdx.x*256+threadIdx.x]=r;
}

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

static unsigned* dsrc; static int* douti; static float* doutf; static long long* dcyc;
static int SMs;

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    SMs=p.multiProcessorCount;
    unsigned hs[32]; for(int i=0;i<32;i++) hs[i]=0x11223344u+i*0x01010101u;
    CK(cudaMalloc(&dsrc,128)); CK(cudaMemcpy(dsrc,hs,128,cudaMemcpyHostToDevice));
    int MAXC = 82*8*256; CK(cudaMalloc(&douti,MAXC*4)); CK(cudaMalloc(&doutf,MAXC*4)); CK(cudaMalloc(&dcyc,8));
    printf("# device %s  SMs=%d  (measurement clock LOCKED 1860 MHz via nvidia-smi -lgc)\n",p.name,SMs);

    // ---------------- CONTROL 1: NACC sweep at 4 warps/CTA = 1 warp/scheduler ----------------
    printf("\n## CONTROL 1 -- NACC sweep, 1 CTA of 4 warps (1 warp/scheduler), clock64 cycles/warp-MMA\n");
    printf("# if cycles/MMA == LATENCY/NACC it HALVES as NACC doubles; if it is the pipe ISSUE INTERVAL it stays flat\n");
    printf("# %-6s %14s %14s %14s\n","NACC","s8.k16","mxf4.k64","mxf8.k32");
    long long hc; double v[3];
#define RUN1(KERN,OUT,NACC,SLOT) do{ \
      KERN<NACC><<<1,128>>>(dsrc,OUT,dcyc); CK(cudaDeviceSynchronize()); \
      CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost)); v[SLOT]=hc/((double)ITERS*NACC); }while(0)
#define ROW(NACC) do{ RUN1(k_s8,douti,NACC,0); RUN1(k_mxf4,doutf,NACC,1); RUN1(k_mxf8,doutf,NACC,2); \
      printf("  %-6d %14.4f %14.4f %14.4f\n",NACC,v[0],v[1],v[2]); }while(0)
    ROW(1); ROW(2); ROW(4); ROW(8); ROW(16);

    // ---------------- CONTROL 2: full-GPU wall-clock throughput ----------------
    // Grid = SMs * ctas_per_sm; 256 threads = 8 warps/CTA = the SHIPPED warp shape.
    printf("\n## CONTROL 2 -- full-GPU wall-clock throughput (grid = %d CTAs x 256 thr, NACC=8)\n", SMs);
    printf("# MACs/warp-MMA: s8.k16=16*8*16=2048  mxf4.k64=16*8*64=8192  mxf8.k32=16*8*32=4096\n");
    printf("# %-34s %10s %12s %14s %12s\n","form","ms(best5)","cyc/MMA@1860","TOP or TFLOP/s","vs s8");
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double base_ops=0;
    for (int form=0; form<3; ++form) {
        double bestms=1e18;
        for (int r=0;r<5;r++){
            if(form==0) k_s8<8><<<SMs,256>>>(dsrc,douti,dcyc);
            if(form==1) k_mxf4<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==2) k_mxf8<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            if(form==0) k_s8<8><<<SMs,256>>>(dsrc,douti,dcyc);
            if(form==1) k_mxf4<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            if(form==2) k_mxf8<8><<<SMs,256>>>(dsrc,doutf,dcyc);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<bestms) bestms=ms;
        }
        double macs_per_mma = form==0?2048.0:(form==1?8192.0:4096.0);
        double warp_mmas = (double)SMs*8.0*ITERS*8.0;               // CTAs * warps/CTA * ITERS * NACC
        double ops = warp_mmas*macs_per_mma*2.0;                     // 2 ops per MAC
        double secs = bestms/1e3;
        // cycles/warp-MMA per scheduler at the locked 1860 MHz:
        //   total scheduler-cycles available = secs*1.86e9 * SMs * 4 schedulers
        double cyc_per_mma = secs*1.86e9*SMs*4.0/warp_mmas;
        const char* nm = form==0?"m16n8k16.s8.s8.s32":(form==1?"m16n8k64.mxf4nvf4.4X":"m16n8k32.mxf8f6f4 e2m1xe4m3");
        if(form==0) base_ops=ops/secs;
        printf("  %-34s %10.3f %12.4f %14.1f %12.3fx\n",nm,bestms,cyc_per_mma,ops/secs/1e12,(ops/secs)/base_ops);
    }
    printf("\n# NOTE: s8 row is TOP/s (integer); mxf4/mxf8 rows are TFLOP/s (float). The 'vs s8' column\n");
    printf("#       is the ratio of raw MAC-rate, which is the quantity the GEMM's runtime depends on.\n");
    return 0;
}
