// PROBE 2 (c): measure the ISSUE INTERVAL (cycles per warp-MMA) of each candidate MMA form on
// sm_120a, by the same method that found the 16.00 for m16n8k16.s8 -- but directly, with clock64()
// around a tight loop of MUTUALLY INDEPENDENT MMAs (independent accumulators, so latency is hidden
// and what remains is the pipe's issue rate).
//
// Feasibility spike only. Never linked into the engine.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define NACC 8      // independent accumulators -> ILP saturates the pipe
#define ITERS 4096  // outer iterations

// ---- m16n8k16.s8.s8.s32 : the CURRENT kernel's instruction ----
__global__ void __launch_bounds__(256) bench_s8_k16(const uint32_t* __restrict__ src, int* __restrict__ out, long long* __restrict__ cyc) {
    uint32_t a0=src[threadIdx.x&31], a1=src[(threadIdx.x+1)&31];
    uint32_t b0=src[(threadIdx.x+2)&31];
    int d[NACC][4];
#pragma unroll
    for (int i=0;i<NACC;i++){ d[i][0]=0;d[i][1]=0;d[i][2]=0;d[i][3]=0; }
    __syncthreads();
    long long t0 = clock64();
    for (int it=0; it<ITERS; ++it) {
#pragma unroll
        for (int i=0;i<NACC;i++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
              : "+r"(d[i][0]),"+r"(d[i][1]),"+r"(d[i][2]),"+r"(d[i][3]) : "r"(a0),"r"(a1),"r"(b0));
        }
    }
    long long t1 = clock64();
    int s=0;
#pragma unroll
    for (int i=0;i<NACC;i++) s += d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if (threadIdx.x==0 && blockIdx.x==0) { cyc[0]=t1-t0; }
    out[blockIdx.x*256+threadIdx.x]=s;
}

// ---- m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X : THE DOOR ----
__global__ void __launch_bounds__(256) bench_mxf4_k64(const uint32_t* __restrict__ src, float* __restrict__ out, long long* __restrict__ cyc) {
    uint32_t a0=src[threadIdx.x&31],a1=src[(threadIdx.x+1)&31],a2=src[(threadIdx.x+2)&31],a3=src[(threadIdx.x+3)&31];
    uint32_t b0=src[(threadIdx.x+4)&31],b1=src[(threadIdx.x+5)&31];
    uint32_t sa=src[(threadIdx.x+6)&31], sb=src[(threadIdx.x+7)&31];
    float d[NACC][4];
#pragma unroll
    for (int i=0;i<NACC;i++){ d[i][0]=0.f;d[i][1]=0.f;d[i][2]=0.f;d[i][3]=0.f; }
    __syncthreads();
    long long t0 = clock64();
    for (int it=0; it<ITERS; ++it) {
#pragma unroll
        for (int i=0;i<NACC;i++) {
            asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
              : "+f"(d[i][0]),"+f"(d[i][1]),"+f"(d[i][2]),"+f"(d[i][3])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
        }
    }
    long long t1 = clock64();
    float s=0.f;
#pragma unroll
    for (int i=0;i<NACC;i++) s += d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if (threadIdx.x==0 && blockIdx.x==0) { cyc[0]=t1-t0; }
    out[blockIdx.x*256+threadIdx.x]=s;
}

// ---- m16n8k32.kind::mxf8f6f4 (e2m1 x e4m3) : the W4A8-preserving variant ----
__global__ void __launch_bounds__(256) bench_mxf8_k32(const uint32_t* __restrict__ src, float* __restrict__ out, long long* __restrict__ cyc) {
    uint32_t a0=src[threadIdx.x&31],a1=src[(threadIdx.x+1)&31],a2=src[(threadIdx.x+2)&31],a3=src[(threadIdx.x+3)&31];
    uint32_t b0=src[(threadIdx.x+4)&31],b1=src[(threadIdx.x+5)&31];
    uint32_t sa=src[(threadIdx.x+6)&31], sb=src[(threadIdx.x+7)&31];
    float d[NACC][4];
#pragma unroll
    for (int i=0;i<NACC;i++){ d[i][0]=0.f;d[i][1]=0.f;d[i][2]=0.f;d[i][3]=0.f; }
    __syncthreads();
    long long t0 = clock64();
    for (int it=0; it<ITERS; ++it) {
#pragma unroll
        for (int i=0;i<NACC;i++) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
              "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
              : "+f"(d[i][0]),"+f"(d[i][1]),"+f"(d[i][2]),"+f"(d[i][3])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
        }
    }
    long long t1 = clock64();
    float s=0.f;
#pragma unroll
    for (int i=0;i<NACC;i++) s += d[i][0]+d[i][1]+d[i][2]+d[i][3];
    if (threadIdx.x==0 && blockIdx.x==0) { cyc[0]=t1-t0; }
    out[blockIdx.x*256+threadIdx.x]=s;
}

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n", cudaGetErrorString(e), __LINE__); return 1;} }while(0)

int main(int argc, char** argv) {
    // WARPS_PER_CTA controls warps/scheduler: 256 threads = 8 warps = 2/scheduler (the SHIPPED shape).
    int nwarps = argc>1 ? atoi(argv[1]) : 8;
    int threads = nwarps*32;
    uint32_t hsrc[32]; for(int i=0;i<32;i++) hsrc[i]=0x11223344u+i*0x01010101u;
    uint32_t* dsrc; CK(cudaMalloc(&dsrc,32*4)); CK(cudaMemcpy(dsrc,hsrc,32*4,cudaMemcpyHostToDevice));
    int* douti; float* doutf; long long* dcyc;
    CK(cudaMalloc(&douti,1024*4)); CK(cudaMalloc(&doutf,1024*4)); CK(cudaMalloc(&dcyc,8));
    long long hc; double mmas_per_warp = (double)ITERS*NACC;

    printf("# nwarps/CTA=%d (warps/scheduler=%.2f), 1 CTA, NACC=%d independent accumulators, ITERS=%d\n",
           nwarps, nwarps/4.0, NACC, ITERS);
    printf("# %-52s %12s %14s\n","instruction","cycles","cyc/warp-MMA");

    for (int rep=0; rep<3; ++rep) {
      // s8 k16
      bench_s8_k16<<<1,threads>>>(dsrc,douti,dcyc); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost));
      printf("rep%d %-52s %12lld %14.4f\n",rep,"m16n8k16.s8.s8.s32",hc,hc/mmas_per_warp);
      // mxf4 k64
      bench_mxf4_k64<<<1,threads>>>(dsrc,doutf,dcyc); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost));
      printf("rep%d %-52s %12lld %14.4f\n",rep,"m16n8k64.mxf4nvf4.block_scale.4X",hc,hc/mmas_per_warp);
      // mxf8 k32
      bench_mxf8_k32<<<1,threads>>>(dsrc,doutf,dcyc); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&hc,dcyc,8,cudaMemcpyDeviceToHost));
      printf("rep%d %-52s %12lld %14.4f\n",rep,"m16n8k32.mxf8f6f4 e2m1xe4m3",hc,hc/mmas_per_warp);
    }
    return 0;
}
