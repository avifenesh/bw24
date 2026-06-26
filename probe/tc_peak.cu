#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{auto e=(x);if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}}while(0)
// Tight loop of independent mma.sync to saturate tensor cores.
// Each warp issues ITERS mma's into 2 independent accumulators (hide latency).
template<int ITERS>
__global__ void fp16_peak(float* sink){
  unsigned a[4]={1,2,3,4}, b[2]={5,6};
  float c0[4]={0,0,0,0}, c1[4]={0,0,0,0};
  #pragma unroll 1
  for(int i=0;i<ITERS;i++){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  if(c0[0]==-1.0f) sink[0]=c0[0]+c1[0];
}
template<int ITERS>
__global__ void fp8_peak(float* sink){
  unsigned a[4]={1,2,3,4}, b[2]={5,6};
  float c0[4]={0,0,0,0}, c1[4]={0,0,0,0};
  #pragma unroll 1
  for(int i=0;i<ITERS;i++){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  if(c0[0]==-1.0f) sink[0]=c0[0]+c1[0];
}
template<typename F>
double run(F k, int blocks,int threads,double flops_per_iter_per_warp,int iters){
  float* s; cudaMalloc(&s,4);
  k<<<blocks,threads>>>(s); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  int reps=10; cudaEventRecord(a);
  for(int r=0;r<reps;r++) k<<<blocks,threads>>>(s);
  cudaEventRecord(b); cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms,a,b);
  int warps = blocks*threads/32;
  double tflop = (double)warps*flops_per_iter_per_warp*iters*reps/(ms/1e3)/1e12;
  return tflop;
}
int main(){
  const int ITERS=4096;
  int blocks=82*4, threads=256; // saturate
  // fp16 m16n8k16: 2 mma/iter, each 16*8*16*2 FLOP = 4096; x2 = 8192 FLOP/iter/warp
  double t16 = run(fp16_peak<ITERS>, blocks,threads, 2.0*16*8*16*2, ITERS);
  // fp8 m16n8k32: each 16*8*32*2=8192; x2 = 16384 FLOP/iter/warp
  double t8  = run(fp8_peak<ITERS>,  blocks,threads, 2.0*16*8*32*2, ITERS);
  printf("FP16 tensor peak: %.0f TFLOP/s\n", t16);
  printf("FP8  tensor peak: %.0f TFLOP/s  (%.2fx fp16)\n", t8, t8/t16);
  printf("note: FP4 ~2x FP8 expected; block-scale mma needs more setup, separate bench\n");
  return 0;
}
