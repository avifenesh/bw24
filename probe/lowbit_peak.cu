#include <cstdio>
#include <cuda_runtime.h>
template<int IT> __global__ void k_fp4(float* s){          // mxf4 m16n8k64 block-scale, f32 acc
  unsigned a[4]={1,2,3,4},b[2]={5,6},sa=1,sb=1; float c0[4]={0},c1[4]={0};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};":"+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};":"+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
  }
  if(c0[0]==-1.f)s[0]=c0[0]+c1[0];
}
template<int IT> __global__ void k_mxf8(float* s){         // mxf8f6f4 m16n8k32 block-scale FP8, f32 acc
  unsigned a[4]={1,2,3,4},b[2]={5,6},sa=1,sb=1; float c0[4]={0},c1[4]={0};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};":"+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};":"+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
  }
  if(c0[0]==-1.f)s[0]=c0[0]+c1[0];
}
template<typename F> double run(F k,int bl,int th,double fpw,int it){
  float* s; cudaMalloc(&s,4); k<<<bl,th>>>(s); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
  int reps=10; cudaEventRecord(a); for(int r=0;r<reps;r++)k<<<bl,th>>>(s); cudaEventRecord(b); cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms,a,b); int w=bl*th/32;
  return (double)w*fpw*it*reps/(ms/1e3)/1e12;
}
int main(){
  const int IT=4096; int bl=82*4,th=256;
  double f4  = run(k_fp4<IT>, bl,th, 2.0*16*8*64*2, IT);  // 2 mma/iter
  double f8b = run(k_mxf8<IT>,bl,th, 2.0*16*8*32*2, IT);
  printf("FP4 mxf4 block-scale peak  : %.0f TFLOP/s\n", f4);
  printf("FP8 mxf8f6f4 block-scale   : %.0f TFLOP/s\n", f8b);
  printf("(ref measured: FP16=117, plain-FP8-f32acc=219)\n");
  printf("FP4/FP16 = %.2fx ; FP4/FP8plain = %.2fx ; FP8block/FP8plain = %.2fx\n", f4/117.0, f4/219.0, f8b/219.0);
  return 0;
}
