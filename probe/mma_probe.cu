#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// Probe: does the assembler accept these PTX MMA instructions for the target arch?
// We don't need correct math; we test that ptxas emits them (capability gate).
__global__ void probe_fp16_mma(){
  // m16n8k16 f16 -> f32, classic Ampere+ tensor core
  unsigned a[4]={0}, b[2]={0}; float c[4]={0};
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  if(c[0]==123456.0f) printf("x");
}
int main(){ probe_fp16_mma<<<1,32>>>(); cudaDeviceSynchronize();
  printf("fp16_mma launch=%s\n", cudaGetErrorString(cudaGetLastError())); return 0; }
