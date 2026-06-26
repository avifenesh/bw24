#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{auto e=(x); if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}}while(0)
__global__ void readbw(const float4* __restrict__ in, float4* __restrict__ out, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  float4 acc = make_float4(0,0,0,0);
  for(; i<n; i+=stride){ float4 v=in[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
  if(acc.x==-1.0f) out[0]=acc; // sink
}
int main(){
  size_t bytes = 4ull*1024*1024*1024; // 4GB ~ 7B-Q4 footprint
  size_t n = bytes/sizeof(float4);
  float4 *in,*out; CK(cudaMalloc(&in,bytes)); CK(cudaMalloc(&out,1024));
  CK(cudaMemset(in,1,bytes));
  int threads=256, blocks=82*16;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  readbw<<<blocks,threads>>>(in,out,n); CK(cudaDeviceSynchronize()); // warm
  int iters=20; cudaEventRecord(a);
  for(int it=0;it<iters;it++) readbw<<<blocks,threads>>>(in,out,n);
  cudaEventRecord(b); CK(cudaDeviceSynchronize());
  float ms; cudaEventElapsedTime(&ms,a,b);
  double gbps = (double)bytes*iters/(ms/1e3)/1e9;
  printf("read BW: %.0f GB/s  (%.1f%% of 896 peak)  over %.1fGB x%d in %.1fms\n",
         gbps, gbps/896.0*100, bytes/1e9, iters, ms);
  return 0;
}
