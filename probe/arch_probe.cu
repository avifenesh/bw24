#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(){}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int memclk=0, busw=0, l2=0;
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);   // kHz
  cudaDeviceGetAttribute(&busw,   cudaDevAttrGlobalMemoryBusWidth, 0); // bits
  cudaDeviceGetAttribute(&l2,     cudaDevAttrL2CacheSize, 0);
  printf("name=%s\n", p.name);
  printf("cc=%d.%d  sms=%d  warp=%d\n", p.major, p.minor, p.multiProcessorCount, p.warpSize);
  printf("smem/block(optin)=%zuKB  smem/sm=%zuKB  regs/sm=%d  regs/block=%d\n",
         p.sharedMemPerBlockOptin/1024, p.sharedMemPerMultiprocessor/1024, p.regsPerMultiprocessor, p.regsPerBlock);
  printf("maxThreads/sm=%d  maxThreads/block=%d\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
  printf("gmem=%.2fGB  l2=%dMB  membus=%dbit  memclk=%dMHz\n", p.totalGlobalMem/1e9, l2/(1024*1024), busw, memclk/1000);
  double bw = 2.0 * (memclk*1000.0) * (busw/8.0) / 1e9;
  printf("peak_mem_bw~=%.0f GB/s\n", bw);
  printf("clusterLaunch=%d  asyncEngines=%d  cooperativeLaunch=%d\n", p.clusterLaunch, p.asyncEngineCount, p.cooperativeLaunch);
  k<<<1,1>>>(); cudaDeviceSynchronize();
  printf("launch_err=%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
