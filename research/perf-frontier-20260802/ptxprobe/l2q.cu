#include <cstdio>
int main(){
  int dev=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
  int persist=0, maxwin=0, clusterlaunch=0, dsmem=0;
  cudaDeviceGetAttribute(&persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
  cudaDeviceGetAttribute(&maxwin, cudaDevAttrMaxAccessPolicyWindowSize, dev);
  cudaDeviceGetAttribute(&clusterlaunch, cudaDevAttrClusterLaunch, dev);
  printf("name=%s cc=%d.%d\n", p.name, p.major, p.minor);
  printf("L2 size=%d bytes (%.1f MB)\n", (int)p.l2CacheSize, p.l2CacheSize/1048576.0);
  printf("maxPersistingL2CacheSize=%d bytes (%.1f MB)\n", persist, persist/1048576.0);
  printf("maxAccessPolicyWindowSize=%d bytes (%.1f MB)\n", maxwin, maxwin/1048576.0);
  printf("clusterLaunch=%d\n", clusterlaunch);
  int gcSupported=-1;
  #ifdef cudaDevAttrGreenContextSupport
  #endif
  return 0;
}
