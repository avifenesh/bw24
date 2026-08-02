#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2,1,1) dsmem_k(int *out) {
  cg::cluster_group c = cg::this_cluster();
  __shared__ int v;
  if (threadIdx.x == 0) v = 1000 + c.block_rank();
  c.sync();
  // read the OTHER block's shared memory
  int peer = c.block_rank() ^ 1;
  int *pv = c.map_shared_rank(&v, peer);
  int got = *pv;
  c.sync();
  if (threadIdx.x == 0) out[blockIdx.x] = got;
}

int main() {
  setbuf(stdout, NULL);
  int *out; cudaMalloc(&out, 8*sizeof(int));
  dsmem_k<<<8, 64>>>(out);
  cudaError_t e = cudaDeviceSynchronize();
  int h[8]; cudaMemcpy(h, out, 8*sizeof(int), cudaMemcpyDeviceToHost);
  printf("dsmem: %s | vals:", e==cudaSuccess?"OK":cudaGetErrorString(e));
  for (int i=0;i<8;i++) printf(" %d", h[i]);
  printf("\n");
  // expect alternating 1001 1000 1001 1000 ... (each block reads peer's value)
  return 0;
}
