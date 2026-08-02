#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2,1,1) ck(int *out) {
  cg::cluster_group c = cg::this_cluster();
  c.sync();
  if (threadIdx.x == 0) out[blockIdx.x] = c.block_rank();
}
__global__ void ck_dyn(int *out) {
  cg::cluster_group c = cg::this_cluster();
  c.sync();
  if (threadIdx.x == 0) out[blockIdx.x] = c.block_rank();
}

int main() {
  setbuf(stdout, NULL);
  int *out; cudaMalloc(&out, 4096*sizeof(int));
  ck<<<82*2, 128>>>(out);
  cudaError_t e = cudaDeviceSynchronize();
  printf("static cluster(2): %s\n", e==cudaSuccess?"OK":cudaGetErrorString(e));
  for (int cs = 2; cs <= 16; cs *= 2) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cs; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(cs*8); cfg.blockDim = dim3(128);
    cfg.attrs = attrs; cfg.numAttrs = 1;
    cudaError_t le = cudaLaunchKernelEx(&cfg, ck_dyn, out);
    cudaError_t se = cudaDeviceSynchronize();
    printf("dyn cluster(%d): launch=%s sync=%s\n", cs, cudaGetErrorString(le), cudaGetErrorString(se));
  }
  cudaLaunchConfig_t qcfg = {};
  qcfg.gridDim = dim3(82); qcfg.blockDim = dim3(128);
  int maxc=0;
  cudaError_t qe = cudaOccupancyMaxPotentialClusterSize(&maxc, (void*)ck_dyn, &qcfg);
  printf("maxPotentialClusterSize(ck_dyn)=%d (%s)\n", maxc, cudaGetErrorString(qe));
  return 0;
}
