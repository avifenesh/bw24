// Minimal CLC work-stealing test on sm_120a, based on CUDA Programming Guide 4.12
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cstdio>
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ void clc_kernel(int *counts, int n) {
  __shared__ uint4 result;
  __shared__ uint64_t bar;
  int phase = 0;
  if (cg::thread_block::thread_rank() == 0) ptx::mbarrier_init(&bar, 1);

  int bx = blockIdx.x;
  while (true) {
    __syncthreads();
    if (cg::thread_block::thread_rank() == 0) {
      ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
      cg::invoke_one(cg::coalesced_threads(), [&]() {
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
      });
      ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
    }
    // "work": each block increments its slot
    int i = bx * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[i], 1);

    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
    phase ^= 1;
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) break;
    bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  }
}

int main() {
  const int TB = 256;
  const int NB = 4096;           // way more blocks than SMs -> stealing must happen
  const int n = NB * TB;
  int *counts; cudaMalloc(&counts, n * sizeof(int));
  cudaMemset(counts, 0, n * sizeof(int));
  clc_kernel<<<NB, TB>>>(counts, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("KERNEL FAILED: %s\n", cudaGetErrorString(e)); return 1; }
  int *h = (int*)malloc(n * sizeof(int));
  cudaMemcpy(h, counts, n * sizeof(int), cudaMemcpyDeviceToHost);
  long long bad = 0, sum = 0;
  for (int i = 0; i < n; i++) { sum += h[i]; if (h[i] != 1) bad++; }
  printf("CLC RUN: sum=%lld expected=%d wrong_slots=%lld -> %s\n", sum, n, bad, bad == 0 ? "CORRECT (every index executed exactly once)" : "INCORRECT");
  return bad != 0;
}
