// Exact CLC pattern destined for cu/mmq_q4_0.cu: no cooperative_groups, single-thread
// elect via threadIdx, 2D ctaid recovery, 256-thread (32x8) block like the MMQ kernel.
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cstdio>
namespace ptx = cuda::ptx;

__launch_bounds__(256, 1)
__global__ void clc_pattern(int *counts, int nx, unsigned long long *steals) {
  __shared__ uint4 clc_response;
  __shared__ uint64_t clc_bar;
  int clc_phase = 0;
  if (threadIdx.x == 0 && threadIdx.y == 0) ptx::mbarrier_init(&clc_bar, 1);

  int it = blockIdx.x;
  int jt = blockIdx.y;
  int local_steals = 0;
  while (true) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      ptx::fence_proxy_async_generic_sync_restrict(
          ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
      ptx::clusterlaunchcontrol_try_cancel(&clc_response, &clc_bar);
      ptx::mbarrier_arrive_expect_tx(
          ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &clc_bar, sizeof(uint4));
    }
    // === per-tile work (stand-in for mul_mat_q_process_tile) ===
    int tile = jt * nx + it;
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(&counts[tile], 1);
    for (volatile int w = 0; w < 2000; ++w) {}
    // === end work ===
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &clc_bar, clc_phase)) {}
    clc_phase ^= 1;
    if (!ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_response)) break;
    it = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_response);
    jt = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y<int>(clc_response);
    local_steals++;
    ptx::fence_proxy_async_generic_sync_restrict(
        ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0 && local_steals > 0)
    atomicAdd(steals, (unsigned long long) local_steals);
}

int main() {
  const int NX = 96, NY = 14;   // 1344 tiles, ffn_gate-out=12288-class row tiling
  const int n = NX * NY;
  int *counts; cudaMalloc(&counts, n * sizeof(int)); cudaMemset(counts, 0, n * sizeof(int));
  unsigned long long *steals; cudaMalloc(&steals, 8); cudaMemset(steals, 0, 8);
  dim3 grid(NX, NY, 1), block(32, 8, 1);
  clc_pattern<<<grid, block>>>(counts, NX, steals);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("KERNEL FAILED: %s\n", cudaGetErrorString(e)); return 1; }
  int *h = (int*)malloc(n * sizeof(int));
  cudaMemcpy(h, counts, n * sizeof(int), cudaMemcpyDeviceToHost);
  unsigned long long hs = 0; cudaMemcpy(&hs, steals, 8, cudaMemcpyDeviceToHost);
  long long bad = 0, sum = 0;
  for (int i = 0; i < n; i++) { sum += h[i]; if (h[i] != 1) bad++; }
  printf("CLC MMQ-pattern: tiles=%d sum=%lld wrong=%lld steals=%llu -> %s, stealing %s\n",
         n, sum, bad, hs, bad == 0 ? "CORRECT" : "INCORRECT", hs > 0 ? "ACTIVE" : "NEVER ENGAGED");
  return bad != 0;
}
