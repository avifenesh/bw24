// CLC steal-count probe: does try_cancel actually SUCCEED on sm_120a (not just assemble/run)?
// Also probes 2D grids (ctaid.y recovery) — the MMQ grid is (nty, ntx, 1).
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cstdio>
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ void clc_kernel2d(int *counts, int nx, int ny, unsigned long long *steals) {
  __shared__ uint4 result;
  __shared__ uint64_t bar;
  int phase = 0;
  if (cg::thread_block::thread_rank() == 0) ptx::mbarrier_init(&bar, 1);

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int local_steals = 0;
  while (true) {
    __syncthreads();
    if (cg::thread_block::thread_rank() == 0) {
      ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
      cg::invoke_one(cg::coalesced_threads(), [&]() {
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
      });
      ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
    }
    // "work": mark the (bx, by) slot
    int tile = by * nx + bx;
    if (threadIdx.x == 0) atomicAdd(&counts[tile], 1);
    // burn a little so blocks do not all finish instantly
    for (volatile int w = 0; w < 2000; ++w) {}

    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
    phase ^= 1;
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) break;
    bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
    by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y<int>(result);
    local_steals++;
    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  }
  if (threadIdx.x == 0 && local_steals > 0) atomicAdd(steals, (unsigned long long) local_steals);
}

int main() {
  const int NX = 64, NY = 24;             // 1536 tiles > resident capacity
  const int n = NX * NY;
  int *counts; cudaMalloc(&counts, n * sizeof(int));
  cudaMemset(counts, 0, n * sizeof(int));
  unsigned long long *steals; cudaMalloc(&steals, 8);
  cudaMemset(steals, 0, 8);
  dim3 grid(NX, NY, 1), block(256, 1, 1);
  clc_kernel2d<<<grid, block>>>(counts, NX, NY, steals);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("KERNEL FAILED: %s\n", cudaGetErrorString(e)); return 1; }
  int *h = (int*)malloc(n * sizeof(int));
  cudaMemcpy(h, counts, n * sizeof(int), cudaMemcpyDeviceToHost);
  unsigned long long hs = 0;
  cudaMemcpy(&hs, steals, 8, cudaMemcpyDeviceToHost);
  long long bad = 0, sum = 0;
  for (int i = 0; i < n; i++) { sum += h[i]; if (h[i] != 1) bad++; }
  printf("CLC 2D: tiles=%d sum=%lld wrong=%lld steals=%llu -> %s, stealing %s\n",
         n, sum, bad, hs,
         bad == 0 ? "CORRECT" : "INCORRECT",
         hs > 0 ? "ACTIVE" : "NEVER ENGAGED");
  return bad != 0;
}
