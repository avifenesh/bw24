// Does cudaAccessPolicyWindow measurably pin data in L2 on GB203?
#include <cstdio>
__global__ void reader(const float4 *__restrict__ hot, float *out, size_t n4, int iters) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0;
  for (int it = 0; it < iters; it++)
    for (size_t j = i; j < n4; j += stride) {
      float4 v = hot[j];
      acc += v.x + v.y + v.z + v.w;
    }
  if (acc == 12345.f) out[0] = acc;
}
__global__ void polluter(const float4 *__restrict__ big, float *out, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0;
  for (size_t j = i; j < n4; j += stride) { float4 v = big[j]; acc += v.x+v.y+v.z+v.w; }
  if (acc == 12345.f) out[1] = acc;
}
int main() {
  setbuf(stdout, NULL);
  const size_t HOT = 32ull << 20;      // 32 MB hot buffer (fits in 40MB persisting carve)
  const size_t BIG = 2048ull << 20;    // 2 GB streaming polluter
  float4 *hot, *big; float *out;
  cudaMalloc(&hot, HOT); cudaMalloc(&big, BIG); cudaMalloc(&out, 8);
  cudaMemset(hot, 1, HOT); cudaMemset(big, 1, BIG);
  cudaStream_t s; cudaStreamCreate(&s);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto run = [&](bool persist) -> float {
    if (persist) {
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 40 << 20);
      cudaStreamAttrValue v = {};
      v.accessPolicyWindow.base_ptr = hot;
      v.accessPolicyWindow.num_bytes = HOT;
      v.accessPolicyWindow.hitRatio = 1.0f;
      v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v);
    } else {
      cudaCtxResetPersistingL2Cache();
      cudaStreamAttrValue v = {};
      v.accessPolicyWindow.num_bytes = 0;
      cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v);
    }
    // interleave: hot reader then polluter then hot reader again x5
    float total = 0; int N = 5;
    for (int r = 0; r < N; r++) {
      polluter<<<328, 256, 0, s>>>(big, out, BIG/16);
      cudaEventRecord(a, s);
      reader<<<328, 256, 0, s>>>(hot, out, HOT/16, 4);
      cudaEventRecord(b, s);
      cudaStreamSynchronize(s);
      float ms; cudaEventElapsedTime(&ms, a, b); total += ms;
    }
    return total / N;
  };
  run(false); // warmup
  float base = run(false);
  float pers = run(true);
  float base2 = run(false);
  printf("hot-read after pollution: baseline=%.3f ms persisting=%.3f ms baseline2=%.3f ms -> speedup %.2fx\n",
         base, pers, base2, base/pers);
  // effective BW
  double bytes = (double)HOT * 4;
  printf("effective hot BW: baseline=%.0f GB/s persisting=%.0f GB/s\n", bytes/base*1e-6, bytes/pers*1e-6);
  return 0;
}
