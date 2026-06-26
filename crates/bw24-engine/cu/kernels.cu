// bw24 engine Stage-1 kernels: correctness-first, all f32, no tensor cores.
// Math matches llama.cpp ggml CUDA ops node-for-node (norm.cu, rope.cu).
#include <cuda_runtime.h>

// ---- RMSNorm: one block per row. y = x / sqrt(mean(x^2) + eps) * weight ----
// x: [ncols, nrows] row-major (row stride = ncols). weight: [ncols]. dst same shape as x.
extern "C" __global__ void rms_norm_f32(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ dst, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;

    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    // block reduce
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// ---- L2 norm per head_dim (no weight). y = x / sqrt(sum(x^2)+eps). one block per row ----
extern "C" __global__ void l2_norm_f32(const float* __restrict__ x, float* __restrict__ dst,
                                       int ncols, float eps) {
    int row = blockIdx.x; int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols; float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale;
}

// ---- RoPE NEOX (full or partial). Pairs x[i] with x[i+n_dims/2]; dims >= n_dims copied. ----
// data layout: [head_dim, n_heads, n_tokens] (head_dim fastest). pos: [n_tokens].
// One thread per (pair i0/2, head, token). grid.x = n_heads*n_tokens, threads = head_dim/2.
extern "C" __global__ void rope_neox_f32(float* __restrict__ x, const int* __restrict__ pos,
                                         int head_dim, int n_dims, int n_heads,
                                         float theta_scale, float freq_scale) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;                 // pair index within head, 0..hd2-1
    if (j >= hd2) return;
    int hr = blockIdx.x;                 // head*token flattened
    int head = hr % n_heads;
    int tok = hr / n_heads;
    (void)head;
    float* base = x + (size_t)hr * head_dim;
    int half = n_dims / 2;
    if (j >= half) {
        // dims >= n_dims are untouched (copy-through is a no-op since in-place)
        return;
    }
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    float c = cosf(theta), s = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * s;
    base[j + half] = x0 * s + x1 * c;
}

// ---- elementwise ----
extern "C" __global__ void silu_mul_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                        float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; dst[i] = (g / (1.0f + expf(-g))) * up[i]; }
}
extern "C" __global__ void add_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

// ---- naive SDPA for one token-batch, GQA, causal. Correctness oracle (no flash). ----
// Q: [head_dim, n_head, T], K/V: [head_dim, n_head_kv, T_kv]. out: [head_dim, n_head, T].
// One block per (head, query-token). threads cooperate over T_kv. Scores in smem.
extern "C" __global__ void sdpa_naive_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                          const float* __restrict__ V, float* __restrict__ O,
                                          int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                          float scale, int causal) {
    int head = blockIdx.x;
    int qt = blockIdx.y;                 // query token index (0..T-1)
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);   // GQA mapping
    int tid = threadIdx.x;
    extern __shared__ float scores[];    // [T_kv]

    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    // query absolute position = (T_kv - T) + qt  (kv holds past + current)
    int q_pos = (T_kv - T) + qt;

    // scores[t] = scale * dot(q, K[:,kv_head,t])
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    // softmax over scores[0..T_kv) — single thread for simplicity (T_kv small in M0 tests)
    __shared__ float red[1];
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
        red[0] = 0.0f;
    }
    __syncthreads();
    // out[d] = sum_t scores[t] * V[d,kv_head,t]
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}
