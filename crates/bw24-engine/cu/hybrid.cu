// Qwen3.5/3.6 hybrid linear-attention kernels: depthwise causal conv1d + SiLU, and the
// Gated DeltaNet recurrent scan. Ported from llama.cpp ggml-cuda {ssm-conv.cu, gated_delta_net.cu},
// simplified to single sequence (n_seqs=1). All f32, no tensor cores → sm_120-native.
#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

template <int WARP>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    // broadcast lane0 result to all lanes
    return __shfl_sync(0xffffffff, v, 0);
}

// ---- Depthwise causal conv1d + optional SiLU. Single sequence. ----
// x: [conv_dim, T] but stored as [T, conv_dim] token-major? No — ggml ssm_conv input is
// [d_conv-1+T, conv_dim] (time-major per channel). We take a simpler contract for the engine:
//   x_in: [conv_dim, T_with_pad] where T_with_pad = T + (d_conv-1), channel-major
//         (channel c, time j at c*T_with_pad + j). The first d_conv-1 cols are the carried state.
//   w:    [d_conv, conv_dim] kernel-major (channel c tap j at c*d_conv + j).
//   y:    [conv_dim, T] (channel c, time t at c*T + t).
// One thread per channel; loops over T. d_conv small (4).
extern "C" __global__ void ssm_conv1d_silu_f32(
        const float* __restrict__ x, const float* __restrict__ w,
        float* __restrict__ y, int conv_dim, int T, int d_conv, int apply_silu) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int Tp = T + d_conv - 1;
    const float* xc = x + (size_t)c * Tp;
    const float* wc = w + (size_t)c * d_conv;
    float* yc = y + (size_t)c * T;
    for (int t = 0; t < T; t++) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {        // unroll cap; d_conv<=8
            if (j < d_conv) acc += xc[t + j] * wc[j];
        }
        yc[t] = apply_silu ? silu(acc) : acc;
    }
}

// ---- Gated DeltaNet recurrent scan (the !KDA branch). Single sequence. ----
// Layout (all f32, head-major then time):
//   q,k:  [S_v, H, T]  (q[(t*H + h)*S_v + i])      -- already L2-normed, repeated to H v-heads
//   v:    [S_v, H, T]  same indexing
//   g:    [H, T]       (g[t*H + h]) RAW log-gate (kernel does expf)
//   beta: [H, T]       (beta[t*H + h]) pre-sigmoid'd
//   state_in/out: [S_v, S_v, H] per head, TRANSPOSED M[col][i] = S[i][col]
//                 (head h, col, i at h*S_v*S_v + col*S_v + i)
//   o:    [S_v, H, T]  output, o[(t*H+h)*S_v + col]
// Grid: (H, 1, S_v/cols_per_block); block: (warp=32, cols_per_block). Each warp owns one column,
// 32 lanes shard S_v=128 rows -> rows_per_lane=4.
template <int S_v, int WARP>
__device__ void gdn_scan_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        float* __restrict__ o, int H, int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_v) return;
    constexpr int rows_per_lane = S_v / WARP;

    const float* st = state_in + ((size_t)h * S_v + col) * S_v;  // row `col` contiguous
    float s_shard[rows_per_lane];
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) s_shard[r] = st[r * WARP + lane];

    for (int t = 0; t < T; t++) {
        const float* q_t = q + ((size_t)t * H + h) * S_v;
        const float* k_t = k + ((size_t)t * H + h) * S_v;
        const float* v_t = v + ((size_t)t * H + h) * S_v;
        float g_val = expf(g[(size_t)t * H + h]);
        float beta_val = beta[(size_t)t * H + h];

        float k_reg[rows_per_lane], q_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            int i = r * WARP + lane;
            k_reg[r] = k_t[i]; q_reg[r] = q_t[i];
        }
        // kv[col] = sum_i S[i][col]*k[i]
        float kv_shard = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) kv_shard += s_shard[r] * k_reg[r];
        float kv_col = warp_reduce_sum<WARP>(kv_shard);
        // delta[col] = (v[col] - g*kv[col]) * beta
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
        // fused state update + attn
        float attn_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        float attn_col = warp_reduce_sum<WARP>(attn_partial);
        if (lane == 0) o[((size_t)t * H + h) * S_v + col] = attn_col * scale;
    }
    // write state back
    float* so = state_out + ((size_t)h * S_v + col) * S_v;
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) so[r * WARP + lane] = s_shard[r];
}

extern "C" __global__ void gdn_scan_s128(
        const float* q, const float* k, const float* v, const float* g, const float* beta,
        const float* state_in, float* state_out, float* o, int H, int T, float scale) {
    gdn_scan_kernel<128, 32>(q, k, v, g, beta, state_in, state_out, o, H, T, scale);
}

// ---- helpers for the linear-attn glue ----
// sigmoid(x) elementwise
extern "C" __global__ void sigmoid_f32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
// softplus(x + bias_broadcast) then * a_broadcast -> g_log. x:[H,T], bias/a:[H]. out:[H,T].
// alpha layout [H,T] (alpha[t*H+h]); dt_bias/a [H].
extern "C" __global__ void gdn_glog_f32(const float* alpha, const float* dt_bias, const float* a,
                                        float* g_log, int H, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * T) return;
    int h = idx % H;
    float x = alpha[idx] + dt_bias[h];
    float sp = (x > 20.0f) ? x : log1pf(expf(x));   // softplus, numerically safe
    g_log[idx] = a[h] * sp;                          // a holds -exp(A_log) (pre-negated)
}

// transpose [rows, cols] row-major -> [cols, rows] row-major. (token-major <-> channel-major)
extern "C" __global__ void transpose_f32(const float* __restrict__ in, float* __restrict__ out,
                                         int rows, int cols) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * cols) return;
    int r = idx / cols;   // in row
    int c = idx % cols;   // in col
    out[(long)c * rows + r] = in[idx];
}

// gated RMSNorm output: dst = RMSNorm(o, w) * silu(z), per head_dim row.
// o,z,dst: [head_dim, n_rows] row-major; w: [head_dim]. one block per row.
extern "C" __global__ void gated_rmsnorm_f32(const float* __restrict__ o, const float* __restrict__ w,
                                             const float* __restrict__ z, float* __restrict__ dst,
                                             int ncols, float eps) {
    int row = blockIdx.x; int tid = threadIdx.x;
    const float* orow = o + (size_t)row * ncols;
    const float* zrow = z + (size_t)row * ncols;
    float* drow = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = orow[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o2);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_down_sync(0xffffffff, v, o2);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float zz = zrow[i];
        drow[i] = (orow[i] * scale * w[i]) * (zz / (1.0f + expf(-zz)));
    }
}

// Repeat-interleave heads: in [head_dim, n_in_heads, T] -> out [head_dim, n_out_heads, T],
// each in-head replicated rep = n_out_heads/n_in_heads times (contiguous in head axis).
// matches ggml_repeat_4d on the head axis. idx over out elements.
extern "C" __global__ void repeat_heads_f32(const float* __restrict__ in, float* __restrict__ out,
                                            int head_dim, int n_in_heads, int n_out_heads, int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)head_dim * n_out_heads * T;
    if (idx >= total) return;
    int d = idx % head_dim;
    int oh = (idx / head_dim) % n_out_heads;
    int t = idx / ((long)head_dim * n_out_heads);
    int rep = n_out_heads / n_in_heads;
    int ih = oh / rep;
    out[idx] = in[((long)t * n_in_heads + ih) * head_dim + d];
}
