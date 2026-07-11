// Qwen3.5/3.6 hybrid linear-attention kernels: depthwise causal conv1d + SiLU, and the
// Gated DeltaNet recurrent scan. Ported from llama.cpp ggml-cuda {ssm-conv.cu, gated_delta_net.cu},
// simplified to single sequence (n_seqs=1). All f32, no tensor cores → sm_120-native.
#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// All-reduce sum via XOR butterfly: EVERY lane ends with the 32-lane sum in
// WARP/2 == log2(WARP) shuffles (5 for WARP=32) — no separate broadcast op.
// (Replaces the old down-then-shfl(0) form = WARP/2 + 1 shuffles. Bit-identical
// up to f32 add-order; same form already proven in flash_attn.cu:179.)
template <int WARP>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// Down-only sum: result valid ONLY on lane 0; saves the broadcast shuffle when
// the consumer is lane-0-gated (the attn output write).
template <int WARP>
__device__ __forceinline__ float warp_sum_down(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ---- FUSED prefill conv: token-major input, zero left-state, conv+SiLU in ONE kernel. ----
// Replaces the transpose -> zeros -> conv_left_pad -> ssm_conv1d chain (was 4 launches + 2
// scratch buffers + a full channel-major round-trip, ~4.5ms of pp512). Reads the matmul output
// qkv_mixed DIRECTLY in its native [T, conv_dim] token-major layout; the causal window for time
// t is rows t-pad..t (rows < 0 are the zero prefill state). Output stays channel-major [conv_dim,
// T] (what qkv_to_gdn_repack consumes). BIT-IDENTICAL to the old chain: same 8-tap register
// accumulation order as ssm_conv1d_silu_f32 (j ascending), same silu, same values — only the
// addressing changed. Token-major reads are coalesced over c (adjacent threads read adjacent
// channels of the same row). Launch: grid=(ceil(conv_dim/256), T), block=256.
extern "C" __global__ void ssm_conv1d_tm_f32(
        const float* __restrict__ qkv_tm,   // [T, conv_dim] token-major (matmul output as-is)
        const float* __restrict__ w,        // [conv_dim, d_conv] kernel-major
        float* __restrict__ y,              // [conv_dim, T] channel-major, SiLU applied
        int conv_dim, int T, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= conv_dim || t >= T) return;
    int pad = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < d_conv) {
            int tt = t - pad + j;                       // input time for tap j (zero state if <0)
            float xv = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c] : 0.0f;
            acc += xv * wc[j];
        }
    }
    y[(size_t)c * T + t] = silu(acc);
}

// ---- FUSED conv + GDN repack: one kernel from token-major qkv straight to q_g/k_g/v_g. ----
// Extends ssm_conv1d_tm_f32: instead of materializing the channel-major conv_out (16MB at T=512)
// and re-reading it in qkv_to_gdn_repack, each (channel, time) thread computes its conv+SiLU value
// ONCE and scatters it directly to the GDN [d_state, num_v, T] layout:
//   c in [0, key_dim)          -> q: kh = c/d_state, i = c%d_state, written for EVERY vh with
//                                 vh % num_k == kh (the ggml_repeat_4d modulo head-repeat,
//                                 num_v/num_k copies — same VALUE, scatter only).
//   c in [key_dim, 2*key_dim)  -> k: same mapping.
//   c >= 2*key_dim             -> v: vh = (c-2key)/d_state, single write.
// Output index (t*num_v + vh)*d_state + i == qkv_to_gdn_repack's exactly. BIT-IDENTICAL values
// (same 8-tap accumulation as ssm_conv1d_tm_f32; scatter does not change the float).
// Launch: grid=(ceil(conv_dim/256), T), block=256.
extern "C" __global__ void ssm_conv1d_gdn_f32(
        const float* __restrict__ qkv_tm,   // [T, conv_dim] token-major
        const float* __restrict__ w,        // [conv_dim, d_conv]
        float* __restrict__ q_g, float* __restrict__ k_g, float* __restrict__ v_g,
        int conv_dim, int T, int d_conv, int d_state, int num_v, int num_k, int key_dim) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= conv_dim || t >= T) return;
    int pad = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < d_conv) {
            int tt = t - pad + j;
            float xv = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c] : 0.0f;
            acc += xv * wc[j];
        }
    }
    float val = silu(acc);
    if (c < 2 * key_dim) {
        int cc = (c < key_dim) ? c : c - key_dim;
        float* dst = (c < key_dim) ? q_g : k_g;
        int kh = cc / d_state;
        int i  = cc % d_state;
        for (int vh = kh; vh < num_v; vh += num_k) {
            dst[((size_t)t * num_v + vh) * d_state + i] = val;
        }
    } else {
        int cc = c - 2 * key_dim;
        int vh = cc / d_state;
        int i  = cc % d_state;
        v_g[((size_t)t * num_v + vh) * d_state + i] = val;
    }
}

// ---- BATCHED verify conv: token-major input, CARRIED conv state, ring update, T>1. ----
// The spec verify path runs T=K+1 tokens through a linear-attn layer in one pass. This is
// ssm_conv1d_tm_f32 with the zero left-pad replaced by the RESIDENT conv ring (window rows
// t-pad..t; negative rows read conv_state[c*pad + (pad+tt)]), plus the decode kernel's ring roll:
// after the pass conv_state holds the last `pad` input columns (exactly what T sequential
// ssm_conv1d_fused_decode steps would leave). Same 8-tap ascending-j accumulation as BOTH the
// prefill and decode conv kernels -> each output value is BIT-IDENTICAL to the T=1 chain.
extern "C" __global__ void ssm_conv1d_tm_state_f32(
        const float* __restrict__ qkv_tm,   // [T, conv_dim] token-major (the batched matmul output)
        float* __restrict__ conv_state,     // [conv_dim, pad] resident ring (read + rewritten)
        const float* __restrict__ w,        // [conv_dim, d_conv]
        float* __restrict__ y,              // [conv_dim, T] channel-major, SiLU applied
        int conv_dim, int T, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= conv_dim || t >= T) return;
    int pad = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    const float* st = conv_state + (size_t)c * pad;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < d_conv) {
            int tt = t - pad + j;
            float xv = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c]
                                 : st[pad + tt];              // carried state column (tt in -pad..-1)
            acc += xv * wc[j];
        }
    }
    y[(size_t)c * T + t] = silu(acc);
}
// Ring roll companion (separate launch so every window read of the pass sees the OLD state):
// conv_state[c][j] = input column at time T-pad+j. Host guarantees T >= pad, so every source is
// an INPUT column (tt >= 0) — no in-place state read, no race. (T < pad falls back to the T=1
// sequential chain host-side.)
extern "C" __global__ void ssm_conv_ring_update_f32(
        const float* __restrict__ qkv_tm, float* __restrict__ conv_state,
        int conv_dim, int T, int d_conv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = d_conv - 1;
    if (idx >= conv_dim * pad) return;
    int c = idx / pad;
    int j = idx % pad;
    int tt = T - pad + j;                     // >= 0 by the host T>=pad guarantee
    conv_state[(size_t)c * pad + j] = qkv_tm[(size_t)tt * conv_dim + c];
}
// PREFIX ring rebuild (spec REPLAY-FREE partial accept): the ring a T=1 chain would hold after
// processing only the FIRST Tc input columns = the last `pad` entries of [ring_old | cols 0..Tc-1].
// PURE COPIES (the ring stores raw input columns — no arithmetic, so this cannot perturb any FP
// order). ring_old = the pre-round snapshot ring; sources fall back to it when Tc < pad.
extern "C" __global__ void ssm_conv_ring_rebuild_f32(
        const float* __restrict__ qkv_tm, const float* __restrict__ ring_old,
        float* __restrict__ conv_state, int conv_dim, int Tc, int d_conv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = d_conv - 1;
    if (idx >= conv_dim * pad) return;
    int c = idx / pad;
    int j = idx % pad;
    int tt = Tc - pad + j;                    // may be negative when Tc < pad
    conv_state[(size_t)c * pad + j] = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c]
                                               : ring_old[(size_t)c * pad + (pad + tt)];
}

// ---- FUSED decode GDN prep: repack + q/k L2-norm + beta sigmoid + g_log in ONE kernel (T=1). ----
// Replaces qkv_to_gdn_repack + 2x l2_norm + sigmoid + gdn_glog (5 launches, ~8.6us/layer of
// serialized tiny kernels on the decode critical path). One CTA per v-head vh (grid = num_v):
// 4 warps: warp 0 handles q (gather kh-row from conv_out, L2-norm, write q_l2), warp 1 k, warp 2 v
// (straight copy), warp 3 lane 0 computes beta = sigmoid(beta_raw[vh]) and g_log = a*softplus(
// alpha[vh]+dt[vh]). d_state <= 128 = 4 elems/lane. BIT-IDENTICAL math: L2 sum is the same
// ascending serial-order? NO — l2_norm_f32 reduces via strided loop + shfl tree; here each warp
// reduces its 128-elem row with the SAME shfl tree over 4-elem/lane partials accumulated in
// ascending i order == l2_norm_f32's tid-strided order for blockDim=32 (i = lane, lane+32, ...).
// So values match l2_norm_f32 with blockDim=32 exactly; l2_norm_f32 launches use blockDim=256 —
// different reduce shape. To keep BIT-IDENTITY with the shipped path we mirror the 256-thread
// two-level reduce ORDER: lane accumulates i = lane, lane+32*1..., then a 32-lane shfl tree —
// identical to a 32-thread block. kernel_check's fused gate is the arbiter (argmax authority).
extern "C" __global__ void gdn_prep_decode_f32(
        const float* __restrict__ conv_out,   // [conv_dim] (T=1, channel-major)
        const float* __restrict__ beta_raw,   // [num_v]
        const float* __restrict__ alpha,      // [num_v]
        const float* __restrict__ dt_bias,    // [num_v]
        const float* __restrict__ a,          // [num_v]
        float* __restrict__ q_l2, float* __restrict__ k_l2, float* __restrict__ v_g,
        float* __restrict__ beta, float* __restrict__ g_log,
        int d_state, int num_v, int num_k, int key_dim, float eps) {
    int vh = blockIdx.x;
    if (vh >= num_v) return;
    int warp = threadIdx.y;      // 0=q, 1=k, 2=v, 3=scalars
    int lane = threadIdx.x;
    int kh = vh % num_k;

    if (warp == 2) {
        // v: straight copy of channels [2*key_dim + vh*d_state, +d_state)
        const float* src = conv_out + 2 * key_dim + (size_t)vh * d_state;
        float* dst = v_g + (size_t)vh * d_state;
        for (int i = lane; i < d_state; i += 32) dst[i] = src[i];
        return;
    }
    if (warp == 3) {
        if (lane == 0) {
            beta[vh] = 1.0f / (1.0f + expf(-beta_raw[vh]));
            float x = alpha[vh] + dt_bias[vh];
            float sp = (x > 20.0f) ? x : log1pf(expf(x));
            g_log[vh] = a[vh] * sp;
        }
        return;
    }
    // warp 0/1: q/k gather + L2 norm (same math as l2_norm_f32: scale = rsqrt(sum + eps)).
    const float* src = conv_out + (warp == 0 ? 0 : key_dim) + (size_t)kh * d_state;
    float* dst = (warp == 0 ? q_l2 : k_l2) + (size_t)vh * d_state;
    float sum = 0.0f;
    for (int i = lane; i < d_state; i += 32) { float v = src[i]; sum += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float scale = rsqrtf(sum + eps);
    for (int i = lane; i < d_state; i += 32) dst[i] = src[i] * scale;
}

// ---- Depthwise causal conv1d + optional SiLU. Single sequence. ----
// x: [conv_dim, T] but stored as [T, conv_dim] token-major? No — ggml ssm_conv input is
// [d_conv-1+T, conv_dim] (time-major per channel). We take a simpler contract for the engine:
//   x_in: [conv_dim, T_with_pad] where T_with_pad = T + (d_conv-1), channel-major
//         (channel c, time j at c*T_with_pad + j). The first d_conv-1 cols are the carried state.
//   w:    [d_conv, conv_dim] kernel-major (channel c tap j at c*d_conv + j).
//   y:    [conv_dim, T] (channel c, time t at c*T + t).
// One thread per channel; loops over T. d_conv small (4).
// Depthwise causal conv1d + optional SiLU. Parallel over BOTH (channel, time): grid.x=channel,
// grid.y * blockDim.x covers T. Was 1 thread/channel SERIAL over all T (512 serial iters/thread
// at T=512 -> 1.14ms, 11% of prefill). Math identical -> bit-stable argmax. d_conv (<=8) taps
// cached in registers. Launch: grid=(conv_dim, ceil(T/256)), block=256 (decode T=1 -> grid.y=1).
extern "C" __global__ void ssm_conv1d_silu_f32(
        const float* __restrict__ x, const float* __restrict__ w,
        float* __restrict__ y, int conv_dim, int T, int d_conv, int apply_silu) {
    int c = blockIdx.x;
    if (c >= conv_dim) return;
    int Tp = T + d_conv - 1;
    const float* xc = x + (size_t)c * Tp;
    const float* wc = w + (size_t)c * d_conv;
    float* yc = y + (size_t)c * T;
    float wreg[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wreg[j] = (j < d_conv) ? wc[j] : 0.0f;
    for (int t = blockIdx.y * blockDim.x + threadIdx.x; t < T; t += gridDim.y * blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // d_conv < 8: xc[t+j] past the window is an OOB read (PR #1, adopted) — the
            // predicated select keeps the unroll and zeroes the tail lanes.
            float xv = (j < d_conv) ? xc[t + j] : 0.0f;
            acc += xv * wreg[j];
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
        float attn_col = warp_sum_down<WARP>(attn_partial);   // lane-0-valid only (write below)
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

// ROUND-STREAM stage (b) 3b twins: j (the accepted prefix length) from DEVICE (acc[0] = n_acc,
// j = base + n_acc). Full accept (j == t_v) early-exits — the verify already advanced the state
// to exactly what a j == t_v restore would recompute (same kernel, same order). Bodies are the
// host-param kernels VERBATIM at Tc/T = j.
extern "C" __global__ void ssm_conv_ring_rebuild_f32_dc(
        const float* __restrict__ qkv_tm, const float* __restrict__ ring_old,
        float* __restrict__ conv_state, int conv_dim,
        const unsigned int* __restrict__ acc, int base, int t_v, int d_conv) {
    int Tc = base + (int)acc[0];
    if (Tc >= t_v) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = d_conv - 1;
    if (idx >= conv_dim * pad) return;
    int c = idx / pad;
    int j = idx % pad;
    int tt = Tc - pad + j;                    // may be negative when Tc < pad
    conv_state[(size_t)c * pad + j] = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c]
                                               : ring_old[(size_t)c * pad + (pad + tt)];
}
extern "C" __global__ void gdn_scan_s128_dc(
        const float* q, const float* k, const float* v, const float* g, const float* beta,
        const float* state_in, float* state_out, float* o, int H,
        const unsigned int* acc, int base, int t_v, float scale) {
    int T = base + (int)acc[0];
    if (T >= t_v) return;
    gdn_scan_kernel<128, 32>(q, k, v, g, beta, state_in, state_out, o, H, T, scale);
}

// =====================================================================================
// A4 (SOTA-ADOPTION rank 6.0): CHUNKED WY / BLOCKWISE-INVERSE GDN PREFILL.
// Chunk-parallel matmul form of the gated delta rule (the flashinfer/fla chunked
// formulation), PREFILL-ONLY — decode and the spec verify keep gdn_scan_s128 (the
// decode==verify dispatch-identity law). Same input/output/state layouts as gdn_scan_s128.
//
// MATH (exact in infinite precision; f32 accumulation ORDER differs from the sequential
// scan, so outputs/states are ~1e-6-rel, NOT bit-identical — argmax battery is the gate).
// Sequential recurrence per head (S is [d_k x d_v], memory M[col][i] = S[i][col]):
//   a_t = exp(g_t);  S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T;  o_t = scale S_t^T q_t
// Per chunk of C tokens with inclusive log-gate cumsum G_j = sum_{i<=j} g_i, b_j = exp(G_j),
// and rows y_j solving the unit-lower-triangular system (WY representation):
//   (I + A) Y = V - diag(b) K S_0,   A[j,i] = beta_i exp(G_j - G_i) (k_j . k_i)  (i < j)
// then
//   o_j   = scale [ b_j q_j^T S_0 + sum_{i<=j} beta_i exp(G_j - G_i) (q_j . k_i) y_i^T ]
//   S_C   = b_C S_0 + sum_i beta_i exp(G_C - G_i) k_i y_i^T
// All exponent arguments are gate-log DIFFERENCES with j >= i; g_t < 0 always (a*softplus,
// a = -exp(A_log)) so every exp() is in (0,1] — no overflow paths. Verified vs the
// sequential scan at C=1 symbolically and by the kernel_check/gdn_bench oracles.
//
// Kernel split (5 launches per layer; K1-K3+K5 chunk-parallel, K4 sequential over chunks):
//   K1 gdn_chunk_cumgate: per-chunk inclusive cumsum of log gates (serial per (chunk,head)).
//   K2 gdn_chunk_attn:    A (strictly-lower) and P[j,i] = beta_i exp(G_j-G_i)(q_j.k_i) (incl).
//   K3 gdn_chunk_solve:   forward substitution of (I+A)^{-1} on BOTH right-hand sides at once:
//                         U = (I+A)^{-1} V, W = (I+A)^{-1} diag(b) K  -> the state-dependent
//                         solve becomes Y_c = U_c - W_c S_c (a GEMM), removing the triangular
//                         solve from the sequential inter-chunk path.
//   K4 gdn_chunk_state:   inter-chunk state pass, S in smem (col-split blocks): per chunk
//                         writes o_inter = b_j q_j^T S_c and Y_c = U_c - W_c S_c, then
//                         S <- b_C S + sum_j (beta_j exp(G_C-G_j) k_j) y_j^T.
//   K5 gdn_chunk_output:  o_j = scale (o_inter_j + sum_{i<=j} P[j,i] y_i)  (chunk-parallel).
// Layouts: q,k,v,o as gdn_scan_s128 ([T,H,128], (t*H+h)*128+i); g,beta,gcum [T,H] (t*H+h);
// A,P [NC,H,C,C] (((c*H+h)*C+j)*C+i); U,W,Y [NC,H,C,128]. C <= 128 (runtime, default 64).
// =====================================================================================

#define GDN_D 128
#define GDN_NSPLIT 4   // K4 state col-split (blocks per head); 128/4 = 32 cols/block

// K1: inclusive per-chunk cumsum of log gates. grid (NC, H), block 32 (lane-0 serial scan
// in ascending t order — deterministic, matches the derivation's G_j definition exactly).
extern "C" __global__ void gdn_chunk_cumgate_f32(
        const float* __restrict__ g, float* __restrict__ gcum, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int j = 0; j < Cc; j++) {
            acc += g[(size_t)(t0 + j) * H + h];
            gcum[(size_t)(t0 + j) * H + h] = acc;
        }
    }
}

// K2 (C <= 64): chunk gate/attention matrices, register-tiled — each thread owns a 2x2
// (j,i) output tile of BOTH A and P and runs a scalar-smem dot over d (no shuffles; the
// warp-per-pair butterfly version was issue-bound at ~10 shfl/pair). Whole-chunk k rows +
// the block's 32 q/k j-rows live in smem (+1 row pad -> even-row reads land on distinct
// banks). P is written FULL-width: zeros above the diagonal — K5's rectangular inner loop
// relies on P[j][i>j] == 0. grid (NC, H, ceil(C/32)), block 256.
extern "C" __global__ void gdn_chunk_attn_f32(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ gcum, const float* __restrict__ beta,
        float* __restrict__ A, float* __restrict__ P, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int jb = blockIdx.z * 32;
    if (jb >= Cc) return;                      // uniform per block (tail chunk)
    __shared__ float kt[64][GDN_D + 1];        // all i-rows of the chunk (C <= 64)
    __shared__ float qt[32][GDN_D + 1];        // this j-block's q rows
    __shared__ float kjt[32][GDN_D + 1];       // this j-block's k rows
    __shared__ float gct[128], bt[128];
    const int tid = threadIdx.x;
    if (tid < Cc) {
        gct[tid] = gcum[(size_t)(t0 + tid) * H + h];
        bt[tid]  = beta[(size_t)(t0 + tid) * H + h];
    }
    for (int idx = tid; idx < Cc * GDN_D; idx += 256) {
        int r = idx / GDN_D, d = idx % GDN_D;
        kt[r][d] = k[((size_t)(t0 + r) * H + h) * GDN_D + d];
    }
    const int jn = min(32, Cc - jb);
    for (int idx = tid; idx < jn * GDN_D; idx += 256) {
        int r = idx / GDN_D, d = idx % GDN_D;
        qt[r][d]  = q[((size_t)(t0 + jb + r) * H + h) * GDN_D + d];
        kjt[r][d] = k[((size_t)(t0 + jb + r) * H + h) * GDN_D + d];
    }
    __syncthreads();
    const int jg = tid / 16, ig = tid % 16;    // 16x2 j-rows x 16x2 i-cols
    const int j0 = jg * 2, i0 = ig * 2;
    for (int ib = 0; ib <= jb; ib += 32) {     // triangular i-blocks (i <= j)
        const int ie = min(ib + 32, Cc);
        float a00 = 0, a01 = 0, a10 = 0, a11 = 0;
        float p00 = 0, p01 = 0, p10 = 0, p11 = 0;
        #pragma unroll 4
        for (int d = 0; d < GDN_D; d++) {
            const float ki0 = kt[ib + i0][d], ki1 = kt[ib + i0 + 1][d];
            const float kj0 = kjt[j0][d], kj1 = kjt[j0 + 1][d];
            const float qj0 = qt[j0][d], qj1 = qt[j0 + 1][d];
            a00 += kj0 * ki0; a01 += kj0 * ki1; a10 += kj1 * ki0; a11 += kj1 * ki1;
            p00 += qj0 * ki0; p01 += qj0 * ki1; p10 += qj1 * ki0; p11 += qj1 * ki1;
        }
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            const int j = jb + j0 + jj;
            if (j >= Cc) continue;
            float* Arow = A + (((size_t)c * H + h) * C + j) * C;
            float* Prow = P + (((size_t)c * H + h) * C + j) * C;
            const float gj = gct[j];
            #pragma unroll
            for (int ii = 0; ii < 2; ii++) {
                const int i = ib + i0 + ii;
                if (i >= ie) continue;
                const float av = jj == 0 ? (ii == 0 ? a00 : a01) : (ii == 0 ? a10 : a11);
                const float pv = jj == 0 ? (ii == 0 ? p00 : p01) : (ii == 0 ? p10 : p11);
                const float sc = bt[i] * expf(gj - gct[i]);
                if (i < j) Arow[i] = sc * av;
                Prow[i] = (i <= j) ? sc * pv : 0.0f;
            }
        }
    }
    // zero-fill the remaining upper columns of this block's P rows (i in (j, Cc))
    for (int jj = tid / 32; jj < jn; jj += 8) {
        const int j = jb + jj;
        float* Prow = P + (((size_t)c * H + h) * C + j) * C;
        for (int i = j + 1 + (tid % 32); i < Cc; i += 32) Prow[i] = 0.0f;
    }
}

// K2 GENERIC (any C, used for C = 128): warp-per-pair butterfly dots with a 64-row smem
// k sub-tile. Slower than the tiled variant — kept for the chunk-size sweep's C=128 leg.
// Also zero-fills P's upper triangle (the K5 contract).
extern "C" __global__ void gdn_chunk_attn_g_f32(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ gcum, const float* __restrict__ beta,
        float* __restrict__ A, float* __restrict__ P, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int lane = threadIdx.x, w = threadIdx.y;
    const int tid = w * 32 + lane;
    __shared__ float kt[64][GDN_D];        // 32KB i-row sub-tile
    __shared__ float gct[128], bt[128];    // chunk gate-cumsum + beta (Cc <= 128)
    if (tid < Cc) {
        gct[tid] = gcum[(size_t)(t0 + tid) * H + h];
        bt[tid]  = beta[(size_t)(t0 + tid) * H + h];
    }
    for (int it0 = 0; it0 < Cc; it0 += 64) {
        const int itn = min(64, Cc - it0);
        __syncthreads();
        for (int idx = tid; idx < itn * GDN_D; idx += 256) {
            int r = idx / GDN_D, d = idx % GDN_D;
            kt[r][d] = k[((size_t)(t0 + it0 + r) * H + h) * GDN_D + d];
        }
        __syncthreads();
        for (int j = w; j < Cc; j += 8) {
            if (j < it0) continue;                    // pairs need i <= j
            const float* kj = k + ((size_t)(t0 + j) * H + h) * GDN_D;
            const float* qj = q + ((size_t)(t0 + j) * H + h) * GDN_D;
            float kjr[4], qjr[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) { kjr[r] = kj[r * 32 + lane]; qjr[r] = qj[r * 32 + lane]; }
            const float gj = gct[j];
            float* Arow = A + (((size_t)c * H + h) * C + j) * C;
            float* Prow = P + (((size_t)c * H + h) * C + j) * C;
            const int iend = min(itn, j - it0 + 1);   // i in [it0, min(j, it0+itn-1)]
            for (int ii = 0; ii < iend; ii++) {
                float dk = 0.0f, dq = 0.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    float kv = kt[ii][r * 32 + lane];
                    dk += kjr[r] * kv; dq += qjr[r] * kv;
                }
                #pragma unroll
                for (int o2 = 16; o2 > 0; o2 >>= 1) {
                    dk += __shfl_xor_sync(0xffffffff, dk, o2);
                    dq += __shfl_xor_sync(0xffffffff, dq, o2);
                }
                if (lane == 0) {
                    const int i = it0 + ii;
                    float sc = bt[i] * expf(gj - gct[i]);
                    if (i < j) Arow[i] = sc * dk;
                    Prow[i] = sc * dq;
                }
            }
        }
    }
    __syncthreads();
    // zero-fill P upper triangle (K5 contract)
    for (int j = w; j < Cc; j += 8) {
        float* Prow = P + (((size_t)c * H + h) * C + j) * C;
        for (int i = j + 1 + lane; i < Cc; i += 32) Prow[i] = 0.0f;
    }
}

// K3: forward substitution R_j = RHS_j - sum_{i<j} A[j,i] R_i for both RHS at once.
// grid (NC, H), block 256: threads 0..127 solve U (RHS = V), 128..255 solve W
// (RHS = diag(b) K). KEY STRUCTURE: column col of the solve only ever reads ITS OWN
// history rows R_i[col] — the whole substitution is thread-private with NO __syncthreads.
// Templated compile-time C keeps the history in REGISTERS (full unroll) with the A tile
// staged to smem — 3.6x over the local-memory generic (which remains for C = 128).
// Sequential depth C per chunk, chunk-PARALLEL grid.
template <int CT>
__device__ void gdn_chunk_solve_kernel(
        const float* __restrict__ v, const float* __restrict__ k,
        const float* __restrict__ A, const float* __restrict__ gcum,
        float* __restrict__ U, float* __restrict__ W, int H, int T) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * CT;
    const int Cc = min(CT, T - t0);
    const int tid = threadIdx.x;
    const int col = tid & (GDN_D - 1);
    const bool is_w = tid >= GDN_D;
    float* R = is_w ? W : U;
    __shared__ float As[CT][CT];
    for (int idx = tid; idx < Cc * CT; idx += 256) {
        int j = idx / CT, i = idx % CT;
        if (i < j) As[j][i] = A[(((size_t)c * H + h) * CT + j) * CT + i];
    }
    __syncthreads();
    const size_t rbase = ((size_t)c * H + h) * (size_t)CT * GDN_D;
    float hist[CT];
    if (Cc == CT) {
        #pragma unroll
        for (int j = 0; j < CT; j++) {
            float acc;
            if (is_w) {
                acc = expf(gcum[(size_t)(t0 + j) * H + h])
                    * k[((size_t)(t0 + j) * H + h) * GDN_D + col];
            } else {
                acc = v[((size_t)(t0 + j) * H + h) * GDN_D + col];
            }
            #pragma unroll
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * GDN_D + col] = acc;
        }
    } else {
        for (int j = 0; j < Cc; j++) {          // tail chunk: dynamic bound
            float acc;
            if (is_w) {
                acc = expf(gcum[(size_t)(t0 + j) * H + h])
                    * k[((size_t)(t0 + j) * H + h) * GDN_D + col];
            } else {
                acc = v[((size_t)(t0 + j) * H + h) * GDN_D + col];
            }
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * GDN_D + col] = acc;
        }
    }
}
extern "C" __global__ void gdn_chunk_solve32_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        float* U, float* W, int H, int T) {
    gdn_chunk_solve_kernel<32>(v, k, A, gcum, U, W, H, T);
}
extern "C" __global__ void gdn_chunk_solve64_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        float* U, float* W, int H, int T) {
    gdn_chunk_solve_kernel<64>(v, k, A, gcum, U, W, H, T);
}
// Generic (any C <= 128): thread-private history in local memory (L1, lane-interleaved).
extern "C" __global__ void gdn_chunk_solve_f32(
        const float* __restrict__ v, const float* __restrict__ k,
        const float* __restrict__ A, const float* __restrict__ gcum,
        float* __restrict__ U, float* __restrict__ W, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int tid = threadIdx.x;
    const int col = tid & (GDN_D - 1);
    const bool is_w = tid >= GDN_D;
    float* R = is_w ? W : U;
    const float* Abase = A + ((size_t)c * H + h) * C * C;
    const size_t rbase = ((size_t)c * H + h) * (size_t)C * GDN_D;
    float hist[128];                       // C <= 128; thread-private column history
    for (int j = 0; j < Cc; j++) {
        float acc;
        if (is_w) {
            acc = expf(gcum[(size_t)(t0 + j) * H + h])
                * k[((size_t)(t0 + j) * H + h) * GDN_D + col];
        } else {
            acc = v[((size_t)(t0 + j) * H + h) * GDN_D + col];
        }
        const float* Aj = Abase + (size_t)j * C;
        for (int i = 0; i < j; i++) acc -= Aj[i] * hist[i];
        hist[j] = acc;
        R[rbase + (size_t)j * GDN_D + col] = acc;
    }
}

// K4: sequential inter-chunk state pass. grid (H, GDN_NSPLIT), block 256; each block owns a
// 32-col slice of the head's state in smem (+1 pad kills bank conflicts) and loops chunks:
//   step A: o_inter[j,col] = b_j sum_i q_j[i] M[col][i]  (written into o, K5 adds intra part)
//           Y[j,col]       = U[j,col] - sum_i W[j,i] M[col][i]
//   step B: M[col][i] = b_C M[col][i] + sum_j (beta_j exp(G_C-G_j) k_j[i]) Y[j,col]
// Blocks are fully independent (col-partitioned); no cross-block traffic. All accumulations
// are ascending serial per thread — deterministic run-to-run.
extern "C" __global__ void gdn_chunk_state_f32(
        const float* __restrict__ k, const float* __restrict__ gcum,
        const float* __restrict__ beta,
        const float* __restrict__ U, const float* __restrict__ W,
        float* __restrict__ Y, float* __restrict__ Ssnap,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        int H, int T, int C) {
    constexpr int COLS = GDN_D / GDN_NSPLIT;   // 32
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    __shared__ float Ms[COLS][GDN_D + 4];      // +4 pad: float4-aligned, bank-spread rows
    __shared__ float wt[32][GDN_D];            // W sub-tile; step B reuses it for k
    __shared__ float ys[32][COLS + 1];         // step-A Y slice (step B reads smem, not L2)
    __shared__ float gk[128];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
        int cl2 = idx / GDN_D, i = idx % GDN_D;
        Ms[cl2][i] = state_in[((size_t)h * GDN_D + col0 + cl2) * GDN_D + i];
    }
    __syncthreads();
    const int NC = (T + C - 1) / C;
    const int cl = tid % COLS, jr = tid / COLS;   // 8 row-groups (A) / 8 i-groups (B) per col
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        if (tid < Cc) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                    * beta[(size_t)(t0 + tid) * H + h];
        }
        // snapshot the chunk-START state for K5's inter-chunk output term (col-fast writes,
        // TRANSPOSED to St[i][col] so K5 reads coalesce). Moves the o_inter dot OFF the
        // sequential path into the fully chunk-parallel output kernel.
        float* sc_out = Ssnap + ((size_t)c * H + h) * GDN_D * GDN_D;
        for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
            int i = idx / COLS, cl2 = idx % COLS;
            sc_out[(size_t)i * GDN_D + col0 + cl2] = Ms[cl2][i];
        }
        float acc[GDN_D / 8];   // step-B accumulators (16 i's/thread), built across sub-tiles
        #pragma unroll
        for (int r = 0; r < GDN_D / 8; r++) acc[r] = 0.0f;
        // Per 32-row sub-tile: step A (Y = U - W S_c, 4 rows/thread, float4 smem dots,
        // U loads HOISTED above the dot chains) then step B (rank update from the smem
        // Y slice + re-staged k rows). The naive global-broadcast form was L2-bound.
        for (int jt = 0; jt < Cc; jt += 32) {
            const int jn = min(32, Cc - jt);
            __syncthreads();
            for (int idx = tid; idx < 32 * (GDN_D / 4); idx += 256) {
                int r = idx / (GDN_D / 4), d4 = idx % (GDN_D / 4);
                *reinterpret_cast<float4*>(&wt[r][d4 * 4]) = (r < jn)
                    ? *reinterpret_cast<const float4*>(
                        &W[(((size_t)c * H + h) * C + jt + r) * GDN_D + d4 * 4])
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
            {
                const size_t yb = (((size_t)c * H + h) * C + jt) * GDN_D + col0 + cl;
                const float u0 = (jr      < jn) ? U[yb + (size_t)jr * GDN_D] : 0.0f;
                const float u1 = (jr + 8  < jn) ? U[yb + (size_t)(jr + 8) * GDN_D] : 0.0f;
                const float u2 = (jr + 16 < jn) ? U[yb + (size_t)(jr + 16) * GDN_D] : 0.0f;
                const float u3 = (jr + 24 < jn) ? U[yb + (size_t)(jr + 24) * GDN_D] : 0.0f;
                float pw0 = 0.0f, pw1 = 0.0f, pw2 = 0.0f, pw3 = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < GDN_D; i += 4) {
                    const float4 m = *reinterpret_cast<const float4*>(&Ms[cl][i]);
                    const float4 w0 = *reinterpret_cast<const float4*>(&wt[jr][i]);
                    const float4 w1 = *reinterpret_cast<const float4*>(&wt[jr + 8][i]);
                    const float4 w2 = *reinterpret_cast<const float4*>(&wt[jr + 16][i]);
                    const float4 w3 = *reinterpret_cast<const float4*>(&wt[jr + 24][i]);
                    pw0 += w0.x * m.x + w0.y * m.y + w0.z * m.z + w0.w * m.w;
                    pw1 += w1.x * m.x + w1.y * m.y + w1.z * m.z + w1.w * m.w;
                    pw2 += w2.x * m.x + w2.y * m.y + w2.z * m.z + w2.w * m.w;
                    pw3 += w3.x * m.x + w3.y * m.y + w3.z * m.z + w3.w * m.w;
                }
                const float y0 = u0 - pw0, y1 = u1 - pw1, y2 = u2 - pw2, y3 = u3 - pw3;
                if (jr      < jn) { Y[yb + (size_t)jr * GDN_D] = y0;        ys[jr][cl] = y0; }
                if (jr + 8  < jn) { Y[yb + (size_t)(jr + 8) * GDN_D] = y1;  ys[jr + 8][cl] = y1; }
                if (jr + 16 < jn) { Y[yb + (size_t)(jr + 16) * GDN_D] = y2; ys[jr + 16][cl] = y2; }
                if (jr + 24 < jn) { Y[yb + (size_t)(jr + 24) * GDN_D] = y3; ys[jr + 24][cl] = y3; }
            }
            __syncthreads();
            for (int idx = tid; idx < 32 * (GDN_D / 4); idx += 256) {
                int r = idx / (GDN_D / 4), d4 = idx % (GDN_D / 4);
                *reinterpret_cast<float4*>(&wt[r][d4 * 4]) = (r < jn)
                    ? *reinterpret_cast<const float4*>(
                        &k[((size_t)(t0 + jt + r) * H + h) * GDN_D + d4 * 4])
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
            for (int jj = 0; jj < jn; jj++) {
                float yv = ys[jj][cl] * gk[jt + jj];
                #pragma unroll
                for (int r = 0; r < GDN_D / 8; r++)
                    acc[r] += wt[jj][jr * (GDN_D / 8) + r] * yv;
            }
        }
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        #pragma unroll
        for (int r = 0; r < GDN_D / 8; r++) {
            int i = jr * (GDN_D / 8) + r;
            Ms[cl][i] = bC * Ms[cl][i] + acc[r];
        }
        __syncthreads();   // Ms/gk stable before the next chunk rewrites them
    }
    for (int idx = tid; idx < COLS * GDN_D; idx += 256) {
        int cl2 = idx / GDN_D, i = idx % GDN_D;
        state_out[((size_t)h * GDN_D + col0 + cl2) * GDN_D + i] = Ms[cl2][i];
    }
}

// K5: full output assembly, chunk-parallel:
//   o[j,col] = scale ( b_j sum_i q_j[i] S_c[i][col]  +  sum_{i<=j} P[j,i] Y[i,col] )
// grid (NC, H, ceil(C/32)): each block owns 32 output rows x 128 cols. Phase 1 streams the
// chunk-start state snapshot (St[i][col], coalesced) through 32-row smem sub-tiles; phase 2
// streams Y the same way. q rows staged once. Accumulators live in registers across phases.
extern "C" __global__ void gdn_chunk_output_f32(
        const float* __restrict__ q, const float* __restrict__ gcum,
        const float* __restrict__ P, const float* __restrict__ Y,
        const float* __restrict__ Ssnap, float* __restrict__ o,
        int H, int T, int C, float scale) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int j0 = blockIdx.z * 32;
    if (j0 >= Cc) return;                      // uniform per block (tail chunk)
    __shared__ float ts[32][GDN_D];            // phase 1: St sub-tile; phase 2: Y sub-tile
    __shared__ float qs[32][GDN_D];            // the block's q rows (zero-padded tail)
    const int tid = threadIdx.x;
    const int cg = tid % 32, rg = tid / 32;    // 4x4 register tile: cols c0=4cg, rows r0=4rg
    const int c0 = cg * 4, r0 = rg * 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;
    for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
        int r = idx / GDN_D, d = idx % GDN_D;
        qs[r][d] = (r < jn) ? q[((size_t)(t0 + j0 + r) * H + h) * GDN_D + d] : 0.0f;
    }
    float acc[4][4];
    #pragma unroll
    for (int rr = 0; rr < 4; rr++)
        #pragma unroll
        for (int cc = 0; cc < 4; cc++) acc[rr][cc] = 0.0f;
    // phase 1: inter-chunk term q_j . S_c[:,col] (4 rows x 4 cols per thread; one float4
    // ts read + 4 qs broadcasts feed 16 FMAs — the m-outer form was smem-issue-bound)
    const float* st = Ssnap + ((size_t)c * H + h) * GDN_D * GDN_D;
    for (int it0 = 0; it0 < GDN_D; it0 += 32) {
        __syncthreads();
        for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
            int r = idx / GDN_D, d = idx % GDN_D;
            ts[r][d] = st[(size_t)(it0 + r) * GDN_D + d];
        }
        __syncthreads();
        #pragma unroll 4
        for (int ii = 0; ii < 32; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float qv = qs[r0 + rr][it0 + ii];
                acc[rr][0] += qv * tv.x; acc[rr][1] += qv * tv.y;
                acc[rr][2] += qv * tv.z; acc[rr][3] += qv * tv.w;
            }
        }
    }
    // gate the inter-chunk term by b_j before the intra-chunk add
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int jj = r0 + rr;
        if (jj < jn) {
            const float b = expf(gcum[(size_t)(t0 + j0 + jj) * H + h]);
            #pragma unroll
            for (int cc = 0; cc < 4; cc++) acc[rr][cc] *= b;
        }
    }
    // phase 2: intra-chunk term P @ Y (rectangular: P upper triangle is ZERO by the K2
    // contract, so no per-row bounds in the inner loop)
    for (int it0 = 0; it0 < jend; it0 += 32) {
        const int itn = min(32, jend - it0);
        __syncthreads();
        for (int idx = tid; idx < 32 * GDN_D; idx += 256) {
            int r = idx / GDN_D, d = idx % GDN_D;
            ts[r][d] = (r < itn) ? Y[(((size_t)c * H + h) * C + it0 + r) * GDN_D + d] : 0.0f;
        }
        __syncthreads();
        const float* P0 = P + (((size_t)c * H + h) * C + j0 + r0) * C + it0;
        for (int ii = 0; ii < itn; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float pv = (r0 + rr < jn) ? P0[(size_t)rr * C + ii] : 0.0f;
                acc[rr][0] += pv * tv.x; acc[rr][1] += pv * tv.y;
                acc[rr][2] += pv * tv.z; acc[rr][3] += pv * tv.w;
            }
        }
    }
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int j = j0 + r0 + rr;
        if (j < jend) {
            const float4 ov = make_float4(scale * acc[rr][0], scale * acc[rr][1],
                                          scale * acc[rr][2], scale * acc[rr][3]);
            *reinterpret_cast<float4*>(&o[((size_t)(t0 + j) * H + h) * GDN_D + c0]) = ov;
        }
    }
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

// gated RMSNorm with FUSED q8_1 quantize epilogue (launch-arc 2026-07-07): same math as
// gated_rmsnorm_f32 (identical reduce + normalize + swish gate), the normalized row emitted
// directly as q8_1 blocks for the ssm_out matvec (matmul_pre) instead of a f32 write + a separate
// quantize_q8_1 launch. ncols (d_state) % 32 == 0 -> per-32 blocks never straddle rows, so the
// global block index is row*(ncols/32)+blk: BIT-IDENTICAL bytes to quantize_q8_1 over the flat
// [nrows*ncols] vector.
extern "C" __global__ void gated_rmsnorm_q8_1(const float* __restrict__ o, const float* __restrict__ w,
                                              const float* __restrict__ z,
                                              signed char* __restrict__ out_q, float* __restrict__ out_d,
                                              int ncols, float eps) {
    int row = blockIdx.x; int tid = threadIdx.x;
    const float* orow = o + (size_t)row * ncols;
    const float* zrow = z + (size_t)row * ncols;
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
    // quantize per-32 block, warp-per-block (ncols=128 -> 4 blocks/row; block never straddles rows)
    int nblk = ncols / 32;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    for (int blk = tid >> 5; blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float zz = zrow[i];
        float v = (orow[i] * scale * w[i]) * (zz / (1.0f + expf(-zz)));
        float amax = fabsf(v);
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o2));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
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

// dst[i] += alpha * src[i]
extern "C" __global__ void axpy_f32(const float* src, float* dst, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * src[i];
}

// dst[r*ncols + c] += src[r*ncols + c] * scale[r]   (r = i / ncols)
extern "C" __global__ void add_scaled_rows_f32(const float* src, const float* scale,
                                               float* dst, int ncols, int nrows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ncols * nrows;
    if (i < total) {
        int r = i / ncols;
        dst[i] += src[i] * scale[r];
    }
}

// =====================================================================================
// On-device repack kernels: eliminate the per-token decode dtoh->host-scatter->htod.
// These move the layout shuffles from full_attn/linear_attn onto the GPU. The index math
// MATCHES the host loops in decode.rs / hybrid_forward.rs EXACTLY (this is a layout move,
// not a math change). Constants for the validated 9B/35B: head_dim=256, n_head=16,
// conv_dim=8192, d_state=128, num_v=32, num_k=16, key_dim=2048.
// =====================================================================================

// ---- 1. q|gate split. ----
// qf: [T, n_head*2*head_dim] token-major, head hh's fused block at offset hh*stride, stride=2*head_dim.
//     q = first head_dim of the block, gate = next head_dim.
// q_out, gate_out: [head_dim, n_head, T] i.e. dst row (tok*n_head+hh) of head_dim, contiguous.
// One thread per output element of q (and the matching gate element). idx over [T*n_head*head_dim).
// Matches hybrid_forward.rs:86-92 (prefill) and decode.rs:98-103 (T=1).
extern "C" __global__ void q_gate_split_f32(
        const float* __restrict__ qf, float* __restrict__ q_out, float* __restrict__ gate_out,
        int head_dim, int n_head, int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * n_head * head_dim;
    if (idx >= total) return;
    int d  = idx % head_dim;
    int hh = (idx / head_dim) % n_head;
    int tok = idx / ((long)head_dim * n_head);
    int stride = 2 * head_dim;
    long src = (long)tok * (n_head * stride) + (long)hh * stride;   // head block base
    q_out[idx]    = qf[src + d];
    gate_out[idx] = qf[src + head_dim + d];
}

// ---- 2. qkv -> GDN repack (q/k head-repeat via MODULO kh = vh % num_k). ----
// conv_out: channel-major [conv_dim, T] (channel c, time tt at c*T + tt). For decode T=1 -> index c.
//   q channels [0,key_dim), k [key_dim,2*key_dim), v [2*key_dim,conv_dim). head_k = d_state.
// q_g/k_g/v_g: [d_state, num_v, T], dst (tt*num_v+vh)*d_state + i.
//   kh = vh % num_k ; qc = kh*head_k + i ; kc = key_dim + kh*head_k + i ; vc = 2*key_dim + vh*d_state + i.
// One thread per output element. idx over [T*num_v*d_state). head_k == d_state.
// Matches decode.rs:195-206 (T=1) and hybrid_forward.rs:176-190 (general T).
extern "C" __global__ void qkv_to_gdn_repack_f32(
        const float* __restrict__ conv_out,
        float* __restrict__ q_g, float* __restrict__ k_g, float* __restrict__ v_g,
        int d_state, int num_v, int num_k, int key_dim, int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * num_v * d_state;
    if (idx >= total) return;
    int i  = idx % d_state;
    int vh = (idx / d_state) % num_v;
    int tt = idx / ((long)d_state * num_v);
    int head_k = d_state;
    int kh = vh % num_k;                                   // MODULO head-repeat (validated mapping)
    long qc = (long)kh * head_k + i;                       // q channel
    long kc = (long)key_dim + (long)kh * head_k + i;       // k channel
    long vc = (long)2 * key_dim + (long)vh * d_state + i;  // v channel
    q_g[idx] = conv_out[qc * T + tt];
    k_g[idx] = conv_out[kc * T + tt];
    v_g[idx] = conv_out[vc * T + tt];
}

// ---- 2b. conv left zero-pad (prefill from zero state). ----
// src: [conv_dim, T] channel-major (channel c, time tt at c*T + tt).
// dst: [conv_dim, T+pad] channel-major, cols 0..pad-1 = 0, cols pad..pad+T-1 = src.
// dst MUST be pre-zeroed (e.zeros) so we only write the data cols. One thread per src element.
// Matches hybrid_forward.rs conv_in build (conv_in[c*tp + pad + tt] = qkv_cm[c*t + tt]).
extern "C" __global__ void conv_left_pad_f32(
        const float* __restrict__ src, float* __restrict__ dst, int conv_dim, int T, int pad) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)conv_dim * T;
    if (idx >= total) return;
    int tt = idx % T;
    int c  = idx / T;
    int tp = T + pad;
    dst[(long)c * tp + pad + tt] = src[idx];
}

// ---- 3. conv-state assemble + ring roll (decode T=1). ----
// conv_state: resident [conv_dim, pad] (channel c, tap j at c*pad + j). pad = d_conv-1.
// qkv_col:    [conv_dim] new token (channel c at index c) -- the matmul output, token-major T=1.
// conv_in:    [conv_dim, pad+1] (channel c, time j at c*(pad+1)+j). cols 0..pad-1 = state, col pad = new.
// AND roll the ring: conv_state[c*pad + j] = conv_in[c*(pad+1) + 1 + j]  (keep last pad cols).
// We assemble into conv_in first (read state), then roll state in the SAME thread using the
// just-built conv_in (which still holds the OLD state in cols 0..pad-1 + the new col). The roll
// reads conv_in (not conv_state) so there is no read-after-write hazard across threads.
// One thread per channel c. Matches decode.rs:175-185 EXACTLY.
extern "C" __global__ void conv_assemble_and_roll_f32(
        const float* __restrict__ qkv_col, float* __restrict__ conv_state,
        float* __restrict__ conv_in, int conv_dim, int pad) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int tp = pad + 1;
    const float* st = conv_state + (size_t)c * pad;
    float* ci = conv_in + (size_t)c * tp;
    // assemble: [state cols | new col]
    for (int j = 0; j < pad; j++) ci[j] = st[j];
    ci[pad] = qkv_col[c];
    // roll: keep last `pad` cols of conv_in (cols 1..=pad) -> conv_state
    float* so = conv_state + (size_t)c * pad;
    for (int j = 0; j < pad; j++) so[j] = ci[1 + j];
}

// RANK3 LEVER (conv fuse, T=1 DECODE): fuse conv_assemble_and_roll + ssm_conv1d_silu into ONE
// kernel. The two-kernel path materializes conv_in[conv_dim, pad+1] to HBM then reads it straight
// back; here one thread per channel assembles the conv window [state | new] IN REGISTERS, computes
// the depthwise conv + SiLU, writes conv_out[c], and rolls the ring — never touching conv_in HBM.
// Saves 1 launch + the conv_in write/read per linear-attn layer per token.
// BIT-IDENTICAL to conv_assemble_and_roll_f32 -> ssm_conv1d_silu_f32(T=1, apply_silu=1): the conv
// window equals the assembled conv_in (cols 0..pad-1 = state, col pad = new), and the accumulation
// reproduces ssm_conv1d's EXACT 8-wide order (acc += win[j]*wreg[j], j=0..7, wreg[j]=0 for j>=d_conv).
//   qkv_col:    [conv_dim] new token (channel c at index c), the matmul output (token-major T=1).
//   conv_state: resident [conv_dim, pad] (channel c, tap j at c*pad + j). pad = d_conv-1.
//   w:          [d_conv, conv_dim] kernel-major (channel c tap j at c*d_conv + j).
//   conv_out:   [conv_dim] (channel c at index c), SiLU(conv).
// One thread per channel c. Launch: grid=ceil(conv_dim/256), block=256.
extern "C" __global__ void ssm_conv1d_fused_decode_f32(
        const float* __restrict__ qkv_col, float* __restrict__ conv_state,
        const float* __restrict__ w, float* __restrict__ conv_out, int conv_dim, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int pad = d_conv - 1;
    float* st = conv_state + (size_t)c * pad;
    const float* wc = w + (size_t)c * d_conv;
    // assemble the conv window in registers: win[0..pad-1] = state, win[pad] = new.
    float win[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) win[j] = (j < pad) ? st[j] : 0.0f;
    win[pad] = qkv_col[c];          // pad <= 7 (d_conv <= 8); the new column
    float wreg[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wreg[j] = (j < d_conv) ? wc[j] : 0.0f;
    // depthwise causal conv — SAME 8-wide accumulation order as ssm_conv1d_silu_f32 (t=0).
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) acc += win[j] * wreg[j];
    conv_out[c] = silu(acc);
    // roll the ring: conv_state[j] = win[1 + j] for j in 0..pad-1 (drop oldest, append new).
    #pragma unroll
    for (int j = 0; j < 8; j++) if (j < pad) st[j] = win[1 + j];
}
// MoE grouped-prefill gather/scatter kernels (A2 prototype — RESIDENT case).
// These are appended to hybrid.cu (same fatbin).

// gather_rows_f32: gather m_e rows from src[T, ncols] into dst[m_e, ncols],
// using an index array idx[m_e] (each in 0..T-1).
// Grid: (ceil(ncols*m_e / 256), 1, 1), block: (256, 1, 1).
extern "C" __global__ void gather_rows_f32(
    const float* __restrict__ src,  // [T, ncols]
    const int*   __restrict__ idx,  // [m_e] indices into src rows
    float*       __restrict__ dst,  // [m_e, ncols]
    int ncols, int m_e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m_e * ncols;
    if (i < total) {
        int row = i / ncols;
        int col = i % ncols;
        dst[i] = src[(size_t)idx[row] * ncols + col];
    }
}


// scatter_slot_f32: copy each of m_e rows in src[m_e, ncols] into the slot buffer
// dst[tok_idx[row], slot_idx[row], col] = src[row, col] (RAW, no weight multiply).
// Weight is stored into wbuf[tok_idx[row] * n_used + slot_idx[row]] by the col==0 thread.
// This separation allows the reduce step to use FMA for bit-identity with the axpy path.
// Grid: (ceil(ncols*m_e / 256), 1, 1), block: (256, 1, 1).
extern "C" __global__ void scatter_add_slot_f32(
    const float* __restrict__ src,       // [m_e, ncols] expert output
    const int*   __restrict__ tok_idx,   // [m_e] original token indices (0..T-1)
    const int*   __restrict__ slot_idx,  // [m_e] top-k slot (0..n_used-1)
    const float* __restrict__ weight,    // [m_e] expert weights
    float*       __restrict__ dst,       // [T, n_used, ncols] slot buffer
    float*       __restrict__ wbuf,      // [T, n_used] weight buffer
    int ncols, int n_used, int m_e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m_e * ncols;
    if (i < total) {
        int row = i / ncols;
        int col = i % ncols;
        int t = tok_idx[row];
        int s = slot_idx[row];
        // Copy raw expert output (weight applied in reduce via FMA for bit-identity).
        dst[(size_t)t * n_used * ncols + (size_t)s * ncols + col] = src[i];
        // Store weight once per row (col==0 avoids redundant stores).
        if (col == 0) {
            wbuf[t * n_used + s] = weight[row];
        }
    }
}

// reduce_slots_fma_f32: weighted sum of n_used slots per token using FMA.
// dst[t, col] = sum_{s=0}^{n_used-1} FMA(wbuf[t,s], slots[t,s,col], acc).
// The FMA matches axpy_f32 semantics (dst[i] += alpha * src[i] compiles to FMA).
// Grid: (ceil(T * ncols / 256), 1, 1), block: (256, 1, 1).
extern "C" __global__ void reduce_slots_f32(
    const float* __restrict__ slots,  // [T, n_used, ncols]
    const float* __restrict__ wbuf,   // [T, n_used] weights per slot
    float*       __restrict__ dst,    // [T, ncols]
    int ncols, int n_used, int T)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * ncols;
    if (i < total) {
        int t = i / ncols;
        int col = i % ncols;
        float acc = 0.0f;
        const float* base = slots + (size_t)t * n_used * ncols + col;
        for (int s = 0; s < n_used; s++) {
            acc = __fmaf_rn(wbuf[t * n_used + s], base[(size_t)s * ncols], acc);
        }
        dst[i] = acc;
    }
}
