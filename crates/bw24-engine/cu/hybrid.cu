// Qwen3.5/3.6 hybrid linear-attention kernels: depthwise causal conv1d + SiLU, and the
// Gated DeltaNet recurrent scan. Ported from llama.cpp ggml-cuda {ssm-conv.cu, gated_delta_net.cu},
// simplified to single sequence (n_seqs=1). All f32, no tensor cores → sm_120-native.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// STATE twin (task #18 conv-fuse, 2026-07-26): the prime path's conv+repack chain
// materialized conv_out [conv_dim, T] (67MB at T=2048) with uncoalesced channel-major
// writes, then re-read it transposed — 11.8ms of the 86.8ms T=2048 prime. This fuses
// the CARRIED-ring window conv + SiLU + GDN scatter into one pass: negative window rows
// read the resident ring (== ssm_conv1d_tm_state_f32's st[pad+tt]), outputs land
// directly in q_g/k_g/v_g token-major. BIT-IDENTICAL values (same 8-tap ascending
// accumulation, same SiLU, same scatter mapping). Ring update stays a separate launch.
// hk (task #21 de-broadcast): q_g/k_g head count — num_k stores each distinct GQA head
// ONCE ([T, num_k, 128]); passing hk == num_v reproduces the broadcast layout exactly.
extern "C" __global__ void ssm_conv1d_gdn_state_f32(
        const float* __restrict__ qkv_tm, const float* __restrict__ conv_state,
        const float* __restrict__ w,
        float* __restrict__ q_g, float* __restrict__ k_g, float* __restrict__ v_g,
        int conv_dim, int T, int d_conv, int d_state, int num_v, int num_k, int key_dim,
        int hk) {
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
            float xv = (tt >= 0) ? qkv_tm[(size_t)tt * conv_dim + c] : st[pad + tt];
            acc += xv * wc[j];
        }
    }
    float val = silu(acc);
    if (c < 2 * key_dim) {
        int cc = (c < key_dim) ? c : c - key_dim;
        float* dst = (c < key_dim) ? q_g : k_g;
        int kh = cc / d_state;
        int i  = cc % d_state;
        for (int vh = kh; vh < hk; vh += num_k) {
            dst[((size_t)t * hk + vh) * d_state + i] = val;
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
// task #14 pad-proofing piece 2: ring update from a DEVICE true length (padded prime
// graphs — the ring must hold the last real rows, not the pad tail). Same math, len from
// len_d[0]; host guarantees true_len >= pad (PRIME_MIN_T).
extern "C" __global__ void ssm_conv_ring_update_dev_f32(
        const float* __restrict__ qkv_tm, float* __restrict__ conv_state,
        const int* __restrict__ len_d, int conv_dim, int d_conv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = d_conv - 1;
    if (idx >= conv_dim * pad) return;
    int c = idx / pad;
    int j = idx % pad;
    int tt = len_d[0] - pad + j;
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

// varlen twin (task #18 increment 2): grid (max_nc, H, B). Chunks past a seq's
// nc no-op via the Cc guard (loop bound goes non-positive). gdnseq_t is declared
// later in this file for the K4/K5 twins; forward-declare its use here via a
// generic pointer table would hurt readability — the vl twins for K1-K3 live
// AFTER the gdnseq_t declaration instead (see gdn_chunk_k123_vl kernels below).

// K2 (C <= 64): chunk gate/attention matrices, register-tiled — each thread owns a 2x2
// (j,i) output tile of BOTH A and P and runs a scalar-smem dot over d (no shuffles; the
// warp-per-pair butterfly version was issue-bound at ~10 shfl/pair). Whole-chunk k rows +
// the block's 32 q/k j-rows live in smem (+1 row pad -> even-row reads land on distinct
// banks). P is written FULL-width: zeros above the diagonal — K5's rectangular inner loop
// relies on P[j][i>j] == 0. grid (NC, H, ceil(C/32)), block 256.
#if !defined(BW24_PORTABLE_CUDA) || defined(BW24_HOPPER_MMA)
__device__ __forceinline__ void gdn_k2_body(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ gcum, const float* __restrict__ beta,
        float* __restrict__ A, float* __restrict__ P, int H, int T, int C,
        int c, int h, int jb, int hk) {
    const int hq = h % hk;   // q/k head (task #21 de-broadcast; hk == H reproduces old)
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
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
        kt[r][d] = k[((size_t)(t0 + r) * hk + hq) * GDN_D + d];
    }
    const int jn = min(32, Cc - jb);
    for (int idx = tid; idx < jn * GDN_D; idx += 256) {
        int r = idx / GDN_D, d = idx % GDN_D;
        qt[r][d]  = q[((size_t)(t0 + jb + r) * hk + hq) * GDN_D + d];
        kjt[r][d] = k[((size_t)(t0 + jb + r) * hk + hq) * GDN_D + d];
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

extern "C" __global__ void gdn_chunk_attn_f32(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ gcum, const float* __restrict__ beta,
        float* __restrict__ A, float* __restrict__ P, int H, int T, int C, int hk) {
    gdn_k2_body(q, k, gcum, beta, A, P, H, T, C, blockIdx.x, blockIdx.y, blockIdx.z * 32, hk);
}
#endif

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
        float* __restrict__ U, float* __restrict__ W, int H, int T, int c,
        __nv_bfloat16* __restrict__ Wb16, int hk) {
    const int h = blockIdx.y;
    const int hq = h % hk;   // task #21: k head map (hk == H reproduces old)
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
                    * k[((size_t)(t0 + j) * hk + hq) * GDN_D + col];
            } else {
                acc = v[((size_t)(t0 + j) * H + h) * GDN_D + col];
            }
            #pragma unroll
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * GDN_D + col] = acc;
            if (is_w && Wb16 != nullptr) Wb16[rbase + (size_t)j * GDN_D + col] = __float2bfloat16(acc);
        }
    } else {
        for (int j = 0; j < Cc; j++) {          // tail chunk: dynamic bound
            float acc;
            if (is_w) {
                acc = expf(gcum[(size_t)(t0 + j) * H + h])
                    * k[((size_t)(t0 + j) * hk + hq) * GDN_D + col];
            } else {
                acc = v[((size_t)(t0 + j) * H + h) * GDN_D + col];
            }
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * GDN_D + col] = acc;
            if (is_w && Wb16 != nullptr) Wb16[rbase + (size_t)j * GDN_D + col] = __float2bfloat16(acc);
        }
    }
}
extern "C" __global__ void gdn_chunk_solve32_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        float* U, float* W, __nv_bfloat16* Wb16, int H, int T, int hk) {
    gdn_chunk_solve_kernel<32>(v, k, A, gcum, U, W, H, T, blockIdx.x, Wb16, hk);
}
extern "C" __global__ void gdn_chunk_solve64_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        float* U, float* W, __nv_bfloat16* Wb16, int H, int T, int hk) {
    gdn_chunk_solve_kernel<64>(v, k, A, gcum, U, W, H, T, blockIdx.x, Wb16, hk);
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


// ---- task #14 pad-proofing (design v3) ----
// Pads beyond the true length become IDENTITY steps in the GDN recurrence: the update law
// is state' = exp(g)*state + beta*(...), so beta=0 AND g_log=0 at pad rows leaves state
// untouched and contributes nothing. beta/g layout [T,H] (t*H+h); len from a device int.
extern "C" __global__ void gdn_pad_mask_f32(float* __restrict__ beta, float* __restrict__ g_log,
                                            const int* __restrict__ len_d, int H, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    if (i / H >= len_d[0]) { beta[i] = 0.0f; g_log[i] = 0.0f; }
}

// Gather row (len_d[0]-1) of a [T, ncols] buffer into dst[ncols] — the padded prime
// graph's h_seed/hlast source (the true last token's row, not the pad tail).
extern "C" __global__ void row_gather_dev_f32(const float* __restrict__ src, float* __restrict__ dst,
                                              const int* __restrict__ len_d, int ncols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncols) dst[i] = src[(size_t)(len_d[0] - 1) * ncols + i];
}

// ---- helpers for the linear-attn glue ----
// sigmoid(x) elementwise
extern "C" __global__ void sigmoid_f32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
// attn out-gate fused epilogue (task #17): dst = attn * sigmoid(gate) in ONE launch plus the
// fp16 GEMM operand for wo. BIT-IDENTICAL to sigmoid_f32(gate)->gsig; mul_f32(attn,gsig)->dst;
// bw24_f16_cvt(dst): the f32 store/reload of gsig is value-exact, so composing the expressions
// yields the same floats, and dst16 uses the same __float2half.
extern "C" __global__ void sig_mul_f16out_f32(const float* __restrict__ a, const float* __restrict__ g,
                                              float* __restrict__ dst, __half* __restrict__ dst16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.0f / (1.0f + expf(-g[i]));
        float o = a[i] * s;
        dst[i] = o;
        dst16[i] = __float2half(o);
    }
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

// gated RMSNorm with FUSED fp16 GEMM-operand epilogue (task #17): same math as
// gated_rmsnorm_f32 (identical reduce + normalize + swish gate); the epilogue also writes
// dst16[i] = __float2half(dst[i]) — exactly the bytes bw24_f16_cvt_kernel would emit — so the
// ssm_out fp16 GEMM consumes an identical operand without the standalone convert pass.
extern "C" __global__ void gated_rmsnorm_f16out_f32(const float* __restrict__ o, const float* __restrict__ w,
                                                    const float* __restrict__ z, float* __restrict__ dst,
                                                    __half* __restrict__ dst16, int ncols, float eps) {
    int row = blockIdx.x; int tid = threadIdx.x;
    const float* orow = o + (size_t)row * ncols;
    const float* zrow = z + (size_t)row * ncols;
    float* drow = dst + (size_t)row * ncols;
    __half* hrow = dst16 + (size_t)row * ncols;
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
        float ov = (orow[i] * scale * w[i]) * (zz / (1.0f + expf(-zz)));
        drow[i] = ov;
        hrow[i] = __float2half(ov);
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
// ==== B2' batched decode state ops (ARCHITECTURE-H100.md B1/B2) ====
// One launch serves B sequences. Per-seq recurrent state lives at per-cache pointers
// (host ping-pong swaps them), so the batched kernels take DEVICE POINTER ARRAYS [B]
// built per step. Batched activations are row-major [B, ...] from the batched
// projections. Bodies are the single-seq kernels VERBATIM per sequence — bit-identical
// per row (same accumulation order); only the launch geometry changes.

extern "C" __global__ void ssm_conv1d_fused_decode_b_f32(
        const float* __restrict__ qkv_cols,           // [B, conv_dim] row-major
        float* const* __restrict__ conv_states,       // [B] device ptrs, each [conv_dim, pad]
        const float* __restrict__ w,                  // shared [d_conv, conv_dim]
        float* __restrict__ conv_outs,                // [B, conv_dim]
        int conv_dim, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (c >= conv_dim) return;
    const float* qkv_col = qkv_cols + (size_t)b * conv_dim;
    float* conv_out = conv_outs + (size_t)b * conv_dim;
    int pad = d_conv - 1;
    float* st = conv_states[b] + (size_t)c * pad;
    const float* wc = w + (size_t)c * d_conv;
    float win[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) win[j] = (j < pad) ? st[j] : 0.0f;
    win[pad] = qkv_col[c];
    float wreg[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wreg[j] = (j < d_conv) ? wc[j] : 0.0f;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) acc += win[j] * wreg[j];
    conv_out[c] = silu(acc);
    #pragma unroll
    for (int j = 0; j < 8; j++) if (j < pad) st[j] = win[1 + j];
}

extern "C" __global__ void gdn_prep_decode_b_f32(
        const float* __restrict__ conv_outs,   // [B, conv_dim]
        const float* __restrict__ beta_raws,   // [B, num_v]
        const float* __restrict__ alphas,      // [B, num_v]
        const float* __restrict__ dt_bias,     // shared [num_v]
        const float* __restrict__ a,           // shared [num_v]
        float* __restrict__ q_l2, float* __restrict__ k_l2, float* __restrict__ v_g,
        float* __restrict__ beta, float* __restrict__ g_log,   // [B, ...] rows
        int d_state, int num_v, int num_k, int key_dim, float eps, int conv_dim) {
    int vh = blockIdx.x;
    int b = blockIdx.z;
    if (vh >= num_v) return;
    int warp = threadIdx.y;
    int lane = threadIdx.x;
    int kh = vh % num_k;
    const float* conv_out = conv_outs + (size_t)b * conv_dim;
    const float* beta_raw = beta_raws + (size_t)b * num_v;
    const float* alpha = alphas + (size_t)b * num_v;
    size_t vrow = (size_t)b * num_v * d_state;

    if (warp == 2) {
        const float* src = conv_out + 2 * key_dim + (size_t)vh * d_state;
        float* dst = v_g + vrow + (size_t)vh * d_state;
        for (int i = lane; i < d_state; i += 32) dst[i] = src[i];
        return;
    }
    if (warp == 3) {
        if (lane == 0) {
            beta[(size_t)b * num_v + vh] = 1.0f / (1.0f + expf(-beta_raw[vh]));
            float x = alpha[vh] + dt_bias[vh];
            float sp = (x > 20.0f) ? x : log1pf(expf(x));
            g_log[(size_t)b * num_v + vh] = a[vh] * sp;
        }
        return;
    }
    const float* src = conv_out + (warp == 0 ? 0 : key_dim) + (size_t)kh * d_state;
    float* dst = (warp == 0 ? q_l2 : k_l2) + vrow + (size_t)vh * d_state;
    float sum = 0.0f;
    for (int i = lane; i < d_state; i += 32) { float v = src[i]; sum += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float scale = rsqrtf(sum + eps);
    for (int i = lane; i < d_state; i += 32) dst[i] = src[i] * scale;
}

// Batched T=1 GDN scan: grid (H, B, S_v/COLS_PER_BLOCK) — blockIdx.y picks the sequence
// (free axis; the template uses x=head, z=col-group). Per-seq state in/out pointer arrays.
extern "C" __global__ void gdn_scan_s128_b(
        const float* q, const float* k, const float* v, const float* g, const float* beta,
        const float* const* state_ins, float* const* state_outs,
        float* o, int H, float scale) {
    int b = blockIdx.y;
    size_t row = (size_t)b * H * 128;      // [B, H*S_v] activation rows (T=1)
    size_t sc = (size_t)b * H;             // [B, H] scalar rows
    gdn_scan_kernel<128, 32>(q + row, k + row, v + row, g + sc, beta + sc,
                             state_ins[b], state_outs[b], o + row, H, 1, scale);
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

// ===================================================================================
// K4-MMA (BW24_GDN_MMA opt-in seam, 2026-07-26 — tools/bench_gdn_k4.cu arc, harness
// verdict 68.3us vs the f32 K4's 119.4 = 1.75x at (H=32,T=512,C=32)): the chunked WY
// state pass with M resident in mma accumulator fragments, step A/B as m16n8k16 bf16
// warp tiles, W/k pre-converted bf16 through a 2-deep cp.async ring. REQUIRES C == 32
// (the Rust seam guards). Numerics: bf16 operand rounding WITHIN the gated chunked
// config — BW24_GDN_DIFF oracle + argmax battery arbitrate. mma helpers duplicated
// from flash_attn.cu (k4-prefixed; cu TUs are separate fatbins, no shared header).
#if !defined(BW24_PORTABLE_CUDA) || defined(BW24_HOPPER_MMA)
#include <cuda_bf16.h>
namespace k4mma {
struct CTile { float x[4]; };
struct ATile { nv_bfloat162 x[4]; };
struct BTile { nv_bfloat162 x[2]; };
static __device__ __forceinline__ void ld_A(ATile& t, const __nv_bfloat16* xs0, int stride_pairs, int lane){
    int* xi = (int*)t.x;
    const unsigned* xs = (const unsigned*)xs0 + (lane % 16)*stride_pairs + (lane / 16)*4;
    unsigned addr = (unsigned)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[1]),"=r"(xi[2]),"=r"(xi[3]) : "r"(addr));
}
static __device__ __forceinline__ void ld_A_trans(ATile& t, const __nv_bfloat16* xs0, int stride_pairs, int lane){
    int* xi = (int*)t.x;
    const unsigned* xs = (const unsigned*)xs0 + (lane % 16)*stride_pairs + (lane / 16)*4;
    unsigned addr = (unsigned)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[2]),"=r"(xi[1]),"=r"(xi[3]) : "r"(addr));
}
static __device__ __forceinline__ void mma_bf16(CTile& D, const ATile& A, const BTile& B){
    const int* Ax=(const int*)A.x; const int* Bx=(const int*)B.x; float* Dx=D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(Dx[0]),"+f"(Dx[1]),"+f"(Dx[2]),"+f"(Dx[3])
        : "r"(Ax[0]),"r"(Ax[1]),"r"(Ax[2]),"r"(Ax[3]),"r"(Bx[0]),"r"(Bx[1]));
}
static __device__ __forceinline__ void mma_f16(CTile& D, const ATile& A, const BTile& B){
    // same fragment shapes; operands are IEEE half (the coupled Y/Ssnap channel needs
    // 11 mantissa bits — bf16's 8 compounded K4->K5 error past the config pin).
    const int* Ax=(const int*)A.x; const int* Bx=(const int*)B.x; float* Dx=D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(Dx[0]),"+f"(Dx[1]),"+f"(Dx[2]),"+f"(Dx[3])
        : "r"(Ax[0]),"r"(Ax[1]),"r"(Ax[2]),"r"(Ax[3]),"r"(Bx[0]),"r"(Bx[1]));
}
}  // namespace k4mma
using k4mma::CTile; using k4mma::ATile; using k4mma::BTile;
using k4mma::ld_A; using k4mma::ld_A_trans; using k4mma::mma_bf16; using k4mma::mma_f16;
#define MB_PAD 40

// bulk f32 -> bf16 convert (the W/k mirrors the mma K4 consumes; float4-vectorized).
extern "C" __global__ void f32_to_bf16_bulk(const float* __restrict__ x,
                                            __nv_bfloat16* __restrict__ o, long n) {
    long base = (blockIdx.x * (long)blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 v = *(const float4*)(x + base);
        o[base + 0] = __float2bfloat16(v.x);
        o[base + 1] = __float2bfloat16(v.y);
        o[base + 2] = __float2bfloat16(v.z);
        o[base + 3] = __float2bfloat16(v.w);
    } else {
        for (long i = base; i < n; i++) o[i] = __float2bfloat16(x[i]);
    }
}

// ---------------- v2: v1 + bf16 W/k inputs + cp.async double-buffered staging ----------------
// Probe verdict on v1: synchronous global->bf16 staging = 72us of 133 (54%); Ssnap 15us.
// v2 takes W and k PRE-CONVERTED to bf16 (engine side: K3 casts W on store for free; k gets
// a bf16 mirror pass) and pipelines the 8KB tiles through a 2-deep cp.async ring.
__device__ __forceinline__ void cp_async16_g(void* dst, const void* src, int src_size) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// task #18 (varlen): per-seq args for the batched-prime varlen twins — one launch runs
// ALL B sequences' K4/K5 (grid gains a seq dim; each block's math is IDENTICAL to the
// per-seq launch, so the varlen path is strictly bit-gateable). Passed BY VALUE like
// wptr8_t (Rust GdnSeqVl/GdnVl8, #[repr(C)]).
typedef struct {
    const __nv_bfloat16* kb16; float* gcum; const float* beta;
    float* U; const __nv_bfloat16* Wb16; __half* Y; __half* Ssnap;
    const float* state_in; float* state_out;
    const float* q; float* P; float* o;
    const float* k; const float* v; const float* g; float* a; float* w;
    int T; int nc;
} gdnseq_t;
typedef struct { gdnseq_t s[8]; } gdnvl_t;

__device__ __forceinline__ void
gdn_k4_body(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
            const float* __restrict__ beta,
            const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
            __half* __restrict__ Y, __half* __restrict__ Ssnap,
            const float* __restrict__ state_in, float* __restrict__ state_out,
            int H, int T, int C, int h, int col0, int hk) {
    constexpr int D = GDN_D;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;

    __shared__ __nv_bfloat16 Wb[2][32 * D];
    __shared__ __nv_bfloat16 kb[2][32 * D];
    __shared__ __nv_bfloat16 Mb[32 * (D + 8)];
    constexpr int MB_STR = D + 8;
    __shared__ __nv_bfloat16 ys[32 * MB_PAD];
    __shared__ float gk[32];

    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / 4, nq = warp % 4;
    CTile Macc[4];
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            Macc[t4].x[l] = state_in[((size_t)h * D + col0 + col) * D + i];
        }

    const int NC = (T + C - 1) / C;
    // stage(chunk, buf): W tile 32xD bf16 (8KB) + k tile (8KB) = 8 x 16B per thread
    // zfill guards: W rows exist for the full nc*C buffer; k rows past T zero-fill
    #define V2_STAGE(c_, buf_) do {                                                       \
        int t0_ = (c_) * C;                                                               \
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {                               \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_g(&Wb[buf_][r * D + seg * 8],                                      \
                         Wb16 + (((size_t)(c_) * H + h) * C + r) * D + seg * 8, 16);      \
            cp_async16_g(&kb[buf_][r * D + seg * 8],                                      \
                         kb16 + ((size_t)(t0_ + r) * hk + (h % hk)) * D + seg * 8,        \
                         (t0_ + r < T) ? 16 : 0);                                         \
        }                                                                                 \
        cp_commit();                                                                      \
    } while (0)

    V2_STAGE(0, 0);
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        const int cur = c & 1;
        // chunk top: Ssnap + Mb mirror + gk (independent of the in-flight stage)
        // COUPLED PAIR (2026-07-26): Ssnap and Y are written BF16 — K5-mma is their only
        // consumer and rounds them to bf16 anyway; writing bf16 directly is numerically
        // identical to the uncoupled chain and halves the K5-side traffic (harness: K5
        // 63.0 -> 35.3us). The f32 K4/K5 pair keeps f32 buffers (the seam switches both).
        // Ssnap COLUMN-BLOCK layout (round 32): [4 col-blocks][128 rows][32 cols] per
        // (c,h) — this CTA's 32-col slice writes one contiguous 8KB block (the fragment
        // scatter's 256B-strided 4B pairs were the K4 tail slack). Same values; K5's
        // ST stage reads the matching addressing below.
        __half* sc_out = Ssnap + ((size_t)c * H + h) * D * D + (size_t)(col0 >> 5) * (D * 32);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
                int i = nq * 32 + t4 * 8 + fc + (l & 1);
                sc_out[(size_t)i * 32 + col] = __float2half(Macc[t4].x[l]);
                Mb[col * MB_STR + i] = __float2bfloat16(Macc[t4].x[l]);
            }
        if (tid < Cc) {
            float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                    * beta[(size_t)(t0 + tid) * H + h];
        } else if (tid < 32) {
            gk[tid] = 0.0f;
        }
        cp_wait<0>();
        __syncthreads();
        if (c + 1 < NC) V2_STAGE(c + 1, cur ^ 1);

        // step A
        {
            const int mj = warp / 4, colg = (warp % 4) / 2, half = warp % 2;
            CTile Sc; Sc.x[0] = Sc.x[1] = Sc.x[2] = Sc.x[3] = 0.0f;
            #pragma unroll
            for (int k16 = 0; k16 < D / 16; k16++) {
                ATile A;
                ld_A(A, Wb[cur] + (mj * 16) * D + k16 * 16, D / 2, lane);
                ATile Bt;
                ld_A(Bt, Mb + (colg * 16) * MB_STR + k16 * 16, MB_STR / 2, lane);
                BTile B;
                if (half == 0) { B.x[0] = Bt.x[0]; B.x[1] = Bt.x[2]; }
                else           { B.x[0] = Bt.x[1]; B.x[1] = Bt.x[3]; }
                mma_bf16(Sc, A, B);
            }
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int j = mj * 16 + fr + ((l < 2) ? 0 : 8);
                int col = colg * 16 + half * 8 + fc + (l & 1);
                if (j < Cc) {
                    float u = U[(((size_t)c * H + h) * C + j) * D + col0 + col];
                    float y = u - Sc.x[l];
                    Y[(((size_t)c * H + h) * C + j) * D + col0 + col] = __float2half(y);
                    ys[j * MB_PAD + col] = __float2bfloat16(y * gk[j]);
                } else {
                    ys[j * MB_PAD + col] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // step B
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++)
            #pragma unroll
            for (int l = 0; l < 4; l++) Macc[t4].x[l] *= bC;
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A_trans(A, ys + (k16 * 16) * MB_PAD + mh * 16, MB_PAD / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, kb[cur] + (k16 * 16) * D + nq * 32 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(Macc[p2 * 2 + 0], A, Blo);
                mma_bf16(Macc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 32 + t4 * 8 + fc + (l & 1);
            state_out[((size_t)h * D + col0 + col) * D + i] = Macc[t4].x[l];
        }
}

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_state_mma(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
                     const float* __restrict__ beta,
                     const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
                     __half* __restrict__ Y, __half* __restrict__ Ssnap,
                     const float* __restrict__ state_in, float* __restrict__ state_out,
                     int H, int T, int C, int hk) {
    gdn_k4_body(kb16, gcum, beta, U, Wb16, Y, Ssnap, state_in, state_out,
                H, T, C, blockIdx.x, blockIdx.y * 32, hk);
}

// varlen twin (task #18): grid (H, D/32, B); block (h, col-tile) of seq blockIdx.z runs
// the EXACT per-seq body on that seq's buffers/state — bit-identical per block.
extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_state_mma_vl(gdnvl_t v, int H, int C, int hk) {
    const gdnseq_t a = v.s[blockIdx.z];
    gdn_k4_body(a.kb16, a.gcum, a.beta, a.U, a.Wb16, a.Y, a.Ssnap, a.state_in, a.state_out,
                H, a.T, C, blockIdx.x, blockIdx.y * 32, hk);
}


// ---- v2: coupled form — St and Y arrive as BF16 (written by K4-mma directly; identical
// numerics to v1 which rounds them anyway) through a cp.async ring. P stays f32->bf16.
__device__ __forceinline__ void cp_async16_k5(void* dst, const void* src, int src_size) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit_k5() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait_k5() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ void
gdn_k5_body(const float* __restrict__ q, const float* __restrict__ gcum,
            const float* __restrict__ P, const __half* __restrict__ Yb,
            const __half* __restrict__ Stb, float* __restrict__ o,
            int H, int T, int C, float scale, int c, int h, int j0, int hk) {
    constexpr int D = GDN_D;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    if (j0 >= Cc) return;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / 4, nq = warp % 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;

    __shared__ __half qs[32 * D];
    __shared__ __half ts[2][32 * D];    // double-buffered B sub-tiles (bf16, 8KB each)
    __shared__ __half ps[32 * 40];

    // stage sub-tile: St rows it0..+31 (phase 1) or Y rows (phase 2); 32 rows x 128 bf16 = 16B x 16/row
    const __half* stb = Stb + ((size_t)c * H + h) * D * D;
    #define K5_STAGE_ST(it0_, buf_) do {                                                  \
        for (int idx = tid; idx < 32 * (D / 8); idx += 256) {                             \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_k5(&ts[buf_][r * D + seg * 8],                                     \
                          stb + (size_t)(seg >> 2) * (D * 32)                             \
                              + (size_t)((it0_) + r) * 32 + (seg & 3) * 8, 16);           \
        }                                                                                 \
        cp_commit_k5();                                                                   \
    } while (0)
    #define K5_STAGE_Y(it0_, itn_, buf_) do {                                             \
        for (int idx = tid; idx < 32 * (D / 8); idx += 256) {                             \
            int r = idx / (D / 8), seg = idx % (D / 8);                                   \
            cp_async16_k5(&ts[buf_][r * D + seg * 8],                                     \
                          Yb + (((size_t)c * H + h) * C + (it0_) + r) * D + seg * 8,      \
                          (r < (itn_)) ? 16 : 0);                                         \
        }                                                                                 \
        cp_commit_k5();                                                                   \
    } while (0)

    K5_STAGE_ST(0, 0);
    for (int idx = tid; idx < 32 * D; idx += 256) {
        int r = idx / D, d = idx % D;
        float v = (r < jn) ? q[((size_t)(t0 + j0 + r) * hk + (h % hk)) * D + d] : 0.0f;
        qs[r * D + d] = __float2half(v);
    }
    // P staged up front too (phase 2 A operand; independent of the ring)
    for (int idx = tid; idx < 32 * 32; idx += 256) {
        int r = idx / 32, i = idx % 32;
        float v = (r < jn && i < min(32, jend) && i <= j0 + r)
            ? P[(((size_t)c * H + h) * C + j0 + r) * C + i] : 0.0f;
        ps[r * 40 + i] = __float2half(v);
    }

    CTile acc[4];
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) { acc[t4].x[0]=acc[t4].x[1]=acc[t4].x[2]=acc[t4].x[3]=0.0f; }

    for (int it = 0; it < 4; it++) {           // phase 1: 4 x 32-i sub-tiles
        cp_wait_k5<0>();
        __syncthreads();
        int cur = it & 1;
        if (it < 3) K5_STAGE_ST((it + 1) * 32, cur ^ 1);
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, (const __nv_bfloat16*)(qs + (mh * 16) * D + it * 32 + k16 * 16), D / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, (const __nv_bfloat16*)(ts[cur] + (k16 * 16) * D + nq * 32 + p2 * 16), D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_f16(acc[p2 * 2 + 0], A, Blo);
                mma_f16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
        if (it == 3) K5_STAGE_Y(0, min(32, jend), 0);   // prefetch phase-2's first tile
        __syncthreads();
    }
    {
        float b_lo = 0.0f, b_hi = 0.0f;
        int j_lo = mh * 16 + fr, j_hi = j_lo + 8;
        if (j_lo < jn) b_lo = expf(gcum[(size_t)(t0 + j0 + j_lo) * H + h]);
        if (j_hi < jn) b_hi = expf(gcum[(size_t)(t0 + j0 + j_hi) * H + h]);
        #pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            acc[t4].x[0] *= b_lo; acc[t4].x[1] *= b_lo;
            acc[t4].x[2] *= b_hi; acc[t4].x[3] *= b_hi;
        }
    }
    const int nit2 = (jend + 31) / 32;
    for (int it = 0; it < nit2; it++) {        // phase 2 (j0=0,C=32 -> usually 1 sub-tile)
        cp_wait_k5<0>();
        __syncthreads();
        int cur = it & 1;
        if (it + 1 < nit2) K5_STAGE_Y((it + 1) * 32, min(32, jend - (it + 1) * 32), cur ^ 1);
        // refresh ps for sub-tiles beyond the first (P cols shift by it*32)
        if (it > 0) {
            for (int idx = tid; idx < 32 * 32; idx += 256) {
                int r = idx / 32, i = idx % 32;
                int gi = it * 32 + i;
                float v = (r < jn && gi < jend && gi <= j0 + r)
                    ? P[(((size_t)c * H + h) * C + j0 + r) * C + gi] : 0.0f;
                ps[r * 40 + i] = __float2half(v);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A(A, (const __nv_bfloat16*)(ps + (mh * 16) * 40 + k16 * 16), 20, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, (const __nv_bfloat16*)(ts[cur] + (k16 * 16) * D + nq * 32 + p2 * 16), D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_f16(acc[p2 * 2 + 0], A, Blo);
                mma_f16(acc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t4 = 0; t4 < 4; t4++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int j = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int col = nq * 32 + t4 * 8 + fc + (l & 1);
            if (j < jn)
                o[((size_t)(t0 + j0 + j) * H + h) * D + col] = scale * acc[t4].x[l];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_output_mma(const float* __restrict__ q, const float* __restrict__ gcum,
                      const float* __restrict__ P, const __half* __restrict__ Yb,
                      const __half* __restrict__ Stb, float* __restrict__ o,
                      int H, int T, int C, float scale, int hk) {
    gdn_k5_body(q, gcum, P, Yb, Stb, o, H, T, C, scale,
                blockIdx.x, blockIdx.y, blockIdx.z * 32, hk);
}

// varlen twin (task #18): grid (max_nc, H, B); requires C == 32 (j-tile = 1, the mma
// seam's chunk size). Chunks past a seq's nc early-return via the body's Cc guard.
extern "C" __global__ void __launch_bounds__(256, 2)
gdn_chunk_output_mma_vl(gdnvl_t v, int H, int C, float scale, int hk) {
    const gdnseq_t a = v.s[blockIdx.z];
    gdn_k5_body(a.q, a.gcum, a.P, a.Y, a.Ssnap, a.o, H, a.T, C, scale,
                blockIdx.x, blockIdx.y, 0, hk);
}

// varlen K1-K3 (task #18 increment 2): grid (max_nc, H, B) each; chunks past a seq's
// nc no-op via the Cc guards. Per-block math identical to the per-seq launches.
extern "C" __global__ void gdn_chunk_cumgate_vl(gdnvl_t v, int H, int C) {
    const gdnseq_t s = v.s[blockIdx.z];
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, s.T - t0);
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int j = 0; j < Cc; j++) {
            acc += s.g[(size_t)(t0 + j) * H + h];
            s.gcum[(size_t)(t0 + j) * H + h] = acc;
        }
    }
}

extern "C" __global__ void gdn_chunk_attn_vl(gdnvl_t v, int H, int C, int hk) {
    const gdnseq_t s = v.s[blockIdx.z];
    gdn_k2_body(s.q, s.k, s.gcum, s.beta, s.a, s.P, H, s.T, C,
                blockIdx.x, blockIdx.y, 0, hk);
}

extern "C" __global__ void gdn_chunk_solve32_vl(gdnvl_t v, int H, int C, int hk) {
    const gdnseq_t s = v.s[blockIdx.z];
    if (blockIdx.x * 32 >= s.T) return;
    // mirror-fold: W's bf16 twin (the K4 wb16 mirror) is emitted on store
    gdn_chunk_solve_kernel<32>(s.v, s.k, s.a, s.gcum, s.U, s.w, H, s.T, blockIdx.x,
                               (__nv_bfloat16*)s.Wb16, hk);
}

// ---- task #18 increment 3: varlen PREP + TAIL (one launch per stage for all B seqs).
// Every twin reproduces the per-seq kernel's per-element/per-block math exactly; only
// the grid gains a seq dim with T-guards, so the whole chain stays bit-gateable.
typedef struct {
    const float* qkv;          // [T, conv_dim] token-major view (concat row offset)
    float* conv_state;         // resident per-seq ring
    float* conv_out;           // [conv_dim, T]
    float* q_g; float* k_g; float* v_g;
    float* q_l2; float* k_l2;
    const float* beta_raw; const float* alpha;
    float* beta; float* g_log;
    const float* o;            // K5 output (tail input)
    const float* z;            // post-norm gate view (concat row offset)
    float* gn; __half* gn16;   // tail outputs
    __nv_bfloat16* kb16;       // k mirror emitted by the l2 v2 epilogue (mirror-fold)
    int T; int pad_;
} gdnprep_t;
typedef struct { gdnprep_t s[8]; } gdnprepvl_t;

extern "C" __global__ void ssm_conv1d_tm_state_vl(gdnprepvl_t v, const float* __restrict__ w,
                                                  int conv_dim, int d_conv) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= conv_dim || t >= sq.T) return;
    int pad = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    const float* st = sq.conv_state + (size_t)c * pad;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < d_conv) {
            int tt = t - pad + j;
            float xv = (tt >= 0) ? sq.qkv[(size_t)tt * conv_dim + c]
                                 : st[pad + tt];
            acc += xv * wc[j];
        }
    }
    sq.conv_out[(size_t)c * sq.T + t] = silu(acc);
}

extern "C" __global__ void ssm_conv1d_gdn_state_vl(gdnprepvl_t vv, const float* __restrict__ w,
        int conv_dim, int d_conv, int d_state, int num_v, int num_k, int key_dim, int hk) {
    const gdnprep_t sq = vv.s[blockIdx.z];
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= conv_dim || t >= sq.T) return;
    int pad = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    const float* st = sq.conv_state + (size_t)c * pad;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < d_conv) {
            int tt = t - pad + j;
            float xv = (tt >= 0) ? sq.qkv[(size_t)tt * conv_dim + c] : st[pad + tt];
            acc += xv * wc[j];
        }
    }
    float val = silu(acc);
    if (c < 2 * key_dim) {
        int cc = (c < key_dim) ? c : c - key_dim;
        float* dst = (c < key_dim) ? sq.q_g : sq.k_g;
        int kh = cc / d_state;
        int i  = cc % d_state;
        for (int vh = kh; vh < hk; vh += num_k) {
            dst[((size_t)t * hk + vh) * d_state + i] = val;
        }
    } else {
        int cc = c - 2 * key_dim;
        int vh = cc / d_state;
        int i  = cc % d_state;
        sq.v_g[((size_t)t * num_v + vh) * d_state + i] = val;
    }
}

extern "C" __global__ void ssm_conv_ring_update_vl(gdnprepvl_t v, int conv_dim, int d_conv) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = d_conv - 1;
    if (idx >= conv_dim * pad) return;
    int c = idx / pad;
    int j = idx % pad;
    int tt = sq.T - pad + j;
    sq.conv_state[(size_t)c * pad + j] = sq.qkv[(size_t)tt * conv_dim + c];
}

extern "C" __global__ void qkv_to_gdn_repack_vl(gdnprepvl_t v, int d_state, int num_v,
                                                int num_k, int key_dim) {
    const gdnprep_t sq = v.s[blockIdx.z];
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)sq.T * num_v * d_state;
    if (idx >= total) return;
    int i  = idx % d_state;
    int vh = (idx / d_state) % num_v;
    int tt = idx / ((long)d_state * num_v);
    int head_k = d_state;
    int kh = vh % num_k;
    long qc = (long)kh * head_k + i;
    long kc = (long)key_dim + (long)kh * head_k + i;
    long vc = (long)2 * key_dim + (long)vh * d_state + i;
    sq.q_g[idx] = sq.conv_out[qc * sq.T + tt];
    sq.k_g[idx] = sq.conv_out[kc * sq.T + tt];
    sq.v_g[idx] = sq.conv_out[vc * sq.T + tt];
}

// fused q+k l2 (grid.y: 0 = q, 1 = k) — the reduction body IS l2_norm_f32's (block 256).
extern "C" __global__ void gdn_l2_vl(gdnprepvl_t v, int ncols, int num_v, float eps) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int row = blockIdx.x;
    if (row >= num_v * sq.T) return;
    const float* x = blockIdx.y == 0 ? sq.q_g : sq.k_g;
    float* dst = blockIdx.y == 0 ? sq.q_l2 : sq.k_l2;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols; float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v2 = xr[i]; sum += v2 * v2; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale;
}

// l2 v2 varlen twin (same numeric config as l2_norm_pp_v2_f32; warp-per-row float4).
extern "C" __global__ void gdn_l2_v2_vl(gdnprepvl_t v, int ncols, int num_v, float eps) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= num_v * sq.T) return;
    const float* x = blockIdx.y == 0 ? sq.q_g : sq.k_g;
    float* dst = blockIdx.y == 0 ? sq.q_l2 : sq.k_l2;
    int lane = threadIdx.x & 31;
    const float* xr = x + (size_t)row * ncols;
    float4 val = *(const float4*)(xr + lane * 4);
    float sum = val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    float scale = rsqrtf(sum + eps);
    float4 o4 = make_float4(val.x * scale, val.y * scale, val.z * scale, val.w * scale);
    *(float4*)(dst + (size_t)row * ncols + lane * 4) = o4;
    // mirror-fold: the k side also emits its bf16 twin (the K4 kb16 mirror)
    if (blockIdx.y == 1 && sq.kb16 != nullptr) {
        __nv_bfloat16* h = sq.kb16 + (size_t)row * ncols + lane * 4;
        h[0] = __float2bfloat16(o4.x); h[1] = __float2bfloat16(o4.y);
        h[2] = __float2bfloat16(o4.z); h[3] = __float2bfloat16(o4.w);
    }
}

// fused sigmoid(beta_raw) + glog(alpha) — both elementwise over [T, H].
extern "C" __global__ void gdn_gate_prep_vl(gdnprepvl_t v, const float* __restrict__ dt_bias,
                                            const float* __restrict__ a, int H) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * sq.T) return;
    sq.beta[idx] = 1.0f / (1.0f + expf(-sq.beta_raw[idx]));
    int h = idx % H;
    float x = sq.alpha[idx] + dt_bias[h];
    float sp = (x > 20.0f) ? x : log1pf(expf(x));
    sq.g_log[idx] = a[h] * sp;
}

// varlen bf16 mirrors over the gdnseq_t table: k (q_l2's sibling k_l2 -> kb16) and w -> wb16.
// float4 body == f32_to_bf16_bulk per element.
extern "C" __global__ void gdn_mirror_vl(gdnvl_t v, int elems_per_t, int which) {
    const gdnseq_t sq = v.s[blockIdx.z];
    long n = which == 0 ? (long)sq.T * elems_per_t
                        : (long)sq.nc * elems_per_t * 32;   // w: nc*H*C*D with elems_per_t = H*D
    const float* x = which == 0 ? sq.k : sq.w;
    __nv_bfloat16* o = (__nv_bfloat16*)(which == 0 ? (void*)sq.kb16 : (void*)sq.Wb16);
    long base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 val = *(const float4*)(x + base);
        o[base + 0] = __float2bfloat16(val.x);
        o[base + 1] = __float2bfloat16(val.y);
        o[base + 2] = __float2bfloat16(val.z);
        o[base + 3] = __float2bfloat16(val.w);
    } else {
        for (long i = base; i < n; i++) o[i] = __float2bfloat16(x[i]);
    }
}

// varlen gated-norm tail (+f16out): per-row body == gated_rmsnorm_f16out_f32 (block 128).
extern "C" __global__ void gated_rmsnorm_f16out_vl(gdnprepvl_t v, const float* __restrict__ w,
                                                   int ncols, int num_v, float eps) {
    const gdnprep_t sq = v.s[blockIdx.z];
    int row = blockIdx.x;
    if (row >= num_v * sq.T) return;
    int tid = threadIdx.x;
    const float* orow = sq.o + (size_t)row * ncols;
    const float* zrow = sq.z + (size_t)row * ncols;
    float* drow = sq.gn + (size_t)row * ncols;
    __half* hrow = sq.gn16 + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v2 = orow[i]; sum += v2 * v2; }
    __shared__ float s[32];
    for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o2);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o2 = 16; o2 > 0; o2 >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o2);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float zz = zrow[i];
        float ov = (orow[i] * scale * w[i]) * (zz / (1.0f + expf(-zz)));
        drow[i] = ov;
        hrow[i] = __float2half(ov);
    }
}

#endif  // !BW24_PORTABLE_CUDA || BW24_HOPPER_MMA


// ================= K4+K5 fused wgmma (BW24_GDN_WGMMA, task #22) =================
// Harness-proven (tools/bench_gdn_wgmma.cu v5, ledger 1f08b997): persistent-M chunk
// kernel absorbing K5's output pass. CTA (head, 32-col block) x 256 threads, 2
// warpgroups x i-halves. gk folds into the k^T staging so plain Y^T serves step B's
// A-operand and phase 2's B-operand; Y and Ssnap globals have no consumer here and
// are never written. q/P arrive as bf16 mirrors (P pre-masked) for cp.async staging.
// sm_90a only (wgmma); the dispatch is env+cfg gated so other arches never call it.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define BW24_K45_REAL 1
#endif

__device__ __forceinline__ unsigned long long k45_desc(const void* smem_ptr,
                                                       unsigned lead_bytes, unsigned stride_bytes) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (unsigned long long)((addr & 0x3FFFF) >> 4);
    d |= (unsigned long long)((lead_bytes >> 4) & 0x3FFF) << 16;
    d |= (unsigned long long)((stride_bytes >> 4) & 0x3FFF) << 32;
    return d;
}
__device__ __forceinline__ size_t k45_canon(int st, int r, int kk) {
    return (size_t)st * 2048 + (r / 8) * 256 + (kk / 8) * 128 + (r % 8) * 16 + (kk % 8) * 2;
}
#ifdef BW24_K45_REAL
__device__ __forceinline__ void k45_wgmma(float acc[32], unsigned long long da,
                                          unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n}"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void k45_fence()  { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void k45_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void k45_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;"); }
__device__ __forceinline__ void k45_cp16(void* dst, const void* src, int sz) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(sz));
}
#endif

// P masked bf16 mirror: out[j][j2] = j2 <= j ? bf16(p) : 0   (layout [nc][h][C][C])
extern "C" __global__ void gdn_p_bf16_masked(const float* __restrict__ p,
                                             __nv_bfloat16* __restrict__ out,
                                             int C, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j2 = (int)(i % C), j = (int)((i / C) % C);
    out[i] = (j2 <= j) ? __float2bfloat16(p[i]) : __float2bfloat16(0.0f);
}

extern "C" __global__ void __launch_bounds__(256, 1)
gdn_k45_wgmma(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
              const float* __restrict__ beta,
              const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
              const __nv_bfloat16* __restrict__ qb16, const __nv_bfloat16* __restrict__ Pb16,
              float* __restrict__ o, float scale,
              const float* __restrict__ state_in, float* __restrict__ state_out,
              int H, int T, int C, int hk) {
#ifdef BW24_K45_REAL
    constexpr int D = GDN_D;
    const int h = blockIdx.x;
    const int hq = h % hk;
    const int col0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int warp = wtid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;
    const int ih = wg * 64;

    __shared__ __align__(128) __nv_bfloat16 sM[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 sW[64 * 128];
    __shared__ __align__(128) __nv_bfloat16 sK[2][64 * 32];
    __shared__ __align__(128) __nv_bfloat16 sQ[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 sP2[64 * 32];
    __shared__ __align__(128) __nv_bfloat16 sYs[64 * 32];
    float* sS = (float*)&sM[0][0];
    float* sO = (float*)&sQ[0][0];

    float Macc[32];
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        bool re0 = cll < 32, re1 = cll + 8 < 32;
        Macc[q + 0] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 0] : 0.0f;
        Macc[q + 1] = re0 ? state_in[((size_t)h * D + col0 + cll) * D + ih + il + 1] : 0.0f;
        Macc[q + 2] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] : 0.0f;
        Macc[q + 3] = re1 ? state_in[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] : 0.0f;
    }

    const int NC = (T + C - 1) / C;
    for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
        int r = 32 + seg / (D / 8), s8 = seg % (D / 8);
        *(uint4*)((char*)sW + k45_canon(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int seg = tid; seg < 32 * 4; seg += 256) {
        int r = 32 + seg / 4, s8 = seg % 4;
        *(uint4*)((char*)sP2 + k45_canon(s8 / 2, r, (s8 % 2) * 8)) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        #pragma unroll
        for (int q = 0; q < 32; q += 4) {
            int n8 = q / 4;
            int cll = warp * 16 + fr;
            int il = fc + n8 * 8;
            if (cll < 32) {
                *(__nv_bfloat162*)((char*)sM[wg] + k45_canon(il / 16, cll, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 0], Macc[q + 1]);
                *(__nv_bfloat162*)((char*)sM[wg] + k45_canon(il / 16, cll + 8, il % 16)) =
                    __floats2bfloat162_rn(Macc[q + 2], Macc[q + 3]);
            }
        }
        for (int seg = tid; seg < 32 * (D / 8); seg += 256) {
            int r = seg / (D / 8), s8 = seg % (D / 8);
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            k45_cp16((char*)sW + k45_canon(st, r, kk8),
                     Wb16 + (((size_t)c * H + h) * C + r) * D + st * 16 + kk8, (r < Cc) ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;");
        {
            const float gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            for (int idx = wtid; idx < 32 * 8; idx += 128) {
                int j = idx / 8, i8l = (idx % 8) * 8;
                float gkj = 0.0f;
                __nv_bfloat16 kv8[8];
                if (j < Cc) {
                    gkj = expf(gC - gcum[(size_t)(t0 + j) * H + h]) * beta[(size_t)(t0 + j) * H + h];
                    *(uint4*)kv8 = *(const uint4*)(kb16 + ((size_t)(t0 + j) * hk + hq) * D + ih + i8l);
                } else *(uint4*)kv8 = make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int e2 = 0; e2 < 8; e2++)
                    *(__nv_bfloat16*)((char*)sK[wg] + k45_canon(j >> 4, i8l + e2, j & 15)) =
                        __float2bfloat16(gkj * __bfloat162float(kv8[e2]));
            }
        }
        for (int seg = tid; seg < 2 * 32 * (D / 16); seg += 256) {
            int half = seg / 256, rem = seg % 256;
            int j = rem / 8, s8 = rem % 8;
            int st = s8 / 2, kk8 = (s8 % 2) * 8;
            k45_cp16((char*)sQ[half] + k45_canon(st, j, kk8),
                     qb16 + ((size_t)(t0 + j) * hk + hq) * D + half * 64 + st * 16 + kk8,
                     (j < Cc) ? 16 : 0);
        }
        for (int seg = tid; seg < 32 * 4; seg += 256) {
            int j = seg / 4, s8 = seg % 4;
            k45_cp16((char*)sP2 + k45_canon(s8 / 2, j, (s8 % 2) * 8),
                     Pb16 + (((size_t)c * H + h) * C + j) * C + (s8 / 2) * 16 + (s8 % 2) * 8,
                     (j < Cc) ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;");
        float2 uPre[16];
        if (wg == 0) {
            const int j0p = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4, cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0p + pr * 8;
                    uPre[n8 * 2 + pr] = (j < Cc && cl < 32)
                        ? *(const float2*)(U + (((size_t)c * H + h) * C + j) * D + col0 + cl)
                        : make_float2(0.0f, 0.0f);
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");

        float acc[32], Oacc[32];
        k45_fence();
        for (int st = 0; st < 4; st++) {
            unsigned long long da = k45_desc((char*)sW + (wg * 4 + st) * 2048, 128, 256);
            unsigned long long dq = k45_desc((char*)sQ[wg] + st * 2048, 128, 256);
            unsigned long long db = k45_desc((char*)sM[wg] + st * 2048, 128, 256);
            k45_wgmma(acc, da, db, st == 0 ? 0 : 1);
            k45_wgmma(Oacc, dq, db, st == 0 ? 0 : 1);
        }
        k45_commit();
        k45_wait();
        __syncthreads();
        if (wg == 1) {
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int r = warp * 16 + fr, cc = fc + n8 * 8;
                sS[(r + 0) * 64 + cc + 0] = acc[q + 0];
                sS[(r + 0) * 64 + cc + 1] = acc[q + 1];
                sS[(r + 8) * 64 + cc + 0] = acc[q + 2];
                sS[(r + 8) * 64 + cc + 1] = acc[q + 3];
                sO[(r + 0) * 64 + cc + 0] = Oacc[q + 0];
                sO[(r + 0) * 64 + cc + 1] = Oacc[q + 1];
                sO[(r + 8) * 64 + cc + 0] = Oacc[q + 2];
                sO[(r + 8) * 64 + cc + 1] = Oacc[q + 3];
            }
        }
        __syncthreads();
        const float bC = expf(gcum[(size_t)(t0 + Cc - 1) * H + h]);
        if (wg == 0) {
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    float yv0 = 0.0f, yv1 = 0.0f;
                    if (j < Cc && cl < 32) {
                        float2 u2 = uPre[n8 * 2 + pr];
                        yv0 = u2.x - (acc[q + pr * 2 + 0] + sS[j * 64 + cl + 0]);
                        yv1 = u2.y - (acc[q + pr * 2 + 1] + sS[j * 64 + cl + 1]);
                    }
                    if (j0 < 32 && cl < 32) {
                        *(__nv_bfloat16*)((char*)sYs + k45_canon(j / 16, cl + 0, j % 16)) = __float2bfloat16(yv0);
                        *(__nv_bfloat16*)((char*)sYs + k45_canon(j / 16, cl + 1, j % 16)) = __float2bfloat16(yv1);
                    }
                }
            }
            const float b0 = (j0 < Cc) ? expf(gcum[(size_t)(t0 + j0) * H + h]) : 0.0f;
            const float b1 = (j0 + 8 < Cc) ? expf(gcum[(size_t)(t0 + j0 + 8) * H + h]) : 0.0f;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4;
                int r = j0, cc = fc + n8 * 8;
                Oacc[q + 0] = (Oacc[q + 0] + sO[(r + 0) * 64 + cc + 0]) * b0;
                Oacc[q + 1] = (Oacc[q + 1] + sO[(r + 0) * 64 + cc + 1]) * b0;
                Oacc[q + 2] = (Oacc[q + 2] + sO[(r + 8) * 64 + cc + 0]) * b1;
                Oacc[q + 3] = (Oacc[q + 3] + sO[(r + 8) * 64 + cc + 1]) * b1;
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;");
        #pragma unroll
        for (int q = 0; q < 32; q++) Macc[q] *= bC;
        k45_fence();
        for (int st = 0; st < 2; st++) {
            unsigned long long da = k45_desc((char*)sYs + st * 2048, 128, 256);
            unsigned long long db = k45_desc((char*)sK[wg] + st * 2048, 128, 256);
            k45_wgmma(Macc, da, db, 1);
        }
        // phase 2 on BOTH wgs (C7519: divergent wgmma serializes); wg1 result discarded
        for (int st = 0; st < 2; st++) {
            unsigned long long da = k45_desc((char*)sP2 + st * 2048, 128, 256);
            unsigned long long db = k45_desc((char*)sYs + st * 2048, 128, 256);
            k45_wgmma(Oacc, da, db, 1);
        }
        k45_commit();
        k45_wait();
        if (wg == 0) {
            const int j0 = warp * 16 + fr;
            #pragma unroll
            for (int q = 0; q < 32; q += 4) {
                int n8 = q / 4, cl = fc + n8 * 8;
                #pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    int j = j0 + pr * 8;
                    if (j < Cc && cl < 32)
                        *(float2*)(o + ((size_t)(t0 + j) * H + h) * D + col0 + cl) =
                            make_float2(scale * Oacc[q + pr * 2 + 0], scale * Oacc[q + pr * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < 32; q += 4) {
        int n8 = q / 4;
        int cll = warp * 16 + fr;
        int il = fc + n8 * 8;
        if (cll < 32) {
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 0] = Macc[q + 0];
            state_out[((size_t)h * D + col0 + cll) * D + ih + il + 1] = Macc[q + 1];
        }
        if (cll + 8 < 32) {
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 0] = Macc[q + 2];
            state_out[((size_t)h * D + col0 + cll + 8) * D + ih + il + 1] = Macc[q + 3];
        }
    }
#endif  // BW24_K45_REAL
}
