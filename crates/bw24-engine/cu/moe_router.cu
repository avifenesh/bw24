// EDGE-1 §A: fused MoE router. Replaces the host dtoh + softmax-256 + stable DESC top-8 sort +
// renorm in hybrid_forward.rs (~281-298). One CTA per token row, blockDim = n_expert (256) =
// one thread per expert. Reproduces the Stage-1 host numerics EXACTLY so the selected experts +
// renormalized weights are bit-identical (the argmax-1178 gate depends on this).
//
// Host path (the oracle this matches):
//   maxl = max over 256 logits
//   probs[i] = exp(logit[i] - maxl);  den = sum;  probs[i] /= den          (softmax over 256)
//   sort idx DESC by (probs[b].total_cmp(probs[a]).then(a.cmp(b)))         (prob DESC, idx ASC)
//   sel = idx[..8]
//   w[j] = probs[sel[j]];  ws = sum(w);  ws = max(ws, 6.103515625e-5);  w[j] /= ws
//
// Tie handling: iterative argmax over n_used rounds. Each round picks the expert with the
// largest prob; ties broken by SMALLEST index (matches the host `.then(a.cmp(b))`). The chosen
// expert is masked to -INF for the next round. This reproduces the stable DESC sort's top-k.
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Block reduce to find argmax of `val` with smallest-index tiebreak. Each thread brings (val, idx).
// Returns the winning (val, idx) to ALL threads via shared memory.
// We encode the comparison as: a beats b iff (a.val > b.val) || (a.val == b.val && a.idx < b.idx).
extern "C" __global__ void moe_router_topk_f32(
    const float* __restrict__ logits,   // [t, n_expert]
    int*   __restrict__ sel_idx,         // [t, n_used]  (out)
    float* __restrict__ sel_w,           // [t, n_used]  (out)
    int n_expert,                        // 256
    int n_used)                          // 8
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;         // one thread per expert, tid in [0, n_expert)
    const float* lg = logits + (size_t)row * n_expert;

    // shared scratch: per-warp partials for reductions + the running prob array.
    extern __shared__ float smem[];      // unused (we use static below)
    (void)smem;
    __shared__ float s_val[32];          // per-warp reduce scratch (max 32 warps = 1024 threads)
    __shared__ int   s_idx[32];
    __shared__ float s_max;              // block max logit
    __shared__ float s_den;              // softmax denominator
    __shared__ float s_pick_val;         // winning prob this round
    __shared__ int   s_pick_idx;         // winning expert this round
    __shared__ float s_wsum;             // accumulated weight sum over picked experts

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (n_expert + 31) >> 5;
    const unsigned FULL = 0xffffffffu;

    // thread's own logit (threads with tid < n_expert only; blockDim == n_expert so all valid).
    float my_logit = (tid < n_expert) ? lg[tid] : -FLT_MAX;

    // ---- 1. block-max reduce over 256 (matches row.iter().fold(NEG_INF, max)) ----
    float v = my_logit;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    if (lane == 0) s_val[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(FULL, t, o));
        if (lane == 0) s_max = t;
    }
    __syncthreads();
    const float maxl = s_max;

    // ---- 2. exp(l - max), block-sum denom ----
    float my_exp = (tid < n_expert) ? expf(my_logit - maxl) : 0.0f;
    float sden = my_exp;
    for (int o = 16; o > 0; o >>= 1) sden += __shfl_down_sync(FULL, sden, o);
    if (lane == 0) s_val[warp] = sden;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(FULL, t, o);
        if (lane == 0) s_den = t;
    }
    __syncthreads();
    const float den = s_den;

    // my probability (the unbiased softmax prob used both as the top-k key AND the weight).
    float my_prob = my_exp / den;     // tid >= n_expert -> 0 (never picked: prob 0, masked below)
    // masked working copy for iterative argmax (winner -> -INF so it can't be re-picked).
    float work = (tid < n_expert) ? my_prob : -FLT_MAX;

    if (tid == 0) s_wsum = 0.0f;
    __syncthreads();

    // ---- 3. iterative argmax: n_used rounds, prob DESC, smallest-index tiebreak ----
    for (int j = 0; j < n_used; ++j) {
        // warp-level argmax with smallest-index tiebreak
        float bv = work;
        int   bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(FULL, bv, o);
            int   oi = __shfl_down_sync(FULL, bi, o);
            // pick other if its val is strictly greater, OR equal val with smaller index
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float t  = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
            int   ti = (lane < nwarps) ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_down_sync(FULL, t, o);
                int   oi = __shfl_down_sync(FULL, ti, o);
                if (ov > t || (ov == t && oi < ti)) { t = ov; ti = oi; }
            }
            if (lane == 0) { s_pick_val = t; s_pick_idx = ti; }
        }
        __syncthreads();

        int   pick_idx = s_pick_idx;
        // gather the WINNER's unbiased prob (== work value before masking == my_prob at pick_idx)
        // s_pick_val is exactly that (work==my_prob for picked, not yet masked this round).
        float pick_prob = s_pick_val;

        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            sel_w[(size_t)row * n_used + j]   = pick_prob;   // raw prob; renormalized below
            s_wsum += pick_prob;
        }
        // mask the winner for the next round
        if (tid == pick_idx) work = -FLT_MAX;
        __syncthreads();
    }

    // ---- 4. renorm: ws = max(sum, F16_MIN_NORMAL) BEFORE divide ----
    if (tid == 0) {
        float ws = fmaxf(s_wsum, 6.103515625e-5f);   // F16 smallest normal, clamp before divide
        for (int j = 0; j < n_used; ++j) {
            sel_w[(size_t)row * n_used + j] /= ws;
        }
    }
}

// gemma4 twin: per-expert output scale folded into the renorm write.
extern "C" __global__ void moe_router_topk_scaled_f32(
    const float* __restrict__ logits,   // [t, n_expert]
    int*   __restrict__ sel_idx,         // [t, n_used]  (out)
    float* __restrict__ sel_w,           // [t, n_used]  (out)
    int n_expert,                        // 256
    int n_used,                          // 8
    const float* __restrict__ ex_scale)  // [n_expert] gemma4 per-expert output scale
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;         // one thread per expert, tid in [0, n_expert)
    const float* lg = logits + (size_t)row * n_expert;

    // shared scratch: per-warp partials for reductions + the running prob array.
    extern __shared__ float smem[];      // unused (we use static below)
    (void)smem;
    __shared__ float s_val[32];          // per-warp reduce scratch (max 32 warps = 1024 threads)
    __shared__ int   s_idx[32];
    __shared__ float s_max;              // block max logit
    __shared__ float s_den;              // softmax denominator
    __shared__ float s_pick_val;         // winning prob this round
    __shared__ int   s_pick_idx;         // winning expert this round
    __shared__ float s_wsum;             // accumulated weight sum over picked experts

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (n_expert + 31) >> 5;
    const unsigned FULL = 0xffffffffu;

    // thread's own logit (threads with tid < n_expert only; blockDim == n_expert so all valid).
    float my_logit = (tid < n_expert) ? lg[tid] : -FLT_MAX;

    // ---- 1. block-max reduce over 256 (matches row.iter().fold(NEG_INF, max)) ----
    float v = my_logit;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    if (lane == 0) s_val[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(FULL, t, o));
        if (lane == 0) s_max = t;
    }
    __syncthreads();
    const float maxl = s_max;

    // ---- 2. exp(l - max), block-sum denom ----
    float my_exp = (tid < n_expert) ? expf(my_logit - maxl) : 0.0f;
    float sden = my_exp;
    for (int o = 16; o > 0; o >>= 1) sden += __shfl_down_sync(FULL, sden, o);
    if (lane == 0) s_val[warp] = sden;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(FULL, t, o);
        if (lane == 0) s_den = t;
    }
    __syncthreads();
    const float den = s_den;

    // my probability (the unbiased softmax prob used both as the top-k key AND the weight).
    float my_prob = my_exp / den;     // tid >= n_expert -> 0 (never picked: prob 0, masked below)
    // masked working copy for iterative argmax (winner -> -INF so it can't be re-picked).
    float work = (tid < n_expert) ? my_prob : -FLT_MAX;

    if (tid == 0) s_wsum = 0.0f;
    __syncthreads();

    // ---- 3. iterative argmax: n_used rounds, prob DESC, smallest-index tiebreak ----
    for (int j = 0; j < n_used; ++j) {
        // warp-level argmax with smallest-index tiebreak
        float bv = work;
        int   bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(FULL, bv, o);
            int   oi = __shfl_down_sync(FULL, bi, o);
            // pick other if its val is strictly greater, OR equal val with smaller index
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float t  = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
            int   ti = (lane < nwarps) ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_down_sync(FULL, t, o);
                int   oi = __shfl_down_sync(FULL, ti, o);
                if (ov > t || (ov == t && oi < ti)) { t = ov; ti = oi; }
            }
            if (lane == 0) { s_pick_val = t; s_pick_idx = ti; }
        }
        __syncthreads();

        int   pick_idx = s_pick_idx;
        // gather the WINNER's unbiased prob (== work value before masking == my_prob at pick_idx)
        // s_pick_val is exactly that (work==my_prob for picked, not yet masked this round).
        float pick_prob = s_pick_val;

        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            sel_w[(size_t)row * n_used + j]   = pick_prob;   // raw prob; renormalized below
            s_wsum += pick_prob;
        }
        // mask the winner for the next round
        if (tid == pick_idx) work = -FLT_MAX;
        __syncthreads();
    }

    // ---- 4. renorm: ws = max(sum, F16_MIN_NORMAL) BEFORE divide ----
    if (tid == 0) {
        float ws = fmaxf(s_wsum, 6.103515625e-5f);   // F16 smallest normal, clamp before divide
        for (int j = 0; j < n_used; ++j) {
            // gemma4 R3 fold: (w / ws) * ex_scale[sel] — the moe_w_exscale chain verbatim.
            sel_w[(size_t)row * n_used + j] = sel_w[(size_t)row * n_used + j] / ws
                * ex_scale[sel_idx[(size_t)row * n_used + j]];
        }
    }
}
