// Sampled speculative decoding — device sampling primitives (piece A of the arc;
// research/sampled-spec-impl-map.md). All randomness is Philox4x32-10 counter-based:
// (seed, stream_pos) -> deterministic values, independent of launch geometry, CUDA-graph-replay
// safe (the counter is data, not state). temp -> 0 continuity: gumbel noise is scaled by temp,
// so at temp=0 the perturbed argmax IS the plain argmax (greedy limit, gate (1)).
//
// Kernels:
//   1. gumbel_perturb_f32   : y[i] = x[i]/T + T_gumbel * G_i  (G from Philox; caller then runs the
//                             existing validated 2-pass argmax on y — Gumbel-max sampling).
//   2. softmax_gather_f32   : one block per row: max + sumexp + prob gather of ONE token id
//                             (the p_j / q_j acceptance inputs), temp-applied.
//   3. residual_sample_f32  : sample from norm(max(0, softmax(p)-softmax(q))) with one uniform —
//                             two passes over vocab in one block (sum, then inverse-CDF walk).
//                             q_logits == nullptr -> plain sample from softmax(p) (bonus token).
#include <cstdint>

// ---------------- Philox4x32-10 (Salmon et al. 2011), 32-bit counter-based ----------------
static __device__ __forceinline__ void philox_round(uint32_t& c0, uint32_t& c1, uint32_t& c2,
                                                    uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    uint32_t h0 = __umulhi(M0, c0), l0 = M0 * c0;
    uint32_t h1 = __umulhi(M1, c2), l1 = M1 * c2;
    uint32_t n0 = h1 ^ c1 ^ k0, n1 = l1, n2 = h0 ^ c3 ^ k1, n3 = l0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}
static __device__ __forceinline__ uint4 philox4(uint32_t seed_lo, uint32_t seed_hi,
                                                uint32_t ctr_lo, uint32_t ctr_hi) {
    uint32_t c0 = ctr_lo, c1 = ctr_hi, c2 = 0u, c3 = 0u, k0 = seed_lo, k1 = seed_hi;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        philox_round(c0, c1, c2, c3, k0, k1);
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return make_uint4(c0, c1, c2, c3);
}
// u32 -> (0,1] uniform (never 0 -> log is finite).
static __device__ __forceinline__ float u01(uint32_t v) {
    return ((float) v + 1.0f) * (1.0f / 4294967296.0f);
}

// ---------------- 1. Gumbel perturbation ----------------
// stream_pos identifies the sampling EVENT (one per drafted/bonus token); i indexes vocab.
// G_i = -log(-log(u_i)). y = x/temp + G (temp folded into x so noise magnitude is temp-invariant
// relative to scaled logits — equivalently argmax(softmax_T sample)). temp<=0 -> y = x (greedy).
extern "C" __global__ void gumbel_perturb_f32(
        const float* __restrict__ x, float* __restrict__ y, int n,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t stream_pos, float temp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (temp <= 0.0f) { y[i] = x[i]; return; }
    // one Philox block yields 4 lanes; use lane (i & 3) of counter (stream_pos, i >> 2)
    uint4 r = philox4(seed_lo, seed_hi, (uint32_t) (i >> 2), stream_pos);
    uint32_t v = (i & 3) == 0 ? r.x : (i & 3) == 1 ? r.y : (i & 3) == 2 ? r.z : r.w;
    float g = -__logf(-__logf(u01(v)));
    y[i] = x[i] / temp + g;
}

// ---------------- 2. softmax prob gather ----------------
// One block per (row, id) pair; rows are verify columns (p) or draft head rows (q).
// out[pair] = exp((x[id]-max)/temp) / sumexp((x-max)/temp). temp<=0 -> out = (id==argmax) ? 1 : 0.
extern "C" __global__ void softmax_gather_f32(
        const float* __restrict__ x, const int64_t row_stride, const uint32_t* __restrict__ ids,
        const int* __restrict__ rows, float* __restrict__ out, int n, int npair, float temp) {
    const int pair = blockIdx.x;
    if (pair >= npair) return;
    const float* xr = x + (int64_t) rows[pair] * row_stride;
    __shared__ float red[32];
    const int tid = threadIdx.x, nth = blockDim.x;
    // pass 1: max (+ argmax for the temp==0 arm)
    float m = -3.4e38f; int am = 0;
    for (int i = tid; i < n; i += nth) { float v = xr[i]; if (v > m || (v == m && i < am)) { m = v; am = i; } }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float mo = __shfl_down_sync(0xFFFFFFFF, m, off);
        int   ao = __shfl_down_sync(0xFFFFFFFF, am, off);
        if (mo > m || (mo == m && ao < am)) { m = mo; am = ao; }
    }
    if ((tid & 31) == 0) { red[tid >> 5] = m; ((int*) red)[16 + (tid >> 5)] = am; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (nth + 31) / 32; ++w) {
            float mo = red[w]; int ao = ((int*) red)[16 + w];
            if (mo > m || (mo == m && ao < am)) { m = mo; am = ao; }
        }
        red[0] = m; ((int*) red)[16] = am;
    }
    __syncthreads();
    m = red[0]; am = ((int*) red)[16];
    if (temp <= 0.0f) { if (tid == 0) out[pair] = (ids[pair] == (uint32_t) am) ? 1.0f : 0.0f; return; }
    // pass 2: sumexp
    float s = 0.0f;
    for (int i = tid; i < n; i += nth) s += __expf((xr[i] - m) / temp);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if ((tid & 31) == 0) red[tid >> 5] = s;
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (nth + 31) / 32; ++w) s += red[w];
        out[pair] = __expf((xr[ids[pair]] - m) / temp) / s;
    }
}

// ---------------- 3. residual / plain categorical sample ----------------
// Single block. residual r_i = max(0, softmax_T(p)_i - softmax_T(q)_i); sample via inverse CDF
// with uniform u from Philox(stream_pos). q == nullptr -> r_i = softmax_T(p)_i (plain sample).
// Deterministic sequential CDF walk done by thread 0 over block-aggregated chunk sums:
// pass A computes chunk partial sums (one per thread, strided chunks of 1024), thread 0 walks
// chunks then elements — O(n/1024 + 1024) serial work, exact, order-fixed (reproducibility).
extern "C" __global__ void residual_sample_f32(
        const float* __restrict__ p, const float* __restrict__ q, int has_q, int n, float temp,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t stream_pos,
        uint32_t* __restrict__ out_tok) {
    // Self-contained row stats (max + sumexp for p and q) — the reject event is rare (<=1/round),
    // two extra vocab passes cost nothing measurable and keep the call-site API stats-free.
    const int tid0 = threadIdx.x, nth0 = blockDim.x;
    __shared__ float sred[1024];
    float p_max, p_sumexp, q_max = 0.0f, q_sumexp = 1.0f;
    const float invT0 = temp > 0.0f ? 1.0f / temp : 0.0f;
    {
        float m = -3.4e38f;
        for (int i = tid0; i < n; i += nth0) m = fmaxf(m, p[i]);
        sred[tid0] = m; __syncthreads();
        if (tid0 == 0) { for (int t = 1; t < nth0; ++t) m = fmaxf(m, sred[t]); sred[0] = m; }
        __syncthreads(); p_max = sred[0]; __syncthreads();
        float su = 0.0f;
        for (int i = tid0; i < n; i += nth0) su += __expf((p[i] - p_max) * invT0);
        sred[tid0] = su; __syncthreads();
        if (tid0 == 0) { for (int t = 1; t < nth0; ++t) su += sred[t]; sred[0] = su; }
        __syncthreads(); p_sumexp = sred[0]; __syncthreads();
    }
    if (has_q) {
        float m = -3.4e38f;
        for (int i = tid0; i < n; i += nth0) m = fmaxf(m, q[i]);
        sred[tid0] = m; __syncthreads();
        if (tid0 == 0) { for (int t = 1; t < nth0; ++t) m = fmaxf(m, sred[t]); sred[0] = m; }
        __syncthreads(); q_max = sred[0]; __syncthreads();
        float su = 0.0f;
        for (int i = tid0; i < n; i += nth0) su += __expf((q[i] - q_max) * invT0);
        sred[tid0] = su; __syncthreads();
        if (tid0 == 0) { for (int t = 1; t < nth0; ++t) su += sred[t]; sred[0] = su; }
        __syncthreads(); q_sumexp = sred[0]; __syncthreads();
    }
    // r_i evaluated on the fly: pi = exp((p[i]-p_max)/T)/p_sumexp ; qi likewise (q may be null).
    const int tid = threadIdx.x, nth = blockDim.x;
    float* chunk_sum = sred;                             // reuse the stats reduction buffer
    const float invT = temp > 0.0f ? 1.0f / temp : 0.0f;
    auto r_at = [&](int i) -> float {
        float pi = __expf((p[i] - p_max) * invT) / p_sumexp;
        float qi = has_q ? __expf((q[i] - q_max) * invT) / q_sumexp : 0.0f;
        float r = pi - qi; return r > 0.0f ? r : 0.0f;
    };
    // chunked layout: thread t owns contiguous chunk [t*len, (t+1)*len) — CDF order == index order.
    const int len = (n + nth - 1) / nth;
    const int lo = tid * len, hi = min(lo + len, n);
    float s = 0.0f;
    for (int i = lo; i < hi; ++i) s += r_at(i);
    chunk_sum[tid] = s;
    __syncthreads();
    if (tid != 0) return;
    float total = 0.0f;
    for (int t = 0; t < nth; ++t) total += chunk_sum[t];
    if (total <= 0.0f) {                                 // p==q everywhere (or temp==0 exact match)
        // fall back to argmax of p (deterministic, matches greedy limit)
        float m = -3.4e38f; int am = 0;
        for (int i = 0; i < n; ++i) if (p[i] > m) { m = p[i]; am = i; }
        *out_tok = (uint32_t) am; return;
    }
    uint4 rv = philox4(seed_lo, seed_hi, 0xFFFFFFFFu, stream_pos);  // ctr_lo tag: sampler stream
    float u = u01(rv.x) * total;
    float acc = 0.0f; int t = 0;
    for (; t < nth; ++t) { if (acc + chunk_sum[t] >= u) break; acc += chunk_sum[t]; }
    if (t == nth) t = nth - 1;
    int i = t * len, ihi = min(i + len, n);
    for (; i < ihi; ++i) { acc += r_at(i); if (acc >= u) break; }
    *out_tok = (uint32_t) min(i, n - 1);
}

// ---------------- 4. trimmed-head q scatter ----------------
// FR-Spec trims reshape the draft head's vocab: q lives on d_vocab trimmed rows, the verify
// p on the full n_vocab. The residual needs q in TARGET-id space with q=0 (logit -inf) off-trim:
// dst[t] = -3.4e38 for all t, then dst[d2t[i]] = src[i]. Correct by construction — the trimmed
// head cannot propose off-trim tokens, so their residual mass is p(x) untouched.
extern "C" __global__ void scatter_trim_logits_f32(
        const float* __restrict__ src, const uint32_t* __restrict__ d2t,
        float* __restrict__ dst, int d_vocab, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int t = i; t < n_vocab; t += gridDim.x * blockDim.x) dst[t] = -3.4e38f;
    // grid-wide sync not available: scatter in a SECOND launch (see host wrapper) — this kernel
    // only fills. Kept as one entry with a mode flag to avoid two fatbin symbols:
}
extern "C" __global__ void scatter_trim_logits_pass2_f32(
        const float* __restrict__ src, const uint32_t* __restrict__ d2t,
        float* __restrict__ dst, int d_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_vocab) dst[d2t[i]] = src[i];
}

// ---------------- 6. filtered rejection sampling (feat/filtered-spec) ----------------
// Truncation filters make the TARGET distribution the filtered one; rejection sampling stays
// distribution-exact iff p and q see the SAME transform. These kernels compute, per logits row,
// the PROB THRESHOLD th and renormalization mass Z of the filtered softmax:
//   top_k>0:   th = prob of the k-th largest element (binary search on count)
//   top_p<1:   smallest set of highest probs with mass >= top_p (binary search on mass)
//   min_p>0:   th = max(th, min_p * p_max)
// Downstream, filtered_prob(x) = (p(x) >= th) ? p(x)/Z : 0 — used by the gather and residual.
// One block per row; the binary search runs block-internally (no host round trips). Bit-stable:
// thresholds derive from reductions in a fixed order.
extern "C" __global__ void filter_stats_f32(
        const float* __restrict__ x, const int64_t row_stride, const int* __restrict__ rows,
        float* __restrict__ out_th, float* __restrict__ out_z, float* __restrict__ out_max,
        int n, int nrow, float temp, int top_k, float top_p, float min_p) {
    const int r = blockIdx.x;
    if (r >= nrow) return;
    const float* xr = x + (int64_t) rows[r] * row_stride;
    const int tid = threadIdx.x, nth = blockDim.x;
    __shared__ float sred[1024];
    __shared__ float s_max, s_sum, s_th;
    const float invT = temp > 0.0f ? 1.0f / temp : 1.0f;

    // pass 1: max + sumexp (full softmax denom)
    float m = -3.4e38f;
    for (int i = tid; i < n; i += nth) m = fmaxf(m, xr[i]);
    sred[tid] = m; __syncthreads();
    if (tid == 0) { for (int t = 1; t < nth; ++t) m = fmaxf(m, sred[t]); s_max = m; }
    __syncthreads(); m = s_max;
    float su = 0.0f;
    for (int i = tid; i < n; i += nth) su += __expf((xr[i] - m) * invT);
    sred[tid] = su; __syncthreads();
    if (tid == 0) { float t2 = 0.0f; for (int t = 0; t < nth; ++t) t2 += sred[t]; s_sum = t2; }
    __syncthreads();
    const float denom = s_sum;                    // full softmax sum (unfiltered)

    // binary search on the UNNORMALIZED exp value e = exp((x-m)/T), th_e in [0, 1] (e_max = 1).
    // predicate for top_k: count(e >= th_e) >= top_k ; for top_p: mass(e >= th_e)/denom >= top_p.
    float lo = 0.0f, hi = 1.0f;
    if (top_k > 0 || top_p < 1.0f) {
        for (int it = 0; it < 24; ++it) {
            const float mid = 0.5f * (lo + hi);
            float cnt = 0.0f, mass = 0.0f;
            for (int i = tid; i < n; i += nth) {
                const float e0 = __expf((xr[i] - m) * invT);
                if (e0 >= mid) { cnt += 1.0f; mass += e0; }
            }
            sred[tid] = cnt; __syncthreads();
            if (tid == 0) { float c = 0.0f; for (int t = 0; t < nth; ++t) c += sred[t]; sred[0] = c; }
            __syncthreads(); const float tot_cnt = sred[0]; __syncthreads();
            sred[tid] = mass; __syncthreads();
            if (tid == 0) { float z = 0.0f; for (int t = 0; t < nth; ++t) z += sred[t]; sred[0] = z; }
            __syncthreads(); const float tot_mass = sred[0]; __syncthreads();
            bool keep_more = false;   // does the set at `mid` still satisfy the constraint?
            if (top_k > 0 && tot_cnt >= (float) top_k) keep_more = true;
            if (top_p < 1.0f && tot_mass / denom >= top_p) keep_more = true;
            if (keep_more) lo = mid; else hi = mid;
        }
    }
    if (tid == 0) {
        float th_e = (top_k > 0 || top_p < 1.0f) ? lo : 0.0f;
        // min_p: threshold on prob relative to max prob: e >= min_p (since e_max == 1).
        if (min_p > 0.0f) th_e = fmaxf(th_e, min_p);
        s_th = th_e;
    }
    __syncthreads();
    // final Z: mass of the kept set (e >= th, with th chosen so the set INCLUDES the boundary).
    const float th = s_th;
    float z = 0.0f;
    for (int i = tid; i < n; i += nth) {
        const float e0 = __expf((xr[i] - m) * invT);
        if (e0 >= th) z += e0;
    }
    sred[tid] = z; __syncthreads();
    if (tid == 0) {
        float zt = 0.0f; for (int t = 0; t < nth; ++t) zt += sred[t];
        out_th[r]  = th;       // threshold in e-units (e_max == 1 at the row max)
        out_z[r]   = zt;       // filtered renorm mass (e-units)
        out_max[r] = s_max;    // row max (the residual kernel needs it to reconstruct e-units)
    }
}

// filtered prob gather: out = (e(id) >= th) ? e(id)/Z : 0  — the filtered-softmax probability.
extern "C" __global__ void softmax_gather_filtered_f32(
        const float* __restrict__ x, const int64_t row_stride, const uint32_t* __restrict__ ids,
        const int* __restrict__ rows, const float* __restrict__ th, const float* __restrict__ z,
        float* __restrict__ out, int n, int npair, float temp) {
    const int pair = blockIdx.x;
    if (pair >= npair) return;
    const float* xr = x + (int64_t) rows[pair] * row_stride;
    const int tid = threadIdx.x, nth = blockDim.x;
    __shared__ float sred[32];
    const float invT = temp > 0.0f ? 1.0f / temp : 1.0f;
    float m = -3.4e38f;
    for (int i = tid; i < n; i += nth) m = fmaxf(m, xr[i]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, off));
    if ((tid & 31) == 0) sred[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (nth + 31) / 32; ++w) m = fmaxf(m, sred[w]);
        const float e0 = __expf((xr[ids[pair]] - m) * invT);
        out[pair] = (e0 >= th[pair]) ? e0 / z[pair] : 0.0f;
    }
}

// filtered residual sample: r_i = max(0, fp(p,i) - fq(q,i)) with fp/fq the FILTERED softmaxes
// (thresholds/masses precomputed by filter_stats). Same fixed-order CDF walk as kernel 3.
extern "C" __global__ void residual_sample_filtered_f32(
        const float* __restrict__ p, const float* __restrict__ q, int has_q, int n, float temp,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t stream_pos,
        float p_max, float p_th, float p_z, float q_max, float q_th, float q_z,
        uint32_t* __restrict__ out_tok) {
    const int tid = threadIdx.x, nth = blockDim.x;
    __shared__ float chunk_sum[1024];
    const float invT = temp > 0.0f ? 1.0f / temp : 1.0f;
    auto r_at = [&](int i) -> float {
        const float ep = __expf((p[i] - p_max) * invT);
        const float pi = (ep >= p_th) ? ep / p_z : 0.0f;
        float qi = 0.0f;
        if (has_q) {
            const float eq = __expf((q[i] - q_max) * invT);
            qi = (eq >= q_th) ? eq / q_z : 0.0f;
        }
        const float r = pi - qi; return r > 0.0f ? r : 0.0f;
    };
    const int len = (n + nth - 1) / nth;
    const int lo = tid * len, hi = min(lo + len, n);
    float s = 0.0f;
    for (int i = lo; i < hi; ++i) s += r_at(i);
    chunk_sum[tid] = s;
    __syncthreads();
    if (tid != 0) return;
    float total = 0.0f;
    for (int t = 0; t < nth; ++t) total += chunk_sum[t];
    if (total <= 0.0f) {
        float m2 = -3.4e38f; int am = 0;
        for (int i = 0; i < n; ++i) if (p[i] > m2) { m2 = p[i]; am = i; }
        *out_tok = (uint32_t) am; return;
    }
    uint4 rv = philox4(seed_lo, seed_hi, 0xFFFFFFFDu, stream_pos);
    float u = u01(rv.x) * total;
    float acc = 0.0f; int t = 0;
    for (; t < nth; ++t) { if (acc + chunk_sum[t] >= u) break; acc += chunk_sum[t]; }
    if (t == nth) t = nth - 1;
    int i = t * len, ihi = min(i + len, n);
    for (; i < ihi; ++i) { acc += r_at(i); if (acc >= u) break; }
    *out_tok = (uint32_t) min(i, n - 1);
}

// Keskar-style repetition/frequency/presence penalties applied IN PLACE to a logits copy —
// history-dependent, so both the verify column (p) and the draft head (q) get the SAME pass
// before filtering (symmetry keeps rejection sampling exact for the penalized+filtered target).
// hist holds the last `n_hist` generated ids; counts computed on the fly (n_hist is small).
extern "C" __global__ void penalize_logits_f32(
        float* __restrict__ x, const uint32_t* __restrict__ hist, int n_hist,
        float rep, float freq, float present, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_hist) return;
    const uint32_t id = hist[i];
    if (id >= (uint32_t) n) return;
    // first occurrence in hist does the whole adjustment (dedup without atomics)
    for (int j = 0; j < i; ++j) if (hist[j] == id) return;
    int cnt = 1;
    for (int j = i + 1; j < n_hist; ++j) if (hist[j] == id) ++cnt;
    float v = x[id];
    if (rep != 1.0f) v = v > 0.0f ? v / rep : v * rep;
    v -= freq * (float) cnt + present;
    x[id] = v;
}

// ---------------- 5. graph-capturable gumbel (device counter) ----------------
// The graph-draft chain replays with fixed kernel args — the sampling event counter must live
// in DEVICE memory and advance in-graph. bw24_sctr_inc bumps it; gumbel_perturb_ctr_f32 reads it.
extern "C" __global__ void bw24_sctr_inc(uint32_t* __restrict__ ctr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ctr[0] += 1;
}
extern "C" __global__ void gumbel_perturb_ctr_f32(
        const float* __restrict__ x, float* __restrict__ y, int n,
        uint32_t seed_lo, uint32_t seed_hi, const uint32_t* __restrict__ ctr, float temp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (temp <= 0.0f) { y[i] = x[i]; return; }
    uint4 r = philox4(seed_lo, seed_hi, (uint32_t) (i >> 2), ctr[0]);
    uint32_t v = (i & 3) == 0 ? r.x : (i & 3) == 1 ? r.y : (i & 3) == 2 ? r.z : r.w;
    y[i] = x[i] / temp + (-__logf(-__logf(u01(v))));
}

// Gumbel-max over the FILTERED distribution: y[i] = (e(x_i) >= th) ? x_i/T + G_i : -inf.
// Sampling argmax(y) == one draw from the filtered softmax — the draft must propose from
// the SAME filtered q the acceptance test uses (th from filter_stats on this row).
extern "C" __global__ void gumbel_perturb_filtered_f32(
        const float* __restrict__ x, float* __restrict__ y, int n,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t stream_pos, float temp,
        float row_max, float th) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float invT = temp > 0.0f ? 1.0f / temp : 1.0f;
    const float e0 = __expf((x[i] - row_max) * invT);
    if (e0 < th) { y[i] = -3.4e38f; return; }
    if (temp <= 0.0f) { y[i] = x[i]; return; }
    uint4 r = philox4(seed_lo, seed_hi, (uint32_t) (i >> 2), stream_pos);
    uint32_t v = (i & 3) == 0 ? r.x : (i & 3) == 1 ? r.y : (i & 3) == 2 ? r.z : r.w;
    y[i] = x[i] / temp + (-__logf(-__logf(u01(v))));
}

// Rows variant: penalize nrow contiguous rows of length n in one launch (grid.y = row).
extern "C" __global__ void penalize_logits_rows_f32(
        float* __restrict__ x, const uint32_t* __restrict__ hist, int n_hist,
        float rep, float freq, float present, int n, int nrow) {
    const int r = blockIdx.y;
    if (r >= nrow) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_hist) return;
    const uint32_t id = hist[i];
    if (id >= (uint32_t) n) return;
    for (int j = 0; j < i; ++j) if (hist[j] == id) return;
    int cnt = 1;
    for (int j = i + 1; j < n_hist; ++j) if (hist[j] == id) ++cnt;
    float* xr = x + (size_t) r * n;
    float v = xr[id];
    if (rep != 1.0f) v = v > 0.0f ? v / rep : v * rep;
    v -= freq * (float) cnt + present;
    xr[id] = v;
}

// ROUND-STREAM stage (a) (2026-07-10, HANDOVER design): the greedy accept walk ON DEVICE —
// verbatim replication of the host rule (walk j in 0..k_round, accept while t_pred(j) ==
// draft[j]; t_pred(0) = last_pred when base==0; bonus = t_pred(n_acc)). Stage (a) value is
// machinery, not speed (host still reads 8B back); stages (b)/(c) consume n_acc on-device.
extern "C" __global__ void spec_accept_greedy(
        const unsigned int* __restrict__ preds,   // [t_v] verify device argmaxes
        const unsigned int* __restrict__ draft,   // [k_round] draft token ids
        unsigned int last_pred, int base, int k_round,
        unsigned int* __restrict__ out) {         // out[0] = n_acc, out[1] = bonus
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n_acc = 0;
    for (int j = 0; j < k_round; j++) {
        unsigned int tp = (j == 0 && base == 0) ? last_pred : preds[base + j - 1];
        if (tp == draft[j]) n_acc++; else break;
    }
    unsigned int bonus = (n_acc == 0 && base == 0) ? last_pred : preds[base + n_acc - 1];
    out[0] = (unsigned int)n_acc;
    out[1] = bonus;
}

// ROUND-STREAM stage (b) piece 1: next-round seed gather ON DEVICE. Unifies the three commit
// arms' seed rules (spec.rs §5): j = base + n_acc; j >= 1 -> seed = vx[col j-1] (partial) which
// at full accept (j == t_v) is col t_v-1 — the same expression; j == 0 -> seed = fill_prev
// (zero-round fold). Writes BOTH h_seed slots (h_seed_buf and fill_prev get the same value in
// every arm). acc = the spec_accept_greedy output (out[0] = n_acc).
extern "C" __global__ void spec_seed_gather(
        const float* __restrict__ vx,          // [t_v, n_embd] verify hiddens
        const float* __restrict__ fill_prev,   // [n_embd] carried predecessor hidden
        const unsigned int* __restrict__ acc,  // acc[0] = n_acc
        float* __restrict__ h_seed,            // [n_embd] out (caller D2Ds into fill_prev after)
        int base, int n_embd) {
    int j = base + (int)acc[0];
    const float* src = (j >= 1) ? vx + (size_t)(j - 1) * n_embd : fill_prev;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        h_seed[i] = src[i];
}

// ROUND-STREAM stage (b) piece 3a: per-layer KV-len rollback ON DEVICE. Unified rule for all
// three non-replay commit arms: len[il] = saved[il] + j where j = base + n_acc (full accept
// j == t_v rewrites the value the verify already left — harmless; partial truncates; zero-fold
// restores the snapshot len). len_ptrs = device table of each full-attn layer's kvl.len_d;
// saved = snapshot lens; layers without KV carry len_ptr == 0 and are skipped.
extern "C" __global__ void spec_rollback_kv(
        unsigned long long* __restrict__ len_ptrs,   // [n_layer] device ptrs to i32 len_d
        const int* __restrict__ saved,               // [n_layer] snapshot lens
        const unsigned int* __restrict__ acc,        // acc[0] = n_acc
        int base, int n_layer) {
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= n_layer) return;
    int* lp = (int*)len_ptrs[il];
    if (lp == nullptr) return;
    lp[0] = saved[il] + base + (int)acc[0];
}

// ROUND-STREAM stage (c) piece 1: verify-token assembly ON DEVICE. verify_tok[0] = pending
// bonus when pend[0] != 0xFFFFFFFF (the no-pending sentinel), then the draft chain's packed
// slots (tokp[2j] = PRE-remap argmax idx) mapped through d2t (identity when d2t == nullptr).
// Also derives the p-min break vector: brk[0] = k_used = number of draft slots the host walk
// would have kept (first j with p < p_min, respecting j>0-only unless pmin0&&base, capped K).
extern "C" __global__ void spec_assemble_verify(
        const unsigned int* __restrict__ tokp,   // [2K] packed (idx, p-bits) per chain step
        const unsigned int* __restrict__ pend,   // [1] pending bonus or 0xFFFFFFFF
        const unsigned int* __restrict__ d2t,    // trimmed-head map or nullptr
        unsigned int* __restrict__ vtok,         // [K+1] out: verify tokens (used prefix)
        unsigned int* __restrict__ brk,          // [2] out: brk[0]=k_used, brk[1]=base
        float p_min, int k, int pmin0) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int base = (pend[0] != 0xFFFFFFFFu) ? 1 : 0;
    if (base) vtok[0] = pend[0];
    int k_used = 0;
    for (int j = 0; j < k; j++) {
        unsigned int idx = tokp[2 * j];
        float p = __uint_as_float(tokp[2 * j + 1]);
        if (p_min > 0.0f && p < p_min && (j > 0 || (pmin0 && base == 1))) break;
        unsigned int d = (d2t != nullptr) ? d2t[idx] : idx;
        vtok[base + k_used] = d;
        k_used++;
    }
    for (int j = base + k_used; j < base + k; j++) vtok[j] = 0u;  // embed-safe filler
    brk[0] = (unsigned int)k_used;
    brk[1] = (unsigned int)base;
}

// ROUND-STREAM stage (c) 4 epilogue kernels: single-thread device bookkeeping between rounds.
// ring layout: ring[0] = count, tokens from ring[1]. Appends this round's accepted prefix +
// bonus, sets pend <- bonus for the next round's assemble, bumps the pos counter is NOT done
// here (spec_rollback_kv owns every len/pos write).
extern "C" __global__ void spec_ring_commit(
        const unsigned int* __restrict__ vtok,   // [base + K] assembled verify tokens
        const unsigned int* __restrict__ acc,    // acc[0] = n_acc, acc[1] = bonus
        const unsigned int* __restrict__ brk,    // brk[1] = base
        unsigned int* __restrict__ ring,         // [1 + capacity]
        unsigned int* __restrict__ pend) {       // [1]
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int rc = ring[0];
    int base = (int)brk[1];
    int n_acc = (int)acc[0];
    for (int i = 0; i < n_acc; i++) ring[1 + rc + i] = vtok[base + i];
    ring[1 + rc + n_acc] = acc[1];
    ring[0] = rc + (unsigned int)n_acc + 1u;
    pend[0] = acc[1];
}

// i32 counter copy (scratch draft-KV len <- pos counter; draft rope pos <- pos + delta).
extern "C" __global__ void i32_copy_add(const int* __restrict__ src, int* __restrict__ dst,
                                        int delta) {
    if (threadIdx.x == 0) dst[0] = src[0] + delta;
}
// u32[1] copy (g_tok <- pending bonus).
extern "C" __global__ void u32_copy(const unsigned int* __restrict__ src,
                                    unsigned int* __restrict__ dst) {
    if (threadIdx.x == 0) dst[0] = src[0];
}

// ROUND-STREAM stage (c) 2: verify rope positions from a device pos counter (pos evolves by
// n_acc per pre-issued round; the host can't know it at issue time).
extern "C" __global__ void pos_iota_i32(const int* __restrict__ pos0, int* __restrict__ out, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t) out[i] = pos0[0] + i;
}

// ROUND-STREAM stage (c) 3: accept walk with k_round/base from DEVICE (the assemble kernel's
// brk output) and the draft tokens read from the assembled vtok (they ARE the d2t-mapped
// drafts). t_pred rule verbatim from spec_accept_greedy; last_pred_dev is host-seeded once
// (stream invariant: base == 1 on every round after the first, so it is read at most once).
extern "C" __global__ void spec_accept_greedy_dc(
        const unsigned int* __restrict__ preds,     // [t_v] verify device argmaxes
        const unsigned int* __restrict__ vtok,      // [t_v] assembled verify tokens
        const unsigned int* __restrict__ last_pred, // [1]
        const unsigned int* __restrict__ brk,       // [2] = (k_used, base)
        unsigned int* __restrict__ out) {           // out[0] = n_acc, out[1] = bonus
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int k_used = (int)brk[0];
    int base = (int)brk[1];
    int n_acc = 0;
    for (int j = 0; j < k_used; j++) {
        unsigned int tp = (j == 0 && base == 0) ? last_pred[0] : preds[base + j - 1];
        if (tp == vtok[base + j]) n_acc++; else break;
    }
    unsigned int bonus = (n_acc == 0 && base == 0) ? last_pred[0] : preds[base + n_acc - 1];
    out[0] = (unsigned int)n_acc;
    out[1] = bonus;
}
