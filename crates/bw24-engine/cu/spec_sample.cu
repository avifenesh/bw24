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
        const float* __restrict__ p, const float* __restrict__ q, int n, float temp,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t stream_pos,
        float p_max, float p_sumexp, float q_max, float q_sumexp,   // precomputed row stats
        uint32_t* __restrict__ out_tok) {
    // r_i evaluated on the fly: pi = exp((p[i]-p_max)/T)/p_sumexp ; qi likewise (q may be null).
    const int tid = threadIdx.x, nth = blockDim.x;
    extern __shared__ float chunk_sum[];                 // nth entries
    const float invT = temp > 0.0f ? 1.0f / temp : 0.0f;
    auto r_at = [&](int i) -> float {
        float pi = __expf((p[i] - p_max) * invT) / p_sumexp;
        float qi = q ? __expf((q[i] - q_max) * invT) / q_sumexp : 0.0f;
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
