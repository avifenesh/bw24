// bw24 engine Stage-1 kernels: correctness-first, all f32, no tensor cores.
// Math matches llama.cpp ggml CUDA ops node-for-node (norm.cu, rope.cu).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// PDL entry hook (SOTA item 2, 2026-07-13). Under a plain launch this is a documented
// no-op (grid dependencies are complete before any block starts). Under a PROGRAMMATIC
// graph edge (the BW24_PDL post-capture rewrite) it orders this kernel's global reads
// after the producer kernel's writes while still letting the grid launch overlap the
// producer's drain (~120ns/kernel on sm_120, pdl_probe). sm_90+ only; the sm_89
// portable arm compiles it out.
#if !defined(BW24_PORTABLE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BW24_PDL_ENTRY() cudaGridDependencySynchronize()
#else
#define BW24_PDL_ENTRY()
#endif

// ---- GPU-resident greedy argmax over logits[n_vocab] -> token_out[0] (u32). ----
// CUDA-GRAPH-PLAN Phase 1: removes the per-step dtoh(logits)+synchronize host barrier (the hard
// graph-capture blocker). Single CTA, 256 threads. Tie-break = SMALLEST index wins, bit-identical
// to the host argmax (forward.rs `if v > bv` strictly-greater keeps the first max). Each thread
// scans a strided slice keeping (best_val,best_id); reduce keeps the lower id on equal value.
// SUPERSEDED for the live path by the parallel 2-pass kernels below (argmax_partial_f32 +
// argmax_final_f32): one 256-thread block scanning 248K logits on ONE SM is HBM-starved (~448us/
// token, ncu clock-locked). Kept as the bit-exact single-CTA reference (same tie-break contract).
extern "C" __global__ void argmax_logits_f32_to_u32(
        const float* __restrict__ logits, uint32_t* __restrict__ token_out, int n_vocab) {
    int tid = threadIdx.x;
    float best_v = -3.402823466e38f;   // -FLT_MAX (matches f32::NEG_INFINITY seed)
    int   best_i = 0x7fffffff;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        // strictly-greater takes the new value; on a tie keep the smaller index.
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    // warp butterfly reduce: max value, smallest index on tie.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) token_out[0] = (uint32_t)best_i;
    }
}

// ---- PARALLEL argmax (2-pass, multi-CTA). RANK1 LEVER: the single-CTA argmax above scans the full
// 248320-vocab logits with ONE 256-thread block on ONE SM — memory-starved, ~426us/token. This pair
// fans the scan across NB blocks so HBM is saturated, then a 1-block final reduce picks the winner.
// BIT-IDENTICAL to the single-CTA kernel and to host `argmax` (forward.rs `v>bv`): strictly-greater
// takes the new value, ties keep the SMALLEST index. Pass 1 -> (part_v[NB], part_i[NB]); pass 2
// reduces those NB partials into token_out[0]. Both passes are plain launches (graph-capturable).
//
// Greedy-token softmax probability, pass 1: partial sums of exp(logit - max) where max =
// logits[tok] (tok = the argmax token, already on device). p(tok) = 1 / sum. Feeds the
// spec-decode p-min confidence gate (stop drafting when the head is unsure — the mechanism
// behind the serve script's --spec-draft-p-min): one extra ~1-4MB logits read per draft token.
extern "C" __global__ void prob_of_token_partial_f32(
        const float* __restrict__ logits, const uint32_t* __restrict__ tok,
        float* __restrict__ part_s, int n_vocab) {
    const float mx = logits[tok[0]];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = gtid; i < n_vocab; i += gstride) sum += expf(logits[i] - mx);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float ss[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? ss[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) part_s[blockIdx.x] = sum;
    }
}
// pass 2: p = 1 / sum(partials).
extern "C" __global__ void prob_of_token_final_f32(
        const float* __restrict__ part_s, float* __restrict__ p_out, int nb) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < nb; i += blockDim.x) sum += part_s[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float ss[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? ss[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) p_out[0] = 1.0f / sum;
    }
}

// Pass 1: block b, thread tid scans logits[b*blockDim + tid : n_vocab : NB*blockDim] keeping
// (best_v, smallest best_i), block-reduces, writes part_v[b]/part_i[b].
extern "C" __global__ void argmax_partial_f32(
        const float* __restrict__ logits, float* __restrict__ part_v, int* __restrict__ part_i,
        int n_vocab) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    float best_v = -3.402823466e38f;
    int   best_i = 0x7fffffff;
    for (int i = gtid; i < n_vocab; i += gstride) {
        float v = logits[i];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { part_v[blockIdx.x] = best_v; part_i[blockIdx.x] = best_i; }
    }
}

// Pass 2: ONE block reduces the NB partials into token_out[0]. Same tie-break (smallest index).
// nb = number of pass-1 blocks. Launch with block_dim >= 32 (256 used); strided over nb.
extern "C" __global__ void argmax_final_f32(
        const float* __restrict__ part_v, const int* __restrict__ part_i,
        uint32_t* __restrict__ token_out, int nb) {
    int tid = threadIdx.x;
    float best_v = -3.402823466e38f;
    int   best_i = 0x7fffffff;
    for (int i = tid; i < nb; i += blockDim.x) {
        float v = part_v[i];
        int   id = part_i[i];
        if (v > best_v || (v == best_v && id < best_i)) { best_v = v; best_i = id; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) token_out[0] = (uint32_t)best_i;
    }
}

// ---- RMSNorm: one block per row. y = x / sqrt(mean(x^2) + eps) * weight ----
// x: [ncols, nrows] row-major (row stride = ncols). weight: [ncols]. dst same shape as x.
extern "C" __global__ void rms_norm_f32(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ dst, int ncols, float eps) {
    BW24_PDL_ENTRY();
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

// ---- RANK3 LEVER (add+rmsnorm fuse): residual-add THEN RMSNorm in ONE kernel. ----
// res = a + b  (the residual, written out for the next residual-add); dst = rms_norm(res) * w.
// Fuses e.add(a,b,res) + e.rms_norm(res,w,dst) — removes one launch + one HBM read of `res` per
// residual+norm pair. BIT-IDENTICAL to add_f32 then rms_norm_f32: r=a[i]+b[i] is the same IEEE add,
// and the sum-of-squares reduction reads the same r values in the same per-thread/strided order.
// One block per row (row stride = ncols). a,b,res,dst: [ncols, nrows]; w: [ncols].
extern "C" __global__ void add_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
                                            const float* __restrict__ w, float* __restrict__ res,
                                            float* __restrict__ dst, int ncols, float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
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
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = rr[i] * scale * w[i];
}

// ---- E4B glue fusion: rms(a, wa) prologue + the add_rms_norm_f32 program — folds the
// per-layer post-attn rms_norm_f32(o) into the tail's residual-add+ffn-norm launch. ----
extern "C" __global__ void rms_pre_add_rms_norm_f32(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        const float* __restrict__ w, float* __restrict__ res,
        float* __restrict__ dst, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    __shared__ float s[128];
    // SINGLE-PHASE (parity with the q8z twin — verify t>1 and decode t=1 must share the
    // reduction algebra bit-for-bit; the 31B depth-spec 45/128 was this mismatch).
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3];
    float scale = rsqrtf(sumv2 / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] * ascale) * wa[i] + br[i];
        rr[i] = v;
        dr[i] = v * scale * w[i];
    }
}

// ---- E4B glue fusion wave 2: the two tail-entry programs with the ffn-norm output zsh
// ALSO emitted q8_1 (fused2 gate/up consume only the quantized pair at t=1; the f32 zsh
// stays written for the off-class fallbacks). Epilogue = quantize_q8_1's program verbatim. ----
extern "C" __global__ void rms_pre_add_rms_norm_q8z_f32(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        const float* __restrict__ w, float* __restrict__ res,
        float* __restrict__ dst,
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    int nblk = ncols / 32;
    __shared__ float s[128];
    // SINGLE-PHASE (wave 4, same algebra as the closing emit; c == 1 here):
    // sum(v^2) with v = a*ascale*wa + b  ==  ascale^2*S2 + 2*ascale*S3 + S4.
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3];
    float scale = rsqrtf(sumv2 / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        rr[i] = (ar[i] * ascale) * wa[i] + br[i];
    }
    __syncthreads();
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        d4[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- E4B glue fusion wave 2: a + b with the sum ALSO emitted q8_1 (resid feeds inp_gate
// through matmul_pre; f32 resid stays written for the later residual add). ----
// ---- E4B FFN-tail EXIT fusion (glue wave 5): resid = attn_out + rms(f0, post_ffw), ----
// emitted f32 + q8_1 pair in ONE launch — replaces rms_norm_f32(f0 -> sn) + add_q8_1_f32(sn,
// attn_out). BIT-IDENTITY: the rms reduction reads f0 in rms_norm_f32's exact strided order
// (same single sum, same block reduce); the per-element value is ((f0[i]*s)*w[i]) + b[i] — the
// identical op chain of the two-kernel pair (the f32 round-trip of sn is exact, removing it
// changes no bits); the quantize section is add_q8_1_f32's float4-quad walk verbatim.
extern "C" __global__ void rms_pre_add_q8_1_f32(
        const float* __restrict__ a,   // f0 (ffn_down output)
        const float* __restrict__ wa,  // post_ffw_norm weight
        const float* __restrict__ b,   // attn_out (the residual carry)
        float* __restrict__ res,       // resid f32 out
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;

    // phase 1: rms_norm_f32's reduction, verbatim.
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i]; sum += v * v; }
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

    // phase 2: add_q8_1_f32's quad walk verbatim, with v = ((a*scale)*wa) + b inline.
    int lane = tid & 31;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    const float* war = wa;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        const float4 a4 = ((const float4*)ar)[i4];
        const float4 w4 = ((const float4*)war)[i4];
        const float4 b4 = ((const float4*)br)[i4];
        float4 v = make_float4(a4.x * scale * w4.x + b4.x, a4.y * scale * w4.y + b4.y,
                               a4.z * scale * w4.z + b4.z, a4.w * scale * w4.w + b4.w);
        ((float4*)rr)[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

extern "C" __global__ void add_q8_1_f32(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ res,
                                        signed char* __restrict__ out_q, float* __restrict__ out_d,
                                        int ncols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int nblk = ncols / 32;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        const float4 a4 = ((const float4*)ar)[i4];
        const float4 b4 = ((const float4*)br)[i4];
        float4 v = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
        ((float4*)rr)[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- RMSNorm with FUSED q8_1 quantize epilogue (decode glue-fusion lever). ----
// Computes z = rms_norm(x)*w THEN emits z directly as q8_1 (out_q int8 + out_d f32 per-32 scale),
// so the standalone `quantize_q8_1` launch + the f32 `z` HBM round-trip are removed. The normed
// activation has exactly the matvec(s) as consumers, all on the q8_1 fast path — so producing it
// pre-quantized is free (rms_norm already touches every element). BIT-IDENTICAL to
// rms_norm_f32(x,w,z) then quantize_q8_1(z): the scale `s = rsqrt(mean(x^2)+eps)` reduction reads
// the same x in the same strided order; the normed value is the SAME (x[i]*s)*w[i] association;
// the per-32-block amax/d=amax/127/id=1/d/__float2int_rn rounding is quantize_q8_1's exactly.
// One block per row (decode: nrows=1). ncols must be a multiple of 32 (n_embd always is).
extern "C" __global__ void rms_norm_q8_1(const float* __restrict__ x, const float* __restrict__ w,
                                         signed char* __restrict__ out_q, float* __restrict__ out_d,
                                         int ncols, float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    int nblk = ncols / 32;
    // pass 1: sum of squares -> scale (identical reduction to rms_norm_f32)
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
    float scale = rsqrtf(s[0] / ncols + eps);
    // pass 2, WARP-PER-4-BLOCKS float4 (ncu 2026-07-03): lane j reads float4 -> a warp covers 128
    // elements = FOUR 32-blocks per iteration (512B coalesced x/w reads, char4 writes). Each block
    // maps to an 8-lane group; amax reduces within the group (3 shfl_xor steps, width 8). Order of
    // max over the same 32 values is irrelevant -> q8_1 output BIT-IDENTICAL to quantize_q8_1.
    // (Plain warp-per-block regressed here: single-CTA kernel, 8 warps -> too little MLP.)
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)xr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;               // float4 index; 32 lanes * 4 = 128 elems = 4 blocks
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
    // tail (nblk % 4 != 0): scalar warp-per-block for the last <4 blocks.
    for (int blk = (nblk & ~3) + (tid >> 5); blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (xr[i] * scale) * w[i];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
}

// ---- add+RMSNorm with FUSED q8_1 quantize epilogue. res = a+b (written out for the next residual);
// then z = rms_norm(res)*w emitted directly as q8_1. Fuses add_rms_norm + quantize_q8_1 for the FFN
// input path (z feeds ffn_gate/ffn_up matvecs, both q8_1-fast). BIT-IDENTICAL to add_rms_norm_f32
// then quantize_q8_1: r=a[i]+b[i] same IEEE add (and written to `res` for the post-ffn add), the
// sum-of-squares reduction reads the same r, z=(r*scale)*w same association, per-32 q8_1 identical.
// add+RMSNorm emitting BOTH the f32 normed row (z — the MoE router logits input) AND its q8_1
// quantization (the expert dp4a input) in one launch. The MoE layer needs both views of the same
// vector; running add_rms_norm_f32 then quantize_q8_1 costs a launch and re-reads z from HBM.
// BIT-IDENTICAL: z values same IEEE ops as add_rms_norm_f32; q8 blocks same amax/127 rounding as
// quantize_q8_1 over the same z.
extern "C" __global__ void add_rms_norm_zq8(const float* __restrict__ a, const float* __restrict__ b,
                                            const float* __restrict__ w, float* __restrict__ res,
                                            float* __restrict__ z,
                                            signed char* __restrict__ out_q, float* __restrict__ out_d,
                                            int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* zr = z + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
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
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    for (int blk = tid >> 5; blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (rr[i] * scale) * w[i];
        zr[i] = v;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
}

extern "C" __global__ void add_rms_norm_q8_1(const float* __restrict__ a, const float* __restrict__ b,
                                             const float* __restrict__ w, float* __restrict__ res,
                                             signed char* __restrict__ out_q, float* __restrict__ out_d,
                                             int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
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
    // pass 2, WARP-PER-4-BLOCKS float4: same coalesced form as rms_norm_q8_1 (see comment there).
    // Reads the just-written `res` row (rr) — bit-identical (same IEEE values back from cache/HBM).
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* r4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = r4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
    for (int blk = (nblk & ~3) + (tid >> 5); blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (rr[i] * scale) * w[i];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
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

// ---- gemma4: 3-way rms_norm — ONE reduction over x, three weight vectors/outputs
// (gemma's attn_out feeds ffn_norm + router-scale + pre_ffw_norm_2). Numerics per output
// identical to rms_norm_f32 (same block reduction, same scale multiply). ----
extern "C" __global__ void rms_norm3_f32(const float* __restrict__ x,
                                         const float* __restrict__ w0, const float* __restrict__ w1,
                                         const float* __restrict__ w2,
                                         float* __restrict__ d0, float* __restrict__ d1,
                                         float* __restrict__ d2, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    // block reduce — the rms_norm_f32 shuffle tree VERBATIM (per-output bit-identity).
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
    float* o0 = d0 + (size_t)row * ncols;
    float* o1 = d1 + (size_t)row * ncols;
    float* o2 = d2 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float xh = xr[i] * scale;
        o0[i] = xh * w0[i];
        o1[i] = xh * w1[i];
        o2[i] = xh * w2[i];
    }
}

// ---- gemma4: q/k/v head norms in ONE launch — 3 (src,dst,rows) segments of the same width
// (q_norm rows=nh, k_norm rows=nkv, weightless-V rows=nkv). Segment picked by row index;
// per-row chain = rms_norm_f32 verbatim. ----
extern "C" __global__ void rms_norm_qkv_f32(const float* __restrict__ q, const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            const float* __restrict__ wq, const float* __restrict__ wk,
                                            const float* __restrict__ wv,
                                            float* __restrict__ dq, float* __restrict__ dk,
                                            float* __restrict__ dv,
                                            int ncols, int rq, int rk, float eps) {
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;        w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq; xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
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
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// ---- warp-per-row twin of rms_norm_qkv_f32 (prefill T>=16): the block-per-row form runs
// 17k+ tiny blocks of rms_block() threads over 512-col head rows at ~92GB/s (launch/reduce
// latency dominates the 2KB/row payload). Here: 8 warps/block, one ROW per warp, float4
// loads, warp-shuffle reduce only. OWN NUMERIC CONFIG (float4-lane partial sums reduce in a
// different order than the block tree) — battery-gated, BW24_QKVNORM_W=0 reverts. ----
extern "C" __global__ void rms_norm_qkv_w4_f32(const float* __restrict__ q, const float* __restrict__ k,
                                               const float* __restrict__ v,
                                               const float* __restrict__ wq, const float* __restrict__ wk,
                                               const float* __restrict__ wv,
                                               float* __restrict__ dq, float* __restrict__ dk,
                                               float* __restrict__ dv,
                                               int ncols, int rq, int rk, int rv, float eps) {
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rq + rk + rv) return;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;              w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq;      xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    const int nc4 = ncols >> 2;
    const float4* x4 = (const float4*)xr;
    float sum = 0.0f;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i];
        sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float scale = rsqrtf(sum / ncols + eps);
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i]; float4 wv4 = w4[i];
        float4 ov;
        ov.x = xv.x * scale * wv4.x; ov.y = xv.y * scale * wv4.y;
        ov.z = xv.z * scale * wv4.z; ov.w = xv.w * scale * wv4.w;
        d4[i] = ov;
    }
}

extern "C" __global__ void rms_norm_qkv_w4b_f32(const float* __restrict__ q, const float* __restrict__ k,
                                               const float* __restrict__ v,
                                               const float* __restrict__ wq, const float* __restrict__ wk,
                                               const float* __restrict__ wv,
                                               float* __restrict__ dq, float* __restrict__ dk,
                                               float* __restrict__ dv,
                                               __nv_bfloat16* __restrict__ dvb,
                                               int ncols, int rq, int rk, int rv, float eps,
                                               int vf16) {
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rq + rk + rv) return;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;              w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq;      xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    const int nc4 = ncols >> 2;
    const float4* x4 = (const float4*)xr;
    float sum = 0.0f;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i];
        sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float scale = rsqrtf(sum / ncols + eps);
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    // v rows also emit bf16 (the FA V operand; q/k get theirs post-rope).
    __nv_bfloat16* db = (row >= rq + rk) ? dvb + (size_t)(row - rq - rk) * ncols : nullptr;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i]; float4 wv4 = w4[i];
        float4 ov;
        ov.x = xv.x * scale * wv4.x; ov.y = xv.y * scale * wv4.y;
        ov.z = xv.z * scale * wv4.z; ov.w = xv.w * scale * wv4.w;
        d4[i] = ov;
        if (db) {
            if (vf16) {   // f16-P/V door: V operand consumed as __half by the h2/sp16 stamps
                __half* dh = (__half*)db;
                dh[4*i+0] = __float2half(ov.x); dh[4*i+1] = __float2half(ov.y);
                dh[4*i+2] = __float2half(ov.z); dh[4*i+3] = __float2half(ov.w);
            } else {
                db[4*i+0] = __float2bfloat16(ov.x); db[4*i+1] = __float2bfloat16(ov.y);
                db[4*i+2] = __float2bfloat16(ov.z); db[4*i+3] = __float2bfloat16(ov.w);
            }
        }
    }
}

// ---- E4B glue fusion wave 3: rms_norm_qkv + rope_neox2 in ONE launch. Row segments as in
// rms_norm_qkv_f32; after the norm store, q rows (seg 0) and k rows (seg 1) rope in-block
// (rope_neox math verbatim on the normed row; barrier between store and rope read). ----
// cat twin (wave 4b): the q|k|v input is ONE contiguous buffer (the qkv_cat matvec output),
// so the three input segments collapse to base + row*ncols. Outputs stay separate.
extern "C" __global__ void rms_norm_qkv_rope_cat_f32(
        const float* __restrict__ qkv,
        const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int ncols, int rq, int rk,
        const int* __restrict__ pos, int nh_q, int nh_k,
        float theta_scale, float freq_scale, const float* __restrict__ ff,
        float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    const float* xr = qkv + (size_t)row * ncols;
    const float* w; float* dr;
    int seg; int seg_r;
    if (row < rq)           { seg = 0; seg_r = row;           w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { seg = 1; seg_r = row - rq;      w = wk; dr = dk + (size_t)seg_r * ncols; }
    else                    { seg = 2; seg_r = row - rq - rk; w = wv; dr = dv + (size_t)seg_r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
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
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
    if (seg == 2) return;
    __syncthreads();
    int half = ncols / 2;
    int j = tid;
    if (j >= half) return;
    int tok = (seg == 0) ? seg_r / nh_q : seg_r / nh_k;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = dr[j];
    float x1 = dr[j + half];
    dr[j]        = x0 * c - x1 * sn;
    dr[j + half] = x0 * sn + x1 * c;
}

extern "C" __global__ void rms_norm_qkv_rope_f32(
        const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int ncols, int rq, int rk,
        const int* __restrict__ pos, int nh_q, int nh_k,
        float theta_scale, float freq_scale, const float* __restrict__ ff,
        float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    int seg; int seg_r;
    if (row < rq)           { seg = 0; seg_r = row;           xr = q + (size_t)row * ncols;   w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { seg = 1; seg_r = row - rq;      xr = k + (size_t)seg_r * ncols; w = wk; dr = dk + (size_t)seg_r * ncols; }
    else                    { seg = 2; seg_r = row - rq - rk; xr = v + (size_t)seg_r * ncols; w = wv; dr = dv + (size_t)seg_r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
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
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
    if (seg == 2) return;                   // V: norm only, never roped
    __syncthreads();                        // normed row visible before the rope read
    // rope_neox on the normed row (n_dims == ncols == head_dim here; math verbatim).
    int half = ncols / 2;
    int j = tid;
    if (j >= half) return;
    int tok = (seg == 0) ? seg_r / nh_q : seg_r / nh_k;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = dr[j];
    float x1 = dr[j + half];
    dr[j]        = x0 * c - x1 * sn;
    dr[j + half] = x0 * sn + x1 * c;
}

// ---- gemma4: two rms_norms of two DIFFERENT inputs, same width, one launch
// (post_ffw_norm_1(mlp0) + post_ffw_norm_2(moe0)). grid.x = 2*nrows; per-row verbatim. ----
extern "C" __global__ void rms_norm2x_f32(const float* __restrict__ a, const float* __restrict__ b,
                                          const float* __restrict__ wa, const float* __restrict__ wb,
                                          float* __restrict__ da, float* __restrict__ db,
                                          int ncols, int nrows, float eps) {
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    if (row < nrows) { xr = a + (size_t)row * ncols; w = wa; dr = da + (size_t)row * ncols; }
    else { int r = row - nrows; xr = b + (size_t)r * ncols; w = wb; dr = db + (size_t)r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
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
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// ---- gemma4: dst = (a + b) * c — the layer-tail residual add + layer_output_scale in one
// launch. Same IEEE add-then-multiply as add_f32 followed by scale_f32. ----
extern "C" __global__ void add_scale_f32(const float* __restrict__ a, const float* __restrict__ b,
                                         float c, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (a[i] + b[i]) * c;
}

// ---- gemma4 R4: final-logit softcap, y = cap * tanh(y / cap), in place. ----
extern "C" __global__ void softcap_f32(float* __restrict__ y, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(y[i] / cap);
}

// ---- gemma4: suppress-token mask — y[row][ids[j]] = -inf for every logits row, so every
// downstream consumer (host/device argmax, sampler) inherits the model card's forbidden ids. ----
extern "C" __global__ void mask_ids_rows_f32(float* __restrict__ y, const int* __restrict__ ids,
                                             int n_ids, int n_vocab, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_ids * t) y[(size_t)(i / n_ids) * n_vocab + ids[i % n_ids]] = -INFINITY;
}

// ---- gemma4: residual add + layer scale + NEXT layer's attn_norm in one launch.
// res = (a+b)*c (add_scale_f32 verbatim); dst = rms_norm(res, w) (rms_norm_f32 verbatim). ----
extern "C" __global__ void add_scale_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
                                                  float c, const float* __restrict__ w,
                                                  float* __restrict__ res, float* __restrict__ dst,
                                                  int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] + br[i]) * c;
        rr[i] = v;
        sum += v * v;
    }
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
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = rr[i] * scale * w[i];
}

// ---- gemma4: residual add + the THREE attn_out norms in one launch (add then rms_norm3,
// per-element chains verbatim). ----
extern "C" __global__ void add_rms_norm3_f32(const float* __restrict__ a, const float* __restrict__ b,
                                             const float* __restrict__ w0, const float* __restrict__ w1,
                                             const float* __restrict__ w2,
                                             float* __restrict__ res,
                                             float* __restrict__ d0, float* __restrict__ d1,
                                             float* __restrict__ d2, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = ar[i] + br[i];
        rr[i] = v;
        sum += v * v;
    }
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
    float* o0 = d0 + (size_t)row * ncols;
    float* o1 = d1 + (size_t)row * ncols;
    float* o2 = d2 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float xh = rr[i] * scale;
        o0[i] = xh * w0[i];
        o1[i] = xh * w1[i];
        o2[i] = xh * w2[i];
    }
}

// ---- gemma4: residual add + layer scale + next attn_norm EMITTED q8_1 (the mixer input is
// consumed only by quantized matmuls). res = (a+b)*c; norm chain = rms_norm_f32; quantize
// epilogue = rms_norm_q8_1's warp-per-4-blocks float4 form (bit-identical to quantize_q8_1). ----
extern "C" __global__ void add_scale_rms_norm_q8_1(const float* __restrict__ a, const float* __restrict__ b,
                                                   float c, const float* __restrict__ w,
                                                   float* __restrict__ res,
                                                   signed char* __restrict__ out_q, float* __restrict__ out_d,
                                                   int ncols, float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] + br[i]) * c;
        rr[i] = v;
        sum += v * v;
    }
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
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- E4B glue fusion (2026-07-12): rms-normalize `a` FIRST (the PLE tail's post_norm of y),
// then the add_scale_rms_norm_q8_1 program verbatim on (a_normed, b). Replaces the separate
// rms_norm_f32(y) launch per layer. Two full-row reductions, one launch. ----
extern "C" __global__ void rms_pre_add_scale_rms_norm_q8_1(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        float c, const float* __restrict__ w,
        float* __restrict__ res,
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    __shared__ float s[128];
    // SINGLE-PHASE reductions (wave 4): four simultaneous sums in one pass —
    //   S1 = sum(a^2)            -> ascale
    //   S2 = sum((a*wa)^2), S3 = sum(a*wa*b), S4 = sum(b^2)
    // then sum(v^2) with v = (a*ascale*wa + b)*c expands ALGEBRAICALLY to
    //   c^2 * (ascale^2*S2 + 2*ascale*S3 + S4)
    // — one barrier round instead of two full reduction phases. FP-order differs from the
    // sequential two-phase form (expansion rounding); the argmax/chat gates arbitrate.
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = c * c * (ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3]);
    float scale = rsqrtf(sumv2 / ncols + eps);
    // store pass: rr written here (the reduction pass no longer writes it).
    for (int i = tid; i < ncols; i += blockDim.x) {
        float an = (ar[i] * ascale) * wa[i];
        rr[i] = (an + br[i]) * c;
    }
    __syncthreads();
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- gemma4: residual add + the three attn_out norms with TWO outputs emitted q8_1
// (zsh -> the quantized gate/up pair input; moe_in -> the quantized expert input) and the
// router input f32. Chains: add + rms_norm3 + quantize_q8_1 verbatim per element. ----
extern "C" __global__ void add_rms_norm3_q8z_f32(const float* __restrict__ a, const float* __restrict__ b,
                                                 const float* __restrict__ w0, const float* __restrict__ w1,
                                                 const float* __restrict__ w2,
                                                 float* __restrict__ res,
                                                 signed char* __restrict__ q0, float* __restrict__ d0,
                                                 float* __restrict__ out1,
                                                 signed char* __restrict__ q2, float* __restrict__ d2,
                                                 int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = ar[i] + br[i];
        rr[i] = v;
        sum += v * v;
    }
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
    // f32 router output (plain rms_norm write)
    float* o1 = out1 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) o1[i] = rr[i] * scale * w1[i];
    // q8 outputs: the rms_norm_q8_1 warp-per-4-blocks float4 epilogue, once per weight vector.
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    {
        signed char* bq = q0 + (size_t)row * ncols;
        float* bd = d0 + (size_t)row * nblk;
        const float4* w4 = (const float4*)w0;
        for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
            int i4 = quad * 32 + lane;
            float4 xv = x4[i4];
            float4 wv = w4[i4];
            float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                                   (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            float d = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                                  (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
            ((char4*)bq)[i4] = qv;
            if ((lane & 7) == 0) bd[quad * 4 + (lane >> 3)] = d;
        }
    }
    {
        signed char* bq = q2 + (size_t)row * ncols;
        float* bd = d2 + (size_t)row * nblk;
        const float4* w4 = (const float4*)w2;
        for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
            int i4 = quad * 32 + lane;
            float4 xv = x4[i4];
            float4 wv = w4[i4];
            float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                                   (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            float d = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                                  (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
            ((char4*)bq)[i4] = qv;
            if ((lane & 7) == 0) bd[quad * 4 + (lane >> 3)] = d;
        }
    }
}

// ---- gemma4: q AND k roped in ONE launch (two segments on grid.x; per-row math =
// rope_neox_f32 / rope_neox_ff_f32 verbatim; ff = nullptr -> plain). ----
extern "C" __global__ void rope_neox2_f32(float* __restrict__ q, float* __restrict__ k,
                                          const int* __restrict__ pos,
                                          int head_dim, int n_dims, int nh_q, int nh_k,
                                          int n_tokens, float theta_scale, float freq_scale,
                                          const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int total_q = nh_q * n_tokens;
    float* base; int tok;
    if (hr < total_q) { base = q + (size_t)hr * head_dim; tok = hr / nh_q; }
    else { int r = hr - total_q; base = k + (size_t)r * head_dim; tok = r / nh_k; }
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * sn;
    base[j + half] = x0 * sn + x1 * c;
}

// bf16-emit twin (31B glue lane 2026-07-23): identical rope math + stores, ALSO emits the
// post-rope values as bf16 (the exact __float2bfloat16 the FA pre-converter applied) — the FA
// operands come out of this launch, killing the separate q/k f32->bf16 convert + re-read.
extern "C" __global__ void rope_neox2_bf16e_f32(float* __restrict__ q, float* __restrict__ k,
                                          __nv_bfloat16* __restrict__ qb, __nv_bfloat16* __restrict__ kb,
                                          const int* __restrict__ pos,
                                          int head_dim, int n_dims, int nh_q, int nh_k,
                                          int n_tokens, float theta_scale, float freq_scale,
                                          const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int total_q = nh_q * n_tokens;
    float* base; __nv_bfloat16* baseb; int tok;
    if (hr < total_q) { base = q + (size_t)hr * head_dim; baseb = qb + (size_t)hr * head_dim; tok = hr / nh_q; }
    else { int r = hr - total_q; base = k + (size_t)r * head_dim; baseb = kb + (size_t)r * head_dim; tok = r / nh_k; }
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    float y0 = x0 * c - x1 * sn;
    float y1 = x0 * sn + x1 * c;
    base[j]        = y0;
    base[j + half] = y1;
    baseb[j]        = __float2bfloat16(y0);
    baseb[j + half] = __float2bfloat16(y1);
}

// ---- tiny async setters/packers (gemma spec round: zero host-memory transfers) ----
// gemma4-E4B: gather layer il's per-layer-input rows out of the [t][n_layer][n_epl] prologue
// buffer into a dense [t][n_epl] operand (row t at src offset (t*stride + off)).
extern "C" __global__ void copy_rows_strided_f32(
        const float* __restrict__ src, float* __restrict__ dst,
        int row_elems, int n_rows, long src_stride, long src_off) {
    int r = blockIdx.y;
    if (r >= n_rows) return;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < row_elems; j += gridDim.x * blockDim.x)
        dst[(size_t)r * row_elems + j] = src[(size_t)r * src_stride + src_off + j];
}

extern "C" __global__ void u32_set_k(unsigned int* __restrict__ dst, unsigned int v, int idx) {
    dst[idx] = v;
}

// FR-Spec trim id translate (gemma async draft round): buf[idx] holds a TRIM-space argmax --
// map it to the full-vocab token id in place (d2t = ranked-id table). Single-slot, async.
extern "C" __global__ void u32_map_k(unsigned int* __restrict__ buf,
                                     const unsigned int* __restrict__ map, int idx) {
    buf[idx] = map[buf[idx]];
}

// pos-row fill from a device counter: dst[i] = ctr[0] + i (verify-stream rope positions —
// no host pos value; one launch per verify).
extern "C" __global__ void i32_iota_from(const int* __restrict__ ctr, int* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = ctr[0] + i;
}

// counter += v (device-slot append advance for t rows; the +1 twin is inc_seqlen).
extern "C" __global__ void i32_add_k(int* __restrict__ d, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) d[0] += v;
}

// i32 twin (device-len counters, graph arc): async single-slot store, value rides the arg.
extern "C" __global__ void i32_set_k(int* dst, int v, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[idx] = v;
}
// pack: out[0..n1) = a[off_a..], out[n1..n1+n2) = b[0..n2) (single dtoh follows).
extern "C" __global__ void u32_pack2(const unsigned int* __restrict__ a, int off_a, int n1,
                                     const unsigned int* __restrict__ b, int n2,
                                     unsigned int* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n1) out[i] = a[off_a + i];
    else if (i < n1 + n2) out[i] = b[i - n1];
}

// ---- gemma4 R1: GELU(tanh approx) * up GLU epilogue. Constants = ggml's GELU_COEF_A /
// SQRT_2_OVER_PI so the activation matches llama.cpp's CUDA gelu op float-for-float. ----
extern "C" __global__ void gelu_tanh_mul_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                             float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float t = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        dst[i] = 0.5f * x * (1.0f + t) * up[i];
    }
}

// ---- E4B/gemma glue fusion (2026-07-12): GELU(tanh)*up with the activation EMITTED q8_1
// (per-32-block amax quantize, bit-identical to quantize_q8_1's rounding — the add_scale
// emit epilogue's program). The down/proj matmul then rides matmul_pre: one launch replaces
// gelu_tanh_mul_f32 + quantize_q8_1. Row-major [nrows, ncols]; ncols % 128 == 0. ----
extern "C" __global__ void gelu_tanh_mul_q8_1(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              float* __restrict__ act,
                                              signed char* __restrict__ out_q,
                                              float* __restrict__ out_d,
                                              int ncols) {
    BW24_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int nblk = ncols / 32;
    const float* gr = gate + (size_t)row * ncols;
    const float* ur = up + (size_t)row * ncols;
    float* arow = act + (size_t)row * ncols;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;   // float4 index
        const float4 g4 = ((const float4*)gr)[i4];
        const float4 u4 = ((const float4*)ur)[i4];
        float vx[4] = {g4.x, g4.y, g4.z, g4.w};
        float ux[4] = {u4.x, u4.y, u4.z, u4.w};
        float o[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            float x = vx[e];
            float t = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
            o[e] = 0.5f * x * (1.0f + t) * ux[e];
        }
        ((float4*)arow)[i4] = make_float4(o[0], o[1], o[2], o[3]);
        float amax = fmaxf(fmaxf(fabsf(o[0]), fabsf(o[1])), fmaxf(fabsf(o[2]), fabsf(o[3])));
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(o[0] * id), (signed char)__float2int_rn(o[1] * id),
                              (signed char)__float2int_rn(o[2] * id), (signed char)__float2int_rn(o[3] * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- gemma4 R9: RoPE NEOX with per-dim freq factors (rope_freqs.weight, global layers).
// theta = pos * base^(-2j/d) / ff[j] (llama rope_ext freq_factors semantics, freq_scale = 1). ----
extern "C" __global__ void rope_neox_ff_f32(float* __restrict__ x, const int* __restrict__ pos,
                                            int head_dim, int n_dims, int n_heads,
                                            float theta_scale, float freq_scale,
                                            const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int tok = hr / n_heads;
    float* base = x + (size_t)hr * head_dim;
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
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
// FFN SwiGLU epilogue fusion (RANK3 LEVER 2). Folds the per-tensor NVFP4 macro-scale of the gate
// and up matmuls INTO the silu*mul, removing the two separate `scale_f32` launches per dense FFN
// layer. BIT-IDENTICAL to scale_f32(gate,gs); scale_f32(up,us); silu_mul_f32(gate,up,dst): same
// float ops in the same order — multiply by scale, then silu(g'), then multiply by up'. For
// non-NVFP4 weights gs==us==1.0, so this reduces exactly to silu_mul_f32.
extern "C" __global__ void silu_mul_scaled_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                               float gs, float us, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i] * gs; dst[i] = (g / (1.0f + expf(-g))) * (up[i] * us); }
}
// swigluoai (MiniMax-M3 / GPT-OSS): clamped SwiGLU. Math 1:1 vs llama.cpp
// ggml_cuda_op_swiglu_oai_single (unary.cuh:107): gate clamps ABOVE only, up clamps both sides,
// swish uses alpha inside the sigmoid, and the linear term is (1 + up). gs/us fold the NVFP4
// per-tensor macro-scales exactly like silu_mul_scaled_f32 (gs==us==1.0 for non-NVFP4).
extern "C" __global__ void swigluoai_mul_scaled_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                                    float gs, float us, float alpha, float limit,
                                                    float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = fminf(gate[i] * gs, limit);
        float g = fmaxf(fminf(up[i] * us, limit), -limit);
        dst[i] = (x / (1.0f + expf(-x * alpha))) * (1.0f + g);
    }
}
// RANK2 LEVER (q8_1 quant-fold): FFN SwiGLU epilogue that ALSO emits the q8_1 quantization of its
// own output, so ffn_down's activation is produced pre-quantized and the standalone quantize_q8_1
// launch is removed (1 fewer launch + no f32 `act` HBM round-trip per dense FFN layer). The down-proj
// activation has EXACTLY ONE consumer (ffn_down's matvec), so folding the quant into the producer is
// free — silu*mul already touches every element once; here each thread owns one 32-block, computes
// its 32 silu*mul values, finds amax over the block, and writes q8_1 (aq int8 + ad f32 scale).
// BIT-IDENTICAL q8_1 to scale->silu_mul->quantize_q8_1: same float silu*mul (g*gs, up*us), same
// d=amax/127, same id=1/d, same __float2int_rn rounding. n must be a multiple of 32 (n_ff always is).
// WARP-PER-BLOCK (decode elementwise-soup fix, ncu 2026-07-03): lane j of a warp owns element j of
// one 32-block -> fully coalesced 128B gate/up reads + 32B q8 writes. The old thread-owns-block form
// read 32 SEQUENTIAL floats per thread (32-way uncoalesced) on a nblk-thread grid (384 threads for
// n_ff=12288) and measured 22.7us vs ~0.15us of actual DRAM traffic. amax via __shfl_xor max is
// order-independent (max is associative+commutative) -> d and every q8 value stay BIT-IDENTICAL.
extern "C" __global__ void silu_mul_scaled_q8_1(
        const float* __restrict__ gate, const float* __restrict__ up, float gs, float us,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global 32-block index
    int lane = threadIdx.x & 31;
    int nblk = n / 32;
    if (warp >= nblk) return;
    int i = warp * 32 + lane;
    float g = gate[i] * gs;
    float r = (g / (1.0f + expf(-g))) * (up[i] * us);   // silu(g)*up, bit-identical
    float amax = fabsf(r);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[i] = (signed char)__float2int_rn(r * id);
    if (lane == 0) out_d[warp] = d;
}

extern "C" __global__ void add_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}
// y[i] *= s. NVFP4 per-tensor macro-scale broadcast over the whole matmul output.
extern "C" __global__ void scale_f32(float* __restrict__ y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}
// BW24_FULL_PREC dequant-on-use: expand a bf16-resident matmul weight (u16 LE, upper 16 bits of
// f32) to a transient f32 scratch that feeds the SAME cuBLASLt f32 GEMV the Float arm uses. This is
// bit-identical to the load-time bf16->f32 dequant (dequant::bf16_to_f32), just deferred to keep
// VRAM at 2 B/w resident instead of 4.
extern "C" __global__ void bf16_to_f32(const unsigned short* __restrict__ in,
                                       float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float(((unsigned int)in[i]) << 16);
}
extern "C" __global__ void mul_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] * b[i];
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

// Windowed twin of sdpa_naive_f32 (gemma4 R6 SWA): additionally masks keys OLDER than
// q_pos - (window-1) — llama's sliding-window mask (window keys incl self). window <= 0 = none.
extern "C" __global__ void sdpa_naive_w_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                            const float* __restrict__ V, float* __restrict__ O,
                                            int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                            float scale, int causal, int window) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        if (window > 0 && t < q_pos - (window - 1)) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
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

// MoE router GEMV (BW24_ROUTER_KERNEL=1): logits[t][e] = dot(W[e], x[t]) — replaces ~200
// cuBLASLt dispatches/round (4% of the 35B spec round loop, 2026-07-10 BW24_PROFILE_SPEC=2).
// One warp per (expert, token); fixed-stride f32 accumulation + standard warp reduce —
// DETERMINISTIC but a DIFFERENT FP order than cuBLAS: new numeric config, the router feeds
// top-k selection (discontinuous) so the full battery + MOE_GATE oracle arbitrate adoption.
extern "C" __global__ void router_gemv_f32(
        const float* __restrict__ w,   // [n_experts, n_embd] row-major
        const float* __restrict__ x,   // [t, n_embd]
        float* __restrict__ y,         // [t, n_experts]
        int n_embd, int n_experts, int t) {
    const int e = blockIdx.x;
    const int tok = blockIdx.y;
    if (e >= n_experts || tok >= t) return;
    const float* wr = w + (size_t) e * n_embd;
    const float* xr = x + (size_t) tok * n_embd;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += 32) s += wr[i] * xr[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if (threadIdx.x == 0) y[(size_t) tok * n_experts + e] = s;
}

// ROUND-STREAM stage (c) draft-chain pack: (tok, p) into slot j of a u32[2K] buffer — the
// host (or the assemble kernel) reads the whole chain in one go instead of 2 DtoHs per token.
extern "C" __global__ void pack_tok_p(const unsigned int* __restrict__ tok,
                                      const float* __restrict__ p,
                                      unsigned int* __restrict__ out, int slot) {
    if (threadIdx.x == 0) { out[2 * slot] = tok[0]; out[2 * slot + 1] = __float_as_uint(p[0]); }
}

// In-graph trimmed-head token remap: tok[0] = map[tok[0]] — replaces the host read-map-patch
// round trip inside the K-chain draft graph. Exact integer identity with the host map.
extern "C" __global__ void tok_map_u32(unsigned int* __restrict__ tok,
                                       const unsigned int* __restrict__ map) {
    if (threadIdx.x == 0) tok[0] = map[tok[0]];
}

// DSpark semi-AR markov head (dflash lane, 2026-07-13): gather ONE bf16 row of
// markov_w1 [V, rank] by a DEVICE token id into f32 (the rank-256 step vector). The
// sequential draft chain stays on-device (no per-position dtoh).
extern "C" __global__ void gather_row_bf16_f32(const unsigned short* __restrict__ table,
                                               const unsigned int* __restrict__ tok, int idx,
                                               float* __restrict__ dst, int ncols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncols) {
        unsigned short h = table[(size_t)tok[idx] * ncols + i];
        dst[i] = __uint_as_float(((unsigned int)h) << 16);
    }
}

// bias add on ONE logits row: logits[row0*V .. +V] += bias[0..V]
extern "C" __global__ void add_row_inplace_f32(float* __restrict__ logits,
                                               const float* __restrict__ bias,
                                               int n, long row_off) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[row_off + i] += bias[i];
}

// LATENCY-HIDING ARC (owner angles, 2026-07-10): L2 prefetch of a byte range — issued 1-2
// kernels ahead of the consumer (fa's KV stream), so the latency-bound consumer finds its
// lines L2-warm. Pure scheduling: touches no values, changes no numeric config.
extern "C" __global__ void prefetch_l2_bytes(const unsigned char* __restrict__ p, long n) {
    long i = (long)(blockIdx.x * blockDim.x + threadIdx.x) * 128;
    if (i < n) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
    }
}
