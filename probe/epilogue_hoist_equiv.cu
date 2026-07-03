// Probe: prove the EPILOGUE-HOIST on qmatvec_gemm.cu kernel1 is numerically safe BEFORE editing the
// kernel. De-risk pattern mirrors probe/int8_swizzle_aload.cu: prove equivalence STANDALONE first.
//
// The production epilogue (qmatvec_gemm.cu:653-657) runs PER K-block g, on the facc dep chain:
//     float da = sAd[cur][nn];                                            // per-token act scale, INVARIANT in g
//     facc += sWd[cur][rr]*da*(float)dacc[ci] + sWb[cur][rr]*da*sAsum[cur][nn];
// i.e. for one output element o,t:
//     OLD:  y = sum_g ( Wd_g * da * sumi_g  +  Wb_g * da * Asum_g )       // 2 f32 muls + da reuse per g
//
// llama folds the scale at write-out: da is hoisted, ONE scale at the end:
//     STEP1 (da-hoist):  raw = sum_g ( Wd_g*sumi_g + Wb_g*Asum_g );  y = da * raw
//     STEP2 (reg-hoist): bit-identical to STEP1 (Wd_g/Wb_g just read from a reg, same f32 ops/order)
//     STEP3 (split):     mma_acc = sum_g Wd_g*sumi_g ; bias_acc = sum_g Wb_g*Asum_g ; y = da*(mma_acc+bias_acc)
//
// f32 is non-associative: da*(a+b) != da*a+da*b bitwise, and grouping the two terms separately (STEP3)
// regroups the sum. So none of these are bit-identical to OLD. Contract: max rel err < 1e-3 AND argmax
// over outputs (the token's predicted logit row) is PRESERVED. This probe asserts both over many
// randomized realistic (out_f x in_f) GEMM problems.
//
// Build: nvcc -arch=compute_120a -code=sm_120a probe/epilogue_hoist_equiv.cu -o probe/epilogue_hoist_equiv
//        ./probe/epilogue_hoist_equiv
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// One GEMM "problem": OUT rows x NBLK K-blocks of 32, 1 token column (argmax is over the OUT rows for
// that token — exactly the LM-head logit argmax the run-gen MATCH gate checks). Each K-block contributes
// (sumi_g, Asum_g) ints and (Wd_g, Wb_g) per-row f32 scales; da is one per-token f32 scale.
#define OUT  256      // output rows (vocab/feature tile) — argmax dimension
#define NBLK 128      // K-blocks (in_f=4096) — the dep-chain length the hoist targets

// realistic value generators (match decode_q4_k_s / activation-quant ranges):
//   Wd_g = d*sc  : product of two small f16 super-scales, ~[1e-4, 5e-2], strictly >0
//   Wb_g = -dmin*mn : negative bias, ~[-5e-2, 0]
//   da          : per-token act dequant scale, ~[5e-4, 5e-3] (q8 act: amax/127)
//   sumi_g      : dp4a(weight_nibble[0..15-ish signed], act_int8[-127..127]) over 32 lanes -> ~[-4e4, 4e4]
//   Asum_g      : sum of 32 int8 activations -> ~[-4000, 4000]
__device__ __host__ inline float frand(uint32_t& s){ s = s*1664525u + 1013904223u; return (s>>8)*(1.0f/16777216.0f); }

__global__ void run_problem(const float* Wd, const float* Wb, const float* da_in,
                            const int* sumi, const int* Asum,
                            float* y_old, float* y_s1, float* y_s3) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;   // one output row per thread
    if (o >= OUT) return;
    float da = *da_in;

    // OLD: production epilogue, da multiplied every block, single accumulator.
    float facc_old = 0.0f;
    for (int g = 0; g < NBLK; g++) {
        float Wd_g = Wd[g*OUT + o], Wb_g = Wb[g*OUT + o];
        facc_old += Wd_g * da * (float)sumi[g*OUT + o] + Wb_g * da * (float)Asum[g];
    }
    y_old[o] = facc_old;

    // STEP1: hoist da out of the K-loop (== STEP2 bit-for-bit, scales just live in regs).
    float facc_raw = 0.0f;
    for (int g = 0; g < NBLK; g++) {
        float Wd_g = Wd[g*OUT + o], Wb_g = Wb[g*OUT + o];
        facc_raw += Wd_g * (float)sumi[g*OUT + o] + Wb_g * (float)Asum[g];
    }
    y_s1[o] = da * facc_raw;

    // STEP3: split the bias term off the mma-dependent term (two accumulators, combine at write-out).
    float mma_acc = 0.0f, bias_acc = 0.0f;
    for (int g = 0; g < NBLK; g++) {
        float Wd_g = Wd[g*OUT + o], Wb_g = Wb[g*OUT + o];
        mma_acc  += Wd_g * (float)sumi[g*OUT + o];
        bias_acc += Wb_g * (float)Asum[g];
    }
    y_s3[o] = da * (mma_acc + bias_acc);
}

int main() {
    const int NPROB = 4096;   // many randomized problems -> stress argmax flips
    float *Wd, *Wb, *da, *yo, *y1, *y3; int *sumi, *Asum;
    cudaMallocManaged(&Wd,   sizeof(float)*NBLK*OUT);
    cudaMallocManaged(&Wb,   sizeof(float)*NBLK*OUT);
    cudaMallocManaged(&sumi, sizeof(int)  *NBLK*OUT);
    cudaMallocManaged(&Asum, sizeof(int)  *NBLK);
    cudaMallocManaged(&da,   sizeof(float));
    cudaMallocManaged(&yo,   sizeof(float)*OUT);
    cudaMallocManaged(&y1,   sizeof(float)*OUT);
    cudaMallocManaged(&y3,   sizeof(float)*OUT);

    uint32_t s = 0xC0FFEEu;
    double worst_rel_s1 = 0, worst_rel_s3 = 0;        // norm-rel (the real contract)
    double worst_elem_s1 = 0, worst_elem_s3 = 0;      // per-elem rel EXCLUDING near-zero refs (|ref|<1% of vec max)
    int argmax_flips_s1 = 0, argmax_flips_s3 = 0;
    int near_ties = 0;  // problems where top-2 logits are within 1e-3 rel (argmax inherently fragile)

    for (int p = 0; p < NPROB; p++) {
        *da = 5e-4f + frand(s) * 4.5e-3f;
        for (int g = 0; g < NBLK; g++) {
            // Asum: sum of 32 int8 in [-127,127] -> draw centered, realistic spread
            Asum[g] = (int)((frand(s) - 0.5f) * 2.0f * 4000.0f);
            for (int o = 0; o < OUT; o++) {
                Wd[g*OUT + o] = 1e-4f + frand(s) * 5e-2f;           // d*sc > 0
                Wb[g*OUT + o] = -(frand(s) * 5e-2f);                // -dmin*mn <= 0
                sumi[g*OUT + o] = (int)((frand(s) - 0.5f) * 2.0f * 4e4f);
            }
        }
        run_problem<<<(OUT+127)/128, 128>>>(Wd, Wb, da, sumi, Asum, yo, y1, y3);
        cudaDeviceSynchronize();

        // rel error, two ways:
        //  (a) per-element |new-ref|/|ref| — but with sumi symmetric about 0, a single ref can land near
        //      zero (catastrophic cancellation), so this OVERSTATES error for near-zero outputs. This is a
        //      property of THIS synthetic generator, NOT of the LM head (real logits are not zero-clustered),
        //      and NOT of the kernel_check gate (which is a tolerance over a whole reference tensor).
        //  (b) norm-rel over the output VECTOR: ||y_new - y_ref|| / ||y_ref|| — the metric that mirrors a
        //      tensor-wide rel check and is robust to a few near-zero elements. This is the real contract.
        double num1 = 0, num3 = 0, refnorm2 = 0;
        for (int o = 0; o < OUT; o++) {
            float ref = yo[o];
            double d1 = (double)y1[o] - ref, d3 = (double)y3[o] - ref;
            num1 += d1*d1; num3 += d3*d3; refnorm2 += (double)ref*ref;
        }
        double den = sqrt(fmax(refnorm2, 1e-30));
        double nr1 = sqrt(num1)/den, nr3 = sqrt(num3)/den;
        if (nr1 > worst_rel_s1) worst_rel_s1 = nr1;
        if (nr3 > worst_rel_s3) worst_rel_s3 = nr3;
        // per-elem rel, but only for outputs that are NOT near-zero (exclude cancellation noise):
        // threshold = 1% of this vector's max |logit| — a near-zero logit is never the argmax anyway.
        float vmax = 0.0f; for (int o = 0; o < OUT; o++) vmax = fmaxf(vmax, fabsf(yo[o]));
        float zthr = 0.01f * vmax;
        for (int o = 0; o < OUT; o++) {
            float ref = yo[o];
            if (fabsf(ref) < zthr) continue;          // skip cancellation-dominated elements
            double e1 = fabs((double)y1[o]-ref)/fabsf(ref), e3 = fabs((double)y3[o]-ref)/fabsf(ref);
            if (e1 > worst_elem_s1) worst_elem_s1 = e1;
            if (e3 > worst_elem_s3) worst_elem_s3 = e3;
        }
        // argmax over OUT (the LM-head logit row): does the hoist flip the predicted token?
        int am_o = 0, am_1 = 0, am_3 = 0;
        for (int o = 1; o < OUT; o++) {
            if (yo[o] > yo[am_o]) am_o = o;
            if (y1[o] > y1[am_1]) am_1 = o;
            if (y3[o] > y3[am_3]) am_3 = o;
        }
        if (am_1 != am_o) argmax_flips_s1++;
        if (am_3 != am_o) argmax_flips_s3++;
        // is the OLD argmax a near-tie? (top-1 vs top-2 within rel 1e-3 -> any rounding could flip it,
        // which would be a property of the problem, not the hoist)
        float top1 = yo[am_o], top2 = -1e30f;
        for (int o = 0; o < OUT; o++) if (o != am_o && yo[o] > top2) top2 = yo[o];
        if (fabsf(top1 - top2) <= 1e-3f * fmaxf(fabsf(top1), 1e-6f)) near_ties++;
    }

    printf("=== EPILOGUE-HOIST EQUIVALENCE PROBE ===\n");
    printf("problems=%d  out_rows=%d  k_blocks=%d\n", NPROB, OUT, NBLK);
    printf("STEP1 (da-hoist / == STEP2 reg-hoist): max_rel=%.3e  argmax_flips=%d/%d\n",
           worst_rel_s1, argmax_flips_s1, NPROB);
    printf("STEP3 (bias/mma split)               : max_rel=%.3e  argmax_flips=%d/%d\n",
           worst_rel_s3, argmax_flips_s3, NPROB);
    printf("near-tie problems (top1~top2 rel<=1e-3, argmax inherently fragile): %d/%d\n", near_ties, NPROB);

    bool ok1 = worst_rel_s1 < 1e-3 && argmax_flips_s1 == 0;
    bool ok3 = worst_rel_s3 < 1e-3 && argmax_flips_s3 == 0;
    printf("STEP1/STEP2: %s\n", ok1 ? "SAFE (rel<1e-3 + argmax-stable)" : "UNSAFE");
    printf("STEP3:       %s\n", ok3 ? "SAFE (rel<1e-3 + argmax-stable)" : "UNSAFE");
    return (ok1 && ok3) ? 0 : 1;
}
