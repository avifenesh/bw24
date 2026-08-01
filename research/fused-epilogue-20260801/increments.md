# Fused act-epilogue arc — lane 3 (2026-08-01, H100 darklanes-8x GPU 3)

Lever: in the MoE prefill pairs/MMA arms, the down-projection input chain was
gate/up GEMMs -> moe_pairs_{silu,gelu}_mul (f32 elementwise, writes act) ->
mmq_iq_quantize_act (re-reads act, writes block_q8_1_mmq D4 scratch) — two full passes over
the [n_pairs x n_ff] f32 activation. `fused_act_quant_mmq_q8_1_d4_kernel`
(cu/mmq_iq_experts.cu) computes the activation in registers and writes ONLY the quantized
scratch: read gate+up, write scratch — the f32 act buffer never exists on the fused path.

Verified before wiring: in both MMA arms the f32 `act` feeds ONLY `mmq_iq_quantize_act`
(qwen hybrid_forward.rs ~3206; gemma ~4694 mma branch). The non-MMA branches
(quantize_q8_1 consumers) and the f16g door keep the two-pass chain untouched.

Default ON; `MEMRA_MOE_FUSE_ACTQ=0` is the rollback/A-B seam.

## Byte-identity verdict: BYTE-IDENTICAL

- Contract: activation expressions are moe_pairs_{silu,gelu}_mul (qmatvec.cu) VERBATIM;
  quantize fold is quantize_mmq_q8_1_d4_kernel's exact per-thread-amax + 8-lane shfl_xor
  pattern (same fold order); zero-padding beyond in_f preserved (the gemma padded-k down
  GEMM contract rides those zero bytes).
- kernel-check `iq fused act+quant` (new, 3 cases: silu aligned 512, silu padded 768,
  gelu ragged 704): byte_mismatch=0 on all — ALL GREEN with both models' GGUFs
  (kernel-check-q35.log, kernel-check-g26.log).
- Production shapes re-affirmed in the microbench: byte_mismatch=0 at [16384x512] silu and
  [13888x704] gelu (fused-actq-microbench.log).
- run-gen argmax MATCH both models with fused on; g26 logit maxdiff 5.527e0 — the same
  constant the expert-kernel arc reports for byte-identical data movement
  (g26-argmax-gate.log, q35-argmax-gate.log).

## Kernel-level A/B (fused_actq_bench, N=200 reps/arm, single process, GPU 3)

| shape | two-pass | fused | speedup | saved/layer-call |
|---|---|---|---|---|
| q35-silu [16384 x 512] | 72.9 us (1971 GB/s) | 29.3 us (2613 GB/s) | 2.49x | 43.6 us |
| g26-gelu [13888 x 704] | 75.2 us (2293 GB/s) | 38.0 us (2479 GB/s) | 1.98x | 37.2 us |

## e2e prefill A/B (interleaved pairs, N=3 each, MEMRA_NGEN=4, GPU 3; lane 2 active on its own GPU)

g26 pp1736 (depth-prompt-1736-ids):

| rep | fused | twopass |
|---|---|---|
| 1 | 10149.1 | 10003.6 |
| 2 | 10119.3 | 10035.9 |
| 3 | 10117.5 | 10030.2 |

median 10119.3 vs 10030.2 = **+0.89%** (all three fused runs above all three twopass runs).

q35 board-2048 (pp2048):

| rep | fused | twopass |
|---|---|---|
| 1 | 5427.0 | 5459.7 |
| 2 | 5478.5 | 5458.4 |
| 3 | 5469.1 | 5452.9 |

median 5469.1 vs 5458.4 = **+0.20%** — FLAT within run noise (pairwise -0.6%/+0.4%/+0.3%).
Consistent with the arithmetic: 43.6 us saved x 32 MoE layers ~= 1.4 ms on a 375 ms prime
(+0.37% ceiling); q35's prime wall is the gate/up GEMM shape, not the epilogue. The g26
delta also checks out: measured prime shrank ~1.5 ms ~= 37.2 us x its MoE layer count.

All 12 A/B runs argmax MATCH (per-run logs in this directory).

## Stretch: stacked gate+up (one mmq_iq_experts call, one Y-gather) — feasibility: NEGATIVE on paper

The proposal: gate and up read the SAME gathered activation (z_scr); fold both projections
into one call so Y is gathered once.

1. **Doubled-grid form (grid.x = 2*nty, proj from blockIdx) shares nothing.** The Y gather
   is per-(CTA, kb) into that CTA's smem; CTAs cannot share tiles. Zero traffic saved —
   refuted by inspection.
2. **The real form is a per-CTA proj loop** (stage W_gate(kb) and W_up(kb) against one
   gathered Y(kb), two accumulator sets). That halves Y-gather issue traffic but doubles
   accumulators 64 -> 128 regs/thread at MMQ_X=128 — round-45 ncu has this kernel at
   Block Limit Registers = 1 (12.5% occupancy) already; +64 regs = spill or occupancy loss,
   both measured REFUTED classes on this kernel (launch_bounds minblocks=2: -4.7%).
   At MMQ_X=64 the accumulators fit (32 -> 64), but the 64-token tile itself measured -9.1%
   (round 45) and is refuted on paper for q35's ~65-pair average groups (half the groups
   take 2 passes = 2x W dequant, round-46 addendum).
3. **The cost class it attacks is already measured near-flat.** Round-46 inc4 removed ~half
   of all q35 gate/up gather traffic (clamped tail columns of half-empty tiles) for +0.3%;
   post-inc2 the stall profile is `wait` (fixed-latency dep chains), long_scoreboard
   collapsed 123.5k -> 9.3k. The z_scr scratch is L2-resident (~4.7 MB vs 50 MB L2), so the
   duplicate gate/up gather is L2 reads + issue slots, not DRAM. Upper bound by traffic
   arithmetic: ~36 us/layer saved on q35 ~= same magnitude as the epilogue win, but against
   a register wall instead of for free.

Verdict: do not build. The measured fix class for the gate/up shape is expert-BATCHED GEMM
(CUTLASS grouped int8 — round-46 addendum), a separate arc.

## Files

- cu/mmq_iq_experts.cu: fused_act_quant_mmq_q8_1_d4_kernel<ACT> + memra_mmq_iq_fused_act_quant
- src/mmq_ffi.rs: FFI decl + Engine::mmq_iq_fused_act_quant
- src/lib.rs: moe_fuse_actq_on() (MEMRA_MOE_FUSE_ACTQ, default on)
- src/hybrid_forward.rs: qwen + gemma MMA arms wired (two-pass under the seam)
- src/bin/kernel_check.rs: `iq fused act+quant` byte-identity entries (x3)
- src/bin/fused_actq_bench.rs: kernel-level A/B at production shapes
