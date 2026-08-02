# h100-v064-verify: pre-pin H100 verification for v0.64.0 (2026-08-02)

Lane `lane/h100-v064-verify` (from `restructure/public-split`, 1576d8b3 — post q35 board
re-pair merge). Box: Mumbai H100 80GB HBM3 (darklanes-bench, 13.203.16.133), CUDA 13.1,
`MEMRA_CUDA_ARCH=90a`. Model: `/home/ubuntu/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (q35).
Every GPU run under `flock /tmp/gpu-h100.lock`; GPU exclusively ours the whole session
(busy_procs=0 in every jsonl row, temp 38C flat — single thermal window 00:47-01:05Z).
Tree rsync'd over `~/memra` (target/.git excluded), rebuilt release on-box.

Since the last H100 verification (v0.63.0) the tree gained: iq_fast default ON
(arch-global), the gate1 fraction rule (>=4/6), the prime-gate battery + run-gen gate
line, AUTO-KQUANT mode 3 (sm_120a-only), and the q35 board re-pair (5090-only numbers).
This lane verifies each on Hopper before the 8xH100 box pins v0.64.0.

## 1. validate-h100.sh --quick on q35: ALL GATES GREEN (exit 0)

`validate-h100-quick-q35.log`. kernel-check ALL GREEN; decode-batch config B=8 GREEN
under the NEW gate1 fraction rule — the rule is visibly exercised, receipts in
`decode-batch-config-q35.log`:

```
gate1 seed 0 step 6: token diverged — accepted cross-config drift (WARN)
gate1 seed 2 step 15: token diverged — accepted cross-config drift (WARN)
gate1 seed 5 step 8: token diverged — accepted cross-config drift (WARN)
gate1 early draws (step < 3): 0/6 (FAIL threshold >= 4; plumbing class = every draw)
gate1 (B=1 argmax agreement vs decode_step_h, 16 steps, 6 seed(s)): PASS
gate2 (B=8 vs isolated batched-B=1, bit-checked, 16 steps): PASS
gate3 (device sampling: greedy==host-argmax + sampled B=8 vs isolated + lean-logits identity): PASS
ALL GREEN: decode_step_batch exactness battery
```

decode-batch strict (equalized composition) GREEN, decode-dc PASS, graph-decode PASS,
graph-session ALL GREEN. Policy tests: `33 passed; 0 failed; 1 ignored`
(`policy-tests.log`).

## 2. prime-gate + run-gen gate lines on q35 (H100): GREEN

- `prime-gate-q35.log` (mixed 8-prompt set, exit 0):
  `prime-gate SUMMARY: 8 prompts | MATCH=7 FLIP-NEARTIE=1 STRUCTURED=0 det_fails=0 |
  worst maxdiff=1.8037e0 min flip margin=0.0774` — GREEN with the one reported near-tie
  flip (the pp512 probe prompt, the documented cross-config drift class; see §5).
- `run-gen-gate-q35.log` (default prompt, NGEN=8, exit 0):
  `prefill argmax=1178  decode argmax=1178  logit maxdiff=5.268e-1  MATCH`.

## 3. iq_fast H100 impact on q35: BIT-IDENTICAL, perf FLAT (no re-cell needed)

Arms interleaved per rep (naked = iq_fast ON default; iqfast0 = `MEMRA_IQ_FAST=0`
rollback = the pre-flip world). Reps 1-3 naked-first, reps 4-7 iqfast0-first (order-bias
control). NGEN=128, run-gen argmax gate per run, token-stream sha256 per run. Raw:
`iqfast-sweep.jsonl`, `q35-*-rep*.log`, console `iqfast-sweep-console.log`, scripts
`run-iqfast-h100.sh` + `run-iqfast-ext.sh`.

| cell | metric | naked (N med [range]) | iqfast0 (N med [range]) | delta med | streams |
|---|---|---|---|---|---|
| board-2048 | prefill tok/s | N=7 8455.5 [8296.4,8598.0] | N=7 8542.2 [8129.5,8571.3] | -1.0% | BIT-IDENTICAL 7/7 vs 7/7 (sha b2cef91071c6fa0e) |
| board-2048 | decode tok/s | N=7 191.9 [187.7,193.1] | N=7 188.3 [187.8,193.4] | +1.9% | (same) |
| p2-code-medium | prefill tok/s | N=3 7486.5 [7342.3,7542.0] | N=3 7363.6 [7293.4,7383.2] | +1.7% | BIT-IDENTICAL 3/3 vs 3/3 (sha e8276141a07cc5d7) |
| p2-code-medium | decode tok/s | N=3 186.4 [182.7,188.2] | N=3 183.2 [182.4,188.1] | +1.7% | (same) |

- Token streams: bit-identical across arms on every completed run (sha256 of the
  generated-token line). Argmax gate lines identical both arms, zero MISMATCH lines in
  any completed run.
- Every median delta < 2% with heavily overlapping ranges, and signs DISAGREE across
  cells/metrics — noise, not an arm effect. The decode distribution is BIMODAL at the
  box level (~188 vs ~193 modes); BOTH arms visit both modes (naked lows 187.7/187.8/
  188.1; iqfast0 highs 193.25/193.4), so the rep1-3 medians-split was mode luck, killed
  by the order-flipped reps 4-7.
- Static confirmation of WHY it's a no-op on q35: the seam admits only NON-EXPERT
  IQ4_XS matmuls, and q35 carries IQ4_XS exclusively in expert banks
  (`research/kat-anomaly-20260802/ctrl-q35-tensor-mix.txt`: trunk = Q8_0/Q6_K only).
- VERDICT: the published H100 q35 row (217/1.01x, measured pre-flip) needs NO re-cell
  note; iq_fast needs NO arch gate for q35-class models. (KAT-class IQ4_XS-trunk
  models were not measured on H100 — no supported IQ4_XS-trunk artifact is staged on
  this box; that cell stays open, 5090 evidence only.)
- Window note: a v0.63.0 control build briefly ran on the host CPU during ext-rep4
  (killed 00:56:55Z, load peak 1.33/16 vCPU); rep4 values sit in the normal modes both
  arms (187.93/193.1) — no contamination signature, rep retained.

## 4. AUTO-KQUANT (mode 3) Hopper no-op: PROVED (mode 1 naked)

`auto-kquant-mode-probe.log` — temp uncommitted probe bin (deleted from the box after),
built in the same 90a build env, printing the compile-time cfg + the resolved mode:

```
== naked ==            built_cuda_arch=90a  memra_hopper_mma_cfg=true  MEMRA_MOE_F16G_env=None      moe_f16g_mode_resolved=1
== MEMRA_MOE_F16G=0 == built_cuda_arch=90a  memra_hopper_mma_cfg=true  MEMRA_MOE_F16G_env=Some("0") moe_f16g_mode_resolved=0
== MEMRA_MOE_F16G=2 == built_cuda_arch=90a  memra_hopper_mma_cfg=true  MEMRA_MOE_F16G_env=Some("2") moe_f16g_mode_resolved=2
```

Naked on Hopper resolves to mode 1 (the round-49 default) — AUTO-KQUANT mode 3 is
unreachable (`Err(_) => if cfg!(memra_hopper_mma) { 1 } else { 3 }`, lib.rs); the env
door still works both directions. Hopper behavior unchanged, as designed.

## 5. FINDING (report, not decided here): q35 + pp512 prompt trips run-gen's hard
##    argmax assert on H100 — arm-independent, deterministic

Every pp512-cell run, BOTH arms, all 3 reps each, identical to the digit:

```
prefill argmax=365  decode argmax=198  logit maxdiff=5.739e-1  MISMATCH
[gate] prefill: l[365]=14.3006 l[198]=14.1855 | decode: l[365]=13.9139 l[198]=13.9913
assertion `left == right` failed: decode-step diverges from prefill — cache threading bug
```

- This is the DOCUMENTED q35 pp512 near-tie token pair (365 vs 198 "\n" —
  research/residency-cap-20260802 §4, run_gen.rs gap-#46 comment). Margins are the
  near-tie class: prefill 0.115, decode 0.077 on logits ~14. prime-gate's calibrated
  verdict on the SAME prompt (its @pp512 line, §2) classifies it FLIP-NEARTIE
  (min flip margin 0.0774 = this prompt) — accepted, reported, non-fatal.
- On the 5090 the same near-tie flips between batched-prime and tokenwise while
  forward_last vs tokenwise agree (run-gen gate green — the residency-cap receipts);
  on H100 the dice land forward_last=365 / tokenwise=198, and run-gen's hard
  `assert_eq!` has no near-tie allowance, so the process dies. NOT an iq_fast effect
  (iqfast0 arm = pre-flip dispatch, fails identically). NOT non-determinism (maxdiff
  identical across 6 runs).
- v0.63.0 control on the same box/model/prompt: see below.
- Impact: any H100 pipeline that runs `run-gen` on q35 with the pp512 prompt dies at
  the gate. validate-h100 --quick, prime-gate, run-gen default-prompt, board-2048 and
  p2 cells are all green — the assert is prompt-specific.
- Owner options (not decided here): give the run-gen hard assert the same calibrated
  near-tie tolerance prime_gate_verdict already has; or ledger the prompt as a known
  H100 near-tie cell. gate1's fraction rule was built for exactly this class.

### v0.63.0 control (same box, same model, same prompt): PRE-EXISTING

v0.63.0 (tag, 04ff30a3) built fresh in `~/memra-v063` (90a, same nvcc), x2
(`v063-pp512-rep{1,2}.log`):

```
prefill argmax=365  decode argmax=198  logit maxdiff=5.739e-1  MISMATCH   (exit 101, both reps)
```

Identical argmax pair AND identical maxdiff to the digit — the H100 dice have been on
the split side of this near-tie since at least v0.63.0. NOT a v0.63.0..v0.64.0
regression; the cell had simply never been run on H100 (no prior battery line covers
run-gen q35-pp512 on this box). Anchor from the same control build, single run
(`v063-board2048.log`): board-2048 prefill 7862.0 / decode 187.16 tok/s — HEAD naked
medians 8455.5 / 191.9 (N=7), i.e. v0.63.0 -> v0.64.0 moved q35 H100 board prefill
+7.5% (single-run denominator; anchor only, not a board claim).

## Verdict

GO for pinning v0.64.0 on the 8xH100 box:

1. validate-h100.sh --quick (q35): ALL GATES GREEN under the new gate1 fraction rule.
2. prime-gate (mixed set) GREEN (7 MATCH / 1 reported near-tie / 0 structured /
   0 det-fails); run-gen gate MATCH.
3. iq_fast default ON is a proven no-op on H100 q35: token streams bit-identical
   (naked vs MEMRA_IQ_FAST=0, 10 paired runs across 2 shapes), perf deltas <2% with
   overlapping ranges and disagreeing signs (noise; box decode bimodality visited by
   both arms). The published 217/1.01x row stands; no re-cell note, no arch gate.
4. AUTO-KQUANT mode 3 is compile-gated off Hopper: naked resolves to mode 1
   (probe receipt), env door intact.
5. The one red anywhere — run-gen's hard assert on the q35 pp512 near-tie prompt —
   is pre-existing at v0.63.0 (identical failure, receipts), prompt-specific,
   arm-independent, and already classified near-tie by prime-gate's calibrated
   verdict. It does not block the pin; it needs an owner call (assert tolerance vs
   ledger entry) — reported, not decided here.
