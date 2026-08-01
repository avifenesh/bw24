# Hy3 on Hopper (sm_90a) — first light, 2026-08-01

Box: Mumbai darklanes-bench H100 80GB (shared; every GPU run under `flock /tmp/gpu-h100.lock`).
Build: lane/hy3-hopper tree at `/opt/dlami/nvme/hy3-hopper/memra`, `MEMRA_CUDA_ARCH
auto-detected 90a`, release build 3m57s (`logs/build.log`). Includes the one-line
bw24-overlay-format fix (see gaps.md G1) — without it the artifact does not load as an overlay.

## Gate 1: kernel-check

`logs/kernel-check.log` — **ALL GREEN: kernels match CPU reference.** 198 OK lines,
zero failures (MMQ int8 + RP bit-identity, FA decode/prefill views, KV quant round-trips,
MoE router tie-handling, MoE cache-HIT bit-identity, async-prefetch victim protection).

## Gate 2: Hy3 load + short generation (code-path first light)

Command (verbatim, `logs/firstlight-rungen.log`):

    MEMRA_CHAT=1 MEMRA_NGEN=32 MEMRA_PROMPT="Explain in two sentences why the sky is blue." \
      ./target/release/run-gen /opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime

Result: **PASS.**

- `loaded Hy3 from memra repack dir (80 trunk layers; optional MTP skipped)` — overlay
  manifest + safetensors fallback + Hy3 chat template all resolved; `[q8rp] split-plane
  decode mirrors built: 324 tensors`.
- **Argmax gate: `verify-prefill argmax=628  decode argmax=628  logit maxdiff=1.664e0  MATCH`.**
- `generated 32 tokens in 15.270s = 2.10 tok/s (ST greedy decode)` — single cold run,
  spill-dominated (see below), N=1, labeled as such.
- Output text coherent (Rayleigh-scattering answer).
- `MoE cache DECODE-WINDOW: 6572 slots | hits=35742 misses=24930 (hit-rate=58.9%) |
  staged 64.70 GB H2D (1928.3 MB/token)`; `storage DECODE-WINDOW: 0.57 GB physical reads`.
- HBM at steady state: ~70.6 GiB used of 80.

## What this run proves / does not prove

Proves (code paths on sm_90a): mixed-tier expert overlay load (IQ3_S/IQ4_XS/Q2_K/Q3_K/
Q4_K/Q8_0 metadata-aware dispatch), pruned-expert masking, SLRU expert cache + H2D staging,
BF16 safetensors non-expert fallback, GQA attention, greedy decode exactness
(prefill-vs-decode argmax MATCH).

Does NOT prove: performance. One 80GB H100 cannot hold the 73.1 GiB expert payload plus
non-expert weights — decode is expert-staging-bound (1.93 GB/token H2D at 58.9% hit rate).
This is a **code-path gate, not a perf baseline** for the spike: the Aug-2 arrangement is
PP-2 (two H100s per replica), where the expert bank fits fully resident and staging
disappears. The single-H100 numbers here are the *spill-regime floor*, useful for the
$/Mtok table only as the degenerate 1-GPU point.

Note for the record: `logit maxdiff=1.664e0` between the prefill-verify and decode paths is
large in absolute terms (argmax still MATCH). Same-class values were normal on the 5090
spill lane for this mixed-tier artifact; flagged for the spike's exactness battery to keep
an eye on under PP-2 (where the resident path replaces staged dispatch).

## MTP

run-gen's trunk load reports `optional MTP skipped` (greedy decode does not use it), but
the follow-up run-spec probe loaded the full stack — `loaded memra-repack (81 layers,
nextn=1)` — so the MTP block loads and forwards on sm_90a. Spec exactness PASSES
(self-consistency identical to greedy) but draft acceptance is 0/62 = 0.0%, i.e. spec
currently gives slowdown, not speedup, on Hopper. Full evidence + required cross-check:
gaps.md G2 (`logs/spec-k2.log`).
