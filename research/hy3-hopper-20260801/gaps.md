# Hy3-on-Hopper gap list — ranked by must-fix-for-the-Aug-2 spike (2026-08-01)

Classification per finding: coverage gap (sm_90a code path never exercised) / bug /
artifact issue / operational note. Status: FIXED-IN-LANE, OPEN-SPIKE-INPUT, or NOTE.

## G1 — overlay format string rename broke artifact loading [bug, FIXED-IN-LANE]

`crates/memra-gguf/src/source.rs` post-rename accepted only `memra-expert-overlay-v{1,2}`;
the published Hy3 layer103.5 runtime manifest (sha-pinned `b8bdd684…`) carries
`bw24-expert-overlay-v2` on disk. Without the fix the manifest is treated as a complete
repack (`fallback = None`) and load dies on missing non-expert tensors. One-line accept-both
fix, commit `096f212d`; verified end-to-end by first-light PASS. **Must be in the Aug-2
box tree** — any tree cut from restructure/public-split without this cannot load the SKU
artifact at all.

Verbatim pre-fix failure (captured with the box main-tree rename-era binary, run read-only
under flock, `logs/prefix-fail-g1.log`):

    thread 'main' (929658) panicked at crates/memra-engine/src/model.rs:641:32:
    missing embed token_embd.weight

## G2 — MTP spec decode on sm_90a: WORKS (exact, accepting, 1.16x) — acceptance-rate
sweep still owed [coverage gap CLOSED for correctness, rate CLOSED by
`research/hy3-spec-20260802/` (K=1..8, N=3, board-d1736); SPIKE-INPUT]

CLOSURE NOTE (2026-08-01, lane/hy3-spec-sweep): the sweep landed and REFUTES the 1.16x
below — it was a cold-denominator artifact (plain oracle 0.61 tok/s first-ever-touch vs
3.73 warm; same config re-run warm is 0.56x). The 23.8% acceptance replicated bit-exactly
on the merged tip build. At the 1-GPU spill floor spec is a slowdown at every K (best
K=1 = 0.84x median, N=3); accepted count is exactly 10 at all Ks (the nextn=1 head never
chains). Full verdict + PP-2 caveat: `research/hy3-spec-20260802/SUMMARY.md`. The 5090
same-artifact acceptance reference remains owed.

Two probes today, both `MEMRA_SPEC_K=2 run-spec <runtime-dir>`:

1. Default prompt (single token `[55]`, degenerate content — `logs/spec-k2.log`):
   `acceptance: 0/62 = 0.0%  self-consistency: PASS (identical to generate)` plus the
   built-in `WARNING: acceptance == 0 ... MTP head is likely forwarded wrong`. This
   looked like a Hopper draft-path bug for half an hour; it is not — see (2).
2. Real chat-templated prompt (the 25-id first-light prompt —
   `logs/spec-k2-realprompt.log`, verbatim):

       loaded memra-repack (81 layers, nextn=1)
       [generate]   31 tok in 50.754s = 0.61 tok/s (gen-only; this run's prime 30.423s)
       [generate_spec K=2] 32 tok in 43.691s = 0.71 tok/s (1.16x vs generate; this run's prime 15.344s)
         acceptance: 10/42 = 23.8%   self-consistency: PASS (identical to generate)
       === SELF-CONSISTENCY PASS ===

Verdict: the MTP block loads (nextn=1), forwards, drafts get accepted, exactness holds
(spec output token-identical to greedy), and spec already nets 1.16x even in the
single-GPU spill regime. What remains for the spike: a real acceptance-rate sweep
(K=1..8 battery, board-class prompts, and a 5090 same-artifact reference for the
acceptance denominator) before any spec tok/s lands in the $/Mtok table — 23.8% on one
short probe is a floor observation, not a rate claim. All numbers above are single runs
and labeled as such. Operational note for the spike battery: never use run-spec's
default prompt for acceptance measurements on this model.

## G3 — single-H100 decode is expert-staging-bound; PP-2 residency is the actual spike
mode [coverage gap, OPEN-SPIKE-INPUT]

One 80GB card cannot hold the 96.4 GiB logical model: SLRU ran 6572 slots, 58.9% hit rate,
1.93 GB/token H2D. The spike's PP-2 replica (2x80GB) fits the bank fully resident — a
regime this box cannot exercise with one GPU. The resident MoE path is board-proven on
sm_90a for the 35B (kernel-check MoE gates green here), but Hy3's *mixed-tier* bank
(IQ3_S/IQ4_XS/Q2_K/Q3_K/Q4_K/Q8_0) on the *resident/fused* arms is uniform-only by
contract (CLAUDE.md: resident slab, pairs, dev, grouped-decode fused kernels are
uniform-only) — mixed layers stay on staged/SLRU dispatch even when resident. Expect
PP-2 decode to be metadata-aware-dispatch-bound, not fused-kernel-bound; do not promise
fused-resident numbers in the $/Mtok table.

## G4 — `logit maxdiff=1.664e0` prefill-verify vs decode [NOTE, watch under PP-2]

Argmax MATCH (the gate), but the absolute logit delta between the batched-prefill verify
pass and tokenwise decode is large-ish (1.664e0 short prompt; 1.888e0 on the 1818-token
depth prompt, identical across all three baseline runs). Same-class deltas were seen on
the 5090 spill lane for this mixed-tier artifact. Keep the argmax + K=1..8 self-consistency battery mandatory
per PP-2 increment; if maxdiff grows past this class under the transport seam, stop and
bisect before publishing numbers.

## G5 — hf batch `--include` silently skipped one shard [operational note]

First `hf download tencent/Hy3 --include <26 patterns>` fetched 25/26 files
(model-00006 missing, no error in the log); a single-pattern refetch succeeded.
Staging scripts for the Aug-2 box must count files and verify bytes/hashes after
every bulk fetch (the assemble script here does: 237 shards, payload bytes exact,
99/99 shard names resolve).

## G6 — Mumbai root EBS at 94% [operational note]

Weights must never land on `/` (19G free). Everything staged to `/opt/dlami/nvme`
(3.5T). No deletions were needed; nothing was removed. The Aug-2 p5 box: same rule,
stage to instance NVMe, record staged manifest hash (done here), and never report
EBS fault throughput as spill speed.

## Explicitly NOT gaps (checked)

- sm_90a kernel coverage for the artifact's quant mix: kernel-check ALL GREEN (198 gates)
  on this exact lane build, and the real-model run dispatched mixed tiers with no
  unsupported-arm error. The gemma-class "unsupported dispatch arm" failure did not
  materialize on the trunk decode path.
- `manifest_qtype` covers every tier in the artifact (IQ3_S, IQ4_XS, Q2_K, Q3_K, Q4_K,
  Q8_0, plus F32/F16/BF16/NVFP4/Q5_K/Q6_K).
- Tokenizer/chat template resolve from `source_dir` (baked absolute /data path —
  recreated via symlink on the box so the manifest stays byte-identical).
