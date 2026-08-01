# Hy3 MTP spec K-sweep on sm_90a — G2 closure (Mumbai H100, 2026-08-01)

Closes the sweep owed by `research/hy3-hopper-20260801/gaps.md` G2: run-spec K=1..8 on the
staged Hy3 layer103.5 artifact, board-d1736 real-prompt class, spec-vs-plain ratio per K,
N=3. Verdict shape for the Aug-2 8xH100 spike at the bottom.

## Protocol

- Box: Mumbai darklanes-bench H100 80GB (shared; **every** GPU-touching process held under
  `flock /tmp/gpu-h100.lock`). GPU otherwise idle — concurrent compute-apps captured EMPTY
  in every pre/post run bracket, temps 35-38 C at brackets, same-day same thermal regime.
- Build: restructure/public-split tip `2b9a6aa6` (= lane/hy3-spec-sweep base; includes the
  G1 overlay-acceptance fix `096f212d`), fresh tree at `/opt/dlami/nvme/hy3-spec-sweep/memra`,
  `MEMRA_CUDA_ARCH` auto-detected 90a (`logs/build.log`). The first-light lane tree
  `/opt/dlami/nvme/hy3-hopper/memra` was left byte-untouched (its source differs from the
  merged tip in engine core: spec.rs, hybrid*.rs, moe_cache.rs, decode.rs).
- Gate: `kernel-check` ALL GREEN — 206/206 OK on this exact build (`logs/kernel-check.log`).
- Artifact: `/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime` (manifest sha `b8bdd684…`),
  bytes untouched; box left staged exactly per `research/hy3-hopper-20260801/box-state.md`.
- Prompt: `research/gemma4-bringup/depth-prompt-1736.txt`, RAW continuation (no chat
  template) = the baseline.md board-d1736 protocol; encodes to **1818 tokens** (verified in
  every log). `MEMRA_NGEN=128`. NOT run-spec's degenerate default single-token prompt (the
  0%-scare prompt — banned for acceptance numbers per gaps.md G2).
- One sweep run = ONE `run-spec` process (no `MEMRA_SPEC_K`): model load, warmup, plain
  greedy oracle, then the K=1..8 battery — primed state reused within the process
  (within-process primes ~152 s vs the ~17 min fresh-process wall of baseline.md). Each K's
  ratio is against its own process's oracle (same cache regime, same clock). N=3 processes:
  `logs/sweep-r{1,2,3}.log`, driver `spec-sweep-driver.sh` (bracketed GPU state + exit codes
  in-log). All printed tok/s are gen-only (run-spec's in-API prime-subtract timer).

## Result — K-sweep, N=3 (medians; per-run values in sweep-table.md)

Plain-generate oracle: **median 2.49 tok/s** (2.38 / 2.49 / 2.49) — matches the baseline.md
spill floor (2.48 median N=3 on the pre-merge lane build).

| K | self-consistency | acceptance (median, bit-identical x3) | spec tok/s (median) | spec/plain (median) | N |
|---|---|---|---|---|---|
| 1 | PASS x3 | 8.5% (10/117) | 2.08 | 0.84x | 3 |
| 2 | PASS x3 | 4.3% (10/234) | 1.44 | 0.58x | 3 |
| 3 | PASS x3 | 2.8% (10/351) | 1.11 | 0.45x | 3 |
| 4 | PASS x3 | 2.1% (10/468) | 0.91 | 0.37x | 3 |
| 5 | PASS x3 | 1.7% (10/585) | 0.80 | 0.32x | 3 |
| 6 | PASS x3 | 1.4% (10/702) | 0.67 | 0.27x | 3 |
| 7 | PASS x3 | 1.2% (10/819) | 0.61 | 0.25x | 3 |
| 8 | PASS x3 | 1.1% (10/936) | 0.57 | 0.23x | 3 |

Run-to-run spread ≤ 1% on spec tok/s at every K; greedy decode is run-to-run deterministic
on this artifact (identical tokens => bit-identical acceptance counts across runs AND across
builds). Exactness: SELF-CONSISTENCY PASS for every K in every run — the MTP head loads,
forwards, and drafts EXACTLY on sm_90a. Correctness is not in question anywhere here.

## Headline findings

### 1. Spec is a net SLOWDOWN at every K at the 1-GPU spill floor

Best arm K=1 = 0.84x median: spec *subtracts* ~16% on the board-d1736 depth prompt in the
single-H100 spill regime. Mechanism is staging, not drafting: each spec round issues a
(K+1)-token verify batch whose positions route to a *union* of experts, multiplying staged
bytes per round, while the nextn=1 head nets ≤ ~1.09 tokens/round. Measured cost: plain
step 0.40 s/token; spec round 0.52 s at K=1, growing ~0.2-0.27 s per +1 K (sub-linear —
expert unions overlap). Acceptance never pays it back.

### 2. Acceptance count is EXACTLY 10 at every K — the nextn=1 head cannot chain

Drafted = 117 x K exactly, accepted = 10 exactly, at all eight Ks: the same 10 decode
positions accept their first draft token and **no position ever accepts a chained 2nd draft
token**. Expected shape for a single-MTP-layer head applied recursively — draft quality
collapses past +1. K>1 buys zero extra accepts on this content and only multiplies verify
cost. Any future spec deployment of this artifact should only consider K=1.

### 3. The first-light 23.8% / 1.16x point decomposes into (real rate, cold-denominator artifact)

- Probe A (`logs/probe-a-firstlight-repl.log`): the first-light probe re-run on THIS build
  (same 25-id chat prompt, K=2, NGEN=32) — acceptance **10/42 = 23.8%, bit-exact
  replication**. The merge changed nothing in the draft path.
- But the ratio flipped to **0.56x**: first light's 1.16x stood on a 0.61 tok/s plain oracle
  (first-ever-touch cold cache); warm, the same config's oracle runs 3.73 tok/s and spec
  loses. The 1.16x was a cold-denominator artifact — the exact denominator error class the
  H100 lane laws exist for. **Neither 1.16x nor any spec tok/s from this box may enter the
  $/Mtok table as a positive.**
- Probe B (`logs/probe-b-oldbuild-depth-k1.log`): the PRE-merge first-light binary on the
  d1736 K=1 protocol — 8.5% (10/117), 0.84x, identical to the tip build. Cross-build
  agreement is total; the depth acceptance collapse is content-driven, not a merge effect.
- Acceptance is content-class-dependent: chat prose accepts in 47.6% of rounds (10/21);
  the synthetic repeating-story d1736 prompt accepts in 8.5% (10/117). run_spec.rs's own
  warning (synthetic content understates acceptance) applies to d1736 itself; a
  natural-prose board-class corpus would sit somewhere between.

## Verdict for the Aug-2 spike

- **Best K: 1** (least-bad at the floor, and the only K with any acceptance mechanism —
  chained drafts accept exactly zero). K=1: acceptance 8.5% (d1736) / 47.6% of rounds
  (chat prose), 0.84x median at the 1-GPU floor, N=3.
- **Spec adds NOTHING at the 1-GPU floor — it subtracts 16% at best. The $/Mtok exit
  table's 1-GPU floor row must be plain greedy decode (2.49 tok/s median, N=3), spec OFF.**
- **PP-2 is a different regime — do not extrapolate either number to it.** With the bank
  fully resident across 2x80GB the verify batch stops staging experts and becomes
  ~compute-only; spec's ceiling is then set by acceptance alone: ~+8.5% on d1736-class
  content, up to ~+48% on chat-prose-class content (both at K=1, both ceilings assuming a
  free verify pass, before draft-forward overhead). The box team must run its own PP-2
  K=1 sweep before writing any spec row; the floor 0.84x is not the SKU number, and the
  refuted 1.16x must not be quoted at all.
- 5090 same-artifact acceptance reference: **still owed** (local rig saturated with drafter
  corpus work tonight). Acceptance here replicated bit-exactly across two builds on this
  box, so the (artifact, prompt) -> rate mapping is stable evidence, single-box sm_90a.

## Receipts

- `logs/sweep-r{1,2,3}.log` — N=3 full-battery raw logs, GPU brackets + exit codes in-log.
- `logs/probe-a-firstlight-repl.log` — first-light replication on the tip build.
- `logs/probe-b-oldbuild-depth-k1.log` — pre-merge binary, d1736 K=1 cross-check.
- `logs/kernel-check.log` (206/206 ALL GREEN), `logs/build.log` (sm_90a, tip `2b9a6aa6`).
- `sweep-table.md` — machine-parsed per-run table (`parse-sweep.py`).
- `spec-sweep-driver.sh` — the on-box driver (copy of `/opt/dlami/nvme/hy3-spec-sweep/`'s).
- Box additions live only under `/opt/dlami/nvme/hy3-spec-sweep/` (tip build tree + logs,
  left for the spike team); staged artifact dirs untouched; all GPU work off the box at
  18:02Z Aug 1 — 16.5 h before the box-prep hard stop.
