# Ragged token-tile expert MMQ — REFUTED on H100 (lane2, 2026-08-01)

Lever: research-ranked #1 for MoE expert prefill — `mmq_iq_experts_kernel` runs a fixed
MMQ_X=128 token tile while expert groups average ~65 pairs on q35 gate/up (~145 on g26), so a
128 tile "pads 65 to 128 (~50% wasted MMA + gather)". Mechanism built and measured here:
runtime dispatch to the smallest tile in {64,96,128} covering the ceil-average group
(`n_pairs/n_active`), per launch.

**Verdict: REFUTED on this box.** Sub-128 tiles lose ~7.6% q35 pp2048 prefill; g26 is
unaffected (avg 147-236 always picks 128). The padding-waste premise was already half-dead:
the Y gather has skipped clamped tail columns since round 46 inc4, so at tile 128 only the
dead-column MMA remained — and this kernel is latency-bound (round-45 ncu: SM 13-20%, DRAM 3%,
long_scoreboard-dominated), so dead MMA was nearly free *and* was hiding W-stage/Y-gather
latency. Shrinking the tile removed latency-hiding work; the 96-floor probe shows the loss is
NOT the 2-pass W-dequant tax (r96 ≈ ragged within 0.3%).

## Environment

- Box: darklanes-8x (`ip-172-31-11-68`, 18.117.231.105), 8x H100 80GB HBM3, **GPU 2 only**
  (`CUDA_VISIBLE_DEVICES=2` on every run). nvcc from `~/cuda-13.3.1`. sm_90a auto-arch build.
- Tree: `~/lane2` — verified byte-identical to the local branch point (e040e149, "Merge
  restructure/public-split: round 47 cont.") for all touched files before editing.
- Models: `~/models/gemma-4-26B_q4_0-it.gguf` (g26, q4_0 experts, nc=1 arm),
  `~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (q35, IQ4_XS experts, nc=0 arm).
- Thermal regime: datacenter SXM H100; all A/B comparisons interleaved in the same window
  (arms alternate per iteration, minutes apart). Each number below is the **median of 5
  in-process prefill reps** (`MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5`, warmup excluded), 3
  invocations per arm.

## Code under test (commit 1 on this branch; reverted at tip per flags doctrine)

- `crates/memra-engine/cu/mmq_iq_experts.cu`: launcher `memra_mmq_iq_experts` gains `n_pairs`
  (C ABI), computes `avg = ceil(n_pairs/n_active)`, dispatches
  `mmq_iq_experts_kernel<T,nc>` with T = smallest of {64,96,MMQ_X} with T >= avg (MMQ_X build
  seam stays the ceiling; nc arm per `out_f % 128`). Six instantiations in one binary.
  Dynamic smem sized **per dispatch** via `iqexp_smem_bytes(T)` — ids + Y ping-pong scale with
  T, tile_x + W stage ring stay MMQ_Y-sized: T=64: 98,560 B; T=96: 108,928 B; T=128: 117,248 B
  (matches the old fixed formula at 128). Env seams: `MEMRA_MMQ_IQEXP_RAGGED=0` legacy fixed
  tile (rollback), `=96` floor-96 probe arm; `MEMRA_MMQ_IQEXP_DIAG=1` prints first 24 dispatch
  decisions.
- `crates/memra-engine/src/mmq_ffi.rs`: extern + `Engine::mmq_iq_experts` pass `n_pairs`
  (wrapper already received it; call sites unchanged).

## Gates (new binary, GPU 2)

| gate | result | log |
|---|---|---|
| kernel-check | `ALL GREEN: kernels match CPU reference.` (rc=0) | `kernel-check.log` |
| run-gen g26 (depth-prompt-1736 ids as args) | `prefill argmax=623 decode argmax=623 ... MATCH` (rc=0) | `run-gen-g26-new.log` |
| run-gen q35 (board-2048 prompt) | `prefill argmax=485 decode argmax=485 ... MATCH` (rc=0) | `run-gen-q35-new.log` |
| run-gen q35, RAGGED=96 probe arm | `MATCH` (rc=0) | `run-gen-q35-r96-diag.log` |
| run-spec K=1..8 | **N/A on both box artifacts** — quoted: `ERROR: model has no MTP/NextN head (nextn_predict_layers=0, no blk.N.nextn.eh_proj). generate_spec is unavailable for this file.` (rc=2, both models) | `run-spec-q35-new.log`, `run-spec-g26-new.log` |

The refutation is therefore **performance-only**: the mechanism is argmax-exact on both models
and both nc arms, and both staged W loaders (q4_0 w16 path on g26, IQ4_XS 4B path on q35) ran
under it.

## Dispatch evidence (MEMRA_MMQ_IQEXP_DIAG=1)

q35 pp2048: `n_pairs=16384`, `n_active` 219-256 → avg 64-75 → **tiles 64 and 96 live**
(`proj=0/1/2 n_active=256 avg=64 tile=64`, `n_active=252 avg=66 tile=96`, ...).
g26 pp1736: `n_pairs=13888`, `n_active` 59-95 → avg 147-236 → **tile 128 always** (control).
96-floor arm: same layers all pick `tile=96` (`run-gen-q35-r96-diag.log`).

## A/B — q35 pp2048 prefill, interleaved x3, median-of-5 per run

Round 1 (`ab.sh`, logs `ab-q35-*.log`): A=/tmp/lane2-base-run-gen (pre-change binary),
B=new ragged, C=new binary + `MEMRA_MMQ_IQEXP_RAGGED=0` (attribution).

| iter | base (A) | ragged (B) | Δ B/A | ragged0 (C) | Δ C/A |
|---|---|---|---|---|---|
| 1 | 5456.2 | 5024.3 | -7.9% | 5454.5 | -0.03% |
| 2 | 5436.8 | 5025.3 | -7.6% | 5416.9 | -0.4% |
| 3 | 5386.5 | 4998.0 | -7.2% | 5405.5 | +0.4% |
| median | 5436.8 | 5024.3 | **-7.6%** | 5416.9 | -0.4% |

C ≈ A ⇒ the entire loss is the tile choice, not binary/dispatch overhead.

Round 2 (`ab96.sh`, logs `ab96-q35-*.log`): adds D=new binary + `RAGGED=96` (floor 96, never 64).

| iter | base (A) | ragged (B) | Δ B/A | r96 (D) | Δ D/A |
|---|---|---|---|---|---|
| 1 | 5453.1 | 5025.7 | -7.8% | 5046.6 | -7.5% |
| 2 | 5458.0 | 5036.0 | -7.7% | 5037.0 | -7.7% |
| 3 | 5457.6 | 5026.5 | -7.9% | 5038.3 | -7.7% |
| median | 5457.6 | 5026.5 | **-7.9%** | 5038.3 | **-7.7%** |

D ≈ B ⇒ the 64-tile's 2-pass W-dequant tax is NOT the dominant cost; **any** sub-128 tile
loses ~7.6% here.

## A/B — g26 pp1736 prefill (control: ragged picks 128), interleaved x3

| iter | base (A) | ragged (B) | Δ |
|---|---|---|---|
| 1 | 10162.8 | 10165.9 | +0.03% |
| 2 | 10056.4 | 10055.0 | -0.01% |
| 3 | 10069.3 | 10166.6 | +0.97% |
| median | 10069.3 | 10165.9 | flat |

Baseline sanity vs handed-down numbers for this box: g26 ~10.1k ✓ (10.06-10.17k),
q35 board-2048 ~5.4k ✓ (5.39-5.46k) — own baseline re-measured, not imported.

## Mechanism (why the #1-ranked lever loses)

1. The waste model double-counted: gather waste on padded columns was already eliminated at
   tile 128 (round 46 inc4 `token_c > j_max` skip). Only dead-column MMA remained.
2. The kernel is latency-bound, not MMA-bound (round-45 ncu in-file: SM 13-20%, DRAM 3%,
   long_scoreboard 66% of stalls pre-staging). Dead MMA costs ~nothing and its issue slots
   hide the cp.async W-stage + Y-gather latency of the next kb. A 64/96 tile cuts the per-kb
   MMA j-loop by 2x/1.33x, exposing that latency: measured -7.6% even with zero extra passes
   (r96 arm, every group ≤ 96 in one pass on avg-64 layers... groups >96 still 2-pass; the
   64-vs-96 equality bounds the pass-count effect at ≤0.3%).
3. Per-group W dequant re-runs on multi-pass groups add on top for tile 64 — measurable but
   minor (B vs D: ~0.2%).

Transfer note: this is an H100 (sm_90a) result. The mechanism (latency-bound, dead-MMA-as-
latency-hiding) is not H100-specific, but the 5090 balance differs; if the lever is ever
re-tried it must re-run this same interleaved battery on the 5090. Any future ragged attempt
should instead invert the loop nest (dequant W once per kb, walk token sub-tiles inside it)
so smaller token tiles don't multiply W dequant or shrink the latency-hiding window — that is
a different, larger restructure.

## End state (flags doctrine: negative ⇒ kill the flag and the arm)

- Commit 1 on `lane/ragged-tiles`: the mechanism exactly as measured (reproducibility anchor).
- Branch tip: dispatch reverted to the fixed-MMQ_X launcher (byte-equal to base for
  `mmq_iq_experts.cu`, `mmq_ffi.rs`, `FLAGS.md`); `MEMRA_MMQ_IQEXP_RAGGED`/`_DIAG` killed.
  These receipts + raw logs are the record.
- Not pushed (per lane instructions); merge/cherry-pick is the owner's call.

## File inventory (raw runs — evidence discipline)

- `kernel-check.log` — full kernel-check, tail ALL GREEN.
- `run-gen-g26-new.log`, `run-gen-q35-new.log` — argmax gates + dispatch diag (new binary).
- `run-gen-q35-r96-diag.log` — 96-floor arm gate + diag.
- `run-spec-q35-new.log`, `run-spec-g26-new.log` — run-spec N/A (no MTP head), quoted cause.
- `ab.sh`, `ab-{q35,g26}-{base,ragged,ragged0}-{1..3}.log` — round-1 interleaved battery.
- `ab96.sh`, `ab96-q35-{base,ragged,r96}-{1..3}.log` — round-2 96-floor probe battery.
