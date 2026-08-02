# fa-decode-deep: the deep-ctx rewrite of the v4 decode vec kernel (2026-08-02)

Lane `lane/fa-decode-deep` (from `restructure/public-split` 6e03c838 — post depth-decode
merge). Rig: RTX 5090 Laptop 24463 MiB sm_120a, 82 SMs. Every GPU run under
`flock /tmp/gpu5090.lock`; two co-lanes shared the rig this session (their heat is inside
the per-call bench spreads; the quiet-rig nsys cells are labeled). Follow-up to
`research/depth-decode-20260802/` — the class-wide depth decay priced there
(`fa_decode_vec_q_v4_dc` + combine = 12.3 → 44.3 µs/layer-token from d512 → d6144,
162 GB/s effective = 19% of the card).

## 1. Mechanism (ncu, CONFIRMED — receipts `ncu-v4-d6144.txt`)

`fa_v4_smem.k_ints[32][64]`: row stride 64 words ≡ 0 mod 32 banks, so the v4 score
phase's `k_ints[lane][c*8+w]` operand read serializes as a **32-way shared-memory bank
conflict on every dp4a K operand** — measured **1,403,465 load-bank-conflict cycles per
launch** at d6144 (`k_d[32][8]` is 8-way on top). Second order: the byte-wise staging
store stream (477K store-conflict cycles/launch), and the barrier-serialized tile loop
exposing the next tile's DRAM latency. This — not raw DRAM bandwidth — is why the vec
kernel read KV at 19% of the card: the kernel is SM-cycle-bound (same ~56-60K elapsed
cycles at locked 0.8 GHz and real ~2.9 GHz).

## 2. The deep form (BIT-IDENTICAL — the strongly-preferred branch of the exactness law)

`fa_decode_vec_q_v4_deep` / `_deep_dc` run the v4 program VERBATIM — same split
partition (ONE-PARTITION law in the dc twin), same per-key dp4a order, same tile-max /
B2 softmax bookkeeping / B3 accumulation order, same `[head][split]` partials into the
UNCHANGED `fa_decode_combine_f32`. Only the physical smem layout and the load/store
schedule move:

- **A. bank-conflict row pads**: `k_ints[32][68]`, `k_d[32][9]` — read banks
  `(68j+x) % 32` / `(9j+c) % 32` cover all 32; slot mapping and indexing expressions
  unchanged. Load conflicts 1.40M → 3.5K/launch (−99.8%).
- **B. packed staging stores**: the 8 funnelshift K ints and 8 dequanted V bf16 collect
  in registers and land as 16 B stores (all row offsets 16 B-aligned). Store conflicts
  382K → 28.6K/launch.
- **C. L2 tile prefetch**: both first tiles' K/V lines issue `prefetch.global.L2` at walk
  entry (before the q-quant barrier); tile+2 distance inside the loop (sp128 splits).
- **D. hd256/dpl8 specialization** (dispatch is host-gated hd256): compile-time dpl kills
  the per-slot B3 predicates. Measured flat vs C — kept for cost-free codegen.

ptxas: **0 spill stores / 0 spill loads** (68 regs, both twins). smem 12160+16384 B
(3 blocks/SM, same as v4's 27.9 KB class).

Dispatch: `fa_deep_at` in both `fa_decode_kvmod` (eager) and `fa_decode_dc_q8` (graph,
keyed on bucket_max — the fa_v4_at precedent), default KV module only (`!g`), probe modes
excluded. **Rollback seam `MEMRA_FA_DEEP=0`**; `MEMRA_FA_DEEP_MIN` is a sweep/diagnostic
seam. The rows-verify, seqs (batched tick) and combine kernels are untouched — deep's
bits equal v4's bits, so every cross-kernel pin (rows-vs-loop, seqs-vs-loop, graph
bit-identity) holds by construction and is re-verified below.

## 3. Bit-identity verdict: GREEN everywhere measured

- `fa-deep-bench` (production-appended synthetic KV, class geometry nkv=2/gqa=8):
  deep-vs-v4 bitdiff=0 at 11 depths × {eager, dc exact-bucket, dc bucketed-replay},
  including sp8→sp64 rung crossings (3071/3072/3073), tail tiles (513/4097/6143/6200)
  and the ladder-straddle bucket case (deep reproduces v4's documented dc-vs-eager
  straddle EXACTLY, element-for-element). 44/44 OK (`bench-stage2-stores.log`,
  `floor-sweep.log`).
- kernel-check: new standing FA-DEEP pin (deep-vs-v4 BYTE identity, geometries
  nkv=2/gqa=8 AND nkv=8/gqa=4, depths {512,3071,3073,4096,6144,6200} eager+dc+replay) —
  full battery **ALL GREEN** (`kernel-check-full.log`).

## 4. Kernel-level result (quiet-rig nsys, real clocks, N=10 medians)

| kernel @ d6144 | v4 | deep | ratio |
|---|---|---|---|
| vec (dc twin) | 29.7 µs | 20.8 µs | **1.43x** |
| vec @ d2048 (sp8 rung) | 13.5 µs | 11.5 µs | 1.17x |
| combine_f32 | 7.5 µs | 7.5 µs (unchanged kernel) | — |

Effective KV-read bandwidth at d6144: 192 → **277 GB/s** on the same 5.7 MB of q8_0/q5_1
bytes (production receipts had v4 at 162 GB/s under co-running load). The priced 2x+ was
not fully reached in-kernel (1.43x); the residual is split across global-load latency
(long-scoreboard 26%), MIO pressure, and the order-pinned B3 FMA chain — the next
structural levers (cp.async raw pipeline, sV re-layout) all trade against the 3-blocks/SM
smem cliff and were left un-built rather than risk the occupancy regression.

REFUTED in-lane (receipts `nsys-combine-f4-*.txt`, JSONL-of-record = this file):
- combine grid re-tile (n_head×hd/32 blocks of 32): FLAT — 7.66 vs 7.46 µs d6144, 18.6
  vs 19.0 µs d2048. The 1-block/head shape already carries 128 warps.
- combine float4 4-dims/thread: WORSE — 11.2 µs d6144 / 26.1 µs d2048 (32 warps lose
  more latency cover than wide loads buy). Killed per flags doctrine.

## 5. Engagement floor sweep (threshold choice)

`fa-deep-bench sweep`, fine grid 96..6144, production dc call form, N=3 interleaved
medians + per-rep values (`floor-sweep.log`): deep is flat-or-better at EVERY depth
(1.01x–1.26x, no losing cell). **Swept floor = 0: always-on wherever v4 dispatched.**
No engagement boundary ⇒ no new capture/recapture edge (the segmented-recapture law is
satisfied trivially); the sp-ladder rung at 3072 remains the only geometry boundary, as
before.

## 6. Batteries (deep default-on) — ALL GREEN, FAILS=0

`run-battery.sh` → `battery-console.log` + `gate-*.log` (session 2; a harness restart
killed session 1 mid-spec — its three argmax MATCH logs were complete and are superseded
by this full session):

- **kernel-check**: ALL GREEN incl. the FA-DEEP byte pin at BOTH geometries
  (nkv=2/gqa=8 and nkv=8/gqa=4), 36/36 bitdiff=0 (`kernel-check-full.log`).
- **run-gen argmax**: MATCH ×3 models (kat/q35/o35b), d4096 document prompt — the deep
  region (`gate-argmax-*.log`). Additionally every depth-A/B run below carries the gate.
- **run-spec K=1..8 self-consistency**: PASS ×3 models, own-trim drafters, 8/8 per-K
  PASS lines each + final `=== SELF-CONSISTENCY PASS ===` (verify rides these logits;
  `gate-spec-*.log`).
- **decode-batch gates (q35)**: config mode (steps 32, B=8) ALL GREEN; strict mode under
  the equalized env (`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`, steps 16, B=4) ALL GREEN —
  the batched tick rides the same fa class (`gate-decode-batch-*.log`).
- **graph gates** (fa class boundaries are capture-relevant): graph-decode PASS at
  q35 P=6000 N=160 (deep region, "160 steps generate_graph == decode_step
  (BIT-IDENTICAL), buckets=4"), kat P=3000 N=160 (crosses the 3072 sp-ladder rung
  inside a segment), q35 P=500 N=96 (short/vec-floor region); graph-session q35
  step-lift PASS (`gate-graph-*.log`). These ran on the binary carrying the
  `fa_plan` deep-name classification fix (3cd817da), so the segmented exec-update
  path retunes deep nodes exactly as it retunes v4 nodes.

## 7. The depth table — old vs new + vs-llama (e2e)

`run-depth-ab.sh` (+ `run-depth-ab-resume.sh` for rep3 after a harness kill at row 76 —
reps 1-2 complete before it, no cell lost, no duplicates): the depth-decode lane's exact
protocol and prompts. memra arms = naked run-gen gen-only rate over 128 greedy tokens,
OLD = `MEMRA_FA_DEEP=0`, NEW = naked (deep default); arms ADJACENT per (model, depth) in
the same thermal window. llama = fresh same-session `llama-bench -p 0 -n 128 -d ...`
denominators (cross-day denominators are clock-drift-invalid). N=3 per cell, medians;
per-rep values + temps in `depth-ab.jsonl`; argmax gate MATCH in 72/72 memra runs.
Co-lane regime NOTE: this session's rig ran quieter than the depth lane's — the OLD arm
itself reads higher than the depth-lane cells (kat d6144 178.1 here vs 170.8 there), so
cross-lane absolute comparisons are invalid; the in-session pairs are the evidence.

tg128 tok/s (N=3 medians):

| model | arm | d512 | d2048 | d4096 | d6144 |
|---|---|---|---|---|---|
| KAT | old | 189.3 | 177.7 | 181.0 | 178.1 |
| KAT | new | 191.1 | 180.0 | 183.9 | 178.9 |
| KAT | **new/old** | **1.009x** | **1.013x** | **1.017x** | **1.004x** |
| KAT | new/llama | 1.021x | 0.960x | 1.015x | 1.003x |
| q35 | old | 187.0 | 176.9 | 179.9 | 174.4 |
| q35 | new | 187.7 | 177.6 | 180.7 | 176.8 |
| q35 | **new/old** | **1.003x** | **1.004x** | **1.004x** | **1.014x** |
| q35 | new/llama | 1.146x | 1.087x | 1.143x | 1.136x |
| o35b | old | 200.5 | 192.8 | 195.9 | 190.1 |
| o35b | new | 204.6 | 193.7 | 198.0 | 191.6 |
| o35b | **new/old** | **1.020x** | **1.005x** | **1.010x** | **1.008x** |
| o35b | new/llama | 1.114x | 1.061x | 1.120x | 1.107x |

**Verdict vs the win condition:** flat-or-better in ALL 12 cells (min 1.003x, max
1.020x); gains at d4096+ = +0.4..+1.7% e2e. Honest sizing vs the priced expectation:
the lane priced ~+3.7% e2e at d6144 for closing the FULL per-key gap to llama; the
kernel closed 1.43x of the needed ~2x, and this quiet-rig session's old arm runs
faster than the production receipts (attention is a smaller share here), so the e2e
delta lands at ~+1-1.5% at depth with N=3 noise ±1%/cell. The per-rep pairs at d6144
(kat rep1 175.8→179.0 +1.8%, q35 rep3 174.3→176.6 +1.3%, o35b rep2 189.4→192.4 +1.6%)
carry the effect; the kat d6144 median pair (178.1→178.9) is compressed by one hot old
rep. vs llama: q35 1.09-1.15x and o35b 1.06-1.12x at every depth (both arms above
water, new widens every cell); KAT sits at short-ctx parity and stays 0.96-1.02x —
its remaining weak cell is d2048 (0.960), the sp8-rung/combine region (§8), not the
deep-kernel region.

## 8. Stale-verdict finding for a FOLLOW-UP lane (not built here)

At the sp8 rung (d2048: 256 splits) the **combine (19 µs) now exceeds the deep vec
kernel (11.5 µs)** — the 3072 sp8→sp64 rung was calibrated 2026-07-08 on the conflicted
v4 core and is stale under the deep kernel (combine + partial-buffer cost scales with
n_splits; sp64-at-d2048 would cut combine ~8x for a vec-side cost). A ladder move is a
NEW NUMERIC CONFIG (split partition changes combine order) and needs its own full
battery per model — priced, not shipped, in this bit-identical lane.

## Files

`fa_deep_bench.rs` (bit gate + floor sweep + ncu mode) → `bench-stage1-padprefetch.log`,
`bench-stage2-stores.log`, `floor-sweep.log`; `ncu-v4-d6144.txt`;
`nsys-stage2-d{2048,6144}.txt`, `nsys-stage3-d6144.txt`, `nsys-combine-f4-d{2048,6144}.txt`
(quiet-rig kernel medians; `.nsys-rep` binaries local-only per .gitignore);
`kernel-check-full.log`; `run-battery.sh` → `battery-console.log` + `gate-*.log`;
`run-depth-ab.sh` → `depth-ab.jsonl`, `ab-console.log`, `mem-*.log`, `llama-*.log`;
`summarize-ab.py`.
