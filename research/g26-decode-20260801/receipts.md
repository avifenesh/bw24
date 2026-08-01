# g26 decode dig — router w8 flip lands +12.7-12.9% decode; GU slot-packing refuted (lane2, 2026-08-01)

Board context: g26 decode 182.1 vs vLLM 194.6 (0.93x), co-dominant with prefill in the 0.89x
e2e. This lane: (1) honest per-step kernel wall table, (2) re-arbitration of the round-44
router-w8 knife-edge block on multi-prompt real-text evidence, (3) one bounded increment on the
#1 non-router wall.

**Results: the router block was single-prompt roulette — the w8 twin flips ON for gemma4
(+12.7% depth / +12.9% board decode, naked final binary vs naked base binary, x3 interleaved;
g26 decode 182.6 -> 205.7 = 1.06x vs the vLLM 194.6 board denominator*). The gate/up
slot-packing increment measured NEGATIVE (-2.5%) and was killed per flags doctrine.**

*board-denominator caveat: 194.6 is the 2026-07-30 board's vLLM number — per H100-lane law,
a cross-day competitor comparison is clock-drift-invalid; re-run the full board harness for a
publishable cell. The +12.7% memra-vs-memra delta is same-window interleaved and valid.

## Environment

- darklanes-8x (`ip-172-31-11-68`, 18.117.231.105), 8x H100 80GB, **GPU 2 only**. nsys
  2026.1.3 (installed this session via the box's NVIDIA apt repo). sm_90a auto-arch build,
  `~/lane2` tree == branch point e040e149 for all touched files.
- Model: `~/models/gemma-4-26B_q4_0-it.gguf` (30 MoE layers, n_embd=2816, n_ff_exp=704,
  n_expert=128, top-8; experts fully HBM-resident: 428MB x 30).
- Decode number = run-gen `MEMRA_NGEN=128` gen-only tok/s (prime excluded). Every A/B is
  same-window interleaved x3 per arm; datacenter SXM thermal regime.

## 1. The honest wall table (nsys, decode window)

Method: `MEMRA_PROFILE_GEN=1` + `nsys -c cudaProfilerApi`, TWO captures (NGEN=16 and NGEN=528,
identical prompt/prime/gate work) — per-kernel time diff / 512 steps = pure decode shares with
prime contamination subtracted exactly. Raw: `nsys-decode-n{16,528}*.csv|.nsys-rep`.

Pre-flip (base binary), kernel-sum 4823 us/step (step wall ~5.49 ms at 182.3 tok/s -> ~12%
launch/gap overhead):

| us/step | share | inst/step | kernel |
|---|---|---|---|
| 763.9 | 15.8% | 30 | **router_gemv_f32** (25.4 us/layer — the ledger's round-44 number, reproduced) |
| 525.5 | 10.9% | 30 | moe_gate_up_gelu8_dev_q8 |
| 519.0 | 10.8% | 30 | moe_down8_fma_dev_q8_w8r2 |
| 430.0 | 8.9% | 25 | fa_decode_vec_q_rows_v4_w_sp |
| 323.2 | 6.7% | 1 | qmatvec_q6_K_mmvq (LM head) |
| 258.9 | 5.4% | 60 | qmatvec_q4_0_mmvq_rp |
| 235.3 | 4.9% | 25 | fa_decode_combine_rows_w_q8_1 |
| 223.0 | 4.6% | 35 | qmatvec_q4_0_mmvq_fused2_rp |
| 217.6 | 4.5% | 30 | moe_router_topk_scaled_f32 |
| 193.1 | 4.0% | 25 | qmatvec_q4_0_mmvq_fused3_rp |

Router chain (gemv+topk) = 20.3%; expert matvec pair = 21.7%; attention chain ~19.5%.

## 2. Router w8 re-arbitration — verdict: FLIP

Round-44 context: `ROUTER_W8_DEFAULT` stored false for gemma4 because a knife-edge argmax
flipped on ONE synthetic prompt; round-45 (decode-batch gate1) later showed single-synthetic-
prompt knife-edges are roulette. Re-arbitrated here on 6 real prompts (2 canonical + 4 cut
from repo docs — README, CONTRIBUTING, ARCHITECTURE, FLAGS; no synthetic token soup),
each run under BOTH arms (control):

| prompt | lone-warp (default) | w8 (`MEMRA_ROUTER_V2=1`) |
|---|---|---|
| depth-1736 ids | MATCH | MATCH |
| board-2048 | MATCH | MATCH |
| readme (6KB) | **MISMATCH** (965 vs 107) | **MISMATCH** (965 vs 107) |
| contributing (4.5KB) | MATCH | MATCH |
| architecture (8KB) | MATCH | MATCH |
| flags (5KB) | MATCH | MATCH |

The one failing prompt fails BOTH arms with the identical argmax pair (and near-identical
maxdiff 2.203 vs 2.028) — a router-independent prefill-MMA-vs-decode near-tie, not a w8
discriminator. w8's gate outcome is indistinguishable from the lone-warp arm on every real
prompt => the round-44 block was synthetic-prompt roulette. Logs: `gate-*.log`.

Decode A/B (old binary, env-flip arms, x3 interleaved, N=3 medians):

| prompt | lone-warp | w8 | delta |
|---|---|---|---|
| depth-1736 | 182.26 (182.28/181.70/182.26) | 206.16 (205.69/206.16/206.20) | **+13.1%** |
| board-2048 | 180.09 (179.95/181.88/180.09) | 204.32 (206.69/203.61/204.32) | **+13.5%** |

All 12 runs MATCH. Flip implemented: gemma4 no longer stores `ROUTER_W8_DEFAULT=false` at
load (`hybrid.rs`) — it rides the global w8 default like qwen-class; `MEMRA_ROUTER_V2=0`
remains the rollback seam (row added to FLAGS.md — it was previously uncataloged).

Post-flip wall re-profile (same two-capture method, final binary): kernel-sum 4162 us/step
(-661 us, consistent with the e2e delta); `router_gemv_f32_w8` = 114.0 us/step = 3.8 us/layer
(6.7x kernel cut, now 2.7% share). New top walls: gate_up 525.4 (12.6%) + down8 516.1 (12.4%),
fa chain ~20%, LM head q6_K 323.7 (7.8%), topk 217.7 (5.2%). Raw: `nsys-post-*`.

## 3. Bounded increment on the #1 non-router wall — REFUTED

Target: `moe_gate_up_gelu8_dev_q8` (525 us/step) — documented "base geometry only for now":
grid (n_ff=704, 8), block 32 = 5632 lone-warp CTAs (sm_90 caps resident 1-warp CTAs at 32/SM),
~30% of DRAM SOL, the same latency-kernel class as the router. Increment: slot-packed twins
`_j8` (block (32,8), warp j = slot j — the silu family's `_j8` geometry) and `_j8r2` (+2
rows/warp), both preserving the per-(row,slot) FP order bit-exactly (same dot loop, same warp
tree; all three arms produced the IDENTICAL logit maxdiff 2.760e0 — bit-identity confirmed).

Probe (new binary, router w8 in all arms, `MEMRA_MOE_DEVQ8_GGU` forces, x3 interleaved,
depth-1736, N=3 medians): base 208.12 | j8 202.92 (**-2.5%**) | j8r2 202.08 (**-2.9%**).
Logs: `ggu-*.log`.

Refuted: packing 8 slots into one CTA shrinks the grid to 704/352 CTAs (5.3/2.7 per SM) —
coarser scheduling granularity and tail waves beat the 1-warp-CTA occupancy cap it was meant
to fix. Killed per flags doctrine (kernels + dispatch + flag row reverted at tip; the
as-measured mechanism is the parent commit). The next decode rungs per the post-flip table:
the gate_up/down8 pair as a *fusion* candidate (25% combined, both ~15-30% of SOL), the fa
chain, and the 323 us q6_K LM head.

## Gates (final binary: router w8 default, base GU geometry)

| gate | result | log |
|---|---|---|
| kernel-check | `ALL GREEN: kernels match CPU reference.` (rc=0) | `kernel-check-final.log` |
| run-gen g26 depth-1736 ids | MATCH (rc=0) | `gate-final-g26-depth.log` |
| run-gen g26 board-2048 | MATCH (rc=0) | `gate-final-g26-board.log` |
| run-gen q35 board-2048 (cross-model sanity) | MATCH (rc=0) | `gate-final-q35-board.log` |
| run-spec K=1..8 | N/A on this artifact — quoted: `ERROR: model has no MTP/NextN head (nextn_predict_layers=0, no blk.N.nextn.eh_proj).` (rc=2) | `run-spec-g26-final.log` |

## Headline A/B — naked base binary vs naked final binary, x3 interleaved, N=3 medians

| prompt | base (lone-warp) | final (w8 default) | delta |
|---|---|---|---|
| depth-1736 | 182.60 (182.60/181.83/183.49) | 205.73 (205.11/205.73/207.99) | **+12.7%** |
| board-2048 | 180.90 (180.90/179.81/181.85) | 204.17 (207.59/204.17/203.82) | **+12.9%** |

All 12 runs MATCH. Base medians reproduce the board cell (182.1). Logs: `final-*.log`.

5090 note: this is an H100 (sm_90a) result. `ROUTER_W8_DEFAULT` is a runtime default shared
by both arches — per repo law, re-run the correctness battery + decode gates on the 5090
before shipping (the w8 twin was already battery-green on both rigs for qwen-class in round
44; the gemma4 flip needs the same on-rig pass).

## File inventory (raw runs)

- `nsys-decode-n{16,528}_cuda_gpu_kern_sum.csv`, `nsys-n{16,528}.log` — pre-flip capture
  kernel sums (the parseable raw evidence; the binary `.nsys-rep`/`.sqlite` captures are too
  heavy for git and live on the box at `~/lane2/research/g26-decode-20260801/`).
- `nsys-post-n{16,528}_cuda_gpu_kern_sum.csv`, `nsys-post-n{16,528}.log` — post-flip capture kernel sums.
- `gate-battery.sh`, `gate-{depth1736,board2048,readme,contributing,architecture,flags}-{base,w8}.log` — re-arbitration battery.
- `prompt-{readme,contributing,architecture,flags}.txt` — the real-text prompts.
- `ab-router.sh`, `abr-{depth,board}-{base,w8}-{1..3}.log` — router decode A/B.
- `ab-ggu.sh`, `ggu-{base,j8,j8r2}-{1..3}.log` — GU geometry probe.
- `final-battery.sh`, `kernel-check-final.log`, `gate-final-*.log`, `final-{depth,board}-{base,new}-{1..3}.log`, `run-spec-g26-final.log` — final gates + headline A/B.
