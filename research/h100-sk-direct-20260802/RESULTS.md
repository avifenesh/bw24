# h100-sk-direct: does sk + direct-from-quant loaders flip mode 2 past cublas on H100? — NO (2026-08-02)

Lane `lane/h100-sk-direct` (from `restructure/public-split` b9bd9d4c — the tree with
lane/kquant-tile-loaders merged). Rig: darklanes-bench H100 80GB HBM3 (13.203.16.133), tree
rsync'd to `~/memra`, `MEMRA_CUDA_ARCH=90a` nvcc 13.1 release build (3m55s clean). Every GPU
phase under `flock /tmp/gpu-h100.lock`; GPU idle at session start. Model:
`~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (q35), prompt `research/e2e/prompts/board-2048.txt`.

## The question

sm_120a shipped direct-from-quant Q4_K/Q6_K tile loaders on the sk visitor forms
(`research/kquant-tile-loaders-20260802/`, byte-identical maxdiff==0, Ornith pp512 +89%). On
H100 the sk visitor (`MEMRA_MOE_F16G=2`) sat at 96.9% of the cublas mode-1 default
(round 51), with part of the residual priced as the dequant-workspace pass the direct
loaders remove for k-quant layers. Does mode 2 + direct now beat mode 1?

## Gates first

kernel-check sm_90a **ALL GREEN, 0 FAIL** (`kc-skdirect.log`) before any measurement:

- `f16g-kq-direct`: synthetic q4_K + q6_K, all three visitor forms (hybrid/all-128/all-32)
  vs the workspace path — **maxdiff=0.00e0 byte-identical, 6/6**. The real-weight Ornith
  sub-case KC-SKIPs (no ornith gguf on this box; the 5090 receipts carry the real-weight
  green).
- `iq4xs-mmq`: synth T=16/64/128/512 rel <= 1.70e-4, all OK (real KAT sub-case KC-SKIP,
  same reason).

## Three-arm probe — interleaved x5 process rounds, round-robin, one lock hold, same hour

`run-gen MEMRA_NGEN=32`, board-2048 prime (`probe-skdirect.log`). argmax **MATCH 15/15**
(prefill 485 == decode 485 every run; batched-prime gate MATCH every run).

| arm | runs (tok/s) | median |
|---|---|---|
| cublas (naked = Hopper mode-1 default) | 8434.7 / 8547.1 / 8598.3 / 8397.6 / 8635.3 | **8547.1** |
| sk+direct (`MEMRA_MOE_F16G=2`, direct default-on, cross=32) | 8111.5 / 8120.2 / 8112.1 / 8106.3 / 8130.2 | 8112.1 |
| sk-ws (`MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0`, cross=32 — the v0.62 form) | 8084.0 / 8069.6 / 8088.4 / 8055.5 / 8074.2 | 8074.2 |

**VERDICT: NO FLIP.** sk+direct = **94.9% of cublas** (gap 5.1%, ranges disjoint: max
skdirect 8130.2 < min cublas 8397.6). Mode 1 keeps the Hopper naked default; no code or
flag change ships from this lane. The direct loaders DO win over the workspace form —
**+0.47%, zero overlap** (min direct 8106.3 > max ws 8088.4) — but that is the whole
k-quant coverage on this model (next section).

## Cross re-sweep on the winning sk arm (skdirect)

Sweep-grade (1 process per arm, median of 5 in-process reps + 1 warmup, sequential):

| `MEMRA_F16G_SK_CROSS` | pp2048 med (tok/s) |
|---|---|
| 16 | 7999.6 |
| 32 | **8094.9** |
| 64 | 8079.7 |

**cross=32 confirmed** (the sk-bm128 H100 winner) — the direct form did not move the
crossover, and no cross value approaches the cublas arm.

## Pricing the residual (no new nsys — the answer is arithmetic)

q35's expert bank (tensor mix `research/kat-anomaly-20260802/ctrl-q35-tensor-mix.txt`):
direct-eligible Q4_K/Q6_K = **4 of 123 expert projections** (3x Q6_K down + 1x Q4_K down =
0.81 GB of ~15.6 GB = **5.2% of bank bytes**); gate/up are IQ3_S x39 + IQ4_XS x1 + Q3_K x1
each, downs IQ4_XS x37. The +0.47% direct win is coverage-proportional. The IQ3_S/IQ4_XS
bulk (94.8% of bytes) still rides workspace-dequant + the visitor GEMM, so the 5.1% gap to
cublas is round 51's priced residual unchanged (H100 nsys: sk stage 131.9ms vs
cutlass-grouped 101.6 + h2f 10.8; the 32x64 2-stage tail form = 31% of stage) — no kernel
this lane measured differs on that class, so a fresh profile would re-measure known code.

The flip case rested on q35's k-quant stragglers plus the structural no-h2f/zero-sync
edges: measured, those are worth 8074 -> 8112 against a 435 tok/s gap. A Hopper flip needs
**direct-from-quant tile loaders for IQ4_XS/IQ3_S superblocks** (the 94.8%) and/or the
deeper sk32 tail form (BK=64 3-stage small-BM / register double-buffering) — future rungs,
not this lane. Ornith-35B (all-Q4_K bank, where direct is worth +38% pp2048 on the 5090)
is not on the H100 board; if it ever onboards there, re-ask this question for that model
class.

Cross-session note (LAW 1): the sk-vs-cublas gap read 3.1% in the sk-bm128 session
(pre router-exactness fix) and 5.1% today; both are same-session interleaved and the
widening is cross-day + round-52 code motion — no conclusion is built on it.

## Battery

`tools/validate-h100.sh <q35> --quick` run post-probe as the tree-health receipt for the
rsync'd b9bd9d4c tree: **VALIDATE-H100: ALL GATES GREEN, rc=0** — policy tests,
kernel-check, decode-batch config B=8 (gate1 fraction rule), decode-batch strict,
decode-dc, graph-decode, graph-session (`vh100-quick-skdirect.log`). The lane's
measurement gates are the kernel-check + per-run argmax above.

## Board

No board cell ran: no default change, no board-moving number. The H100 board jsonl and the
README/current-board.json (5090 board) are untouched.

## Files

`run-probe3-skdirect.sh`, `run-cross-sweep.sh`; raw logs `probe-skdirect.log`,
`kc-skdirect.log`, `sweep-cross{16,32,64}.log`, `vh100-quick-skdirect.log`;
`receipts.jsonl`.
