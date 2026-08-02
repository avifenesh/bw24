# f16g-default-rearb: sm_120a naked default moe_f16g_mode 3 -> 2 (2026-08-02)

Lane `lane/f16g-default-rearb` (from `restructure/public-split` e42cc8e1; flip commit
d703e8f4). Rig: RTX 5090 Laptop 24463 MiB sm_120a, platform_profile `performance`,
`gpu-full-power on`. Every GPU run under `flock /tmp/gpu5090.lock` (co-resident
`llama-server --embedding` 332 MiB allowlisted, inside every figure). Temps 60-75 C
across the headline battery (stamped per row in `headline.jsonl`); N stated per cell.

## 1. The flip

`moe_f16g_mode()` sm_120a naked default: **3 (AUTO-KQUANT) -> 2** — every expert layer
whose three projections pass `f16g_proj_ok` rides the mode-2 sk visitor with
direct-from-quant tile loaders, not just the layers the int8-MMA MMQ arm rejects. Mode 3
stays reachable as `MEMRA_MOE_F16G=3` (new env arm — `3` previously parsed as mode 1),
the rollback seam. The gemma (gelu) dispatch site is untouched: `moe_f16g_gemma_on()`
keys on env PRESENCE, so the naked default flip leaves it closed (verified §3). Decode
and spec-verify stay on dp4a (t >= 16 floor unchanged). The decode-batch-gate in-binary
`MEMRA_MOE_F16G=0` pin is default-flip-invariant ("0" closes the door under every mode
semantics) — comment updated, pin value untouched, and the gate re-run green (§4).

Why now: the mode-3 ruling "IQ4_XS-bank models keep their measured-faster MMQ tiles" was
priced BEFORE the IQ direct loaders (lane/q4k-expert-prefill). iq-direct-loaders §5
refuted it on the 5090 (q35 mode-2+direct +33.9% over naked mode 3, x3). This lane is
the promotion: headline re-confirmed x5 with whole-binary arms + the full battery under
the new default. #49.

## 2. Headline (x5 process-interleaved, whole-binary arms)

Arms: `mode3` = pre-flip binary (e42cc8e1 naked — the old default), `mode2` = flip binary
(d703e8f4 naked — the candidate). Both binaries stashed provenance: `bin-preflip/`
(the mode2 binary was built from the identical tree that became d703e8f4; the headline
console header stamps e42cc8e1 because the flip was committed mid-session — every value
in `headline.jsonl`, per-run logs `q35-*-r*-{mode2,mode3}.log`).

**q35 board-2048 (`headline.jsonl`, x5, pp = in-process median of 5 reps +1 warmup):**

| cell | mode3 (old default) | mode2 (new default) | delta |
|---|---|---|---|
| pp-only board-2048 tok/s | 4124.3/3715.8/3644.0/4023.4/4052.9 med **4023.4** | 4926.3/4816.1/5355.8/5393.0/5386.4 med **5355.8** | **+33.1%**, zero overlap |
| gen prefill tok/s (board-2048 + 128) | med 4036.2 | med 4949.5 | +22.6% |
| gen decode tok/s | med 175.55 (163.4-177.7) | med 168.01 (164.4-177.3) | flat — both arms span the same range, reps 4-5 read 176.5/177.0 and 175.6/177.3; dispatch identical by construction (t=1 dp4a) |
| gen e2e wall s | med 1.2353 | med 1.1758 | -4.8% |

Confirms the iq-direct §5 stale-verdict cell (4180.1 vs 5598.5 x3 there; 4023.4 vs
5355.8 x5 here — different session/thermal window, same zero-overlap verdict class).

## 3. Per-model arms

**KAT-Coder IQ4_XS (x3 interleaved):** pp2048 2972.9 -> **3886.3 (+30.7%)**; gen512
prefill 2029.4 -> **2987.7 (+47.2%)**; decode 191.48 vs 191.54 (flat); e2e 512+128
0.920 -> 0.840 s. Token shas: mode3 `e5d59ecedc57aa7d` == the kquant-lane mmq anchor
3/3, mode2 `9102ffd0b8241a65` == the iq-direct F16G=2 anchor 3/3.

**Ornith-35B Q4_K_M (x3 interleaved):** pp2048 4843.8 vs 4834.8 (-0.2%, flat); gen512
prefill 3430.9 vs 3410.9 (-0.6%, flat); decode 203.96 vs 205.54 (flat). Its k-quant
expert layers are MMA-rejected, so mode 3 already admitted them to sk — mode 2 changes
nothing: token sha `c0c12c3b350dc7f5` (the o35b anchor) BOTH arms 6/6. No regression.

**g26 (gemma-4-26B a4b Q4_0, gelu MoE, x3 + x2):** pp2048 6735.9 vs 6743.5 (+0.1%,
flat). board-2048-prompt gen (the g26-decode-20260801 known-green gate prompt, x2 per
arm): argmax MATCH every run, decode flat (191.5 vs 191.2), token sha
`84a47adb88ece119` IDENTICAL across arms 4/4 — the naked gemma door stays closed under
the flip, dispatch-unchanged proven at the bit level.

**FOUND IN PASSING (pre-existing, NOT flip-related):** g26 run-gen on the pp512.txt
prompt FAILS the prefill-vs-decode argmax gate on the MERGE HEAD itself — both arms,
bit-identical logits: `prefill argmax=236829 decode argmax=236755 logit maxdiff=4.113e0
MISMATCH` then panic `decode-step diverges from prefill — cache threading bug`
(`[gate] prefill: l[236829]=18.8855 l[236755]=18.6357 | decode: l[236829]=18.6457
l[236755]=19.0074` — identical in all 8 logs `g26-gen-r*-mode*.log` +
`battery-gen-g26-rep1.log`, rc=101). The g26 gate batteries
(research/g26-decode-20260801) ran depth-1736/board-2048/docs prompts, never pp512.txt —
this (model, prompt) cell was never gated. Margin ~0.25/0.36 both sides = the accepted
cross-config FP-composition near-tie class on a previously-undrawn prompt, but the
run-gen gate treats prefill-vs-decode divergence as fatal by design. Owner call whether
to recalibrate that gate for gemma; nothing in this lane moved it (bit-identical across
arms both binaries).

## 4. Exactness battery under the new naked default (all green)

All runs naked on the flip binary (naked = candidate default), `gates.jsonl` +
`battery-*.log`:

- **kernel-check q35**: rc=0, **0 FAIL / 382 OK** (`battery-kernel-check.log`) — the
  f16g-kq-direct iq4_xs/iq3_s real-weight subcases live in this battery.
- **run-gen argmax + token shas**: q35 MATCH x3 sha `e94b6553fde7b9a0` == the mode-2-class
  anchor (research/sk-tail-form + iq-direct F16G=2 stream — naked now IS that stream);
  o35b MATCH x2 sha `c0c12c3b350dc7f5` (UNCHANGED — dispatch identity); KAT MATCH x2 sha
  `9102ffd0b8241a65` (the F16G=2 anchor); g26 board-2048 MATCH x2/arm (§3).
- **decode-batch gates 1/2/3** (q35 + q9j 9B dense judge): config mode ALL GREEN both
  models; strict mode ALL GREEN both models under the equalized-composition protocol
  (`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 --batch 4 --mode strict`, worst draws
  `MEMRA_GATE_SEED` q35=16 / q9j=0 — gate1-recal-20260802 + validate-h100.sh protocol;
  `battery-decode-batch-*-strict-equalized.log`). NOTE: the first strict invocation ran
  NAKED strict (no equalization) — out of protocol; its expected FP-composition-gap
  FAILs are kept as `battery-decode-batch-*-strict-MISFIRE.log`, superseded by the
  equalized re-runs. The in-binary F16G=0 pin held (gate immune to the flip, as
  designed).
- **prime-batch-gate --carried**: q35 + q9j + o35b + KAT — **ALL GREEN x4** (cross-request
  concat prime + carried continuation under the new naked prime config).
- **serve greedy c1-vs-c16** (memra-server naked, 16 prompts, 96 max_tokens, temp 0 seed
  0, fresh refs): q35 **16/16 PASS**, o35b **16/16 PASS** (`greedy-hash-*.jsonl`).
- **run-spec K=1..8 self-consistency** (own-trim drafters, p2 prompt, NGEN=64):
  q35 **PASS x8**, o35b **PASS x8** (`battery-spec-*.log`).
- **graph gates** (q35 + o35b): decode-dc-gate PASS (256 steps BIT-IDENTICAL, 4 buckets),
  graph-decode-gate PASS (256 steps BIT-IDENTICAL, 2 captures), graph-session-gate ALL
  GREEN (session prime = the new f16g path feeding capture; q35 173.3 vs eager 170.8,
  o35b 185.0 vs 184.5) — the SLRU-graph precedent check.

## 5. Board re-cells

Only the q35 speculative row moves (spec gen rides primed state -> acceptance shifts with
the prime numeric class). Plain q35 decode rows: no move — decode dispatch untouched;
session reads at 512 ctx 184.7-189.1 tok/s naked mode-2 (single runs, packed-session
thermal, direction ≥ the published 178.2 cold-start row; not a re-cell protocol match, so
the row stands). Ornith/KAT are not board rows; their hand-written README section numbers
are re-stated from §3 where re-measured.

q35 spec row re-pair: `run-row-repair.sh` — the exact q35-spec-repair-20260802 protocol
(memra run-spec K=2 NGEN=256 naked; llama b9837 c73069749 self-MTP at its per-class
--spec-draft-n-max optimum 3/2/4, re-swept same-day same-build by that lane; N=3 medians,
engines interleaved per rep, one session). Results in `row-repair.jsonl` +
`row-repair-console.log`; board table below filled post-run.

Results (N=3 medians per arm per cell, interleaved per rep, temps 61-71 C, memra
acceptance rep-stable to the decimal, self-consistency PASS every run):

| col | memra runs | med | was | llama runs (n-max) | med | was | ratio old -> new |
|---|---|---|---|---|---|---|---|
| p1 short-code | 300.10/303.99/302.44 (acc 83.9%) | **302.4** | 305.4 | 225.75/239.20/234.66 @3 | **234.7** | 251.4 | 1.21x -> **1.29x** |
| p2 medium-code | 252.41/253.03/254.29 (acc 67.9%) | **253.0** | 241.9 | 196.46/207.22/208.76 @2 | **207.2** | 217.5 | 1.11x -> **1.22x** |
| p3 long-agentic | 270.29/270.72/271.91 (acc 84.7%) | **270.7** | 275.6 | 228.72/237.28/235.77 @4 | **235.8** | 246.5 | 1.12x -> **1.15x** |

Reading: memra p2 +4.6% (the mode-2 prime numeric class lifted p2 acceptance to 67.9%,
rep-stable), p1 -1.0% / p3 -1.8% (window drift; acceptance 83.9/84.7 vs the morning
84.2-class). The llama arm read 5-7% below its morning cells at the same build + same
per-class n-max — cross-session llama-to-llama comparisons are clock-drift-invalid
(LAW 1); the published pair is this session's interleaved memra-vs-llama. All three
columns stay above the 1.05 bold threshold. Board updated:
`research/tune-data/current-board.json` + README/perf-card regenerated in the same
commit; rig5090.jsonl row appended.

## Files

`run-headline.sh` (q35|o35b|kat|g26), `run-battery.sh`
(kc|gen|dbatch|pbatch|serve|spec|graph), `run-followup.sh` (strict|g26gen),
`run-row-repair.sh`; `headline.jsonl`, `gates.jsonl`, `row-repair.jsonl`,
`token-hashes.log`; consoles `headline-console.log`, `battery-console.log`,
`followup-console.log`, `row-repair-console.log`; per-run logs `q35-*`, `o35b-*`,
`kat-*`, `g26-*`, `battery-*`, `memra-*-rep*.log`, `llama-*`; `bin-preflip/`
(GIT_SHA of the mode3 arm binary; binary itself untracked).
