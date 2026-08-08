# v0.72 wave — pair-box release battery (lane/v072-battery)

Validation target: train `restructure/public-split` @ **6afc4f65** (read-only; receipts only).
Rig: AWS pair box 18.195.123.14 — 2x RTX PRO 6000 Blackwell Server 96GB, CUDA 13.2,
sm_120a auto-detected. Tree rsynced to `~/v072/memra`, built `cargo build --release`
(3m36s, 0 errors; `~/v072/build.log`). All GPU windows under `flock /tmp/memra-gpu.lock`.
Raw receipts: `raw/` (driver scripts committed beside them).

Artifacts (box):
- step35 SKU: `~/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`
  + MTP drafter `Step3.7-flash-mtp-Q8_0.gguf`
- q27: `/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
- q9: `/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (+ draft-9b-owntrim staged from the 5090 rig)
- gemma4: `gemma-4-12b-it-qat-q4_0.gguf` staged from the 5090 rig (serve-smoke arm runs, no SKIP)

## Verdicts

### Driver A — exactness core (raw/exactA-20260807T231814Z.log) rc=0
| gate | verdict |
|---|---|
| A1 kernel-check model-backed step35 IQ4_XS | ALL GREEN |
| A2 kernel-check model-backed q27 NVFP4 | ALL GREEN |
| A3 run-gen argmax step35 PP-2 (64 tok) | MATCH (argmax 6776 both paths — same token as the tick-seg baseline receipt) |
| A4 run-gen argmax q27 single-card | MATCH (argmax 1178) |
| A5 ppn-gate stages=2 step35 | BIT-IDENTICAL serial + pipelined |
| A6 ppn-gate stages=2 q27 | BIT-IDENTICAL serial + pipelined |
| A7 run-spec K=1..8 step35+MTP over PP-2 | 8/8 PASS, K=1 acceptance 14/18=77.8% — DIGIT-IDENTICAL to the pinned mtp-draft-PASS-20260806T215132Z + exact-tickseg baselines |
| A8 run-spec K=1..8 q27 single-card | 8/8 PASS, K=1 23/24=95.8% — digit-identical to pp2-spec runspec-q27-doorshut baseline |
| A9 run-spec q9 control (the pinned 82.4% receipt shape) | 8/8 PASS, K=1 14/17=**82.4%** — matches the pinned receipt exactly |

Note on "82.4%": the mission's pinned 82.4% K=1 receipt is the q9 door-shut shape
(research/pp2-hardening-20260806/logs/refusal2/sp-1-doorshut.log et al.), not step35 — step35's
pinned K=1 baseline is 77.8% (mtp-draft-PASS-20260806T215132Z). Both were re-hit exactly.

### Driver B — step35 gate family + decode-batch (raw/gatesB-20260807T232159Z.log) rc=0, ONE RED
| gate | verdict |
|---|---|
| B1 tickinv35 (naked) | PASS — EXACT at budgets 1024/513/512/256/64 + sp64/sp256/sp512 |
| B2 tickinv35c (canary teeth) | **FAIL — CANARY UNEXPECTEDLY MATCHED** (see below) |
| B3 chunkinv35 (axis 1) | PASS |
| B4 chunkinv35c (axis-1 canary) | PASS — canary broke the assertion as required |
| B5 decode-batch q9 NVFP4 config B=8 | ALL GREEN |
| B6 decode-batch q9 NVFP4 strict B=4 equalized | ALL GREEN |
| B7 decode-batch q27 NVFP4 config B=8 | ALL GREEN |
| B8 decode-batch q27 NVFP4 strict B=4 equalized | ALL GREEN |
| decode-batch Q8_0 arm | SKIP — no Q8_0 main-model artifact on this box |

**B2 detail (the tag-blocking find):** with MEMRA_PRIME_CALLLOCAL=1 every arm stayed EXACT
(raw/tickinv35c-canary-probe-raw.log), so flipping the seam no longer changes the bits and the
tickinv35 gate currently CANNOT detect its mechanism. The gate itself says: FIX THE GATE.
Likely regressor: **80f47796 (Merge lane/pp-prefill-serve, LEVER A)** — merged AFTER
lane/tick-seg (f01710ca) validated the canary teeth on the box (gate-tickseg-20260807T113933Z:
canary got=variant). Lever A rewrote the same SWA prefill arm (windowed hd128 FA stamp +
BK=32 aligned view offset, 8b425742/5c523d5e), and this is the EXACT failure class Lever A
already hit on its own axis-1 canary: 82b216b8 "the canary seam went inert under the FA
default" restored both halves of MEMRA_STEP35_SWA_TKV after chunkinv35c matched unexpectedly.
The tick-axis seam (MEMRA_PRIME_CALLLOCAL, hybrid_forward.rs:486 — per-call seq_end) got no
such rework: under the new windowed-FA default, per-call vs request-extent seq_end apparently
selects between bitwise-identical kernels on this prompt/budget set, so the seam is inert.
tickinv35 (the real assertion) is GREEN — the SHIPPED default is tick-invariant; what died is
the gate's teeth, i.e. its ability to catch a future regression of that invariant.

### Driver C — #87 crash gate + serve-smoke (raw/serveC-20260807T232545Z.log) rc=0
| gate | verdict |
|---|---|
| C1 ppspec bit-identity dev01 (q9, stages=2, ts=2,5,9 x3) | rc=0 (0 failing arms) |
| C2 ppspec bit-identity dev10 | rc=0 |
| C3 #87 crash gate: spec+PP-2 serve (naked lift, MEMRA_SPEC_GATE=0), c=2x8 + c=4x100 + c=8x104 | **212/212 ok, 0 err, 0 crashes**; server log clean (no sentinel/illegal/panic/CUDA_ERROR); [spec-acc] live throughout — the fences hold on the release build |
| C4 serve-smoke FULL battery q9+draft, GEMMA_MODEL staged | **0 failed** (47 ok-checks incl. cache-metering, spec==plain, sampled truncation matrix, gemma4 arm: default-scheduler clean + thinking separation + zero panics). No SKIPs — the gemma artifact was staged to the box for this run |

### Drivers D+E — the crash gate's throughput anomaly, attributed (raw/perfD-*, raw/perfE-*)
C3 passed its correctness bar but read agg **17.5 tok/s** where the pp2spec lane's identical
gate shape read **112.2** (same box, same tool, same model, receipts
research/pp2spec-crash-20260807/raw/gates-final/crash-points.jsonl). Localization, one lock
hold each, train binary only:
- D (placement A/B, spec forced): dev10 17.5 / dev01 17.5 (x2 each, identical) — placement-independent.
- D (defaults c=4): 186.3 (spec demoted by #89 — healthy); door-shut c=4 smoke: **547.3** (matches the lane's 548 receipt — single-card serving unregressed).
- E1 (ENGINE run-spec q9 over PP-2 dev10, lane gate shape): generate 115.0, spec K=1 **164.7 tok/s** — the engine spec+PP-2 path is FAST.
- E2 (naked defaults PP-2 c=1 — #89 admits spec here): **17.5** agg, p50 5.65s. E3 (spec-off control): **223.2** (digit-match to the lane's N c=1 receipt 223.3).

So: engine fine, spec-off serving fine, single-card serving fine — the regression is the
SERVING-layer spec path over PP-2 only, worth ~6.4x vs the lane binary (112.5 -> 17.5) and
12.7x vs spec-off. It is USER-FACING on a naked PP-2 boot: #89's LOW=2 admits spec at c<=2,
so default c=1 PP-2 traffic gets 17.5 where the lane shipped 112.5. Likely regressor:
**5f27c55c (Merge lane/cx-bare503 round 2: per-PP-device residency + the serving worker
follows the PP primary device — 768fcffa/238beae0)** — it is the only worker-touching merge
after the lane's perf receipts, and the boot line changed accordingly
("Engine ready (device=1, ...)" on dev10; the lane binary booted without a device pin).
Not debugged further per the battery contract.

## What blocks the tag (owner calls)
1. **B2 tickinv35c canary inert** — the tickinv35 gate cannot currently fail; a future
   regression of the tick-invariance contract would ship green. Fix shape is known and small
   (the 82b216b8 precedent: make MEMRA_PRIME_CALLLOCAL restore the full pre-fix arithmetic
   under the Lever A windowed-FA default). Likely regressor: 80f47796 (Lever A), which
   post-dates the canary's last green teeth (gate-tickseg-20260807T113933Z).
2. **Spec+PP-2 serving 112.5 -> 17.5 tok/s** — correctness green (crash gate 212/212,
   bit-identity, run-spec exact), but the naked PP-2 default at c<=2 now serves at 8% of the
   lane's receipt. Likely regressor: 5f27c55c. Either fix, or gate spec off on PP-2
   (T_LOW=0 placement-aware), or tag with the regression documented — owner call.

Everything else on the battery list is GREEN. Box left clean (0 MiB both cards, lock released).
