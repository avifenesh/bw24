# prime-gate-coverage: gap #46 closed — batched-prime first-token drift measured, classified FP, gated (2026-08-02)

Lane `lane/prime-gate-coverage` (from `restructure/public-split` 5f15f838). Rig: RTX 5090
Laptop 24463 MiB, every GPU run under `flock /tmp/gpu5090.lock` (one co-lane shares; the
allowlisted `llama-server --embedding` 332 MiB is co-resident throughout). Single runs
unless stated — this lane measures bit-level agreement, not throughput; every comparison
is within one process where noted DETERMINISM x2 was also checked.

## 0. The gap (from research/residency-cap-20260802 §4)

run-gen's argmax gate compares `forward_last` (prefill config) vs the tokenwise
`decode_step` loop (m=1). Real generation — `generate`/`generate_with`, and therefore
serving — seeds its first token from **`prime_cache`, a THIRD numeric config the gate
never covered**. On the pp512 probe prompt, naked Qwen3.6-35B greedy-emits `"\n"` + EOS
at 2 tokens while the gate is green (argmax=365 both sides).

## 1. Repro at HEAD (post fast-router merge)

`repro-q35-pp512.log` (prime-gate, q35 IQ4_XS, pp512 probe, T=512):

```
tw=365 (margin 0.6146)  bp=198 (margin 0.6054)  fl=365  maxdiff=2.4287e0  det=BIT-IDENTICAL
stream(16): DIVERGED at step 0   tw_eos=None  bp_eos=Some(2)
```

The flip survives the m-dependent-router fix and the fast-router merge — it is a separate
family (cross-CONFIG, prime GEMMs at m=T vs decode GEMV at m=1), as the mission framed it.
Anchor: run-gen's own ACCEPTED forward-vs-decode drift on this exact prompt/model is
maxdiff **1.386** MATCH (residency lane `qwen-pre-rep1.log`) — the tw-vs-bp 2.43 is <2x
the already-accepted cross-config scale.

## 2. Differential: FP-composition class, not a defect

Concat-lane methodology (research/concat-prime-exact-20260802) applied to the SOLO
batched-vs-tokenwise pair:

1. **Determinism** — batched prime x2 bit-identical, on every prompt of the 144-prompt
   sweep (`det=BIT-IDENTICAL`, 144/144). Not nondeterminism.
2. **Content/causality razor** (`causal-q35-pp512.log`, new `concat-prime-probe causal`
   mode): prime(P) vs prime(P+S), P = pp512, S = 79 tokens.
   - c1 chunk boundary at |P|: rows of P **BIT-IDENTICAL** (hid maxdiff 0.0) — later
     content is invisible backwards across a chunk boundary. No leakage defect.
   - c2 monolithic (m=512 vs m=591): **also BIT-IDENTICAL** — the prefill GEMM family is
     m-INVARIANT at these shapes (consistent with the concat lane's post-fix allw census).
     The entire tw-vs-bp divergence is therefore the prefill-KERNEL-family vs
     decode-KERNEL-family switch, not m-sensitivity inside the prime.
3. **Per-position profile** (`twpos-q35-pp512.log`, new `twpos` mode): 28/512 positions
   flip argmax, SCATTERED (no boundary clustering — the prime was monolithic), flip
   margins 0.003–0.91, maxdiff already 5.1e-1 at position 0 (pure per-op config drift,
   zero history), max 9.0e0 at pos 302, non-monotonic in position. Positions that agree
   hold margins 3.5–5.7. Scatter + near-tie-only flips = the FP class amplified by
   discontinuous top-k routing (the t2probe precedent), not structure.
4. **Config-family scan** (`chunkscan-c{496,384,256,128,64,32}-q35.log`): first token
   OSCILLATES 365/198/365/198/365/198 across chunk configs, bp margins 0.03–0.51. A
   defect gives a consistent wrong answer; a near-tie under config roulette flips back
   and forth. Also `knob-MEMRA_FAST-q35.log`: Stage-A oracle kernels → MATCH at 365 both
   sides (maxdiff still 1.0e0); `knob-MEMRA_F16OUT-q35.log`: bit-identical to naked (not
   the f16out lever).
5. **No config is privileged**: across the sweep's 10 real flips, forward_last agrees
   with the BATCHED prime in 8/10 — the tokenwise "oracle" config is usually the outlier.

**Verdict: accepted cross-config FP-composition drift (near-tie roulette), same law as
forward-vs-decode and decode-batch gate1 — NOT a defect.** The distinguishing negatives
(non-determinism, content leakage, boundary structure, wide-margin flips, monotonic or
privileged-direction divergence) all came back clean.

## 3. Coverage: how widespread is first-token config sensitivity?

`run-coverage.sh` — six models x (8 mixed raw prompts: 512-tok code probe, prose,
instruction, multilingual, numbers, repetition, boundary-length, code one-liner
[`prompts-mixed.txt`] + 16 chat-templated questions [concat lane's `prompts16.txt`]),
24 prompts per model, 144 total. Per prompt: tokenwise vs batched-prime last-position
logits, argmax + margins + maxdiff + determinism + 16-step greedy streams.
Raw: `coverage-*-{raw,chat}.{log,jsonl}` (shipped bounds) and `calib0-*` (the
provisional-bounds calibration pass, identical measurements).

| model | quant | raw flips | chat flips | flip margins | worst maxdiff |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (probe) | IQ4_XS MoE | 1/8 | 1/16 | 0.615, 0.125 | 2.43 |
| Qwen3.5-9B judge | Q8_0 dense | 0/8 | 0/16 | — | 1.00 |
| Ornith-1.0-9B | Q8_0 dense | 0/8 | 0/16 | — | 1.13 |
| Ornith-1.0-35B | Q4_K_M MoE | 1/8 | 2/16 | 0.566, 0.439, 0.083 | 1.83 |
| KAT-Coder-V2.5 | IQ4_XS MoE | 0/8 | 2/16 | 0.014, 0.077 | 3.13 |
| Gemma-4 12B | QAT Q4_0 dense | 2-3/8 (see below) | 0/16 | 0.140, 0.317, 0.697 | 5.54 |

- **10/144 first tokens flip (6.9%)** on the calibration pass; every flip at tokenwise
  margin <= 0.70.
- Dense Q8_0 (the H100 fleet class): **0/48** — first-token stability clean.
- MoE 35Bs flip 2-3 of 24 each (routing amplifies per-op drift: maxdiff 0.5 at pos 0
  grows to 2-9 through expert flips); gemma Q4_0 flips on its larger logit scale.
- Early-EOS consequence appeared ONLY on the q35 pp512 probe (bp stream EOS at step 2).
- Streams: a first-token MATCH does not guarantee stream identity for 16 steps (later
  near-ties can flip within a config too — decode-batch gate1's known class), but every
  observed EARLY divergence traces to a near-tie.

### Cross-PROCESS config selection (observed once in the 144-row double pass)

The whole sweep ran twice (calibration bounds, then shipped bounds — same measurements
expected). 143/144 rows reproduced bit-for-bit in every reported field. The one mover:
g12 raw prompt 0 (pp512, T=560, tw margin 0.1399) — `bp` and `fl` flipped TOGETHER
3651 -> 11129 (= tw) between passes, tokenwise side unchanged; pass 1 ran beside a 17.3 GB
co-lane process, pass 2 beside a different co-lane state. Mechanism: the gemma f16/fp8
prefill lanes select their cuBLASLt algorithm per process via
`cublasLtMatmulAlgoGetHeuristic` (`cu/f16_prefill.cu:315`, `cu/fp8_prefill.cu:145`) —
heuristic choice is environment-sensitive, so the prefill-family FP composition can move
per process while staying bit-deterministic within one. Same near-tie roulette class,
one more legal config axis; both flip directions stayed inside the calibrated bounds.

## 4. What shipped

1. **`prime-gate` binary** (`crates/memra-engine/src/bin/prime_gate.rs`): the dedicated
   multi-prompt battery — tw/bp/fl argmax + margins + maxdiff + bp determinism + greedy
   streams + EOS steps, JSONL receipts, verdict per calibrated bounds; exit non-zero on
   STRUCTURED or non-determinism (`--strict` also fails near-tie flips).
2. **run-gen gate line** (#46 closure): text prompts >= 16 tokens now print
   `batched-prime argmax=... tokenwise argmax=... maxdiff=... {MATCH|FLIP-NEARTIE|MISMATCH-STRUCTURED}`
   after the existing gate; STRUCTURED fails the run. Skipped when generation will take
   the tokenwise arm anyway (MEMRA_PRIME_TOKENWISE / frozen CPU-expert serving);
   `MEMRA_PRIME_GATE=0` diagnostics seam.
3. **Shared verdict** `forward::prime_gate_verdict` with calibrated bounds
   `MEMRA_PRIME_GATE_MAXDIFF=8.0` (legal drift measured up to 5.5 on gemma; defects land
   decades above) and `MEMRA_PRIME_GATE_MARGIN=1.0` (legal flips measured <= 0.70;
   >= 1.43x headroom). Recalibrate when kernels move — the H100 stale-verdict law.
4. **Probe modes** `twpos` and `causal` in `concat-prime-probe` (the solo differential
   toolkit; causal's c1 arm is qwen-chunked-prime-specific).
5. **Docs, honestly worded**: docs/SERVING.md "First-token cross-config drift" section
   (serving primes batched; ~7% of near-tie first tokens can differ from the tokenwise
   oracle; dense fleet class clean; MEMRA_PRIME_TOKENWISE pins), FLAGS.md §3/§4 rows,
   CONTRIBUTING.md gate list.
6. **Serving defaults: UNCHANGED.** Batched prime stays the default — the flip class is
   within the cross-config law the engine already accepts elsewhere (forward-vs-decode,
   decode-batch config mode, gate1 near-tie roulette), no config is more "correct" (fl
   sides with bp 8/10), and the tokenwise pin costs 23x TTFT at 6k. The class is now
   MEASURED, BOUNDED (gate), and WRITTEN (docs) instead of silent.

## 5. Batteries (shipped build)

`run-battery.sh` → `battery-*.log`:

- kernel-check: **ALL GREEN**.
- run-gen: existing prefill/decode argmax line **MATCH on all six models**; the new
  batched-prime line MATCH on five, `FLIP-NEARTIE` REPORTED (non-fatal, exit 0) on the
  q35 pp512 probe — exactly the classified behavior.
- decode-batch-gate: q35 **ALL GREEN**. q9j: gate1 seed 0 diverges at step 1 → FAIL,
  gate2 (serving isolation, bit-checked) PASS — **PRE-EXISTING on restructure/public-split**:
  the pristine 5f15f838 binary (lane changes stashed, rebuilt) reproduces the identical
  signature bit-for-bit (`battery-decode-batch-q9j-BASE.log` vs
  `battery-decode-batch-q9j.log`; gate1 lines diff-identical). Untouched by this lane
  (no decode-path edits); logged here as an out-of-band public-split item — same
  handoff class as the residency lane's original #46 note. Per gate1's own calibration
  law (MEMRA_GATE_SEED doc) its seed set may need a re-sweep on this branch's cores.
- prime-batch-gate (fresh + --carried): **ALL GREEN** (q35 + q9j) — cross-request concat
  prime untouched.
- run-spec: **K=1..8 self-consistency PASS** (q35, own-trim drafter, board prompt).
- tools/local-ci.sh now carries a prime-gate leg (q35 mixed prompts, --steps 0) so the
  gate lives INSIDE the battery; leg verified exit-0 with the near-tie report.

## 6. Files

`run-coverage.sh`, `run-battery.sh`, `prompts-mixed.txt`, `repro-q35-pp512.log`,
`twpos-q35-pp512.log`, `causal-q35-pp512.log`, `chunkscan-c*-q35.log`,
`knob-MEMRA_FAST-q35.log`, `knob-MEMRA_F16OUT-q35.log`,
`coverage-{q35,q9j,o9b,o35b,kat,g12}-{raw,chat}.{log,jsonl}` (shipped bounds),
`calib0-coverage-*` (calibration pass, provisional bounds), `coverage-console.log`,
`battery-*.log` (incl. `battery-decode-batch-q9j-BASE.log`, the pristine-base
pre-existence receipt), `battery-console.log`.
