# Serve-path phase 2 — closing the serve c=1 gap

Rig: local RTX 5090 Laptop (sm_120a, **82 SM**), driver-resident daily model down (owner call),
`flock /tmp/gpu5090.lock` held for every measurement. Models: `q9` =
Qwen3.5-9B-NVFP4-MTP, `q27` = Qwen3.6-27B-NVFP4-Q4_K_M-mtp. Branch `lane/serve-path-p2` off
train HEAD `70ce5a0f`.

Spec: `research/q27-deepdive-20260805/PHASE2-SPEC.md`. Phase-1 gap being closed: serve c=1
**−11.74%** vs naked on the PRO 6000 (188 SM).

Harness law obeyed throughout: `memra-server` + `tools/load-serve.py`, **never**
`decode-batch-bench` (the spec's trap note — it overstates batched cost ~35% via a host argmax
over `n_vocab`). All A/B is **order-paired**: arms interleaved within a rep, order alternated
across reps, server restarted per arm, warmup request discarded. `step_p50_ms` from `/metrics`
is the **decode-only** comparator; `agg_tok_s` folds prefill in.

---

## Verdicts

| | hypothesis | verdict |
|---|---|---|
| **H3** | `b_n==1` fast path | **LANDED** — +8.33% (q9) / +5.19% (q27) decode-only at c=1, 5/5 both |
| **H1** | worker graph door | **REFUTED, post-H3** — same mechanism as H3, and H3 wins it for free |

### The spec's premise was wrong on both hypotheses

**H1's premise** ("the serve path has *no CUDA-graph door at all*") is factually false. The
worker has had one since round 35: `worker.rs` phase **(a0)**, lines 1222-1362, promotes a lone
cold greedy interactive session to `GraphSession` replay via `graph_session_from_cache_masked`,
degrading back to batched-eager the moment concurrency arrives. It simply never fired at the
phase-1 measured config, which failed two of its gates: `s.sampler.is_greedy()` (phase 1 ran
temp=0.7) and `s.budget >= gs_min` where `MEMRA_GS_MIN` defaults to **384** (phase 1 ran
max_tokens=128).

**H3's premise** ("b_n=1 dispatches the dense-FFN gate+up pair through `matmul_pre`, so lever
1's fused arm never fires") mis-identifies the mechanism. `matmul_pre` at `m==1` **already**
routes to `qmatvec_mmvq` — the m=1 *kernel family* was never bypassed. What `decode_step_batch`
lacks is the m=1 **fusion chain**:

- cross-layer `add_rms_norm_q8_1` (3 launches → 1),
- the fused SwiGLU epilogue `silu_mul_scaled_q8_1` (folds `ffn_down`'s quantize into its
  producer) and, with it, `matmul_pre_dual_noscale`'s gate+up pair — i.e. phase-1 **lever 1**.

So the fix is not a batched twin per lever. It routes `b_n==1` through `decode_layers_eager`
**verbatim** — the same trunk `decode_step_h` uses — keeping the batched path's own serving
epilogue (grammar-mask park, device sample, lean-logits park). Every future m=1 lever now
reaches serve for free. `decode_layers_eager` widened `fn` → `pub(crate)` for the share.

---

## H3 — the number

q9 NVFP4-MTP, temp=0.7 (the phase-1 config), mt=128, `MEMRA_SERVE_SPEC=0`, **N=5** order-paired.
Arm A = `MEMRA_SERVE_B1FAST=0` (shipped batched body), arm B = `=1` (H3).

| metric | per-pair delta | A | B | pair mean | wins |
|---|---|---|---|---|---|
| c=1 decode-only `step_p50` | +8.15 +9.13 +7.90 +8.62 +7.85 | 123.68 | 134.06 | **+8.33%** | 5/5 |
| c=1 e2e `agg_tok_s` | +7.05 +7.55 +7.00 +7.04 +7.01 | 123.13 | 131.74 | **+7.13%** | 5/5 |
| c=8 decode-only | −0.41 +0.20 −0.02 +0.18 +0.03 | 63.89 | 63.88 | −0.00% | 3/5 |
| c=8 e2e | −0.57 +0.12 −0.09 +0.06 −0.32 | 505.57 | 505.09 | −0.16% | 2/5 |

q27 NVFP4-MTP (the primary target), same protocol, N=5:

| metric | per-pair delta | A | B | pair mean | wins |
|---|---|---|---|---|---|
| c=1 decode-only | +4.41 +5.67 +5.55 +5.10 +5.25 | 43.56 | 45.79 | **+5.19%** | 5/5 |
| c=1 e2e | +4.55 +5.64 +5.41 +5.26 +5.33 | 43.46 | 45.76 | **+5.24%** | 5/5 |
| c=8 decode-only | −0.61 −0.10 −0.12 +0.00 −0.07 | 22.72 | 22.70 | −0.18% | 1/5 |
| c=8 e2e | −0.58 −0.12 −0.16 −0.01 −0.08 | 179.47 | 179.26 | −0.19% | 0/5 |

**c=8 no-regression: PASS.** q9 is flat with per-pair deltas straddling zero. q27's −0.18% is
carried by a single r1 outlier (−0.61) with the remaining four at ≈−0.1%; the fast path only
fires at `b_n==1`, which a c=8 run reaches only as the batch drains at the tail. Neither is a
saturation-throughput regression.

Same-board naked denominator (q9, this rig, N=3): `run-gen` n=128 = 134.83 / 134.51 / 133.96
tok/s. Serve c=1 decode-only went **123.7 → 134.1**, i.e. from ~8% below the naked board to
level with it — the class of gap phase 1 measured is closed on this rig.

---

## H1 — refuted, and *why* (the interesting part)

Pre-H3 the graph door was a real win, and the shipped `GS_MIN=384` was miscalibrated. Measured
crossover (q9, N=3, order-paired, `MEMRA_SERVE_SPEC=0`), arm A = `GS_MIN=100000` (shut) vs
arm B = `GS_MIN=1` (open), `agg_tok_s`:

| max_tokens | per-pair delta | shut | open | pair mean | wins |
|---|---|---|---|---|---|
| 32 | −11.56 −10.96 −10.38 | 121.21 | 108.63 | −10.97% | 0/3 |
| 64 | −4.99 −2.77 −3.26 | 121.44 | 117.87 | −3.67% | 0/3 |
| 128 | −0.79 +0.30 −0.17 | 121.84 | 122.19 | −0.22% | 1/3 |
| 256 | +2.15 +3.54 +3.22 | 121.56 | 125.86 | +2.97% | 3/3 |
| 512 | +3.73 +4.48 +4.56 | 121.02 | 126.44 | +4.26% | 3/3 |

Crossover sat between 128 and 256 — so 384 was too conservative, honest key ~256 on 82 SM.

**Then H3 landed and that entire sweep went stale** (the H100 lane's law: thresholds calibrated
on old kernels must be re-swept when the code under them moves). Re-swept on top of H3:

| max_tokens | per-pair delta | shut | open | pair mean | wins |
|---|---|---|---|---|---|
| 128 | −8.54 −5.88 −6.38 | 129.78 | 122.15 | −6.93% | 0/3 |
| 256 | −5.29 −3.22 −3.40 | 129.67 | 125.50 | −3.97% | 0/3 |
| 512 | −3.33 −2.19 −2.26 | 129.06 | 126.23 | −2.60% | 0/3 |
| 1024 | −1.47 −1.21 −1.05 | 128.50 | 126.94 | −1.24% | 0/3 |

0/3 at **every** length, out to 1024. The arm decomposition says why — decode-only `step_p50`
tok/s per rep (server-lifetime p50, so each blob blends that arm's mt list):

```
             pre-H3                post-H3
EAGER (shut) 122.8/121.4/121.3     129.0/128.7/124.7
GRAPH (open) 130.8/130.7/130.8     128.8/128.5/125.3
```

The graph arm **did not change** (130.8 → 128.8, within run noise). The eager arm rose to meet
it. **H1 and H3 target the same cost — per-step launch overhead.** The graph door amortizes it
by replaying a captured launch sequence; H3 removes it outright by fusing the launches. Once
fused, there is nothing left for the capture to buy, and the ~340 ms one-time
capture+snapshot becomes pure tax.

That model is quantitatively exact. If the only residual difference is the capture, then
`delta% = −capture / total_gen_time`, which must halve as gen length doubles:

| max_tokens | predicted `−0.340s / (4 × mt × 7.76ms)` | observed |
|---|---|---|
| 128 | −8.56% | −6.93% |
| 256 | −4.28% | −3.97% |
| 512 | −2.14% | −2.60% |
| 1024 | −1.07% | −1.24% |

Independent corroboration from a gate that knows nothing about this lane:
`graph-session-gate` now self-reports **`perf: session 125.2 tok/s vs eager 128.5 tok/s
(−2.6%)`** — the graph arm losing to eager on its own gate, on the same tree.

**Action: none.** `MEMRA_GS_MIN=384` stays exactly as shipped — the door is already
(correctly) shut at every length where it now loses, and it is not on by default anywhere it
would hurt. No flag was added and none is killed: `MEMRA_SERVE_GS` remains a legitimate
machine-config/rollback seam, and the 384 key's *estimate* provenance is now replaced by a
measured verdict. Lowering it, which the pre-H3 sweep would have justified, would have been a
regression. Recorded so nobody re-derives the stale +2.97%/+4.26% crossover and "fixes" the key.

Not re-tested on 188 SM. Per the rig-divergence law (phase-1 lever 2's key does not transfer
from 188 → 82 SM), this refutation is an **82-SM** verdict; the PRO 6000 keeps its own.

---

## Exactness

**H3 is bit-identical to `decode_step_h`** — which is the direction the lever wants: a solo
serve request now computes exactly what `run-gen` computes for the same prompt.

| gate | H3 ON | H3 OFF |
|---|---|---|
| `decode-batch-gate` **strict** gate1 (bit-identity vs `decode_step_h`), q9 | **PASS** | **FAIL** maxdiff 1.591e-1 |
| `decode-batch-gate` strict B=4 equalized, Q8_0 (`ornith-9b`) | **ALL GREEN** | — |
| `decode-batch-gate` config B=8, q9 NVFP4 | **ALL GREEN** | ALL GREEN |
| `decode-batch-gate` config B=8, Q8_0 | **ALL GREEN** | — |

Config-mode gate1 also **improved**: 6/6 seeds now agree for all 32 steps (pre-H3 the q9 draw
flipped at step 1 — the accepted near-tie dice, now gone because B=1 rides the reference's own
program).

**Stream identity through the real server** (`scripts/stream-identity.sh`, native shape so
`/v1/completions` returns raw ids): greedy **150 ids IDENTICAL to the `run-gen` oracle** on
both arms *and* cross-arm; seeded-sampled (temp=0.7 seed=12345) identical cross-arm and
reproducible within arm. The accepted decode-config FP gap stays **sub-token** here, so the
serve path changed speed and not tokens — the spec's hard requirement.

### `decode-batch-gate` had to be repaired to keep its teeth

gate2 and gate3b build their "isolated" reference by calling `decode_step_batch` at **B=1**.
Left alone, that reference would have followed B=1 onto the fused trunk, and their bit/stream
checks would have silently degraded from *"batchmates must not perturb your logits"* (their
real teeth) into a cross-config FP comparison — the class gate1's config mode already tolerates
by design. They now **pin** the reference arm to the batched body through a new
`HybridModel::set_b1_fast` seam, keeping jurisdiction exactly where it was: the batched m≥2
body. The gate prints which arm it pinned.

The seam reads through an `AtomicU8` memo rather than a `OnceLock` because the gate flips it
**between gates in-process** — gate1 needs the fast path ON to prove bit-identity, gate2 needs
it OFF. A latch-once read would bake whichever gate ran first and the gate could never test
both sides.

### Pre-existing find: strict-mode equalization does not cover NVFP4

`--mode strict` **FAILS on q9 NVFP4 at the unmodified train HEAD** — proven by stashing this
lane, rebuilding pristine, and re-running: gate1 bit-diff maxdiff **1.639e-1 @ step 2**, gate2
seq 3 token divergence **@ step 8**, the identical signature seen with the lane applied
(`logs/dbg-strict-b4-TRAINHEAD.log`). Cause: the equalizing env `MEMRA_MMVQ=0
MEMRA_NO_FUSE_NORMQ=1` is **Q8/dp4a-shaped**; on an NVFP4 model the fused arms survive it, so
the two configs are never actually equalized. The same strict battery is **ALL GREEN on Q8_0**,
where the env does bite.

Not caused by this lane, and not a regression — a **coverage gap**. Documented in the gate
header so a future NVFP4 strict FAIL is not misread as a break. Equalizing the NVFP4 arms is
open work.

---

## Full battery (post-H3 tree, on-box)

| gate | result |
|---|---|
| `kernel-check` | **ALL GREEN** (untouched by this lane) |
| `run-gen` argmax, q9 n=128 | **MATCH** (prefill 271 == decode 271) |
| `run-spec` K=1..8 | **8/8 SELF-CONSISTENCY PASS** |
| `decode-batch-gate` config B=8 (q9 NVFP4 + Q8_0) | ALL GREEN both |
| `decode-batch-gate` strict B=4 equalized (Q8_0) | ALL GREEN |
| `graph-decode-gate` | PASS — 256 steps bit-identical, buckets=30 captures=2 |
| `graph-session-gate` | ALL GREEN |
| `decode-dc-gate` | PASS — 256 steps bit-identical, buckets=4 |
| `serve-smoke` | **0 failed** (16 ok) |
| `serve-st-gate` | **0 failed** — incl. CLI-vs-server greedy token identity |
| stream identity (greedy 150 + seeded-sampled) | identical to oracle **and** cross-arm |

## Affinity interaction

No new interaction. Graph promotion requires `s.prefill_done && s.generated.is_empty()` and
captures **over** an already-primed cache (`graph_session_from_cache`), so a resumed/rewound
session promotes only at a clean generation start; concurrency arrival degrades via
`s.graph.take()` → `s.cache = Some(g.cache)`, and dc==eager bit-identity makes that handoff
seamless. H3 adds nothing here — it is a straight-line dispatch choice inside one tick with no
captured buffers and no cross-tick state. `serve-smoke`'s affinity assertions pass (`affinity
fired (3 rewind(s) on a rewritten history)`, `no failed rewinds`).

## Files

- `crates/memra-engine/src/decode_batch.rs` — H3 fast path + `b1_fast_on`/`set_b1_fast` seam
- `crates/memra-engine/src/decode.rs` — `decode_layers_eager` → `pub(crate)`
- `crates/memra-engine/src/bin/decode_batch_gate.rs` — reference pin + NVFP4 strict note
- `scripts/serve-ab.sh` — order-paired serve A/B driver (+ per-arm `/metrics` capture)
- `scripts/gsmin-sweep.sh` — H1 crossover sweep
- `scripts/stream-identity.sh` — the token-stream gate
- `scripts/parse.py`, `scripts/pairs.py` — parsing / order-paired analysis
- `serve-points.jsonl`, `serve-metrics.jsonl` — post-H3 raw points
- `serve-points-preH3.jsonl`, `serve-metrics-preH3.jsonl` — pre-H3 raw points (the stale H1 sweep)
- `logs/` — every per-run log, server log, and gate output behind the tables above
