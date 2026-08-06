# q8-argmax — MECHANISM NAMED: a three-way near-tie at board-2048's final position, and a one-position gate that could not tell that from corruption

Lane `lane/q8-argmax`, 2026-08-06, task #77. The standing pre-existing exactness red carried
into the v0.71.0 release battery: on the 188-SM RunPod RTX PRO 6000 Blackwell WS, the 27B Q8_0
artifact fails `run-gen`'s prefill-vs-decode argmax gate on one prompt of the fast-gate set.

## Verdict in one paragraph

The failing check is **`run-gen`'s intra-run assert** that `forward_last` (batched prefill) and
the `decode_step` tokenwise loop agree on the **last position's** argmax — one rig, one binary,
one process; not a cross-rig golden and not a fast-gate probe row. On the `board-2048` prompt
the 27B Q8_0's final-position logits carry a **three-way near-tie**: ids 332, 485 and 266 sit
inside **0.05** of each other on a scale whose median top-2 gap over the same prompt's other
positions is **1.25** (25x larger; the flip's margin 0.0307 ranks **3rd of 24** positions). The
two configs are two legitimate arithmetics whose spread at those ids is **0.2117 — 6.9x the
margin** — so they pick differently. **Verdict class: FLIP-NEARTIE** (the documented
cross-config drift class, same class as k27's), **not a numeric defect, and not a cache
threading bug** despite what the assert message said. Every mechanism hypothesis is refuted by
one-variable kills, including all fast kernels at once. **The real defect this lane found and
fixed is in the GATE, not the engine**: checking a single position makes greenness depend on
whether a prompt's last token happens to be a near-tie, and the 27B NVFP4 arm the release
battery calls GREEN carries a flip at position 2042 whose margin (**0.0184**) is *smaller* than
the failing arm's.

## The failure, quoted (reproduced on this lane's tip, 3d34b4e4)

```
prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
[gate] prefill: l[332]=12.8352 l[485]=12.8045 | decode: l[332]=12.6235 l[485]=12.7234
thread 'main' panicked at crates/memra-engine/src/bin/run_gen.rs:896:5:
decode-step diverges from prefill — cache threading bug
```

2/2 runs byte-identical to every printed digit, and byte-identical to the four release-battery
binaries (v0.71 candidate, v0.70.0, v0.69.0, 08-04 control) — **pre-existing, not a regression**.
Rig: 188 SM, driver 570.211.01, CUDA 13.1 compat. Artifact md5 `985893c8fdff2fb46095853462608c1e`,
prompt md5 `806a1d9c9d19a47fe014738912c64d0f`, run-gen md5 `4f9228667cc1724fc8029767ba71c5e7`.

## `logit maxdiff` is not evidence of anything — kill this reading first

The gate prints `logit maxdiff` and it is the number a reader reaches for. It is the max over a
~250k-wide vocabulary of `|prefill - decode|`, dominated by whatever tail logit is noisiest
anywhere, and it does **not** discriminate:

| arm | maxdiff | verdict |
|---|---|---|
| 9B Q8_0, board-2048, 5090 (arm L4, N=3) | **1.165** (batched-prime 1.916) | **MATCH** |
| 27B Q8_0, board-2048, pod (the failure) | **0.4659** | **MISMATCH** |
| 27B NVFP4, board-2048, pod (control) | 0.3537 | MATCH |
| repo-wide MATCH population | up to 2.438 | MATCH |

A passing run at **2.5x the failing run's maxdiff** settles it. The discriminator is the
**top-2 margin at the decision position versus the config spread at the contending ids** — a
flip is only possible where `spread > margin`, and it is only a *defect* where it is not.

## One-variable kills — every mechanism hypothesis refuted (N>=1 each, quoted receipts in RESULTS.jsonl)

| # | hypothesis | arm | result |
|---|---|---|---|
| 1 | the k27 `fa_split_keys` SM rung (the prior-art playbook's prime suspect) | `MEMRA_FA_SPLIT=8` | **REFUTED** — still MISMATCH, and it moves the decode winner to a *third* id (266) |
| 2 | same, big-rig value | `MEMRA_FA_SPLIT=16` | REFUTED — byte-identical to naked (188-SM default *is* 16) |
| 3 | FA split-K reduction segmentation at all | `MEMRA_FA_SPLIT=1` | **REFUTED** — split-K off entirely, byte-identical MISMATCH |
| 4 | the Q8_0 g2 grid-fill door (`lib.rs:7333`, the only other SM-count-keyed door on a Q8_0 path) | `MEMRA_Q80_G2=0` | REFUTED — byte-identical |
| 5 | chunked-prefill reduction order | `MEMRA_PRIME_CHUNK=100000` | REFUTED — byte-identical (also a priori: this gate compares `forward_last`, not `prime_cache`) |
| 6 | **any fast kernel — int8/dp4a, MMQ, MMVQ, FA, fused FFN** | `MEMRA_FAST=0` (Stage-A f32 oracle) | **REFUTED, and this is the decisive one** — a wholly different arithmetic class flips the *same pair the same direction*; decode margin collapses to 0.0176 |
| 7 | artifact-specific (this Q8_0 file) | 9B Q8_0 `q9base`, same prompt | **REFUTED** — a 3x smaller model fires the *same* `{332,485}` pair, opposite orientation, margin 0.0208 |
| 8 | 188-SM-specific / this silicon | H100 sm_90a, 9B Q8_0, 07-31, N=5 (inherited receipt) | **REFUTED** — same `{332,485}` pair on a third silicon class at margin **0.0038** |

Kills 6, 7 and 8 together leave no kernel, no artifact and no rig. What remains is the prompt.

## The mechanism, measured (`argmax-margin-probe`, new this lane)

The probe walks the last N positions under **both** configs and reports each position's top-2
margin against the config spread at the contending ids — so "near-tie" is a measurement, not an
adjective. On the failing pair (pod, 27B Q8_0, window 24):

```
2047     332           0.0307       485          0.0999       0.2117         NO <-- FLIP
margin distribution (decode config, 24 positions): min 0.0214  p10 0.0999  p50 1.2531  p90 9.2268  max 13.4456
  -> the decision margin is BELOW 2/24 sampled positions' margins (rank 3 of 24)
  -> flip is ARITHMETICALLY POSSIBLE iff config_delta > margin: 0.2117 > 0.0307 = true
agreement across sampled positions: 23/24 (1 flip(s))
```

**23 of 24 positions agree.** The one that does not sits at margin 0.0307 — rank 3/24, against a
p50 of 1.2531 — and the config spread there is 0.2117, **6.9x** the margin. Under the k27 pin the
decode side lands on a *third* id at margin 0.0434: **three ids inside 0.05** at this position.

Why this prompt: `board-2048.txt` is truncated mid-word (`...into a **fa`), so its final
position is a mid-token continuation — exactly where a vocabulary's continuation candidates
bunch up. Cross-rig/cross-model, the same position keeps producing the same contenders with a
margin that varies just enough to decide whether the coin lands:

| rig | model | margin at pos 2047 | spread | outcome |
|---|---|---|---|---|
| pod 188 SM | 27B Q8_0 | **0.0307** (rank 3/24) | 0.2117 | **flips** |
| pod 188 SM | 9B Q8_0 | 0.0208 | — | **flips** |
| H100 sm_90a | 9B Q8_0 | 0.0038 | — | **flips** (N=5) |
| 5090 82 SM | 9B Q8_0 | 0.0686 (**rank 1/24** — the minimum) | 0.1064 | exposed, holds |
| 5090 82 SM | 9B NVFP4 | 0.1002 | 0.2311 | exposed, holds |
| pod / 5090 | 27B NVFP4 | 0.5602 | 0.1871 | structurally safe |
| 5090 82 SM | 27B NVFP4 | 0.8223 | 0.1871 | structurally safe |

## The actual defect: the gate, not the engine

`run-gen`'s assert inspects **one position** — the last. So which prompts pass is decided by
prompt length modulo where the near-ties fall, and the guarantee is weaker than it reads.
Measured on the arm the release battery calls **GREEN** (27B NVFP4, pod, arm P3):

```
2042     11            0.0684       681          0.0184       0.0684         NO <-- FLIP
...
2047     332           0.5602       332          0.6513       0.0612         yes
agreement across sampled positions: 23/24 (1 flip(s))
```

A config flip at position 2042 at margin **0.0184** — *smaller* than the failing arm's 0.0307 —
shipping green since it is not the last position. Identical mechanism, opposite gate verdict.
That asymmetry, not the Q8_0 red, is what needed fixing.

## Fix shipped (not a pin)

1. **`tools/argmax-margin-gate.sh`** (new, wired into `tools/fast-gate/models.tsv` as `amargin`
   + `amarginc`). Over the last N positions under both configs it asserts
   `flip(p) => margin(p) < spread(p)`: every disagreement must be explained by a margin the
   config spread can reach across, and a **wide-margin flip fails at any position** — that is
   the genuine cache/threading/kernel-bug signature the original assert *meant* to catch. Also
   fails if flips exceed `--max-flips` (default 1), since a near-tie coin is isolated by nature
   while a broken kernel disagrees repeatedly. Strictly stronger than the assert it calibrates:
   it catches the pos-2042 class the current gate sleeps through, and stops calling a
   0.03-margin coin corruption. 31s per arm on a 9B at window 12; `amargin`/`amarginc` both
   PASS through `fast-gate --tier 1` in 63s.
2. **`crates/memra-engine/src/bin/argmax_margin_probe.rs`** (new bin `argmax-margin-probe`) —
   the measurement instrument; the numbers in this verdict are its output.
3. **`run_gen.rs` diagnostic + assert message.** The gate now prints the top-2 margins and the
   config spread beside the logits and labels the class itself (`NEAR-TIE class` vs
   `WIDE-MARGIN flip — a real defect`), and the panic no longer asserts "cache threading bug"
   as the only explanation. The old wording is precisely why this red was carried across three
   releases without a mechanism. No behavior change on the MATCH path (verified: 9B Q8_0
   board-2048 unchanged, `MATCH` at maxdiff 1.165).

**NOT done: pinning the Q8_0 row.** A `MEMRA_FA_SPLIT`-style pin would be theatre here — arm 1
proves the pin does not even change the verdict, and arm 6 proves no kernel door does. The
prompt's final position is a genuine three-way tie; the honest fix is a gate that measures
margins instead of one that guesses from a single position.

### Recommended follow-up (owner call, not taken unilaterally)

`board-2048.txt` is truncated mid-word. If the intent was a clean 2048-token prefill probe, it
should end at a token boundary — that would remove the near-tie from the *gate's* decision
position without weakening anything. Left alone here because the prompt is a pinned artifact
used by other lanes' comparisons, and changing it silently invalidates their rows.

## Stage-3 verification — on the failing rig, with the failing artifact (arms V1-V3)

The fix is not verified by a green run on a healthy model; it is verified against the actual red.
Rebuilt on the pod (188 SM) against `/root/models/Qwen3.6-27B-Q8_0.gguf` + `board-2048`:

```
=== VERIFY: the calibrated diagnostic on the real failing pair ===
  EXIT=101
prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
[gate] prefill: l[332]=12.8352 l[485]=12.8045 | decode: l[332]=12.6235 l[485]=12.7234
[gate] top-2 margin: prefill 0.0307 decode 0.0999 | config spread at these ids 0.2117 -> NEAR-TIE class
assertion `left == right` failed: decode-step diverges from prefill at the last position (see the
[gate] lines above for the near-tie-vs-defect diagnosis; ...)

=== GATE: amargin on the 27B Q8_0 (window 12), 188-SM pod ===   EXIT=0
  explained pos=2047 margin=0.0307 < delta=0.2117
  SUMMARY flips=1 bad=0
  PASS: every prefill/decode argmax flip is explained by a margin the config spread covers

=== GATE canary on the 27B Q8_0 ===                              EXIT=0
  [canary] injected a WIDE-margin flip (pos 9999, margin 5.0000, delta 0.1000)
  explained pos=2047 margin=0.0307 < delta=0.2117
  UNEXPLAINED pos=9999 margin=5.0000 delta=0.1000
  TOO-MANY-FLIPS 2 > 1
  SUMMARY flips=2 bad=2
  CANARY PASS: the injected wide-margin flip broke the assertion as required (teeth)
```

Three things confirmed on the rig that produced the red: (1) the diagnostic classifies the genuine
failure itself, **exit 101 unchanged** — this lane did not silence the assert; (2) the calibrated
gate reaches the same position, quotes its own margins, and passes it for a stated reason;
(3) the canary has teeth **on this artifact**, rejecting the injected wide-margin row on both the
UNEXPLAINED and TOO-MANY-FLIPS axes. Raw: `pod/verify-diagnostic-q8.log`,
`pod/gate-q8-pod.log`, `pod/gate-q8-pod-canary.log`, `pod/VERIFY-SUMMARY.out`.

The remaining question for the owner is what to do with `run-gen`'s own assert now that its
verdict is a *measured* label. It still panics — deliberately left that way here, because
downgrading a hard exactness assert is a release-policy call, not a lane call. The battery's
27B Q8_0 row therefore stays red until the owner picks one of: (a) accept `amargin` as the
authoritative prefill-vs-decode gate and let `run-gen` warn instead of panic on a self-labeled
NEAR-TIE, (b) fix the prompt's token boundary (see above), or (c) keep the red as a known,
now-explained standing item. Recommendation: **(a)**, since `amargin` is the strictly stronger
check and (c) is how this sat undiagnosed for three releases.

## Gate wiring (LAW 3 — gates outside the battery rot)

```
amargin	cmd	tools/argmax-margin-gate.sh	-	-	-
amarginc	cmd	tools/argmax-margin-gate.sh	--canary	-	-
```

The canary took two designs to get honest, both recorded in the script header so the next lane
does not repeat them:

- **(a) relabel the expectation** — re-runs identical data, perfectly correlated with the
  default gate, proves nothing. This is the trap already documented on `lane/chunkinv-flip`.
- **(b) raise a margin floor above the prompt median** — only touches rows that *already*
  flipped, so on a clean model there is nothing to reject; the canary correctly reported "no
  teeth" and was useless as a teeth check. (It reported this on its first run here, which is
  how it got replaced.)
- **(shipped) fault-inject a wide-margin flip row** into the parsed table (margin 5.0 vs spread
  0.1) and require the comparator to reject it — a mutation test of the comparator on its own
  parse path, so it fires whether or not the model under test produces a natural flip.

The gate also emits the `^name: SKIP` shape when inputs are absent, per the hole that reported
`chunkinv` as "PASS (0s)" on this same pod during the v0.70.0 battery.

## Hashes / receipts

- artifact 27B Q8_0 (pod): `985893c8fdff2fb46095853462608c1e`
- prompt `board-2048.txt`: `806a1d9c9d19a47fe014738912c64d0f`
- run-gen (pod, lane tip): `4f9228667cc1724fc8029767ba71c5e7`
- raw logs: `pod/` (10 arms + 3 probes + verify diagnostic + 2 gate runs + 2 summaries, rsynced
  from the pod), `local5090/` (3 margin probes + 3 run-gen reps). Per-arm rows:
  `RESULTS.jsonl` (21).
- Community pod: every pod cell here is an **exactness** row. No perf is claimed anywhere in
  this lane, on either rig.
