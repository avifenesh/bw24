# Draft-side grammar masking — results (lane/draft-mask, 2026-08-04, RTX 5090)

Mask the DRAFT model's per-position sampling with the grammar's legal set, so proposals are
legal by construction and the verify-side truncation (which stays as the correctness backstop)
stops cutting every tight-schema round.

Model `q9` = `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` + `draft-9b-owntrim-nvfp4head-q4blk.gguf`
(FR-Spec trimmed head, so the target-id mask is permuted through `d2t` before upload).
Rig: RTX 5090, GPU serialized under `flock /tmp/gpu5090.lock`, warm (steady-state) regime.
Rollback/A-B seam: `MEMRA_DRAFT_MASK=0` (winner is the default, per flags doctrine).

Raw artifacts in this directory: `battery.log` (full battery stdout), `perf.jsonl` (per-run
rows), `serve-dm-{on,off}.log` + `serve-perf-{on,off}.log` (engine receipts incl. per-round
`gram_cuts`), `dm-{on,off}-*.txt` (every emitted stream compared), `probe-*.txt` /
`probe-*.serve.log` (the two probes below), `local-ci-correctness.log`, `serve-smoke.log`,
`kernel-check.log`.

## Mechanism receipt: the cut count goes to zero

`gram_cuts=C/R` = verify-side grammar truncation fired in C of R spec rounds. Same binary,
same prompts, mask ON vs OFF (from `serve-dm-{on,off}.log`, Phase A):

| cell | mask OFF | mask ON |
|---|---|---|
| tight3 greedy (bounded fleet schema) | 3/12, 3/15, 1/10 | 0/12, 0/15, 0/10 |
| tightschema greedy (unbounded) | 28/30, 18/25 | 0/30, 0/25 |
| person schema / json_object / loose | 0–2 per session | 0 |

Every cut round is wasted draft work: the proposals past the cut are discarded and the cut slot
pays a ~1 MB logits D2H plus a host masked-argmax recompute. Masking ON eliminates the cut
entirely on every cell measured — never partially, because the drafter can no longer propose an
illegal id in the first place.

## Cost of the speculative Matcher clone

One `Matcher::clone()` per spec round, advanced per proposed token, dropped at round end; the
real session Matcher is untouched until verify commits.

```
[draft-mask] mask_rounds=14 clone_total=0.029ms clone_per_round=0.0021ms
[draft-mask] q9: 102 clones 0.21 ms (0.002 ms/clone), 306 draft masks 0.46 ms (0.001 ms/mask)
```

**0.002 ms/round clone, 0.001 ms/position mask.** At ~4 ms/round that is 0.05% of a round —
below the measurement floor, and it buys the whole cut elimination above.

## Exactness — the gate that matters

Bar: with masking ON vs OFF the FINAL emitted stream must be byte-identical (the mask changes
what gets PROPOSED; verify + target sampling decide what gets EMITTED). Compared field-wise on
`{reasoning, content, completion_tokens}`.

| gate | result |
|---|---|
| ON == OFF, `obj-greedy` (json_object, greedy) | PASS byte-identical |
| ON == OFF, `schema-greedy` (person schema, greedy) | PASS |
| ON == OFF, `schema-temp` (person schema, temp 0.8 seed 42) | PASS |
| ON == OFF, `loose-obj` (prose under json_object) | PASS |
| ON == OFF, `tight3` (bounded fleet schema, greedy) | PASS |
| ON == OFF, `tight3-temp` (bounded fleet, temp 0.8 seed 42) | PASS |
| ON == OFF, `tightschema-temp` (unbounded fleet, sampled) | PASS |
| Unconstrained vs PRE-LANE binary (`0a7349f6`), 3 prompts x {greedy, temp} | **6/6 byte-identical** |
| Constrained correctness with masking ON: json_object parses, json_schema validates | PASS |
| `tools/local-ci.sh` correctness stage: kernel-check, prime-gate 8/8, run-gen argmax MATCH (31B + 12B), VERIFY-GATE K=7 (both), spec self-consistency 64/64 | GREEN, exit 0 |
| `tools/serve-smoke.sh` (incl. spec-vs-plain serving exactness) | 8/8, exit 0 |

Battery: `bash research/draft-mask-20260804/run-battery.sh` -> `0 failure(s)`, exit 0,
reproduced twice.

### The one cell that is NOT gated, and why (measured, not inferred)

`tightschema` GREEDY — the *unbounded* fleet schema (`minItems: 6`, no `maxItems` anywhere) at a
400-token cap — is reported as info, never gated. The 9B runs out of distinct fleet entries
around char 600 and then degenerates into unbounded whitespace (the JSON grammar permits
arbitrary whitespace between tokens), riding the cap. That tail sits in a near-tie logit regime,
and `crates/memra-engine/src/spec.rs:2151` already documents that verify batch shape T "changes
FP summation order and can flip argmax at tight logit margins".

`probe-shape.sh` settles it on the **PRE-LANE binary at `0a7349f6`, which contains no
draft-mask code at all**:

```
== pre-lane binary, tightschema greedy, K sweep (no draft-mask code in this binary) ==
  K3 != K2 (SHAPE-DEPENDENT, pre-lane)
  K3 != K1 (SHAPE-DEPENDENT, pre-lane)
== lane binary, draft-mask ON, same sweep ==
  K3 != K2 (SHAPE-DEPENDENT, mask ON)
  K3 != K1 (SHAPE-DEPENDENT, mask ON)
== K=1 cross-arm (shortest chain: pre-lane vs mask ON) ==
  prelane-K1 == mask-K1
```

Every one of those divergences begins at the **same character (603)** — the point where the
degeneration starts — and all variants agree on the 603 chars before it. So on this cell the
emitted stream is a function of draft-chain shape in main already; ON-vs-OFF byte identity is not
achievable through a shape-varying verify and would not have been achievable before this lane
either. With shape held fixed at K=1, pre-lane and mask-ON are byte-identical.

`probe-tight3.sh` is the corresponding fix to the *measurement*: bounding the schema
(`minItems == maxItems`, `tags maxItems`) makes the model close its JSON at
`finish_reason=stop`, ~241 tokens inside a 320 budget, so the run never enters the degenerate
regime — while still being genuinely tight (OFF: 3/12, 3/15, 1/10 cut rounds; ON: 0). That
bounded cell is the gated tight cell, and it passes greedy AND sampled.

## Perf — N=3 same-session per arm, interleaved by arm, warm

Acceptance = engine `[spec-acc] cum` at request end; tok/s = end-to-end wall including HTTP.
Medians of N=3 (`perf.jsonl`, `battery.log`).

| cell | acceptance OFF | acceptance ON | tok/s OFF (med N=3) | tok/s ON (med N=3) | delta |
|---|---|---|---|---|---|
| **tight3** (bounded fleet schema, the tight cell) | 0.561 | **0.651** | 216.6 | **227.5** | **+5.0%** |
| **tightschema** (unbounded fleet schema) | 0.678 | **0.703** | 258.6 | **265.4** | **+2.6%** |
| obj (json_object, "spacecraft ten keys") | 0.423 | 0.429 | 194.2 | 194.5 | +0.2% |
| loose (prose under json_object, control) | 0.492 | 0.500 | 215.0 | 216.7 | +0.8% |
| unconstrained (no grammar, control) | 0.659 | 0.659 | 257.4 | 258.0 | +0.2% |

Reading it: the gain scales with how tight the grammar actually is for this drafter.
+0.090 acceptance / +5.0% tok/s where truncation was firing; the json_object and loose cells move
inside noise because their `gram_cuts` were already ~0 (the drafter proposes legal tokens there
on its own — the json_object "spacecraft" prompt is **not** a tight cell for this drafter, which
is why it is kept only as the merged-battery comparison point). Unconstrained is bit-for-bit
inert: no grammar, no clone, no mask.

`tight3` run1 reads low in both arms (166.0 ON / 154.7 OFF) — first request in a fresh session
pays the draft-graph capture; runs 2-3 are the steady state. The ON/OFF comparison holds at every
run index.

## Side finding: `tools/serve-smoke.sh` was a rotted gate in main

The first `local-ci.sh` run on this lane failed serve-smoke 4/8: `chat non-stream`,
`greedy determinism`, `concurrency (0/3)`, `spec-vs-plain text mismatch`. Running the **same
script from the pre-lane worktree at `0a7349f6` (v0.68.0) reproduced the identical 4 failures** —
so it was red in main, not broken by this lane.

Cause, measured directly: the default smoke model q9 is a REASONING model. At the 32-64 token
budgets the battery uses, every emitted token is still inside the thinking block, so
`message.content` is legitimately `''` with `finish_reason=length`:

```
max_tokens= 49 fin= length
  content= ''
  reasoning= 'Thinking Process:\n\n1.  **Analyze the Request:** ...'
```

Four checks asserted non-empty `content`, making them structurally unpassable on the default
model — including check 8, the spec-vs-plain **serving exactness** contract, which was silently
comparing `"" == ""`... and still failing, because `[ -n "$SA" ]` rejects the empty string.

Fix: those checks now compare the full emitted stream, `reasoning + content`, which is what
"deterministic non-empty output" and "spec emits the same text as plain" actually mean on a
thinking model. Deliberately not fixed by raising `max_tokens` until the model happens to close
its thinking block — that would make the gate a model-verbosity coin flip. serve-smoke is now
8/8 GREEN and `local-ci.sh` exits 0, so the spec-vs-plain serving-exactness check is live again.
