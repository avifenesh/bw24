# Dogfood F4: temperature default + sampled-spec state — 2026-08-04

Lane `lane/sampled-spec` off `restructure/public-split` @ `8aa1eb1e`. Rig: RTX 5090 Laptop
(sm_120a), GPU work serialized under `flock /tmp/gpu5090.lock` (shared with the v0.69 release
wave and the specpool lane).

## The bug (Part 1) — CONFIRMED, fixed

`crates/memra-server/src/main.rs`: both request structs carried

```rust
#[serde(default)]
temperature: f32,     // -> 0.0
```

`serde(default)` on `f32` is `0.0`, and `Sampler::is_greedy()` is `temperature <= 0.0`. So
**every client that omits `temperature` got greedy argmax**, not OpenAI's documented 1.0.

Confirmed against the owner's actual client, not inferred. `~/.pi/agent/models.json`
provider `local-memra` (the pill's daily 27B):

```json
{"baseUrl": "http://127.0.0.1:8002/v1", "api": "openai-completions", ...
 "models": [{"id": "qwen36-27b", "name": "memra 27B (daily)", ...}]}
```

No `temperature` key anywhere in the file (`'temperature' in json → False`). pi never sends
it, so every pill request decoded greedily. Deterministic argmax + a repeating agentic
context = the same tool call forever, which is the `npm view version` cycle in the transcript.

Fix: `#[serde(default = "default_temperature")]` → `1.0` on `CompletionReq` and
`ChatCompletionReq`. Explicit `"temperature": 0` still means greedy.

Neighbouring defaults audited against OpenAI semantics while there:

| param | default | verdict |
|---|---|---|
| `top_p` | 1.0 (`one()`) | correct, = disabled |
| `top_k` | 0 | correct — not an OpenAI param (OpenRouter/HF), 0 = keep all |
| `min_p` | 0.0 | correct — not an OpenAI param, 0.0 = disabled |
| `repetition_penalty` | 1.0 | correct, = off |
| `frequency_penalty` / `presence_penalty` | 0.0 | correct, = off |

So the fixed default is *pure* temperature-1.0 sampling with every filter off — which is also
the fastest sampled-spec regime (`pure_temp`, keeps the in-graph sampled draft chain).

### Gate-script ripple: NONE required

Grepped every request body in `tools/` and the gate batteries. Everything that depends on
greedy already sends `temperature:0` explicitly:

| file | status |
|---|---|
| `tools/serve-smoke.sh` | `"temperature":0` both bodies |
| `tools/serve-st-gate.sh` | `temperature:0` all 3 bodies |
| `tools/check-batch-exact.py` | `"temperature": 0, "seed": 0` |
| `research/serve-compat-20260802/sdk_gate.py` | `temperature=0` x9 (+1 deliberate 0.7) |
| `research/pc-iso-20260802/salt_gate.py` | `"temperature": 0, "seed": 0` |
| `research/constrained-full-20260803/run-battery.sh` | temperature is a required `req()` arg |
| `research/integrate-cache-20260802/*.py` | explicit |
| `tools/load-serve.py` | explicit (0.0 under `--greedy`, else 0.7) |

No script anywhere relied on omitted-temp==greedy. The fix cannot silently un-greedy a gate.

Unit test `omitted_temperature_is_openai_default_not_greedy` pins all four corners
(omitted / explicit-0 x chat / completions) through to the `SamplerConfig`, plus the filter
defaults and `is_greedy()` / `is_spec_sampling()`. memra-server suite **47/47 PASS**.

## Part 2 — sampled spec-decode ALREADY EXISTS (task premise was stale)

The brief asked to implement rejection-sampling spec decode. **It was already implemented and
default-live**, merged 2026-07-09/10 — a month before this lane:

| commit | date | what |
|---|---|---|
| `97c974ee` | 07-09 | `feat(spec): sampled speculative decoding — rejection-sampling verify (eager path)` |
| `1b431e37` | 07-09 | `feat(serve): sampled-spec serve path — temp-only requests ride rejection-sampling spec bursts` |
| `d7135f37` | 07-10 | `feat: sampled drafting in the CUDA-graph draft chain` |
| `54902376` | 07-10 | `merge feat/filtered-spec: filtered+penalized rejection-sampling spec` |
| `6e2742de` | 07-10 | `feat(spec): penalties in the sampled-spec path (v2.1) — legacy serve path fully retired` |

The brief's two premises were both already false:

- *"today spec sessions require sampler.is_greedy() (worker.rs ~557)"* — line 557 is a **stale
  doc comment**, not the predicate. The real predicate, `spec_eligible` (worker.rs ~1726), is
  `(sampler.is_greedy() || sampler.temperature() > 0.0)` — sampled has been eligible for a month.
- *"worker.rs ~1011"* is the **plain single-session `GraphSession`** promotion, unrelated to
  spec; it is already excluded for every spec session by `s.spec.is_none()`.

What is implemented is also *stronger* than the brief's proposed simplification. The brief
suggested greedy-draft + "sample-verify". The shipped path is full Leviathan/Chen with a
**sampled** draft:

- draft proposes `x ~ filtered q` via Gumbel-max (`gumbel_perturb_filtered` + device argmax)
- accept test `u*q(x) < p(x)` — division-free, no `q=0` blowup (spec.rs ~4007)
- on reject: residual `x ~ norm(max(0, fp - fq))` (`residual_sample_filtered`), with the
  FR-Spec trimmed-head q lifted into target-id space via `scatter_trim_logits`
- on full accept: bonus ~ Gumbel-max from the last verify column
- top-k/top-p/min-p **and** repeat/freq/presence penalties applied **symmetrically to p and
  q** before filtering, so the verify is exact for the *filtered penalized* target
- counter-based Philox everywhere (`sess.sctr`/`uctr`), graph-replay safe

Nothing to implement. So this lane's Part-2 contribution is the **missing gate** instead.

## Part 2 (real work): the composition gate — the arc's gate (c)

`HANDOVER.md` "SAMPLED-SPEC ARC" defines three gates. (a) seeded reproducibility exists
(run-spec seeded-rerun, `research/graph-sampled-logs/gateB-verdicts.log`). (b) exists at
kernel level. **(c) "aggregate distribution equality vs plain sampling" did NOT exist** — no
chi-square, KL, or TV anywhere in the tree.

The pre-existing `sample-check` arms 1-5 oracle each primitive **in isolation**. A spec decode
can pass all of them and still emit the wrong distribution if the primitives are **composed**
wrong — and composition is exactly what the isolation arms cannot see. That gap matters much
more now: the F4 fix makes sampled spec the **default** decode path.

New arm 6 in `crates/memra-engine/src/bin/sample_check.rs` runs the real device primitives in
spec.rs's order, with spec.rs's own host-Philox accept test, and checks the **composed output
distribution** against the CPU softmax of p. Leviathan/Chen: for one slot, `x ~ q` then
accept-or-residual emits `x ~ p` exactly, for *any* q. Deliberately mismatched q (different
logits and scale) so the accept rate sits near 0.11 — both branches get exercised.

Three checks: L-inf on the empirical PMF (20k draws), total-variation distance, and a
non-degenerate-acceptance guard so the arm can never go vacuous. Thresholds: L-inf 0.012
(~10 binomial sd at the modal mass), TV 0.05.

### Gate result (5090, `flock`ed) — `sample-check.log`

```
composed accept-walk output ~ p (20k draws, acc=0.114): maxabs=0.0032 tv=0.0184 OK
composed walk exercises BOTH branches (acc=0.114 in 0.05..0.95): OK
self-draft (q==p) accept rate == 1 (got 1.0000): OK
=== sample-check ALL GREEN ===
```

All 16 arms green (full log in `sample-check.log`).

### Negative controls — the gate DEMONSTRABLY catches, not just passes

A gate that has never failed is not evidence. Both controls perturb **only the composition**;
every kernel stays untouched, and arms 1-5 pass in both.

| control | perturbation | result | log |
|---|---|---|---|
| 1 | accept test inverted (`u*p < q`) | **FAIL** — acc 0.114→0.970, maxabs 0.0032→0.0719, **tv 0.0184→0.8826**; the vacuity guard also tripped | `negctl1-inverted-accept.log` |
| 2 | reject samples from p alone instead of `norm(max(0,p−q))` (the classic "forgot the residual") | **FAIL** — maxabs 0.0032→0.0210, tv 0.0184→**0.0881**; acceptance unchanged at 0.114, so *only* the distribution check caught it | `negctl2-no-residual.log` |

Control 2 is the important one: identical accept rate, every kernel individually correct, and
the bug is invisible to all five isolation arms. Only arm 6 sees it.

`sample-check` is wired into `tools/fast-gate` (`models.tsv` probe `samp`, mapped from
`spec_sample.cu` + `spec.rs` + `crates/memra-sampling/` in `map.tsv`), so arm 6 runs on every
touch of those paths. Gate = exit 0; `--refresh-goldens` skips it.

## Stale comments corrected (they are what made the brief wrong)

The misdiagnosis was caused by doc comments that outlived the code. Fixed, so the next reader
isn't sent down the same path:

- `worker.rs:554` — `spec` field said "greedy sessions ... `Some` only when: sampler greedy".
  Now states both arms, names `greedy_penalized` as the one excluded class, and points at
  `spec_eligible` as authoritative.
- `worker.rs:2300` — spec-burst comment claimed exactness == "byte-identical to fresh greedy"
  for all bursts. Now separates greedy (byte-identical) from sampled (distributionally exact,
  own Philox streams, reproducible per (seed, session)) — "that is the contract, not a gap".
- `worker.rs:1011` — added the note the brief asked for: sampled sessions do not graph-promote,
  and it costs nothing today because this promotion only fires for `s.spec.is_none()`, while
  every sampled session on an MTP model already rides the faster sampled spec burst. It would
  only matter for sampled on a **non-MTP** model.
- `memra-sampling/src/lib.rs:59` — `is_spec_sampling()` claimed "penalties are NOT yet wired
  into the spec verify: penalized requests take the legacy path". Both clauses were false
  (penalties wired since `6e2742de`; eligibility is decided by `spec_eligible`, which never
  calls this). Now documents what the predicate actually identifies — the **pure-temp regime**
  that keeps the in-graph sampled draft chain — and tightened it to match (added the
  top_k/top_p/min_p conditions), since filters force the eager draft. Only caller is the new
  unit test.

## Known gaps (NOT closed by this lane — named, not hidden)

- `MEMRA_SPEC_TEMP` is **undocumented** in `docs/FLAGS.md` (0 occurrences) despite switching
  the whole sampled path on. Its CLI default seed is 42 (`spec.rs:2762`) while `docs/FLAGS.md`
  documents `MEMRA_SEED` default 0 for run-gen — a real doc/code mismatch.
- No `tools/` script runs an **end-to-end** sampled spec decode; every serve gate sends
  `temperature:0`, and both fast-gate spec probes (`q35spec`, `g31spec`) are greedy. Arm 6
  closes the math, not the e2e path.
- End-to-end temp→0 == greedy-spec continuity (arc gate (b)) exists only at kernel level.
- llama matched-temp pairing protocol: still absent.
- `sample_check` is not declared as an explicit `[[bin]]` in `crates/memra-engine/Cargo.toml`
  (builds via edition-2024 autobin discovery), unlike its 40+ sibling gate binaries — so
  `tools/validate-h100.sh`, which builds gates by explicit `--bin` list, does not build it.
- Filters/penalties silently drop the sampled draft to the **eager** chain (`pure_temp` gate,
  spec.rs:3054-3063) — a real perf cliff, documented but not gated.
