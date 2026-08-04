# Dogfood F4: omitted temperature AND seed + sampled-spec state — 2026-08-04

Lane `lane/sampled-spec` off `restructure/public-split` @ `8aa1eb1e`. Rig: RTX 5090 Laptop
(sm_120a), GPU work serialized under `flock /tmp/gpu5090.lock` (shared with the v0.69 release
wave and the specpool lane).

**Verdict in one line:** F4 was TWO pinned defaults, not one — `temperature` (0.0 = greedy)
*and* `seed` (0 = a fixed stream). Both fixed; the owner's exact serve config verifies
no-loop, greedy byte-identity, and seed semantics all PASS on the new binary, with sampled
sessions keeping **1.72x** over plain-sampled (73.92 vs 43.00 tok/s, N=5). The brief's Part 2
premise was stale — rejection-sampling spec decode shipped 2026-07-09/10 — so Part 2 became
the missing distribution gate (arc gate (c)) plus two negative controls that prove it catches.
Pre-merge battery green: fast-gate tier 0+1 0 fail, `run-spec` K=1..8 PASS in **both** the
greedy (token-identical) and sampled (seeded-rerun) regimes, serve-smoke 8/8 — raw in `gates/`.

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

## The bug's SECOND HALF (found in Part 3) — `seed` also defaulted to a pinned 0

Fixing the temperature default is **not sufficient**. Both structs also carried

```rust
#[serde(default)]
seed: u64,        // -> 0
```

and `0` is a perfectly valid FIXED seed. So a temperature-1.0 request that omits `seed`
replays one single sampled stream forever — same context in, same tokens out. The loop
survives the temperature fix untouched.

Found by driving the LIVE pre-fix server (the owner's own 8002 daily driver, old binary from
`bw24-unified`) rather than by reading code. Raw: `prefix-control.log`, per-run JSON in
`raw-prefix/`. Content sha256[:16] over `reasoning + content`:

| arm | request | runs | result |
|---|---|---|---|
| loop repro | agentic tool-check prompt, **temp omitted, seed omitted** (pi's exact shape) | 3 | `ecb102d458e0bc04` x3 — **identical** |
| seed isolation | `temperature: 1.0`, **seed omitted** | 3 | `98ce5c171213e4fe` x3 — **identical** |
| sampler control | `temperature: 1.0`, `seed: 1/2/3` | 3 | `bac60db73f70ab0a` / `df9687784f89f44b` / `0b50c59940e5da4f` — **differ** |

Row 2 is the finding: temperature 1.0 alone still loops. Row 3 proves the sampler and the
rejection-sampling spec path are fine — the seed was the remaining pin. Same bug class as
F4, same file, same `#[serde(default)]`-on-a-meaningful-zero mistake.

Fix (`6f51d4a1`): `seed: Option<u64>` on both structs, resolved as
`seed.unwrap_or_else(fresh_seed)`. `fresh_seed()` mixes the nanosecond clock with a
process-lifetime `AtomicU64` counter through SplitMix64's finalizer, so two requests in the
same nanosecond tick (batched arrivals) still get distinct streams, and it never returns 0.

**Explicit seeds — including an explicit `0` — are honored exactly.** Audited the ripple the
same way as the temperature default: every determinism-sensitive script sends BOTH
`temperature:0` and (where it matters) `seed:0` explicitly, so no gate changes behavior.
Scripts that send a nonzero temperature (`sdk_gate.py` 0.7, the two `run-row-repair.sh`
rebaseline p3 lines at 0.7) all pin `seed` explicitly (0 / 42). Scripts that omit `seed`
entirely all send `temperature:0`, where the seed is unused — `devsample_meta` returns
`(0.0, 0, 0)` for greedy rows and the spec path builds `SpecSampling` only when
`temperature() > 0.0`, so a greedy request never reads the seed at all.

### One real interaction, documented at the site (spec.rs `s_key`)

The sampled draft CUDA graph bakes `(seed, temp.to_bits(), k)` as capture-time constants
(`DraftGraphCtx::s_key`). A seed-omitting request that RESUMES a parked spec session now
finds an `s_key` from the previous request's seed and pays one recapture. Bounded: a
session's seed is fixed for its lifetime (`worker.rs` reads `s.sampler.seed()` per burst),
so the miss happens at most **once per resumed request** — the first burst recaptures, every
later burst replays. This does not reopen the ~16 ms/burst regression the persistent
`DraftGraphCtx` exists to fix. A client wanting both the parked graph and reproducibility
supplies an explicit `seed`, which keeps `s_key` stable across its whole conversation.

Also documented the corrected omitted-param semantics in `docs/SERVING.md`'s OpenAI
compatibility contract, next to the structurally identical `max_tokens`-omitted bullet.

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

## Part 3 — daily-driver verification on the NEW binary: ALL PASS

Rig: RTX 5090 Laptop (sm_120a), GPU serialized under `flock /tmp/gpu5090.lock`. Server started
from the owner's exact `serve-qwen36-27b-memra` config (same MEMRA_MODELS incl. the regime
draft, MEMRA_CTX=131072, MEMRA_MAX_SESSIONS=1, MEMRA_REUSE_POOL=1, MEMRA_PRIME_CHUNK=2048,
MEMRA_API_KEY) with `MEMRA_BIN`→the lane binary and `PORT=8102` so the owner's 8002 is
untouched. Battery: `run-battery.sh`; raw per-run JSON in `raw/`; server log
`server-8102.log`. Thermal regime: 68C at start, steady 83-85C under the perf arm at
151 W, SM clock 1582-1665 MHz — i.e. warm/steady-state, not a cold-clock burst.

| gate | request shape | runs | result | verdict |
|---|---|---|---|---|
| A. no loop | agentic tool-check prompt, **temp + seed omitted** (pi's exact shape) | 4 | 4 distinct sha16 | **PASS** — pre-fix this was 3/3 identical |
| B. greedy unchanged | explicit `temperature: 0, seed: 0` | 3 | `fa1d69e4c568ca5e` x3 | **PASS** — byte-identical, gate contract intact |
| C. seed semantics | `temperature: 1.0, seed: 4242` | 2 | `be86e4268547d039` x2 | **PASS** — explicit seed reproduces |
| C. seed semantics | `temperature: 1.0`, seed omitted | 4 | 4 distinct sha16 | **PASS** — omitted seed varies |

Gate A is the exact pattern from the transcript (repeated `npm view` tool-check turn with a
"don't repeat a command you already ran" instruction) at 400 tokens — the shape that produced
~10 identical cycles. Four runs, four different completions.

Spec **does** engage on the sampled default: 266 `[spec-acc]` bursts in `server-8102.log`,
cumulative acceptance settling at **0.59** (e.g. `ctx=526 burst=12/18 cum=327/552=0.592`).
So the fix does not silently cost the owner spec decode.

### Throughput (N=5 each, medians, 27B daily driver, same prompt, ngen=512)

| arm | median tok/s | min–max | vs plain-sampled |
|---|---|---|---|
| **sampled-spec** (the new default) | **73.92** | 72.39–77.12 | **1.72x** |
| greedy-spec (`temperature: 0`) | 88.28 | 88.08–88.40 | 2.05x |
| plain-sampled (`MEMRA_SERVE_SPEC=0`) | 43.00 | 42.75–43.19 | 1.00x |

Reading: an omitted-temperature session — what every OpenAI-SDK and pi/pill client sends —
now gets **1.72x** the plain-sampled rate, i.e. sampled requests keep essentially the whole
spec win. It sits **16% below greedy-spec** (0.84x), which is the expected cost of
rejection-sampling verify vs argmax verify: acceptance is 0.59 sampled where greedy on this
draft runs higher, and each round pays the extra gather/uniform/residual work. That gap is a
*sampling* cost, not a regression — the pre-fix alternative wasn't "greedy-spec at 88 tok/s",
it was greedy-spec output stuck in a loop.

Protocol notes (so these numbers are usable):
- The sampled-spec and greedy-spec arms are **interleaved within each repetition** on ONE
  server process, so clock/thermal drift hits both equally. Their comparison is valid.
- The plain-sampled arm required `MEMRA_SERVE_SPEC=0`, which is a **server restart**, so it
  is a *separate process* (`server-8102-nospec.log`, `raw-nospec/`) and therefore a
  cross-run comparison. Its clocks ran *higher* (1785-1852 MHz vs 1582-1665) at similar
  power, so if anything it flatters the denominator — the 1.72x is conservative, not
  inflated. Zero `[spec-acc]` lines in that log confirms spec was genuinely off.
- Greedy-spec produced the identical hash `4ae03fff0e11f032` on all 5 perf reps at 512
  tokens, which is an extra byte-identity data point beyond gate B.

### First attempt died of a real OOM (recorded, not silently retried)

The first launch failed with, quoted from `server-8102.log`:

```
[server] FATAL: worker init failed: load qwen36-27b: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Concurrent-GPU state at failure (`nvidia-smi --query-compute-apps`): PID 586229
(`bw24-unified/target/release/memra-server`, the 8002 daily driver, **started 40 s after this
lane took `/tmp/gpu5090.lock`**) holding 22498 MiB, plus llama-server 260 MiB and a colbert
`mqar_loop.py` 1142 MiB, leaving 22 MiB free of 24463. Two 27B servers do not fit 24 GB —
this is the documented `MEMRA_MAX_SESSIONS=1` VRAM math, not a new bug. No number was
reported from that attempt; the battery was re-run after the box freed, and the medians above
come entirely from the clean run.

## Pre-merge correctness battery — raw logs in `gates/`

The three repo-mandated gates (CONTRIBUTING.md / CLAUDE.md "CI is compile-only") plus the
serving-surface smoke, run on the lane binary. Every log is committed under `gates/`.

### 1. fast-gate `--diff 8aa1eb1e` — `gates/fast-gate.log` (+ per-probe logs)

```
== fast-gate tier 0 ==
  build: OK (0s)
  kernel-check (synthetic arms): GREEN (2s)
tier 0: GREEN (4s total)
== fast-gate tier 1 ==
  samp: PASS (self-gating check green, 2s)
  g12: PASS (gates green + golden token-identical, 154s)
  sampt: PASS (self-gating check green, 0s)
  q35spec: PASS (gates green + golden token-identical, 12s)
  g31spec: PASS (stream agreement 32/32, 8s)
tier 1: 0 fail (180s total)
```

`samp` is the arm-6 composition gate added by this lane; `g12`/`sampt` are the argmax
(`run-gen`) probes; `q35spec` (`gates/probe-q35spec.log`: `acceptance: 23/30 = 76.7%
self-consistency: PASS (identical to generate)`) and `g31spec`
(`gates/probe-g31spec.log`: `stream agreement 32/32`) are the spec probes. fast-gate printed
`NOTE: memra-server touched — run tools/serve-smoke.sh`, which is gate 3 below.

### 2. `run-spec` K=1..8 self-consistency — BOTH regimes PASS

Model: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (`run-spec` drafts from the model's own MTP head;
`MEMRA_DRAFT` is not read by this binary). `MEMRA_NGEN=48 MEMRA_CHAT=1`, real text prompt.

**Greedy arm** (the gate as CONTRIBUTING.md defines it — every K token-identical to plain
decode) — `gates/run-spec-greedy-K1-8.log`:

| K | acceptance | self-consistency |
|---|---|---|
| 1 | 22/25 = 88.0% | PASS (identical to generate) |
| 2 | 31/36 = 86.1% | PASS (identical to generate) |
| 3 | 34/39 = 87.2% | PASS (identical to generate) |
| 4 | 36/48 = 75.0% | PASS (identical to generate) |
| 5 | 38/55 = 69.1% | PASS (identical to generate) |
| 6 | 38/66 = 57.6% | PASS (identical to generate) |
| 7 | 38/77 = 49.4% | PASS (identical to generate) |
| 8 | 39/80 = 48.8% | PASS (identical to generate) |

```
=== SELF-CONSISTENCY PASS ===
```

**Sampled arm** (`MEMRA_SPEC_TEMP=1.0 MEMRA_SEED=4242`) — the arm this lane makes the serve
default, so it is gated too. Greedy identity is *undefined* here by construction
(Leviathan/Chen give distribution equality, not stream equality), so run-spec switches to the
seeded-reproducibility gate: same `(seed, prompt, K)` must reproduce the identical stream on a
second generation. `gates/run-spec-sampled-K1-8.log`:

| K | acceptance | self-consistency |
|---|---|---|
| 1 | 23/24 = 95.8% | PASS (seeded rerun identical) |
| 2 | 29/36 = 80.6% | PASS (seeded rerun identical) |
| 3 | 35/42 = 83.3% | PASS (seeded rerun identical) |
| 4 | 35/56 = 62.5% | PASS (seeded rerun identical) |
| 5 | 38/60 = 63.3% | PASS (seeded rerun identical) |
| 6 | 42/60 = 70.0% | PASS (seeded rerun identical) |
| 7 | 40/63 = 63.5% | PASS (seeded rerun identical) |
| 8 | 38/80 = 47.5% | PASS (seeded rerun identical) |

```
=== SELF-CONSISTENCY PASS ===
```

Acceptance is non-zero at every K in both arms, so run-spec's second gate (a wrong MTP head
passes identity via the bonus token while accepting nothing) is also clear.

**Timing honesty:** both run-spec sweeps ran with the owner's live 8002 server resident
(14786 MiB of 24463; PID 739917 holding `/tmp/gpu5090.lock`, captured in the log headers).
This is a *correctness* gate needing VRAM, not exclusivity, so it ran unlocked — but the
`tok/s` and `x vs generate` figures in those logs are **contended, not perf evidence**, and
are recorded here only because they are part of the verbatim output. The perf numbers in this
document come from the Part-3 battery, not from these logs.

### 3. `tools/serve-smoke.sh` — `gates/serve-smoke.log`

fast-gate does not cover the serving surface, and this lane changed `memra-server` request
parsing, so this is the gate that actually exercises the fix's blast radius.

```
== serve-smoke: plain serving ==
  ok: /models lists the model
  ok: chat non-stream (text + usage + finish_reason)
  ok: chat stream (SSE chunks + [DONE])
  ok: /v1/completions
  ok: greedy determinism (2 runs identical)
  ok: 3 concurrent chats
  ok: long generation (>=100 tok)
== serve-smoke: spec serving (draft attached) ==
  ok: spec == plain greedy text (serving exactness)
serve-smoke: 0 failed
```

`greedy determinism (2 runs identical)` and `spec == plain greedy text` are the two that prove
the F4 defaults did not leak into the greedy path: both scripts send `temperature:0`
explicitly, and both still get byte-identical output.

Not run on this lane: `tools/local-ci.sh` tier 2 (the full battery + perf stage) — it needs
exclusive GPU, and the box is shared with the v0.69 release wave, the specpool lane, and the
owner's live daily driver. fast-gate's own footer says tier 2 gates every merge and tag; this
lane's code is already merged into `restructure/public-split` (as `c716954b`) by the release
wave, so tier 2 falls to that wave's pre-tag battery, with these results as the lane's evidence.

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
