# Constrained decoding (lane/constrained) — battery receipts, 2026-08-03

Rig: local RTX 5090. Model: q9 = Qwen3.5-9B NVFP4 MTP GGUF (no draft attached — plain
decode; spec x constrained is gated OFF loudly). Server: MEMRA_COMPAT=openai, default env.
Battery: `run-battery.sh` (this dir), GPU serialized via `flock /tmp/gpu5090.lock`.
Full run log: `battery-run.log`; server stderr: `serve-lane.log`.

## Phase A — no-op exactness (the isolation contract)

Baseline binary = release build of merge-base 464c9da7 (pre-lane, saved before any edit).
Lane binary = release build of this lane. Six unconstrained requests each side
(3 prompts x {greedy temp=0 seed=0, sampled temp=0.8 seed=42}, max_tokens=96), comparing
the FULL generated stream (reasoning + content + completion token count):

    exact-baseline-*.txt == exact-new-*.txt  — 6/6 byte-identical  (PASS)

A request without `response_format` builds no factory, no matcher, and takes no new
branch; this is the measured proof.

## Phase B — constrained correctness

- `{"type":"json_object"}` greedy: output parses as a JSON object (json-object-out.txt).
- `{"type":"json_schema"}` greedy: parses AND validates (python jsonschema) against
  a schema with required keys, integer minimum, array minItems, additionalProperties:false
  (json-schema-out.txt).
- Same schema, sampled temp=0.8 seed=7: parses AND validates (json-schema-sampled-out.txt).
- `{"type":"yaml"}`: 400 with a named error (honesty gate — no silent downgrade).

## Phase C — mask cost (N=3, same session, single run each, warm)

256-token greedy generation, same prompt:

| arm                       | run1  | run2  | run3  | tok/s median |
|---------------------------|-------|-------|-------|--------------|
| unconstrained             | 170.8 | 194.1 | 194.3 | 194.1        |
| constrained (json_object) | 117.6 | 117.4 | 117.4 | 117.4        |

llguidance host-side mask compute: **0.055-0.058 ms/step** (worker `[constrained]` lines,
217 masked steps -> ~12 ms total per 217-token generation). The mask itself is ~1% of a
step; the 194->117 tok/s gap is the v1 integration cost, dominated by:
  1. constrained rows are excluded from device-side sampling + lean logits, so every step
     pays the [n_vocab]=248320 f32 D2H (~9-30% of a tick per the inc2 profile) plus a
     host-side row clone + O(n_vocab) host sample;
  2. graph promotion is off for constrained sessions (the +34% B=1 door).
Known lever for a follow-up: device-side mask application (H2D the packed SimpleVob,
-inf on device, keep device sampling). NOT done in v1 — shippable subset first.

Note run1 vs run2/3 of the unconstrained arm shows the usual first-request warmup; the
constrained arm was already warm. Numbers are single-session serve-level, not
interleaved-x5 — recorded as the lane's mask-cost receipt, not a board row.

## Cross-checks

- `cargo test -p memra-server`: 38/38 PASS (incl. new constrained unit tests:
  schema->mask->forced-walk parses+validates, outside-mask consume errors, padding-tail
  ban, response_format parse forms, grammar-only-when-present no-op contract).
- `tools/serve-smoke.sh`: 4 failures on this box — IDENTICAL failure set on the baseline
  binary (chat non-stream / greedy determinism / concurrency / spec-vs-plain are broken
  by think-tail content routing at small max_tokens on q9, a pre-existing box/battery
  condition, verified by running the same script on the pre-lane binary). Not introduced
  by this lane; Phase A covers the regression question with byte-level receipts.

## Grammar x think interaction (found live, fixed in this lane)

The grammar masks from the first generated token, so an open `<think>` tail could never
close — forced JSON landed in the `reasoning` field and `content` came back empty.
Constrained requests now force the template's no-think switch; a think-tail template
WITHOUT `enable_thinking` is a loud 400.
