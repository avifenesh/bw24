# Constrained decoding FULL (lane/constrained-full) — battery receipts, 2026-08-03

Rig: local RTX 5090 (shared, flock-serialized). Model: q9 = Qwen3.5-9B NVFP4 MTP GGUF.
Server: MEMRA_COMPAT=openai. Baseline binary = release build of the merged v1 HEAD
c2ed69ed (`/tmp/memra-server-c2ed69ed`). Battery: `run-battery.sh` (this dir); full run
log `battery-run.log`; per-run rows `perf.jsonl`; server logs `serve-*.log`.

What FULL means vs v1 (c2ed69ed): v1 excluded constrained rows from device sampling
(full 248k-row D2H + host -inf mask + host sample), skipped graph promotion, and gated
spec OFF. FULL uploads the packed llguidance bitset per step (~31KB H2D, stable
per-session device buffer) and bans on device (`mask_logits_f32`, -FLT_MAX) BEFORE the
existing device samplers — constrained rows now ride the SAME device-sample / lean-logits
/ graph / spec paths as unconstrained rows.

## Phase A — no-op exactness (v1-HEAD baseline vs FULL binary)

Six unconstrained requests each side (3 prompts x {greedy temp=0 seed=0, sampled
temp=0.8 seed=42}, max_tokens=96), full stream compared (reasoning + content + count):

    exact-baseline-*.txt == exact-new-*.txt — 6/6 byte-identical  (PASS)

## Phase B — constrained correctness on EVERY path

| path | config | receipt | verdict |
|---|---|---|---|
| spec burst (default) | json_object + json_schema greedy | b1-spec-{obj,schema}.txt | parses + validates; spec bursts confirmed in log |
| plain batched | SPEC=0, greedy | b2-plain-{obj,schema}.txt | parses + validates |
| sampled (device gumbel + mask) | SPEC=0, temp=0.8 seed=7 | b2-sampled-schema.txt | parses + validates |
| graphed | SPEC=0 GS_MIN=32, prefix-hit promotion | b3-graph-schema.txt | parses + validates; 2 captures (graph-census); graphed == eager byte-identical |
| host oracle | SPEC=0 CONSTRAIN_HOST=1 | b4-host-schema.txt | parses + validates |

Cross-path identity (same prompt/schema, greedy):
- device-mask plain == host-oracle (v1 path): **byte-identical** — the masked device
  argmax is the exact twin of the host -inf mask + host argmax.
- spec constrained == plain constrained: **byte-identical** — verify-side grammar
  truncation preserves the plain constrained-greedy stream token-for-token.
- graphed constrained == eager constrained: **byte-identical** (same prompt, prefix-hit
  second request vs cold first).
- Unknown response_format type: still a loud 400 (unchanged from v1, unit-covered).

## Phase C — the three-way perf table (N=3 interleaved, same session, warm after run1)

256-token greedy generation, same prompt ("long JSON object … spacecraft"). Medians of
runs 2-3 quoted alongside (run1 carries the usual first-request warmup):

| arm | run1 | run2 | run3 | median |
|---|---|---|---|---|
| plain unconstrained (SPEC=0) | 122.0 | 124.5 | 124.4 | **124.4** |
| plain constrained v1-path (CONSTRAIN_HOST=1) | 93.7 | 117.8 | 117.7 | **117.7** |
| plain constrained FULL | 98.3 | 123.8 | 123.7 | **123.7** |
| spec unconstrained (default) | 163.1 | 194.4 | 194.4 | **194.4** |
| spec constrained FULL | 119.6 | 153.4 | 153.8 | **153.4** |

- Plain: constrained-full = **99.4% of unconstrained** (123.7 vs 124.4). The v1 host path
  measures 117.7 here (its remaining gap is the full-row D2H + host mask+sample per
  step). v1's merged receipts were 117 vs 194 because v1 ALSO lost spec for constrained
  sessions — the FULL spec arm closes that: **153.4 vs the v1 117 = +31%**, and every
  constrained request now takes the fastest eligible path by default.
- Spec constrained (153.4) vs spec unconstrained (194.4): the gap is DRAFT ACCEPTANCE
  under the grammar, not mask overhead — the unconstrained drafter proposes tokens the
  grammar rejects at verify. Measured acceptance this prompt/schema: cum 0.467-0.513
  ([spec-acc] rows, serve-perf-spec.log) vs 0.62-0.82 on the looser Rex schema
  (serve-spec.log). Expected and documented; draft-side masking is the optional future
  lever if tight-grammar acceptance ever matters.
- Per-step mask cost (llguidance compute only now — apply is on device):
  **0.006-0.007 ms/step** ([constrained] lines), down from v1's 0.055-0.058 ms/step
  (which included the host 248k-row apply). H2D of the packed mask rides the step's
  existing transfers (~31KB — noise on PCIe).

## Graph fingerprints

Untouched. `fa_class_of` / `fa_segment_end` (fa_vec pick, v4 max, fa512 floor, ladder
rung) don't know the mask exists; the mask node is kernel-class-invariant (same kernel,
same launch shape at every t_kv — only buffer contents change). The mask buffer is a
capture-time stable pointer (the KV `len_d` pattern): baked once, contents re-uploaded
per step (`GraphSession::upload_mask`), carried across kernel-class recaptures
(re-baked, same address). graph-session/graph-decode gates' shape pins see identical
graphs for unconstrained sessions (Phase A byte-identity is the e2e proof).

## Kernel-check arm

`mask_logits_col`: synthetic packed bitset, vocab 4099 (non-multiple-of-32), mask
shorter than the row (padded-lm_head tail rule), stacked-column addressing —
mismatch=0 vs host reference, argmax equal, **OK (byte-identical)** (5090).

## Merge gates (q9, this HEAD)

- kernel-check: ALL GREEN (0 FAIL) incl. the new mask arm — gates.log.
- run-gen argmax: MATCH — gates.log.
- run-spec K=1..8 self-consistency: PASS — gates.log.
- cargo test -p memra-server: 38/38.

## Flags doctrine

The device path is the ONLY constrained path (no opt-in flag). `MEMRA_CONSTRAIN_HOST=1`
is the rollback oracle (documented in docs/FLAGS.md §3): v1 host mask + host sample, no
graph, no spec. Fallback sampler configs (penalties/top-k/top-p/min-p) host-sample with
the host mask automatically — same behavior as their unconstrained twins.
