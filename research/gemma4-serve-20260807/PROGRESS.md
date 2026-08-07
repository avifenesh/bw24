# lane/gemma4-serve-gaps — the two receipted gemma4 serving gaps

Predecessor receipts: `research/step-sku-20260807/PROGRESS.md` §"Pre-existing gaps" items 1–2
(`raw/think-smoke-20260807T093918Z.log` = the panic, `raw/think-smoke-20260807T094506Z.log` =
the thought/content leak). This lane re-receipted gap 1 on its own binary before touching code:
`raw/repro-panic-server-20260807T114502Z.log` (commit 6e94b2ab) — ONE request to
gemma-4-12b-it-qat-q4_0 on the DEFAULT scheduler, worker panics at decode_batch.rs:553,
respawn re-panics on the queued request, process FATALs.

## Gap 1 — where the panic actually fires, and every adjacent hole found on the same walk

The default batched tick (worker.rs phase (c)) chunks decoding sessions per model and calls
`decode_step_batch_sampled_lean_masked`. That body's B=1 fast path EXCLUDES gemma4
(`decode_batch.rs:506-507`), so even a lone gemma4 request falls into the batched body and hits
`assert!(... gemma4.is_none(), "decode_step_batch v1 covers the hybrid non-gemma4 trunk only")`
at :553 — a Rust panic, which kills the worker thread. `MEMRA_SERVE_BATCH=0` (legacy
round-robin, `step_session` → `decode_step_h` → `gemma4_decode_step_h`) serves gemma4 fine —
the eager gemma4 arm exists and is gated; only the SCHEDULER routes into a body with no arm.

Adjacent holes in the same class, found while mapping the fix (all reachable once requests stop
panicking, so they are in-scope for the fail-closed shape):

1. **Graph promotion is silently WRONG for gemma4.** Phase (a0) promotes a solo greedy
   session (budget >= 384) via `graph_session_from_cache_masked` →
   `decode_step_dc_cap_masked` (decode.rs:1363) — which has NO gemma4 route: it captures the
   generic qwen-class layer walk over gemma weights (the exact "argmax-INIT passthrough"
   class the round-45 g12 gate caught on the dc path). Not a panic — wrong logits. The
   promotion must exclude gemma4.
2. **`prime_cache_batch` is silently WRONG for fresh gemma4 pairs.** It refuses only
   `carried && gemma4` (hybrid_forward.rs:987); two concurrent FRESH gemma4 prompts >=
   PRIME_MIN_T batch into the generic concat attn core (no per-layer swa geometry, no
   softcapped head). Must refuse gemma4 unconditionally, like the step35 refusal right below.
3. **Chunked/continuation prime PANICS.** `gemma4_prime` opens with
   `assert_eq!(cache.pos, 0)` (hybrid_forward.rs:5385). Any gemma4 prompt > PREFILL_TICK_T
   (1024) chunks in `prefill_tick`/`step_session` and the second chunk kills the worker —
   in BOTH scheduler modes. Any reuse-pool resume with a suffix >= PRIME_MIN_T does the same.

## Gap-1 fix shape: (b) fail-closed per-model route — and WHY not (a)

A real batched gemma4 decode arm means a batched twin of `gemma4_decode_step_h`: per-layer
SWA/global geometry (hd 256 vec / hd 512 MQA globals), weightless V-norm, softcapped head,
suppress mask — a new engine arm plus its own decode-batch-gate battery (config + strict,
isolation contract at B>1). That is an engine lane of its own, not something this lane can
gate honestly today. The brief's option (b) is explicitly for this case. Same shape the
step-sku lane shipped for step35 B>1 (chunk pin + fail-closed body), one step further:
per-model scheduler selection so the batched bodies are never entered at all.

The fix, smallest honest form:

- **worker.rs**: models with `cfg.gemma4.is_some() || is_gemma4_e4b()` are EAGER-ONLY —
  computed once at spawn next to `chunk_caps`, loud log line. Their sessions:
  * never enter the phase-(c) batched chunks — a dedicated per-session `step_session` loop
    (the legacy tick body verbatim) steps them inside the batched scheduler;
  * never graph-promote (hole 1);
  * never join prime-batch candidates (hole 2's worker side);
  * prime monolithically — no PREFILL_TICK_T chunking for fresh gemma4 (engine cannot chunk,
    hole 3), continuation suffixes >= PRIME_MIN_T take the tokenwise path.
- **decode_batch.rs**: the two gemma4 asserts (:553 unsplit, :669 ppn) become returned
  `Err` — defense in depth: any residual/future path that reaches them refuses PER-REQUEST
  (`Event::Error` to that session) instead of killing the process.
- **hybrid_forward.rs**: `prime_cache_batch` refuses gemma4 unconditionally (hole 2);
  `gemma4_prime`'s `assert_eq!(pos, 0)` becomes `Err` (hole 3's backstop).

## Gap 2 — thought/content separation: two defects, not one

Receipt (`think-smoke-20260807T094506Z.log`, gemma arms):
- thinking ON: `content='<|channel>thought\nThe user wants me to r…'` — thought text in
  content, because the reasoning splitter (`ToolStreamParser`) knows only the qwen `</think>`
  dialect and is armed on `caps.qwen_think`, which is FALSE for gemma4 (its template has
  `<|think|>`/`<|channel>thought`, never the `<think>` substring).
- thinking OFF: `content='ok<turn|><turn|><turn|>thought\n<channel|'` — `<turn|>` turn-end
  tokens leak as text AND generation does not stop, because `params.eos` carries only the
  GGUF `eos_id` (1) and gemma4's turn-end token `<turn|>` is not in it. The tokenizer already
  exposes `eog_ids()` (`<|im_end|>`, `<turn|>`, `<end_of_turn>`) — run_gen and gemma_gate use
  it; the SERVE path never did.

Fix shape:
- **worker step_admit**: `params.eos` unions `tok.eog_ids()` (llama's special_eog set) — the
  turn token stops generation and, per the existing EOS-text-never-streamed rule, never
  reaches the client as text. Covers ON and OFF arms and every model class (qwen's
  `<|im_end|>` is already its eos; the union is idempotent there).
- **toolcall.rs**: the reasoning splitter gains the gemma dialect — model-emitted opening tag
  `<|channel>thought\n` is syntax (stripped; a stream that does not open a channel is pure
  content), `<channel|>` closes the thought segment (its leading `\n` is syntax), content
  follows directly (no separator newlines). Same holdback/streaming discipline as the qwen
  dialect, unit-tested char-by-char.
- **main.rs**: `ModelCaps.gemma_think` (template contains `<|channel>thought`);
  think-open for the gemma dialect = ThinkMode::Think (template default is OFF, absent =
  Default = OFF), arming the gemma-dialect reasoning-only parser.

## Gates (all local 5090, CPU-capped)

- serve-smoke gains a gemma4 arm: 1 request on the DEFAULT scheduler must succeed (gap-1
  regression), thinking-ON request asserts reasoning non-empty + content free of
  `<|channel>`/`<channel|>`/`<turn|>`, thinking-OFF asserts clean content.
- run-gen argmax MATCH on gemma4 12B local (the local-ci depth-prompt arm).
- kernel-check ALL GREEN (no kernel touched — proven, not assumed).
- memra-server + memra-tokenizer test suites green.

## Results

**Gap 1 — SHIPPED, fail-closed shape (option b).** `eager_only_model()` predicate
(gemma4 + e4b), computed once at spawn, loud `EAGER-ONLY serving` log line. Sessions on
those models: per-session `step_session` eager decode inside the batched scheduler
(phase c-), excluded from batched chunks, never graph-promote, never join prime batches,
fresh prompts prime WHOLE (no chunked prime exists), carried suffixes tokenwise, LCP split
off. Engine backstops: the two `decode_step_batch*` gemma4 asserts → per-request `Err`,
`prime_cache_batch` refuses gemma4 unconditionally (was carried-only — fresh pairs walked
the generic concat core silently), `gemma4_prime`/`e4b_prime` pos==0 asserts → `Err`.
Receipts: `raw/postfix-*` (1-request 200 on DEFAULT scheduler, 3-concurrent all served,
EAGER-ONLY log line, zero panic lines). Commit 3809ae56.

**Gap 2 — SHIPPED, both defects.** (1) `params.eos` unions `tok.eog_ids()` — `<turn|>`
now stops generation (finish=stop at 2 tokens on the OFF arm, was finish=length with tag
soup) and never streams as text. (2) `ToolStreamParser::gemma_thought()` — the
`<|channel>thought\n…\n<channel|>` dialect routes to `reasoning`, tags/label/newlines are
syntax, channels split at any stream position; armed via `ModelCaps.gemma_think`
(template contains `<|channel>`) on every non-tools gemma4 chat request. 4 new unit tests
(one-shot == char-by-char, mid-stream channel, unclosed flush, partial-tag holdback).
Receipts: `raw/gap2-*` (OFF/ON/none/stream arms all clean, stream deltas separated).
Commit a97f3b03.

## Gates (all local 5090, CPU-capped via systemd-run, GPU verified idle before/after)

| gate | result |
|---|---|
| serve-smoke incl. NEW gemma4 arm (default-scheduler 1-request + thinking separation + alive + zero panics) | **0 failed** — `raw/serve-smoke-20260807T*.log`, 20/20 checks |
| run-gen argmax, gemma4 12B q4_0, depth prompt 1736 ids | **MATCH** (prefill==decode==batched-prime argmax 623) — `raw/rungen-g12-*.log` |
| kernel-check | **ALL GREEN** — `raw/kernel-check-*.log` |
| memra-server tests | **102/102** (98 + 4 new gemma-dialect) |
| memra-engine lib tests | 46 passed, 1 ignored |
| memra-tokenizer tests | 26 passed |

## Ledger

| item | state |
|---|---|
| gap-1 repro receipt on this binary | **DONE** — commit 6e94b2ab |
| gap-1 fix (fail-closed eager route + engine Err backstops) | **DONE** — commit 3809ae56 |
| gap-2 fix (eog stop union + gemma reasoning dialect) | **DONE** — commit a97f3b03 |
| gates (serve-smoke gemma4 arm, run-gen, kernel-check, test suites) | **ALL GREEN** |
| docs (SERVING.md reasoning-separation + gemma dialect) | **DONE** |
