# serve-st lane — safetensors-checkpoint serving in memra-server (FP8-ST program item 7)

Branch `lane/serve-st` off `restructure/public-split` @ 9461c7bf. Rig: local RTX 5090,
GPU work under `flock /tmp/gpu5090.lock`. Dates: 2026-08-03/04.

## What landed

1. **Model plan** (`21fd710c`): a `MEMRA_MODELS` path may be an HF safetensors checkpoint
   DIR (`config.json` + `model.safetensors[.index.json]`) or a repack dir
   (`manifest.json`), validated at parse time (`validate_model_path` in
   crates/memra-server/src/main.rs) — bogus dirs fail fast naming what was expected.
   The worker's dir branch (SafetensorsSource -> HybridModel::load_from_source +
   Tokenizer::from_hf_dir) predates this lane; the plan seam + validation + tests are new.
2. **Chat template** (`040dbcfa`): dir checkpoints resolve their template through the
   existing from_hf_dir seam (tokenizer_config `chat_template`, else
   `chat_template.jinja`). New `ModelCaps::chat_ok`: a template-less DIR checkpoint 400s
   on /v1/chat/completions with a message naming the missing files and pointing at
   /v1/completions; GGUF keeps the historical ChatML fallback.
3. **Gate + quarantine** (`163328e5`): `tools/serve-st-gate.sh` (see below) + the
   ST-spec quarantine.

## CLI-vs-server exactness (gate item 3)

Same checkpoint, same prompt ("What is the capital of France? Answer in one short
sentence."), same template render, greedy, 64 new tokens. CLI arm = `run-gen <dir>
--prompt` (MEMRA_CHAT=1, tokenwise decode); server arm = `/v1/completions` with
`chat:true` on `MEMRA_SERVE_SPEC=0` (native response carries raw token ids).

| checkpoint | verdict |
|---|---|
| qwen35-4b-hf (BF16, 2 shards) | IDENTICAL 64/64 ids (`serve-st-gate-run3.log`) |
| qwen35-9b-nvfp4-st-modelopt (NVFP4, single file) | IDENTICAL 64/64 ids (`serve-st-gate-9bst-run1.log`) |

Server Token events stop BEFORE the EOS id, CLI includes it — the gate tolerates only
that trailing-EOS length difference (neither run hit EOS inside the window here).

## ST-SPEC QUARANTINE (found live in this lane)

Serving a dir checkpoint with spec ON diverged from plain greedy; single runs, but the
divergence is deterministic per arm (spec rep1 == rep2 byte-identical):

- 4B BF16, default spec (graph draft): text corrupts outright ~250 tok in
  (`"The capital capital of...":":}}"\\:}\\}{}` …) — plain 1573 chars vs spec 1083.
  `MEMRA_SPEC_NOGRAPH=1` arm: MATCH 1573/1573. So the 4B corruption is the DRAFT GRAPH
  arm on ST weights.
- 9B NVFP4 ST: diverges at a near-tie token ("questions" vs "queries", char 551) even
  at `MEMRA_SPEC_K=1` with nograph — a logits-delta class, not corruption.
- CONTROL: run-spec CLI self-consistency on the SAME dir checkpoints PASSES K=1..8
  (4B: `runspec-4b-selfcheck.log`, 200 and 400 tok; 9B ST: `runspec-9bst-selfcheck.log`).
- CONTROL: GGUF 9B NVFP4 served spec-vs-plain: MATCH (same binary, same day).

So the fault is specific to the WORKER's generate_spec_session path on dir-loaded
weights — root cause OPEN (suspects: SpecSession continuation prime vs the CLI's
fresh-prompt spec, draft-graph capture interacting with ST-loaded MTP head tensors).
Quarantine: dir checkpoints are spec-INELIGIBLE in serve unless MEMRA_SERVE_SPEC=1 is
explicit (loud notice at load). GGUF spec serving untouched. Follow-up owner: lift only
on a green ST serve-spec exactness gate.

## Gate battery (tools/serve-st-gate.sh)

Both checkpoints ALL GREEN (runs 3 + 9bst-run1): /models lists, chat coherent (Paris)
through the ckpt's own template, CLI-vs-server 64/64 ids, quarantine notice logged,
default server text == MEMRA_SERVE_SPEC=0 text.

## serve-smoke regression check (GGUF, post-change)

`tools/serve-smoke.sh` on the 9B NVFP4 GGUF pair: fail set must remain EXACTLY the 4
receipted pre-existing fails (research/constrained-20260803/RESULTS.md — chat
non-stream / greedy determinism / concurrency / spec-vs-plain, the think-tail content
routing at small max_tokens condition). See `serve-smoke-post.log`.

## Unit tests

`cargo test -p memra-server` 40/40 PASS, incl. new:
- `model_plan_accepts_st_dir_and_rejects_bogus_dir` (accept single/sharded/repack/
  gguf-file; reject empty dir / missing config.json / nonexistent, message content pinned)
- `chat_on_templateless_dir_checkpoint_is_rejected_with_clear_message` (400 wording pinned)

## CLOSURE (2026-08-04, lane/fp8-ship): quarantine LIFTED — #68 root-caused and fixed

The fault was never ST-specific: the per-session persistent draft graph (2026-08-01)
replayed with dangling pool addresses (capture transients not retained on the session +
`fa_part_pool` freeing grown-past buffers the capture baked). Reproduced on GGUF session
bursts at n>=600 with the new `spec-st-probe` harness; the GGUF "MATCH" control above was
a 400-token window under the corruption onset. The 9B "near-tie flip" was reclassified:
the serve-spec arm matches the CLI tokenwise oracle — the batched PLAIN serve arm is the
outlier (the accepted decode-config near-tie class). Full root cause, elimination table,
fix, and post-fix gates: `research/fp8ship-20260804/RESULTS.md`. serve-st-gate item 4 now
pins default-serve (spec ON) text against the tokenwise serve oracle; both checkpoints
0 failed post-lift.
