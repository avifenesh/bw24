# SFT trace corpus — sft-corpus-20260802

Agentic/coding trace corpus generated with the paid Kimi K3 subscription (window: 18 days
from 2026-08-02), for distillation into Qwen3.6-35B-A3B. Tooling lives in
`tools/sft-pipeline/`. **This lane covers trace creation only.** TRAINING spend is gated
on the distribution verdict from `lane/finetune-sku` (running separately) **plus the
owner's explicit spend approval** — no training run starts from this lane.

## Directory layout

```
research/sft-corpus-20260802/
  raw/        as-received K3 trace JSONL (append-only; one file per generation session)
  converted/  Qwen3.6 ChatML training records (output of convert_k3_qwen.py)
  verified/   traces that passed verify_outcome.py (the only training-eligible pool)
  rejects/    rejects.jsonl — every discarded trace WITH its reason and captured
              evidence (owner rule: bad results are recorded too; never delete)
```

Raw traces are the primary artifact — conversion and verification are reproducible from
`raw/` + the pinned tool versions; nothing downstream is hand-edited.

## Pipeline

```
raw/*.jsonl
  -> validate_trace.py    structural gate (exit non-zero on any bad line)
  -> verify_outcome.py    verified-outcome filter (patch-apply + tests, answer checks);
                          failures -> rejects/rejects.jsonl with quoted evidence
  -> convert_k3_qwen.py   K3 -> Qwen3.6 ChatML (serve-template byte parity, --roundtrip)
  -> corpus_stats.py      Markdown report: counts, histograms, dedup, keep-rate
```

One-command batch validation of fresh traces:

```
python3 tools/sft-pipeline/validate_trace.py research/sft-corpus-20260802/raw/*.jsonl
```

Full pass over a fresh batch:

```
python3 tools/sft-pipeline/validate_trace.py raw/batch-N.jsonl \
&& python3 tools/sft-pipeline/verify_outcome.py raw/batch-N.jsonl \
     -o verified/batch-N.jsonl --rejects rejects/rejects-batch-N.jsonl ; \
python3 tools/sft-pipeline/convert_k3_qwen.py --roundtrip verified/batch-N.jsonl \
     -o converted/batch-N.jsonl \
&& python3 tools/sft-pipeline/corpus_stats.py verified/*.jsonl \
     --rejects rejects/rejects-batch-N.jsonl -o stats.md
```

(`verify_outcome.py` exits 1 when it rejected traces — that is a normal outcome, the
rejects file is the record; exit 2 means a harness error and the batch is not done.)

## Trace schema (one JSON object per line)

```jsonc
{
  "task_id":     "swebench-lite-0042",        // unique, stable; stem groups retries
  "seed_source": "swebench-lite",             // where the task came from
  "tools": [ /* OpenAI-shape function schemas offered to the model (optional) */ ],
  "messages": [
    {"role": "system",    "content": "..."},               // optional, index 0 only
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...",                // may be "" when calling
     "reasoning_content": "...",                            // K3 thinking (optional)
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "run_tests",
                                  "arguments": "{\"path\": \"tests/\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "<real captured output>"},
    {"role": "assistant", "content": "final answer / summary"}
  ],
  "outcome": {"verified": false,                     // verify_outcome.py rewrites this
              "method": "tests_pass",                // tests_pass | patch_applies |
              "detail": "pending verification"},     //   answer_check | manual
  "meta": {"model": "kimi-k3", "ts": "2026-08-02T12:00:00+00:00",
           "turns": 6, "sub_session": null},
  "verify": { /* optional spec consumed by verify_outcome.py, see below */ }
}
```

Alternation law (enforced): optional leading `system`; then `user`; `assistant` follows
`user` or a fully-consumed tool batch; every `tool_calls[].id` is answered by exactly one
`tool` message (`tool_call_id`) before the next non-tool message; the trace ends on an
`assistant` turn with no pending calls. Empty contents are illegal (assistant `""` is
allowed only when carrying `tool_calls`). Unknown fields are legal everywhere — the
converter preserves them under `meta.conversion`, never drops them silently.

### verify spec

- Patch tasks: `{"type": "patch", "repo": <pinned checkout>, "rev": <sha>,
  "patch": <unified diff> | "patch_path": <file>, "test_cmd": [argv...], "timeout": s}`.
  The harness applies the patch to a throwaway `git worktree` of the pinned rev and runs
  the test command; verdicts carry the captured exit codes + stdout/stderr tails. Method
  is `tests_pass` when a test command ran, `patch_applies` otherwise.
- Answer tasks: `{"type": "answer", "expected": <exact>}` or `{"pattern": <regex>}`,
  matched against the last assistant message with `<think>` blocks stripped.
- `method: manual` traces pass only under `--allow-manual` AND only when they already
  carry `verified: true` plus non-empty evidence in `detail`.

## Generation protocol (summary)

- **Real tool execution only.** Every `tool` message must contain output actually
  produced by running the tool. Imagined/hallucinated tool outputs poison the corpus —
  a trace whose tool results cannot have come from a real execution is discarded on
  sight (to `rejects/`, with the reason).
- **Verified-outcome filter.** Only traces in `verified/` are training-eligible. "The
  model said it passed" is not verification; the harness re-runs the check and quotes
  the output.
- **Failure-recovery traces are welcome.** A trace where the model hits a real error,
  reads the real error text, and recovers to a verified outcome is high-value signal.
  What gets rejected is an unverified *final* outcome, not mid-trace failure.
- Rejects are never deleted: `rejects/rejects.jsonl` keeps the full trace, the reason,
  and the captured evidence.
- `seed_source` is recorded per trace so the mix can be rebalanced later; dedup runs on
  task-id stem + first-user-message hash (`corpus_stats.py`).

## Target format

Converted records render the qwen3.5/3.6-class ChatML **tools branch** byte-identically
to memra's serve surface (`crates/memra-tokenizer/src/chat.rs::apply_chat_template_tools`)
— same `<tools>` system header + fixed instruction block, same `<tool_call>/<function=/
<parameter=` call rendering, same grouped `<tool_response>` user turns. Parity is pinned
by `convert_k3_qwen.py --selftest` (golden vectors copied from the Rust tests) and
`--roundtrip` re-parses every rendered record. Training text carries no generation
prompt; the final assistant turn is the target. K3 `reasoning_content` is embedded as a
`<think>` block on the final assistant turn only (the qwen3.6 convention); earlier-turn
reasoning moves to `meta.conversion.dropped_reasoning`.

## Pilot target

1–2k **verified** traces (i.e. `verified/` count, not raw count) inside the K3
subscription window. Track keep-rate from `corpus_stats.py`; if keep-rate makes the
pilot infeasible, that is a finding to report, not a reason to loosen verification.

## Gates

- Trace **creation** proceeds now — the K3 subscription is already paid.
- **TRAINING is gated**: distribution verdict from `lane/finetune-sku` + the owner's
  explicit spend approval. Nothing in this lane schedules or pays for a training run.
