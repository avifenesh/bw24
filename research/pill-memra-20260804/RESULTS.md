# pill → memra dogfood wiring — 2026-08-04

The "real client, real usage" test the gap-scan said we never do: the owner's daily-driver
agent harness (`pi`, aliased `pill`) wired to memra-server serving the ACTUAL daily model,
Qwen3.6-27B NVFP4+MTP, on the 5090 rig. Server binary built at train HEAD 2299ee0f
(includes the #68 serve-spec fix ac99e675 — required for long agentic sessions).

## What was wired

- **Serve script**: `~/.local/bin/serve-qwen36-27b-memra` (versioned copy:
  `tools/serve-examples/serve-qwen36-27b-memra`). Port 8002. Serves the SAME artifact the
  llama daily uses (`/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`,
  inode-verified identical to the `~/ai-ml` path the llama script reads — same ext4 volume)
  with the regime drafter attached via `+draft` syntax
  (`draft-daily-owntrim-nvfp4head-q4blk.gguf`, the 2026-07-18 board-move own-gen trimmed
  draft). Spec decode default-on; startup log confirms
  `regime draft attached` and `[spec-acc]` bursts on every request. Supersedes
  `serve-qwen36-27b-bw24` (old bw24-server binary + BW24_* env; left in place untouched —
  owner may delete).
- **Provider**: `local-memra` added to `~/.pi/agent/models.json` (backup:
  `models.json.bak-20260804`; exact diff: `models.json.diff` here — ~/.pi is owner config,
  NOT committed). Same api/compat class as `local-moe` (openai-completions,
  usage-in-streaming, max_tokens), model id `qwen36-27b`, name "memra 27B (daily)",
  reasoning + thinkingFormat qwen-chat-template, contextWindow 131072.

## Config (owner requirements honored)

- **Drafter + spec ON**: binding requirement. `[spec-acc]` engaged on every smoke;
  short-chat acceptance ~0.70-0.74, agent-turn ~0.53-0.65, essay-class ~0.46, 36.5k-deep
  ~0.41-0.43 (matches the depth-acceptance profile in the board data).
- **ctx = 128k default / 160k `full`** (matches the llama twin's modes). KV math @131072:
  17/65 KV-bearing layers × (K q8_0 4h·256d = 1088 B + V q5_1 = 768 B)/tok = 31,552 B/tok
  → 4.13 GB KV; session VRAM observed by the server: **4194 MB** (log line, matches the
  math). Weights ~16 GB (NVFP4 + 1.2 GB trimmed draft) + session 4.2 GB ≈ 21 GB of 24 GB.
  KV formats are memra defaults q8_0/q5_1 — the same formats the llama daily runs, and the
  measured winners (fp8-K FLIP-BLOCKED, q4_0-V quality-taxed; docs/FLAGS.md).
- **MEMRA_MAX_SESSIONS=1, MEMRA_REUSE_POOL=1**: two 128k sessions don't fit 24 GB;
  interactive queues FIFO beyond the cap (never shed). Pool=1 keeps the agent's own
  continuation resume while halving parked-cache pressure.

## Smoke verdicts (real client: `pi -p -ne --provider local-memra --model qwen36-27b`)

pi could NOT initially run at all: default extension load fails on an unrelated broken
extension (`valibot` missing in ~/projects/tools). `-ne` bypasses it. That's a pi-install
issue, not a memra issue; interactive `pill` sessions load extensions the same way, so the
owner's environment already tolerates or fixes this.

| class | prompt | verdict |
|---|---|---|
| short chat | capital of France | PASS — "The capital of France is Paris." (smoke1) |
| tool-use | read /tmp/pill-smoke/config.ini, report beta (pi built-in read tool, full agentic turn loop) | PASS — "The value of `beta` is **2**." Multi-turn: spec sessions engaged per turn (smoke2) |
| long generation | 1200+ word essay (the #68 regime, >800 tok) | PASS — 1812 words emitted, no truncation/garbage, EXIT 0 (smoke3) |
| long context | 36,524-token prompt (worker.rs paste) + question | PASS on final config — TTFT 33.9 s (~1078 tok/s effective prefill @2048 chunks), correct 2-sentence answer, stream + usage correct, 1.29 GB free at end (smoke4-longctx-chunk2048.txt) |

Envelope (raw capture, `stream-envelope-completions.txt`): OpenAI SSE `text_completion`
chunks, terminal chunk carries `usage` (prompt/completion/total + cached_tokens split +
elapsed_s) then `data: [DONE]` — exactly the `supportsUsageInStreaming: true` shape pi
expects. Reasoning/content split is CLIENT-side for pi (thinkingFormat qwen-chat-template
over raw completions): `<think>` blocks arrive as literal text, pi parses them — verified
non-headless-visible thinking did not leak into the final `-p` answers, no template garbage,
no literal `<|im_end|>` in output.

## Findings

### F1 (fixed in config): 36.5k-token prefill OOMs at the default prime chunk (4096)

Reproduced twice — on a mid-life server AND a fresh boot (`server-run2-oom.log`):
`step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` streamed as a clean
SSE error chunk; GPU at failure 23,942/24,463 MiB used, compute-apps = memra-server
23,586 MiB + llama-server 260 MiB (idle 35B moe stub). Cause class: per-chunk prime
transients (PrimeSlabs gate/up/act = T·n_ff·4B ≈ 285 MB apiece at T=4096 on n_ff=17408)
plus the dequant-once FA workspace over the deep past, on top of weights ~16 G + the
4.2 GB session — the margin at 4096 goes negative around ~36k on a 24 GB card.
`MEMRA_PRIME_CHUNK=2048` (halved transients) passes the same probe with 1.29 GB free.
Baked into the serve script. Candidate engine improvement (not taken here): scale the
default chunk down with remaining VRAM, or retry the chunk at half size on alloc failure
— the OOM currently kills the request instead of degrading prefill speed.

### F2 (server behavior, documented): `max_tokens` can overshoot by <burst overshoot> on spec

`max_tokens=50` returned `completion_tokens=53`. Spec bursts commit accepted drafts past
the budget (session-mode contract in spec.rs: committed must match cache rows; worker
streams the whole burst then checks `generated >= budget`). OpenAI semantics treat
max_tokens as a hard cap; pi tolerates it (finish_reason=length is correct, count honest).
Off-by-a-few on a 4-8 token burst tail, worst case K+3. Not fixed here: clamping the
STREAM to budget while committing overshoot to the session is a worker-side change that
touches the park/continuation contract (emitted_bytes vs committed divergence) — flagged
as a serve-compat work item rather than hot-patched into the dogfood lane.

### F3 (pi-side, noted): broken unrelated extension blocks default startup

`pi` (and `pill`) fail hard at startup if any discovered extension fails to load
(`valibot` missing in ~/projects/tools/packages/read). Headless smoke used `-ne`.

## Receipts

- `smoke1-short-chat.txt`, `smoke2-tooluse.txt`, `smoke3-longgen.txt` — pi -p transcripts (final config)
- `smoke4-longctx-output.txt` — the OOM probe on the fresh default-chunk server (F1 evidence)
- `smoke4-longctx-chunk2048.txt` — the passing long-context probe (final config)
- `stream-envelope-completions.txt` — raw SSE capture, pi request shape
- `server-run1.log` — first server run (smokes 1-3 + first OOM, mid-life)
- `server-run2-oom.log` — fresh-boot OOM repro (F1)
- `server-run3-final.log` — final config run (all smokes green)
- `models.json.diff` — the exact ~/.pi/agent/models.json change (owner config, not committed)

## One-liner to live on it

```bash
serve-qwen36-27b-memra            # port 8002, 128k, drafter+spec on (full = 160k)
pill --provider local-memra --model qwen36-27b
```

llama daily stays untouched on 8001 (`serve-qwen36-27b`); switching back is dropping the
`--provider/--model` overrides.
