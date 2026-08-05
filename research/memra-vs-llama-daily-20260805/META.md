# memra vs llama — daily 27B dogfood diagnostic — 2026-08-05

**NOT BOARD MATERIAL.** llama benching is doctrine-stopped for the tracked boards; this
cell is a dogfood-experience diagnostic the owner explicitly asked for ("27B gets 73
tok/s? I think I got better with llama"). Numbers here answer the owner's experienced
comparison, not the competition posture.

## Rig / environment

- Local RTX 5090 laptop rig (24463 MiB), `gpu-full-power on` for every phase
  (boost=25, profile=performance — the owner's serving power state), reverted at exit.
- Co-resident (both phases identically): llama-server embed stub port 8181
  (bge-small, -ngl 0, 332 MiB — CPU-only, present in the owner's daily use too).
- GPU lock: /tmp/gpu5090.lock held per phase, released between phases (F5 lane priority;
  queue marker left in /tmp/gpu5090-coordination.txt).

## Servers (the owner's ACTUAL daily configs)

- Artifact (both): `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
  (inode 41685701 — identical file via both the /data and ~/ai-ml paths).
- **memra**: binary snapshot `~/tmp-dogfood/memra-server-c716954b`
  (md5 48aa272dca86013aef91f025340de131 = bw24-unified target/release build,
  system_fingerprint `memra-c716954b6fef`, the v0.69.0-era train). Exact
  `serve-qwen36-27b-memra` env: regime draft `draft-daily-owntrim-nvfp4head-q4blk.gguf`
  attached (+draft), spec default-ON, MEMRA_CTX=131072, MEMRA_MAX_SESSIONS=1,
  MEMRA_REUSE_POOL=1, MEMRA_PRIME_CHUNK=2048, port 8002.
- **llama**: `~/projects/llama.cpp/build/bin/llama-server` version 9837 (c73069749),
  exact `serve-qwen36-27b` flags: `--model-draft mtp-Qwen3.6-27B-Q4_K_M.gguf
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 --ctx-size 131072
  --ubatch-size 512 -ngl 999 -ngld 999 -fa on --parallel 1 --cache-type-k q8_0
  --cache-type-v q5_1 --cache-ram 0 --jinja`, port 8001.

**llama's daily script DOES run the MTP draft** (draft-mtp, n-max 3, p-min 0.1) — the
comparison is spec-vs-spec, like-for-like AND as-configured. The drafts differ by
design: memra's daily regime draft is the own-gen-trimmed NVFP4-head build (the
2026-07-18 board move); llama's daily is the full Q4_K_M MTP draft. Both are each
stack's owner-tuned daily configuration.

## Arms (per rep; memra phase then llama phase, interleaved x5)

| arm | server | sampling |
|---|---|---|
| memra-t0.8 | memra | temp 0.8, memra defaults (top_p 1, top_k off, min_p 0), seed varied |
| memra-t0.8-lsampler | memra | temp 0.8 + llama's daily truncation (top_k 40, top_p 0.95, min_p 0.05) — isolates the sampler-shape effect on acceptance |
| memra-t1.0 | memra | temp 1.0 untruncated = the ACTUAL daily memra path (pi omits temperature; memra default-when-omitted = 1.0) |
| memra-greedy | memra | temp 0 — control, anchors vs the board rows |
| llama-default-t0.8 | llama | temperature omitted = llama server defaults: temp 0.8, top_k 40, top_p 0.95, min_p 0.05 — the ACTUAL daily llama path |
| llama-t1.0 | llama | temp 1.0, other defaults intact |

Seeds: 1000+rep on every sampled arm, both servers.

## Cells

- **short-agentic** — the owner's tool-check shape (pi system prompt + tool instruction,
  max 160 tok).
- **long-gen** — 512-token essay-class generation.
- **ctx4k** — ~3.9k-token agentic-log continuation, max 256 tok.
- Per-request nonce in the user turn defeats both servers' prefix caches: every TTFT is
  a cold full prefill, and memra exercises the daily-real pool-miss path (F5: pi rewrites
  history, so the owner's real turns miss the reuse pool nearly every time).
- One warmup request per phase (excluded).

## Timing (client-side, identical for both; see scripts/driver.py header)

- ttft_s = first text chunk − send; decode_tok_s = char-weighted tokens-after-first-chunk
  / stream duration; e2e_tok_s = completion_tokens / total wall.
- Server-truth cross-checks captured per row: llama `timings.*`
  (predicted_per_second, draft_n/draft_n_accepted), memra `usage.elapsed_s`.

## Files

- `runs.jsonl` — every row (raw, append-only).
- `logs/server-{memra,llama}-rN.log` — full server logs (memra spec-acc lines; llama
  draft stats), gpustate stamps in `logs/driver-*.log`.
- `RESULTS.md` — table + verdict.
