# darklanes serving v1 — replica-per-GPU multi-user bring-up (2026-08-01)

Box: darklanes-8x (8x H100 80GB), serving lane GPUs 5/6/7.
Model: Qwen3.5-9B-Q8_0.gguf (9.5 GB gguf, 16.7 GB resident per replica).
Engine: memra-server @ box-prebuilt `~/memra/target/release/memra-server`.
Receipts: `load-points.jsonl` (one line per load point), `per-request.jsonl`
(per-request latencies), `logs/` (replica + proxy + sweep raw logs). Every load
point below is a single run (N=1 run; per-point request count shown as `n`),
H100s otherwise idle, no other GPU tenants (verified nvidia-smi).

## 1. Device selection

`memra-server` hardcodes device 0: `Engine::new(0)` at
`crates/memra-server/src/worker.rs:232` (one OS thread owns the CUDA context and
every loaded model). Therefore `CUDA_VISIBLE_DEVICES=<n>` per process is the whole
placement mechanism — confirmed empirically: three processes launched with
`CUDA_VISIBLE_DEVICES=5|6|7` landed 16,659 MiB each on physical GPUs 5, 6, 7
(nvidia-smi), GPUs 0-4 untouched.

Replica invocation (per process):

```
CUDA_VISIBLE_DEVICES=5 MEMRA_COMPAT=openai \
  MEMRA_MODELS="qwen=/home/ubuntu/models/Qwen3.5-9B-Q8_0.gguf" \
  MEMRA_ADDR=127.0.0.1:8085 ./target/release/memra-server
```

Ports 8085/8086/8087 = GPU 5/6/7. `MEMRA_COMPAT=openai` selects the OpenAI
response shapes (same surface serve-smoke.sh gates).

## 2. Front routing v1

`tools/serve-proxy.py` — python3 stdlib ThreadingHTTPServer reverse proxy on
:8080, least-outstanding-requests routing (ties -> lowest index), 2s-interval
health probes that pull dead replicas from rotation, SSE chunked relay.
One fix found under load: stdlib's default listen backlog (`request_queue_size=5`)
dropped 10/256 connections (ECONNRESET) at c=64; raised to 256 -> 0 errors.
Routing balance over the c=64 + c=24 runs: 122/116/116 requests per replica.

## 3. Load harness

`tools/load-serve.py` — N worker threads looping non-streaming
`/v1/chat/completions`: ~200-token prompt, `max_tokens=128`, temperature 0.7 with
per-request seeds (realistic divergent sequences for batched decode; `--greedy`
for determinism checks), 1 warmup request, aggregate tok/s = sum(completion_tokens)/wall.

## 4. Scaling results

Aggregate output tok/s and per-request latency (p50/p95 seconds):

| c  | single replica (GPU5) | p50    | p95    | 3-replica proxy | p50   | p95   |
|----|----------------------:|-------:|-------:|----------------:|------:|------:|
| 1  | 130.7                 | 0.979  | 0.990  | 131.0           | 0.977 | 0.978 |
| 4  | 270.9                 | 1.887  | 1.900  | 395.0           | 1.294 | 1.297 |
| 8  | 308.9                 | 3.311  | 3.369  | 641.5           | 1.568 | 1.610 |
| 16 | 307.3                 | 6.664  | 6.752  | 756.9           | 2.291 | 2.636 |
| 24 | —                     | —      | —      | 904.6           | 3.345 | 3.437 |
| 32 | 305.0                 | 13.411 | 13.546 | 824.3           | 4.877 | 4.937 |
| 64 | 303.6                 | 26.956 | 27.211 | 874.9           | 9.066 | 9.409 |

(proxy c=32 predates the backlog fix but had 0 errors; proxy c=64 shown is the
post-fix clean rerun — the pre-fix run's row, 885.9 with 10 resets, is in the
jsonl. c=24 was added as the matched-saturation point, 8 outstanding/replica.)

Reference — 3 harnesses driven directly at the replicas in parallel (no proxy),
c=8 each: 306.9 + 302.7 + 306.3 = **915.9 tok/s aggregate**.

### Findings

- **Single-replica saturation is c=8, ~308 tok/s.** Throughput is flat
  (303-309) from c=8 to c=64 while p50 latency doubles with each doubling of c —
  pure queueing. This matches the engine internals: the batched scheduler admits
  up to `MEMRA_MAX_SESSIONS` (default 64) sessions but advances decode through
  `decode_step_batch` in **chunks of <= 8** (worker.rs tick phase c). Replica
  /metrics after the sweep: `step_p50_ms=24.36` -> 8 tokens / 24.4 ms = 328 tok/s
  decode ceiling; measured 308 includes prefill ticks. So max useful per-replica
  concurrency today = 8; beyond that clients only buy latency.
- **3-replica scaling: 2.93x at matched saturation.** Proxy c=24 (8/replica) =
  904.6 tok/s vs 3x single-c8 arithmetic 926.7 and measured direct-3x 915.9.
- **Proxy overhead: ~0 at c=1** (p50 0.977 vs 0.979 — within noise of the direct
  path), **~1.2% at saturation** (904.6 vs 915.9 direct). The stdlib
  thread-per-request proxy is not the bottleneck at this scale.
- **Over-admission slightly hurts:** proxy c=64 (874.9) < c=24 (904.6). With
  ~21 outstanding per replica everything past 8 queues; tail waves drain
  unevenly and the deeper queues add scheduler churn. No admission control
  exists to stop this (v2 gap).
- **Correctness:** identical greedy (temperature=0, seed=0) completion — same
  sha256 (`dbd1c98f9fed4efe...`) — from all three replicas on the same prompt.
- Single-stream 130.7 tok/s here is NOT the tuned single-user e2e number (~204):
  this harness uses plain chat decode with no `+draft` regime attach and a
  200-token prompt; the tuned number rides the spec path. Not a regression —
  different config, noted to prevent cross-report confusion.

## 5. v2 gaps

1. **Session affinity / KV reuse.** Routing is stateless; a multi-turn
   conversation re-prefills its whole history on whichever replica it lands on.
   Need consistent-hash or session-id pinning to the replica already holding the
   KV, plus the engine-side prefix-cache reuse story.
2. **Queue admission + backpressure.** Nothing sheds load; at c=64 p95 hits 9.4s
   and throughput sags below the c=24 peak. Proxy should cap per-replica
   outstanding (~8-12), queue with a deadline, and 429 beyond it.
3. **Per-replica batch ceiling is the single-GPU lever.** The chunk-of-8
   decode_step_batch caps a replica at ~308 tok/s aggregate while the GPU holds
   16.7/80 GB. Raising the batch chunk (or graph-batched wider decode) is where
   per-GPU multi-user throughput lives.
4. **Per-GPU model diversity.** memra-server already serves multiple models per
   process (MEMRA_MODELS is a list); the proxy routes blindly by load. v2:
   model-aware routing table (model -> replica set), heterogeneous replicas.
5. **MPS / multi-replica-per-GPU packing.** 60+ GB idle per GPU at 9B-Q8; two+
   replicas per GPU under MPS (or one process with a bigger batch) before
   scaling out to more GPUs.
6. **Proxy hardening.** New TCP connection per forwarded request (no keep-alive
   pool), no retry-on-replica-death mid-request, no TTFT/streaming metrics; SSE
   relay works but was not load-tested.

## 6. Receipts

- `load-points.jsonl` — every load point (including the pre-fix error run).
- `per-request.jsonl` — per-request latency/tokens rows for the sweep points.
- `logs/replica-808{5,6,7}.log`, `logs/proxy.log`, `logs/sweep.log` — raw.
- `run-sweep.sh` — the exact sweep driver (params baked as literals).
- Box copy: `~/darklane-serving-20260801/` on darklanes-8x.
- Code: `tools/serve-proxy.py`, `tools/load-serve.py` (this repo).
