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

---

# Round 2 — the chunk-of-8 ceiling, attacked (2026-08-01, GPU 5 only)

## R2.1 What the cap actually is

Three layers, found in code:

1. **Scheduler chunking** — `group_chunks` (`crates/memra-server/src/worker.rs:1097`)
   hardcoded `c.len() < 8`.
2. **Engine policy assert** — `decode_step_batch` refused `B > 8`
   (`crates/memra-engine/src/decode_batch.rs:42-47`): "crosses the m>=16 GEMM tier
   (a different numeric config) — refused until the batched-tier exactness policy lands".
3. **Kernel-shape reality** (`matmul_pre`/`matmul` dispatch, `lib.rs`): B=2..8 rides the
   batched weight-resident mmvq arms (`_b2/_b4/_b8` — one weight read serves all B rows,
   per-(token,row) bit-identical to isolated m=1 decode). m=9..15 falls to the grid.y=m
   dp4a tail (m FULL weight re-reads per projection + a different 128-thread reduce
   shape). m>=16 crosses into the tensor-core GEMM tier (block-scale f32 rounding).
   `batched_kernel_name` had b16 arms wired for Q4_0/Q6_K only — but
   **`qmatvec_q8_0_mmvq_b16_rp` was already compiled** (`cu/qmatvec.cu:6730`, same
   template family as b2/b4/b8, instantiated for the q8rp split-plane mirror lane that
   this Hopper box builds by default: "[q8rp] split-plane decode mirrors built: 249
   tensors"), just never dispatched.

So: soft cap in code, backed by a real numeric-tier boundary, with an unwired
exactness-grade b16 kernel sitting in the fatbin for exactly our model class.

## R2.2 What was changed (branch lane/serving-v1, measurement door)

- `MEMRA_DECODE_BATCH_CAP` (default 8, clamp 1..32): parameterizes both the engine
  assert and `group_chunks`. Default behavior byte-identical to before.
- Wired Q8_0 b16: `(QT_Q8_0, 16)` added to `batched_kernel_name`; the three m>8
  dispatch gates (`matmul_pre`, `matmul_decode_exact`, `matmul`) admit Q8_0 iff the
  q8rp mirror is present (the b16 kernel exists only as the `_rp` twin).

## R2.3 Measured (single runs each, fresh server restart per config, GPU 5)

Aggregate tok/s (~200-tok prompt, 128 gen, temp 0.7):

| config (binary/chunk)      | c=8   | c=16  | c=32  | c=64  | batched-exactness (24 greedy) |
|----------------------------|------:|------:|------:|------:|-------------------------------|
| pre-b16 / 8                | 304.4 | 305.6 | 261.2 | 302.5 | PASS 24/24                    |
| pre-b16 / 16               | 271.8 | 169.2 | 168.8 | 168.3 | FAIL (3/24 shift)             |
| pre-b16 / 32               | 268.8 | 168.3 | 237.5 | 237.1 | FAIL (4/24 shift)             |
| b16 / 8 (control)          | 302.2 | 305.2 | 241.9 | 297.4 | PASS 24/24                    |
| **b16 / 15 (bit-exact)**   | 263.2 | 260.1 | 269.4 | 274.4 | **PASS 24/24**                |
| b16 / 16                   | 306.5 | 181.5 | 181.1 | 180.4 | FAIL (3/24 shift)             |

Greedy single-prompt hash: `dbd1c98f9fed4efe` in EVERY config (B=1 decode is
unchanged by the cap; only concurrent batches cross tiers).
The c=32 dip on both chunk-8 runs (261/242 vs ~305 at c=16/64) is a repeating
single-run anomaly worth a separate interleaved look, not a conclusion.

### Findings

1. **The exactness contract is real and measurable at the serving surface.** Any
   config that lets chunks cross m=8 without the b16 arm (or cross m=16 at all)
   shifted 3-4 of 24 identical greedy requests — first divergence ~100-170 chars in.
   The wired b16_rp at cap 15 is the only wide config that stayed 24/24 byte-identical
   to isolated, confirming the b16 kernel's bit-exactness contract end-to-end.
2. **Wider is not faster — the ceiling is NOT weight bandwidth.** chunk 15 (bit-exact,
   one weight sweep per tick for 15 sessions) still lands at 260-274 tok/s, BELOW
   chunk 8's ~305. The dp4a tail (m=9..15 pre-b16) is catastrophic (169) and the
   m=16 GEMM tier is worse than 2x8 batched mmvq (181 vs 305 — the GEMM crossover
   was tuned for prefill-scale m, not 16). Arithmetic: one full rp-trunk weight sweep
   is ~2.8ms on this HBM (~9.5GB @ 3.35TB/s) but the B=8 tick p50 is 24.4ms — weight
   streaming is only ~12% of the tick. The tick is dominated by work that scales
   per-sequence (per-seq fa_decode/KV appends, GDN state pointer chases, host
   sample/emit, launch counts), so widening the GEMV amortizes almost nothing and
   the untuned b16 kernel (no r2/pf variants exist at MCOLS=16) gives some of it back.
3. **Verdict: chunk 8 stays the default** — it is simultaneously the fastest AND the
   exact config. `MEMRA_DECODE_BATCH_CAP` stays as a documented measurement door
   (docs/FLAGS.md §7 Serving); if the owner prefers the strict doctrine (kill flags
   for flat experiments), the JSONL here is the record and the flag can go.
4. **Side product worth keeping:** the Q8_0 b16_rp wiring makes
   `matmul_decode_exact` m=9..15 take ONE weight read instead of m on q8rp-mirrored
   models — the same cliff the b8 tier fixed for K=4..7 spec verify. That band is the
   spec-verify t=9..16 tier (adaptive-K > 8), untested here; needs the run-spec
   K=1..8+ battery on an affected model before any default claim.

## R2.4 The real lever for >305 tok/s per GPU

VRAM per admitted session (sampler CSV `vram-gpu5.csv`, identical across all 6
configs): baseline 16,659 MiB (9.5GB weights + ~6.4GB q8rp mirror + ctx), first-load
step to ~23,100 (fixed activation/allocator scratch + first 16 session caches — c=8
and c=16 plateau at the SAME level), then ~150 MiB/session (16->32) and ~214
MiB/session (32->64); peak 32,405 MiB at 64 sessions (8192-ctx floor caches).
Extrapolated sessions-to-80GB: ~290-300. **VRAM is not the limit; the scheduler cap
(MEMRA_MAX_SESSIONS=64) and the per-seq-serial tick are.**

Ranked next steps for per-GPU throughput:
1. **Two replicas per GPU** (2 x 32.4GB peak = 64.8GB, fits; the tick is not
   BW-bound so two processes share HBM cheaply) — projected ~2x305 = ~610 tok/s/GPU
   with zero engine work; needs an MPS/context-share check and a measured run.
2. **Batch the per-seq serial parts** — a blockIdx.z-batched fa_decode over B caches
   (the "v2 fusion" already named in decode_batch.rs's header for GDN), batched KV
   append, device-side sampling for the batched path (dc-style 4B/token traffic
   instead of B full logits rows D2H per tick).
3. Only after (2) moves the tick toward weight-stream-bound does a tuned b16/b32
   family (r2/pf variants at MCOLS=16) become the right investment.

## R2.5 Round-2 receipts

- `chunk-sweep.jsonl` / `chunk-sweep-per-request.jsonl` — all 24 load points.
- `chunk-exact.jsonl` — exactness verdicts + divergence samples; `isolated-refs.json`
  — the 24 isolated greedy references (collected once at chunk 8, reused everywhere).
- `vram-gpu5.csv` — 1 Hz GPU-5 memory.used for the whole campaign.
- `logs/chunk-sweep.log`, `logs/b16-sweep.log`, `logs/replica-8085-*chunk*.log` — raw.
- `run-chunk-sweep.sh`, `run-b16-sweep.sh` — exact drivers.
- Code: `crates/memra-engine/src/decode_batch.rs` (cap door),
  `crates/memra-server/src/worker.rs` (chunking), `crates/memra-engine/src/lib.rs`
  (Q8_0 b16_rp wiring), `tools/check-batch-exact.py` (the serving-level exactness
  harness). End state: GPU 5 replica restored to default (cap 8) on the new binary,
  hash-verified; GPUs 6/7 untouched on the original binary.

---

# Round 3 — two replicas per GPU, measured (2026-08-01, GPUs 5-7)

## R3.1 Setup

Two co-resident memra-server replicas on GPU 5 (ports 8085 + 8088, both
`CUDA_VISIBLE_DEVICES=5`, default chunk 8), GPUs 6/7 unchanged as stability
control. Load = direct 2-harness (one per replica, same protocol as the round-1
direct reference — no proxy confound); pair concurrency {8,16,24,32} = per-replica
{4,8,12,16}. Single runs per point; the timeslice c16/c32 points were re-run
same-conditions (below) and reproduced within 0.1-2.7%.

MPS recipe (scoped so GPUs 0-4 are untouched):

```
export CUDA_MPS_PIPE_DIRECTORY=~/darklane-serving-20260801/mps/pipe   # lane-private
export CUDA_MPS_LOG_DIRECTORY=~/darklane-serving-20260801/mps/log
CUDA_VISIBLE_DEVICES=5 nvidia-cuda-mps-control -d    # daemon scope = GPU 5 only
# clients: set the SAME CUDA_MPS_PIPE_DIRECTORY, do NOT set CUDA_VISIBLE_DEVICES
# (the daemon's device set is the scope; Engine::new(0) lands on GPU 5).
# teardown: echo quit | nvidia-cuda-mps-control   (same pipe env)
```

Worked first try on this DLAMI (driver-default compute mode is fine); verified via
`nvidia-cuda-mps-server` in compute-apps and both clients at 16,648 MiB on GPU 5.

Tenant note (evidence discipline): a VLLM::EngineCore (30.7 GB, another lane)
appeared on GPU 1 at 01:25:24 — AFTER the first timeslice sweep ended (01:25:16).
The MPS arm ran with it present; the timeslice re-run (`pair-timeslice2`) under the
same conditions reproduced the original numbers (491.4 vs 491.8; 396.3 vs 385.7),
so the tenant is not a factor on GPU-5 numbers.

## R3.2 Pair vs single (aggregate tok/s, GPU 5)

| pair-c (per-replica) | timeslice | MPS   | single replica (round-1, same c total) |
|----------------------|----------:|------:|---------------------------------------:|
| 8 (4+4)              | 381.7     | 393.2 | 270.9 (c=4) .. 308.9 (c=8)             |
| 16 (8+8)             | **491.8** | 459.9 | 307.3 (c=16)                           |
| 24 (12+12)           | 386.6     | **447.5** | ~306                                |
| 32 (16+16)           | 385.7     | **451.7** | 305.0 (c=32)                        |

p50/p95 at the pair sweet spot (c16): 4.16/4.37s timeslice, 4.45/4.64s MPS — vs
6.66/6.75s for a single replica carrying the same 16 sessions. Greedy hash
`dbd1c98f9fed4efe` held on BOTH co-resident replicas, sequential and simultaneous,
in BOTH modes — co-residency does not change outputs.

### MPS vs timeslice verdict

- **At <=8 sessions/replica (each replica inside its exactness-tier batch):
  timeslice wins** — 491.8 vs 459.9 (+7%), and it is operationally free.
- **Past 8/replica, timeslice thrashes** (386-396, both replicas symmetrically at
  ~193 — context-switch cost between two saturated contexts), **while MPS holds
  447-452** (+15% over thrashed timeslice). MPS's SM co-scheduling degrades
  gracefully; time-slicing degrades in a step.
- Net: **run pairs in plain timeslice mode WITH admission capped at ~8
  sessions/replica** (the queue-admission v2 gap does double duty here). MPS is
  the safety choice only if over-admission cannot be prevented.
- Pair ceiling is ~490, NOT the naive 2x305=610: co-residency costs each replica
  ~19% (305 -> ~246) even at the sweet spot — consistent with round 2's
  serial-latency-bound tick (two processes' small kernels inflate each other's
  latency; there is no idle bandwidth being handed over). VRAM peaked at 44.1 GiB
  of 81.6 (pair at 16+16 sessions) — no OOM risk at these caps.

## R3.3 Fleet point (6 replicas, 3 GPUs, c=48)

Second replicas added on GPU 6 (8089) and GPU 7 (8090), timeslice mode, 8
sessions/replica (the sweet-spot regime), direct 6-harness:

| replica | GPU | agg tok/s | p50 | p95 |
|---------|-----|----------:|----:|----:|
| 8085 | 5 | 248.1 | 4.12s | 4.42s |
| 8088 | 5 | 248.3 | 4.12s | 4.31s |
| 8086 | 6 | 249.8 | 4.11s | 4.25s |
| 8089 | 6 | 252.2 | 4.09s | 4.25s |
| 8087 | 7 | 241.0 | 4.26s | 4.32s |
| 8090 | 7 | 240.7 | 4.29s | 4.49s |

**Fleet aggregate: 1480.1 tok/s** (0 errors; ~493/GPU — the pair-c16 number
reproduced exactly across all three GPUs; VRAM ~43-44 GiB/GPU). vs round-1
3-replica direct 915.9: **+62% from pair-packing the same three GPUs.** The mixed
binaries (8086/8087 = round-1 binary, rest = round-2 binary at default cap) are
behavior-identical at chunk 8 (control row, round 2) and hash-identical.

## R3.4 Updated v2 gap list (supersedes §5 ranking)

1. **Queue admission at ~8 sessions/replica** — now load-bearing twice: it holds
   the exactness-tier batch AND it is what makes timeslice pairs beat MPS (491
   vs 386 thrashed). Proxy-side cap + 429/deadline queue.
2. **Pair-packed fleet as deployment default** — 6 replicas/3 GPUs = 1480 tok/s
   today with zero engine work. Needs: proxy backend list extended to 6, session
   affinity (gap 3), and a supervisor (systemd units) instead of nohup.
3. **Session affinity / KV reuse** (unchanged from v1).
4. **Batch the per-seq serial parts of the tick** (round-2 finding): batched
   fa_decode across caches, batched KV append, device-side sampling. This is what
   raises the per-replica 305 — and it compounds with pair-packing only if it
   also shrinks kernel-latency sensitivity (fewer, larger launches co-schedule
   better under co-residency).
5. **Third replica per GPU** — 44 GiB used of 81.6 leaves room for a third
   (~66 GiB); diminishing returns expected from the same serial-latency contention
   that priced the second at -19%, but it is a one-command measurement.
6. Per-GPU model diversity + proxy hardening (unchanged from v1).

## R3.5 Round-3 receipts

- `pair-sweep.jsonl` / `pair-sweep-per-request.jsonl` — all pair points (both arms
  + the `pair-timeslice2` same-conditions re-runs).
- `fleet-point.jsonl` / `fleet-per-request.jsonl` — the 6-replica confirming point.
- `vram-gpu5-r3.csv` — 1 Hz GPU-5 VRAM for the whole round.
- `logs/pair-timeslice.log`, `logs/pair-mps.log` — sweep drivers' raw output
  (greedy hashes per arm inside); `logs/control.log`, `logs/server.log` — MPS
  daemon/server logs; replica logs `replica-8088-timeslice.log`,
  `replica-808{5,8}-mps.log`, `replica-8089-gpu6.log`, `replica-8090-gpu7.log`.
- `run-pair-sweep.sh` — the exact driver.
- End state: 3-replica + proxy steady state restored (8085/8086/8087 + :8080),
  greedy hash verified, no MPS processes, extra replicas killed, GPUs 0-4 never
  touched. (Superseded by R4: the pair-packed fleet is now the steady state.)

---

# Round 4 — productized: admission proxy + fleet supervisor (2026-08-01)

## R4.1 What shipped

**Admission proxy** (`tools/serve-proxy.py`, rewrite): least-outstanding routing now
enforces a per-backend outstanding cap (`--cap`, default 8 — the exactness-tier batch
width AND the timeslice anti-thrash bound from R2/R3, enforced AT the proxy so
over-admission can never reach a replica), a bounded FIFO wait queue (`--queue-max`
256) with a deadline (`--queue-deadline` 30s) -> 429 + Retry-After on
overflow/timeout, a `/metrics` endpoint (per-backend counters, queue
depth/peak/waits p50/p95, TTFB/latency p50/p95, 429 + 5xx counts), and a **passive
circuit breaker**: the first connection-level failure (refused/reset/
RemoteDisconnected) marks the backend DOWN immediately — the 2s active probe
restores it. The breaker came out of the chaos check: a killed backend's slots free
instantly, making it least-outstanding, so the router preferentially fed the corpse
for the probe interval — 100/768 fast-fail 502s. With the breaker: 8/768 (exactly
the in-flight cap).

**Fleet supervisor** (`tools/serve-fleet.sh`): declarative config (GPUS x
REPLICAS_PER_GPU x MODEL, env-overridable), gpu-major port layout from BASE_PORT,
pidfile discipline under `$FLEET_RUN`, a health loop (5s interval, 120s model-load
grace) that restarts dead replicas AND the proxy, `start|stop|status|restart`
commands, nohup re-exec so the supervisor survives ssh HUP. systemd-free by design
(userland box). Default config = the measured sweet spot: pairs on GPUs 5/6/7,
cap 8, ports 8085-8090, proxy :8080.

## R4.2 Validation (through the NEW proxy, 6 backends, cap 8)

| point | agg tok/s | p50 | p95 | err | queue evidence |
|---|---:|---:|---:|---:|---|
| c=48 | 1367.9 | 4.43s | 4.65s | 0 | queue mostly empty |
| **c=96** | **1378.6** | 7.94s | 8.97s | 0 | peak depth 49, 240 enqueued, wait p95 4.50s, **zero 429s** |
| c=48 + replica SIGKILL (no breaker) | 1355.6 | 4.49s | 5.03s | 100 | the storm that motivated the breaker |
| c=48 + replica SIGKILL (breaker) | 1387.0 | 4.53s | 4.97s | **8** | = in-flight on victim |

**c=96 HOLD verdict: PASS.** Throughput flat vs c=48 (1378.6 vs 1367.9 — the cap
sheds the surplus 48 to the queue instead of letting them thrash the replicas),
queue-wait visible in /metrics (p95 4.50s = exactly one service generation), p95
client latency 8.97s = queue + service, no 429s (30s deadline never approached at
this depth; 429s begin when queue wait crosses it, i.e. around c ≈ 48 + 30/4.4*48
≈ 375 offered concurrent).

Proxy admission overhead: 1368-1387 proxied vs 1480 direct = ~6-7% (up from ~1%
for the cap-less round-1 proxy at c=24; the admission lock + 96 Python threads is
the cost). Acceptable v1; a Rust/asyncio proxy is the hardening path if this
matters later.

## R4.3 Chaos timelines (measured, from proxy/supervisor logs)

Replica SIGKILL (with breaker), victim :8090 mid c=48 run:
- 02:00:26 kill -9
- 02:00:26 proxy: passive DOWN (RemoteDisconnected — same second)
- 02:00:28 supervisor: unhealthy past grace -> restart (+2s)
- 02:00:33 proxy: backend UP (+7s; model reload is page-cache-warm)
- errors: 8/768 (the in-flight requests on the victim); aggregate 1387 tok/s
  across the run — recovery invisible in throughput.
- post-restart greedy hash on ALL SIX replicas: `dbd1c98f9fed4efe` (match).

Proxy SIGKILL (deploy path doubles as chaos): killed 01:59:35, supervisor
relaunched it 01:59:37 (+2s) with the updated code — zero replica impact.

## R4.4 Fleet ops runbook

```
# start the default fleet (pairs on GPUs 5/6/7, cap 8, proxy :8080)
tools/serve-fleet.sh start

# check / stop / bounce
tools/serve-fleet.sh status
tools/serve-fleet.sh stop
tools/serve-fleet.sh restart

# scale: more/fewer replicas per GPU or different GPUs (declarative)
GPUS="5 6 7" REPLICAS_PER_GPU=3 tools/serve-fleet.sh restart
GPUS="5" REPLICAS_PER_GPU=1 tools/serve-fleet.sh restart

# different model / ports / admission
MODEL=~/models/other.gguf BASE_PORT=9085 PROXY_PORT=9080 CAP=8 tools/serve-fleet.sh start

# observe
curl -s localhost:8080/health   # per-backend up/down + outstanding
curl -s localhost:8080/metrics  # queue depth/waits, TTFB, 429s, 5xx
tail -f ~/darklane-fleet/logs/supervisor.log   # restarts
tail -f ~/darklane-fleet/logs/proxy.log        # UP/DOWN transitions

# on-box paths: binaries from ~/memra/target/release, run state in ~/darklane-fleet/
```

Steady state after R4: the 6-replica pair-packed fleet + admission proxy + supervisor
IS the serving stack (replaces the round-1 3-replica layout). Client-visible number:
**~1380 tok/s sustained through the proxy at any offered concurrency 48-96+, with
byte-exact outputs and self-healing replicas.**

## R4.5 Round-4 receipts

- `r4-points.jsonl` / `r4-per-request.jsonl` — c48/c96/chaos1/chaos2 points.
- `metrics-r4.log` — /metrics sampled at 2 Hz through c48+c96 (queue depth curve).
- `chaos-metrics.log` — healthy-backend set + 5xx counter through the first kill.
- `logs/fleet-supervisor.log`, `logs/fleet-proxy.log` — the chaos timelines quoted
  above.
- Code: `tools/serve-proxy.py` (admission + breaker rewrite),
  `tools/serve-fleet.sh` (supervisor).
- Remaining gaps (unchanged): session affinity/KV reuse; per-seq-serial tick
  batching (the per-replica 305 lever); model diversity; Rust/asyncio proxy if the
  ~7% admission overhead matters.
