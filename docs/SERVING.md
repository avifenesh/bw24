# Multi-user serving — the replica fleet

memra's engine owns one GPU per process (`Engine::new(0)`; `CUDA_VISIBLE_DEVICES` is the
placement mechanism). Multi-GPU serving is therefore a **replica fleet**: N `memra-server`
processes fronted by an admission proxy. Tensor parallelism is a separate in-progress build
(M0 comms floor measured — ARCHITECTURE-H100.md).

## Tools

| tool | what it does |
|---|---|
| `tools/serve-fleet.sh start\|stop\|status\|restart` | declarative fleet supervisor: brings up `REPLICAS_PER_GPU` replicas per GPU in `GPUS`, fronts them with the proxy, health-loop restarts anything that dies. systemd-free; pidfiles under `$FLEET_RUN` |
| `tools/serve-proxy.py` | least-outstanding reverse proxy with per-backend admission cap (default 8 = the engine's exactness-tier batch width and the two-replicas-per-GPU anti-thrash bound). Bounded FIFO queue with deadline → 429 + Retry-After; `/health` + `/metrics` JSON |
| `tools/load-serve.py` | concurrent OpenAI-format load harness: aggregate output tok/s, p50/p95 latency, JSONL per load point |
| `tools/serve-smoke.sh` | OpenAI-surface smoke gate for a single server |

## Measured numbers (Qwen3.5-9B Q8_0; receipts in `research/`)

- **Single replica (H100):** temp-0.7 c=8/16/32 medians **654/657/659 tok/s** after the
  batched decode tick (z-batched FA + KV append, device sampling, lean logits — +25-36%
  over the pre-batched tick; N=4, `research/batched-tick-inc2-20260801/`; chunk-8 era —
  see the exact-16 tier below).
- **Managed fleet, 3 H100s x 2 replicas (v0.60-validated):** **1,477 tok/s** through the
  admission proxy at c=96 (N=2 interleaved passes: 1477.0/1473.1), zero 429s/5xx —
  managed now matches the v0.59-era 1,480 direct number (the ~7% admission-overhead gap
  closed at the fleet level). Chaos-tested: SIGKILL a replica mid-load, breaker DOWN the
  same second, supervisor restart +2s, backend UP +9s, 8/768 requests lost (exactly the
  victim's in-flight cap), aggregate across the kill 1,487 tok/s; greedy hash identical
  on all 6 replicas in every condition, 18/18 (`research/fleet-v060-20260801/SUMMARY.md`).
  The proxy cap (8) was calibrated on the v0.59 core — the cap re-sweep is pending the
  next box window (stale-verdict risk flagged in the validation summary).
- **Single replica (RTX 5090, exact-16 tier):** with the Q8_0 split-plane mirror
  (`MEMRA_Q8RP=1` on 24GB; Hopper default), the worker auto-selects decode chunk 16 —
  c=16 median **494.5 tok/s vs 416.4** at chunk 8, same mirror, interleaved N=4
  (**+18.8%**; +33.8% vs the mirror-less baseline); c=32 at `MEMRA_CTX=2048` runs
  **502.1** with 128/128 ok (single run; `research/batched-tick-inc3-20260801/`).
- **Spec fast lane:** MTP speculative serving is a single-stream latency tier — 1.82x plain
  serving at c=1 on the 27B (131.8 vs 72.5 tok/s); plain batching overtakes between c=2 and
  c=4, so spec and bulk tiers run as separate server processes (`MEMRA_SERVE_SPEC`;
  `research/spec-serving-20260801/`).

## The isolation contract

Greedy serving is **isolated-identical under concurrent load at defaults**: a request's
output tokens are byte-identical whether it arrives alone or inside a full batch. This is
gated, not assumed — the serve gate replays the same prompts at c=1 and c=16 and
byte-compares every stream. It is also a fixed defect, not a freebie: the batched
cuBLASLt prefill router and shared-expert-gate GEMMs were m-dependent, so under
cross-request prefill batching a MoE request's own expert selection changed with its
co-arrivals (the supported Qwen3.6-35B had the same defect as the onboards whose serve
gate exposed it — Ornith-35B 6/16, KAT 7/16, both 16/16 after the fix). Default
`MEMRA_ROUTER_PREFILL_EXACT` routes prefill through decode's m-invariant router/gate
kernels; a bit-identical batched twin recovers most of the prefill cost
(`MEMRA_ROUTER_BATCH`, FLAGS §3). Receipts: `research/concat-prime-exact-20260802/`,
`research/fast-router-20260802/`.

**Admission is VRAM-aware** (2026-08-02): once the first admitted session reveals the
model's per-session VRAM cost, further admissions require free ≥ 2x that cost — otherwise
the request *waits* in the same never-rejected FIFO as the session-count cap instead of
failing with a cache-alloc OOM (the c=16 8192-ctx failure mode under resident-if-fits,
caught by the serve gate as instant HTTP 400s — `research/fast-router-20260802/RESULTS.md`).
The first session always admits; an OOM with no active sessions is real capacity and
still errors loudly, with the CUDA error quoted.

## The exact-16 decode chunk tier

The batched tick decodes sessions in per-model chunks. Default width is **16 on models
where every matmul has a bit-exact 16-batch kernel class** (`decode_batch_exact16_ok`:
the b16 batched-mmvq family — Q8_0 qualifies only through its `_rp` mirror twin), **8
otherwise**; `MEMRA_DECODE_BATCH_CAP` stays the explicit measurement door. Qualifying
steps scope out every m>=16 GEMM/MMQ arm, so chunk-16 output is bit-identical to
isolated decode (gate2 bit-checked at steps 32 and 160). B=32 has no exact kernel class
— chunk policy stays <=16. On the H100 fleet model (9B Q8_0, mirror on by default) the
tier engages automatically on the next deploy; the H100 numbers above are chunk-8-era
and the chunk-16 fleet effect is pending on-box re-validation.

**Capacity envelope (24GB):** the mirror costs ~model-size VRAM, so c=32 sessions at the
default `MEMRA_CTX=8192` exceed VRAM (captured `CUDA_ERROR_OUT_OF_MEMORY` in the
pre-admission-wait receipts; ~27 sessions fit — since the VRAM-aware admission wait the
overflow queues instead of erroring). Set `MEMRA_CTX` to the workload — 2048 clears the
same cell (machine-specific config per the flags doctrine).

## Prompt caching (cross-request prefix cache) — 2026-08-02

Two caching tiers serve prompt tokens without recomputing them:

1. **Continuation pool** (pre-existing, `MEMRA_KV_REUSE`): a retired session parks its whole
   (prompt + generation) state; a new prompt that EXACTLY EXTENDS it resumes. Single-use,
   exact-extension only — a new session that merely shares a system prompt always missed.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`, default 256MB, 0 = off): compact
   device snapshots of primed state at token boundaries, keyed by the exact token-id prefix,
   per model, LRU under the byte budget. Entries are REUSABLE — a hit deep-copies the entry
   into the new session's cache, so one marketplace system prompt serves any number of
   sessions. Learning sequence for a shared-prefix pattern: request 1 seeds its full prompt,
   request 2 split-primes at the longest-common-prefix and inserts the boundary entry,
   request 3+ hit. Hybrid models are safe by construction: GDN conv/ssm state cannot be
   truncated to a shorter prefix, so the state is snapshotted AT the boundary while a fresh
   session primes — never rolled back.

**Exactness contract:** an entry stores the KV/recurrent bytes from WHATEVER prime config ran
(single, chunked, or concat batch-prime); decode from those bytes is deterministic, so a
cached hit is bit-identical to the run that computed the prefix — gated 16/16 partial-prefix
+ 16/16 full-prefix cached-vs-fresh greedy identity across depths
(`research/prompt-cache-20260802/gate-exact.jsonl`). Comparing a cached-hit stream against a
DIFFERENT prime config's fresh stream inherits the batched-prime near-tie first-token law
above — same documented class, reported not gated.

**Policy:** spec sessions bypass the prefix cache entirely (SpecSession owns trunk + draft
caches; a trunk-only prefix restore would leave draft state unprimed — the spec tier keeps
its own continuation pool). Legacy round-robin mode (`MEMRA_SERVE_BATCH=0`) also bypasses.
Sessions always win over the cache: a failed session-cache allocation evicts every entry and
retries before erroring.

**Accounting:** every response shape carries OpenAI-schema usage with the worker-truth split —
`usage.prompt_tokens`, `completion_tokens`, `total_tokens`, and
`prompt_tokens_details.cached_tokens` (tokens resumed from ANY cache tier: continuation pool,
spec resume, or prefix cache). `/metrics` exposes the cumulative split
(`prompt_tokens_in`/`cached_tokens_in`) plus `prefix_cache_hits`/`entries`/`bytes`. Cached
prefill costs ~0 to serve and bills at 25% of input on the OpenRouter hy3 endpoints — the
margin lever (`research/or-provider-20260802/REPORT.md`).

## Knobs

Serving flags (batch cap, device sampling, lean logits, prime batching, spec burst) are
cataloged in [FLAGS.md §7](FLAGS.md) under "Serving (memra-server)"; fleet topology knobs
(`GPUS`, `REPLICAS_PER_GPU`, `CAP`, ports, health cadence) are env-overridable at the top of
`tools/serve-fleet.sh`. The exactness contract holds under batching: the decode-batch gate
battery (gate1-3, gate3c lean-vs-full) runs inside `tools/validate-h100.sh`.

## First-token cross-config drift (batched prime) — stated honestly

Serving primes prompts BATCHED (`prime_cache`, prefill GEMMs) while the historical oracle
stream is tokenwise (`decode_step`, m=1). These are different numeric configs by design —
same law as forward-vs-decode and the decode-batch gate's config mode — so on near-tie
prompts the FIRST generated token of a request can differ from the tokenwise oracle
stream, and everything after it follows the new prefix. Measured on the six-model
2026-08-02 sweep (`research/prime-gate-coverage-20260802/`, 144 prompts): **10/144 first
tokens flip (~7%)**, every flip at a tokenwise top1-top2 margin <= 0.70, batched prime
bit-deterministic, no content leakage across chunk boundaries, and forward_last sides
with the batched prime in 8/10 flips — the tokenwise config is usually the outlier, so
this is config roulette on a near-tie, not a wrong path. On the gemma prefill lanes the
config can even move per PROCESS (cuBLASLt heuristic algo selection; one observed
instance in the 144-row double pass, bit-deterministic within a process). Dense Q8_0
models (9B judge, Ornith-9B — the fleet class) flipped 0/48. Consequences can be visible (the Qwen3.6-35B
pp512 probe greedy-emits `"\n"` + EOS at 2 tokens where the tokenwise stream writes 128):
within contract, but real. `MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill
cost; the run-gen `batched-prime` gate line + the `prime-gate` battery bound the class
(structured divergence fails hard, near-tie flips are reported).
