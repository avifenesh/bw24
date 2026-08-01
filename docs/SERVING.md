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

## Knobs

Serving flags (batch cap, device sampling, lean logits, prime batching, spec burst) are
cataloged in [FLAGS.md §7](FLAGS.md) under "Serving (memra-server)"; fleet topology knobs
(`GPUS`, `REPLICAS_PER_GPU`, `CAP`, ports, health cadence) are env-overridable at the top of
`tools/serve-fleet.sh`. The exactness contract holds under batching: the decode-batch gate
battery (gate1-3, gate3c lean-vs-full) runs inside `tools/validate-h100.sh`.
