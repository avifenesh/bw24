# lane/prompt-cache — marketplace-grade prompt caching (2026-08-02)

Rig: RTX 5090 Laptop (24GB), all GPU runs under `flock /tmp/gpu5090.lock`.
Model under test: q35 = Qwen3.6-35B-A3B-UD-IQ4_XS (the serve-gate board model; embedded MTP
head, so NAKED defaults serve greedy chat on the spec tier — the bulk tier is
`MEMRA_SERVE_SPEC=0`, the config marketplace concurrency runs on). Smoke receipts on the 9B
judge (`smoke-*`). Runner: `run-lane.sh` (phases audit/gate/load/battery); every claim's raw
JSONL/log sits next to this file.

## 1. AUDIT — what existed before this lane (as-is receipts)

Reuse before this lane was **per-session continuation only**: a retired session parked its
whole (prompt + generation) state — legacy pool `ReuseEntry` (exact token-prefix extension,
single-use, `MEMRA_REUSE_POOL`=2/model) and the spec pool (committed-text prefix extension).
There was **no cross-request prefix matching**: a NEW session sharing a system prompt with
earlier traffic always primed from scratch. Usage carried no prompt-token accounting at all
(`completion_tokens` only).

Measured as-is (1,593-token shared system prompt, `audit-asis-{naked,bulk}.jsonl`, single
runs, cold server each):

| request | naked (spec tier) | bulk tier | cached_tokens |
|---|---|---|---|
| R1 cold `[sys,u1]` | TTFT 474ms | 393ms | 0 |
| R2 new session `[sys,u2]` | 453ms | 358ms | **0 — the gap** |
| R2b third session `[sys,u3]` | 418ms | 358ms | **0** |
| R3 exact continuation `[sys,u1,a,u3]` | 447ms | 380ms | **0** |

Same system prompt re-sent = **zero prefill skipped** in every class measured. Even the R3
exact-continuation class missed on q35: the chat template rewrites history (think-block
stripping), which breaks both existing pools' exact-extension match — the documented
weakness, now receipted (no `kv-reuse`/`spec-reuse` lines in either server log).

## 2. WHAT SHIPPED

**Cross-request prefix cache** (`crates/memra-server/src/worker.rs`, `MEMRA_PREFIX_CACHE_MB`
default 256MB, 0=off): token-prefix-keyed compact device snapshots of primed KV + recurrent
state, per model, LRU under the byte budget. Entries are REUSABLE — a hit deep-copies the
entry into the fresh session cache (D2D, stream-ordered on the CUDA owner thread), so one
marketplace system prompt serves any number of sessions. Hybrid-safe by construction: GDN
conv/ssm state cannot truncate, so state is snapshotted AT the boundary while a fresh session
primes. Learning sequence: request 1 seeds its full prompt at prefill-done; request 2
split-primes at the LCP and inserts the boundary entry; request 3+ hit. Sessions win over the
cache (alloc failure evicts all entries and retries). Entry cost on q35: ~68MB fixed
recurrent state + 9.3KB/token (~83MB at 1.6k tokens); 9B-class measured 76.3MB at 1,593
tokens.

**Accounting**: OpenAI-schema usage on every response shape (chat/completions x
blocking/SSE + native): `prompt_tokens`, `completion_tokens`, `total_tokens`,
`prompt_tokens_details.cached_tokens` — worker-truth (tokens resumed from ANY cache tier:
continuation pool, spec resume, prefix cache). `/metrics` adds `prompt_tokens_in`,
`cached_tokens_in`, `prefix_cache_hits/entries/bytes`.

**Policy — spec x cache**: spec sessions bypass the cross-request cache entirely
(SpecSession owns trunk + draft caches; a trunk-only restore would leave draft state
unprimed). The spec tier keeps its own continuation pool; the prefix cache serves the
batched bulk tier. Legacy round-robin (`MEMRA_SERVE_BATCH=0`) also bypasses.

## 3. EXACTNESS GATE (the contract) — ALL GREEN

`cache_exact_gate.py`, 16 cells across depths 96-2048 words (~128-2,700 tokens), greedy c=1,
q35 bulk tier, budget 1024MB (LRU live across cells). Contract: an entry stores KV/recurrent
bytes from WHATEVER prime config ran; decode from them is deterministic — a hit is
bit-identical to the run that computed the prefix.

| gate | result |
|---|---|
| partial-prefix hit == the fresh split-prime that made the entry (B2==B1) | **16/16** |
| full-prefix hit == the cold whole-prime that made the entry (A2==A1) | **16/16** |
| usage truth (hit rows report cached_tokens; full hit == prompt_tokens) | 16/16 |
| cold rows report cached_tokens == 0 | 16/16 |
| control: cache-ON cold path == cache-OFF whole-prime (same config) | **16/16** |

Cross-config REPORT (not gated, the documented batched-prime near-tie law): split-prime
stream vs whole-prime fresh stream moved **6/16** on this MoE — same class as the 2026-08-02
prime-gate sweep, not a new divergence. A cached prefix replayed under a different prime
config inherits exactly that law and nothing more. Raw: `gate-exact.jsonl`, `gate-refs.json`.

## 4. TTFT / savings — the marketplace numbers

`agentic_load.py load`: one shared 1,593-token system prompt, 32 distinct user turns, c=8,
4 waves per pass, **N=3 interleaved arm pairs** (fresh server per arm per rep; cache-on =
naked budget 256MB, cache-off = `MEMRA_PREFIX_CACHE_MB=0`; both `MEMRA_SERVE_SPEC=0`).
Medians of per-rep values; spread across reps < 3%.

| metric (median, N=3) | cache OFF | cache ON | delta |
|---|---|---|---|
| steady-state TTFT p50 (hit waves) | 3.005 s | **0.710 s** | **-76%** |
| steady-state TTFT p95 | 3.017 s | 0.742 s | -75% |
| hit-wave aggregate throughput | 165.6 tok/s | **323.3 tok/s** | **+95%** |
| pass aggregate (incl. cold+learning waves) | 165.3 tok/s | 210.2 tok/s | +27% |
| pass wall (32 req) | 18.6 s | 14.6 s | -21% |
| prompt tokens served from cache | 0% | 49.5% of pass; **99.1% in hit waves** | |

Per-wave (cache-on): wave0 cold 3.05s/0%, wave1 learning 3.59s/0% (8 concurrent split-primes
— the one-time price), wave2 0.74s/99.1%, wave3 0.69s/99.1%. Cache-off holds ~3.0s/165 tok/s
in every wave. Billing lever: hit-wave sessions bill ~99% of input at the 25% cache-read
rate while costing ~0 prefill compute to serve (OpenRouter hy3 endpoints price cache reads
at 25% of input — `research/or-provider-20260802/REPORT.md`).

Single-stream (c=1) demonstration (`audit-cache-on.jsonl`): third session TTFT 393ms ->
**37ms (10.6x)** with 1,577/1,593 tokens cached; the R3 continuation class that missed
as-is also hits via the prefix tier (75ms, 1,577/1,707 cached).

## 5. Batteries

| gate | result |
|---|---|
| kernel-check (q35, untouched engine) | rc=0, **382 OK / 0 FAIL** (matches f16g-rearb count) |
| serve greedy c1-vs-c16, q35 NAKED (spec tier) | **PASS 16/16** (`greedy-hash-q35-naked.jsonl`) |
| serve greedy c1-vs-c16, q35 bulk + cache live | **PASS 16/16** (`greedy-hash-q35-bulk.jsonl`) |
| decode-batch q35 / q9j, config mode | ALL GREEN both |
| decode-batch q35 / q9j, strict EQUALIZED protocol | **ALL GREEN both** (rc=0) |
| run-spec q35 K=1..8 self-consistency | **PASS 8/8** (spec serving unaffected) |

Strict protocol note: the equalized composition is `MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`
+ worst-draw `MEMRA_GATE_SEED` (q35=16, q9j=0) + `--batch 4 --mode strict` (gate1-recal-
20260802 + validate-h100.sh, as executed by f16g-default-rearb/run-followup.sh). Two
out-of-protocol invocations preceded the green runs — the stale bare `--mode strict` form,
then `--batch 4` without the env — and both misfired on the documented accepted
FP-composition gap (gate1/gate2 bit-diffs; config-mode ALL GREEN same session). Kept as
`*-strict-MISFIRE*.log`, superseded by `battery-decode-batch-{q35,q9j}-strict-equalized.log`.

## 6. Metering field spec for the darklane side (describe-only; no cross-repo change)

The memra response now carries the worker-truth split on every shape. The dl meter record
should add one field and one derived charge line:

```
meter_record.cached_tokens   u64   <- usage.prompt_tokens_details.cached_tokens
  (computed prompt tokens = prompt_tokens - cached_tokens; invariants:
   0 <= cached_tokens <= prompt_tokens, and cached_tokens == 0 whenever the serving
   tier was spec or the request was the pattern's first/learning request)
charge = (prompt_tokens - cached_tokens) * input_rate
       + cached_tokens * cache_read_rate      # 25% of input on the hy3 endpoints
       + completion_tokens * output_rate
```

Streaming responses carry usage on the final SSE chunk (OpenAI convention); the non-stream
shapes carry it in `usage`. `/metrics` (`prompt_tokens_in`/`cached_tokens_in`) is the
replica-level reconciliation counter for the meter's per-request sum.

## Files

- `run-lane.sh`, `run-strict-equalized.sh`, `cache_exact_gate.py`, `agentic_load.py` — runners (literals baked)
- `audit-asis-{naked,bulk}.jsonl`, `audit-cache-on.jsonl`, `server-audit-*.log` — audit receipts
- `gate-refs.json`, `gate-exact.jsonl`, `server-gate-*.log` — exactness gate
- `load-cache-{on,off}.jsonl`, `server-load-*.log`, `metrics-*.json` — TTFT/savings (N=3)
- `battery-*.log`, `greedy-hash-*.jsonl`, `greedy-refs-*.json` — batteries
- `smoke-*.{jsonl,log}` — 9B-judge smoke (first light: 1577/1593 cached, TTFT 443->42ms)
- `console.log`, `runner.log` — full run transcript
