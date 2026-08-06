# Serving-density lane — Q1 prefix-duplication census + Q2 c=64 robustness gate — 2026-08-06

Worktree `lane/serving-density` off `restructure/public-split` (train HEAD a85135ae).
Rig: local RTX 5090 (24463 MiB), GPU flock held per phase. Battery: serve-smoke **0 failed**
(logs/serve-smoke.log); no crate code touched this lane (one new tools/ script + research files).

---

## Q1 — prefix-duplication measurement: **RECEIPTED-DEAD at the pi/coding-agent shape**

### Accounting model (verified against source, not assumed)

Session KV is allocated **flat at `ctx_cap`** up front (`memra-kv/src/lib.rs Cache::new_inner`:
`alloc_u8(max_ctx * tok_bytes)`), plus one draft-scratch KvLayer at the same cap
(`spec.rs MtpScratch::new`), plus fixed recurrent state per linear layer. `ctx_cap =
max(prompt + max_new + 8, MEMRA_CTX)` (worker.rs ~2082). Per-token/session bytes:

| model | full-attn layers | per-token KV (trunk+draft) | fixed recurrent/session |
|---|---|---|---|
| q9 (Qwen3.5-9B) | 8 (+1 draft) | 16704 B = 16.3 KiB/tok | 102.3 MiB |
| k27 (Qwen3.6-27B, deployment daily) | 16 (+1 draft) | 31552 B = 30.8 KiB/tok | 299.7 MiB |

**Measured anchor** (c=8 barrier-synchronized sessions, shared 3623-token dogfood-ctx4k prefix +
unique ~120-tok tails, q9+draft spec-ON, MEMRA_CTX=8192): 8/8 ok; `prompt_tokens=3623` each,
`cached_tokens_in=0` — every session re-prefilled and re-stored the full identical prefix
(spec sessions bypass the cross-request prefix cache **by design**, worker.rs ~2126, and the
spec continuation pool is exact-extension/affinity only — no cross-SESSION sharing tier exists).
Worker's own observed session cost 235 MB(SI) = 224 MiB vs analytic 232.8 MiB (−3.8%).
Burst peak VRAM 10388 MiB vs idle-after-load 5842 MiB.

### The census (full table: logs/census-analytic.txt; duplication = (c−1) x prefix KV)

| scenario | duplication | ladder slack |
|---|---|---|
| q9, measured 4k shape, c=8, 24GB card | 390 MiB = **1.60%** | 558 MiB = 2.28% |
| q9, 4k shape, c=16 / c=32, 96GB | 836 / 1728 MiB = **0.85% / 1.76%** | 1.13% / 2.27% |
| q9, 8k prefix, c=16 / c=32, 96GB | 1936 / 4000 MiB = **1.97% / 4.07%** | ~0% (right-sized) |
| k27, 4k shape, c=16 / c=32, 96GB | 1580 / 3265 MiB = **1.61% / 3.32%** | 2.14% / 4.29% |
| k27, 8k prefix, c=16 / c=32, 96GB | 3656 / 7556 MiB = **3.72% / 7.69%** | ~0% |
| k27, 32k prefix, c=16 / c=32, 96GB | 14443 / 29850 MiB = **14.69% / 30.36%** | ~0.3% |

### VERDICT

At the representative agent-trace shape (4–8k shared system prefix, short tails), sealed-prefix
sharing frees **at most 7.7% of the 96GB card** (k27, 8k prefix, c=32; only 3.7% at c=16) —
**below the 10%-of-card bar at c=16+. The idea dies cheap at this shape.** Two receipted riders:

1. **Revive bar (mechanical):** duplication crosses 10% of the 96GB card on k27 at c=16 only when
   the sealed prefix reaches **~22k tokens** (32k prefix: 14.7% at c=16, 30.4% at c=32). That is a
   different workload class (RAG / repo-context agent farm), not the pi/coding-agent trace. If
   darklanes lands a 20k+-sealed-prefix product shape, re-open with this table as the prior.
2. **The bigger, cheaper stranding is CONFIG, not duplication:** `max_tokens` omitted at
   MEMRA_CTX=32768 strands **6.3% (c=16) / 12.6% (c=32)** of the 96GB card in q9 ladder slack —
   exceeding the duplication at trace shapes. Right-sized requests (explicit max_tokens) strand
   ~0%. An ops note (clients must send max_tokens; or default-when-omitted could size down),
   zero new mechanism.
3. Prefix sharing's real win at this shape would be prefill COMPUTE/TTFT (c−1 redundant 3.6k-token
   prefills measured), not VRAM — out of this question's scope, and the affinity/prefix-cache
   tiers already own that lane for the non-spec path.

---

## Q2 — c=64 robustness gate: **FAIL at defaults; mode = quoted step-OOM, worker survives**

Gate built: `tools/serve-stress-gate.sh` — 64 staggered (50 ms) streaming clients, 128–256 tok,
temp 0.7 seeded, asserts: all complete + streams well-formed (finish_reason + [DONE]) + worker
alive + no panic/CUDA/OOM lines in the server log; p50/p95 wall + TTFB recorded informationally.
**Not flock-wrapped** (callers own the GPU lock — the fast-gate self-deadlock lesson).

### Result: 3/3 runs FAIL, identical mode (defaults: MEMRA_MAX_SESSIONS=64, spec-ON, q9+draft)

- **0/64 well-formed in every run.** All 64 admitted ([meter] lines), VRAM climbs
  5.5 GB → 23.98 GB (card full) in ~6 s of admission+prefill, then **9x**
  `[spec] WARN: sampled draft-graph capture failed (CUDA_ERROR_OUT_OF_MEMORY); eager fallback`
  followed by **64x** `step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` — every request errors.
- **Not a hang, not a panic, not a crash:** max wall 5.5 s, every stream carries the error chunk
  and `[DONE]`, the server stays alive and healthy after the burst (clean SIGTERM drain). The
  failure mode is *graceful-but-total request loss* — better than mistral.rs's 6/69 hang class,
  still a red gate.

### Bracketing controls (same 64-client burst)

| config | result | peak VRAM |
|---|---|---|
| MEMRA_MAX_SESSIONS=16, spec-ON | **PASS 64/64** (wall p50 38.3s) | 11400 MiB |
| MEMRA_MAX_SESSIONS=32, spec-ON | **PASS 64/64** (p50 47.5s) | 15948 MiB |
| MEMRA_MAX_SESSIONS=48, spec-ON | **PASS 64/64** (p50 49.4s) | 20528 MiB |
| MEMRA_SERVE_SPEC=0, cap 64 | **PASS 64/64** (p50 26.1s, ttfb p50 0.14s) | (plain path) |

The boundary is between 48 and 64 **concurrent spec sessions** on 24 GB; the plain batched path
survives all 64. The failing component is the **admission VRAM gate's cost model for spec
sessions** (worker.rs ~1257): it admits while `free >= 2x observed session cost` (201 MB,
measured as the free-VRAM delta of the FIRST admit — i.e. the *parked* session cost), but a live
spec session's peak footprint adds the per-session sampled draft-graph capture + burst transients
(VerifyCkpt stashes, verify activations). 64 x 233 MiB parked = 14.6 GiB fits beside 5.8 GiB
weights; the transient layer on top does not. The F5 right-size ladder never fires (0 ladder/evict
lines) because session *allocation* succeeds — the OOM lands at *step* time, where no ladder or
requeue exists, so all in-flight sessions die in the same tick sweep.

### Fix brief (NOT fixed in this lane, per lane contract)

1. **Admission headroom:** enforce the transient reserve at admission, not just at ladder landing —
   admit while `free >= session_cost + max(session_cost, SPEC_SHRINK_RESERVE)` (the same 1.5 GiB
   floor `SPEC_SHRINK_RESERVE` already encodes for exactly this transients-on-panicking-paths
   class), or measure `session_vram_cost` at first-burst peak instead of admit-delta.
2. **Step-time OOM should degrade, not kill:** a spec burst step that OOMs on a card-full
   condition should requeue the session for the next tick (FIFO-wait — the same philosophy the
   admission gate already applies to the count and VRAM axes) and mark the model VRAM-tight
   (pause admissions) instead of erroring all 64 streams. The 9x draft-graph capture OOM →
   eager fallback seam already behaves correctly (warn + degrade).
3. After the fix: wire `tools/serve-stress-gate.sh` into local-ci behind `MEMRA_CI_STRESS=0`
   skip-door + a fast-gate cmd probe row (the gate script already speaks fast-gate's
   SKIP-verdict-word contract). **Not wired now** — wiring a known-red gate into local-ci would
   either block every merge or normalize a red gate; the wiring belongs to the fix lane's
   receipts.

### Receipts

- RESULTS.jsonl (this dir) — one row per run/control, N and thermal stated.
- logs/stress-run{1,2,3}*.{txt,log,jsonl} — raw gate output, server logs (quoted OOM lines),
  per-request rows; run3 adds the 200 ms VRAM trajectory.
- logs/stress-control-{cap16,cap32,cap48,specoff}* — bracketing controls + VRAM peaks.
- logs/q1-* — Q1 measured c=8 anchor; logs/census-analytic.txt + kv_census.py — the census.
- logs/serve-smoke.log — full serving battery green on the lane build.
