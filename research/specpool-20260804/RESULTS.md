# F5 — spec-pool thrash under long-context VRAM pressure

Owner's daily driver (serve-qwen36-27b-memra: 27B NVFP4+MTP, MEMRA_CTX=131072,
MEMRA_MAX_SESSIONS=1, MEMRA_REUSE_POOL=1, 24GB 5090). Live session log:
`owner-session.log` (captured from /tmp/memra-27b.log before rotation).

## Symptom

From the session's early turns onward (first at ctx≈8.6k, EVERY turn from ctx≈14.4k
to the log's end at 21.9k), each new request logs:

    [worker] spec pool evicted (1) after alloc failure; retrying

One evict + full ~4.4GB session realloc per turn. Owner reports "becoming slower
every loop". Acceptance healthy (0.6-0.9) — the spec math is fine; the cost is pure
alloc/evict/realloc churn plus the total loss of the parked session's continuation
value (every turn re-primes the whole conversation instead of resuming).

## Root cause (code-level)

`admit()` in `crates/memra-server/src/worker.rs`:

1. `ctx_cap` floors at MEMRA_CTX (131072) for EVERY request (line ~1642). On this
   config a fresh spec session = trunk KV 17 layers x 31,552 B/tok x 131072 ≈ 4.1GB
   + MTP draft scratch (~0.3GB) ≈ **4.4GB per session ask, always**.
2. On retire, the finished spec session PARKS WHOLE in `spec_reuse` (trunk cache +
   draft scratch, line ~1493) — the parked ghost holds its own ~4.4GB.
3. Next request probes the pool for an exact token-prefix or text-prefix match
   (line ~1833). The owner's client (pi, thinkingFormat qwen-chat-template) rewrites
   history — think-block stripping breaks literal text extension, so the probe
   MISSES nearly every turn (the log shows ~1 miss per turn; the module doc at
   ReuseEntry line ~185 already names this miss class).
4. Miss path: `new_session(ctx_cap)` asks for a NEW 4.4GB while the parked ghost
   still holds 4.4GB. Weights ~16GB + ghost 4.4GB + new 4.4GB > 24GB → alloc FAILS
   → evict the pool → retry the same full-size ask → succeeds.

So the pool is structurally a one-slot cache whose occupant is evicted by every
miss — and every turn is a miss. The evict-retry "reclaim" (serve script comment,
line 41-42 of serve-qwen36-27b-memra) was designed for a genuinely-new session;
in the daily driver it fires every turn.

Why "slower every loop": each turn pays (a) the failed cudaMalloc + pool evict +
4.4GB realloc, and (b) a FULL re-prime of the whole conversation (the parked
session's 20k-token committed history is destroyed on evict, so nothing resumes;
prime cost grows linearly with ctx). (b) dominates and grows with ctx — the
progressive slowdown. The fa_part_retired residual (#68 retire-on-grow) adds
bounded pressure (< final pool size, tens of MB at these shapes) — a contributor
to the tightness but not the driver.

## Fix shape

See fix commit. Candidates considered:
- (a) right-size on failure + cache the shrunken ask: treats the symptom; a smaller
  spec session at 128k config caps the conversation below the serving window.
- (b) pre-size from free VRAM at admission: same cap problem.
- (c) **evict-BEFORE-alloc on miss (chosen, slice 1)**: the miss path knows the
  parked entry is dead weight for this conversation (exact rationale already in the
  POOL MISS comment); evicting it BEFORE the first alloc attempt turns
  fail→evict→realloc into evict→alloc. This removes the doomed 4.4GB cudaMalloc
  attempt per turn but NOT the re-prime cost.
- (d) **park-trim / resume-repair**: make the pool actually HIT for the daily
  driver's miss class. The text-prefix matcher requires literal extension; the
  owner's client strips think blocks. Slice 2 investigates a
  longest-common-prefix resume (token-level LCP >= threshold on the APPEND-ONLY
  committed sequence — spec sessions cannot rewind (GDN in-place states), so LCP
  short of committed length cannot resume a spec session; only full-prefix can).
  => For spec sessions the honest fix is (c) + making the alloc failure path cheap
  and non-degrading. The resume-rate problem is a separate lane (session-id
  affinity API, named as follow-up in the POOL MISS comment).

## Receipts

- `owner-session.log` — the owner's live session (178 lines, ctx 8.6k→21.9k).
- `before-curve.jsonl` / `after-curve.jsonl` — scripted repro, per-turn wall time
  + thrash onset (this rig, flock /tmp/gpu5090.lock).
