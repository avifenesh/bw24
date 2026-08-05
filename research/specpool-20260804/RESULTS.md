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

## Fix landed (commits 86518927 + 72ca6005)

1. **evict-first** (learned per model, process lifetime): after one observed
   "parked ghost + new session don't fit" failure, every later pool miss evicts
   the dead-weight pool BEFORE allocating. Same eviction the failure forced,
   minus the doomed multi-GB cudaMalloc + retry per turn. Roomy rigs never learn
   the flag; pool HITS are probed before the miss arm and untouched.
2. **right-size ladder** on genuine (post-evict) failure: halve from
   learned/ctx_cap÷2 toward `need` = prompt + budget + 64; memoize the landing
   (learned_ctx). MaxNew preempts ContextFull by construction => identical
   emission. Hardened by (a) fallible embed-table residency at landing (the lazy
   get_or_init/expect upload OOM-panicked the worker on the first un-hardened
   ladder run) and (b) a 1.5GiB probe-allocate-drop transient reserve on NEW
   landing sizes (mem_get_info can't see the pinned-threshold async pool's
   cached blocks; only a probe proves fit).

## Measured (this rig, RTX 5090, flock /tmp/gpu5090.lock; single runs, N=1 per
## curve, thermals steady across the pair)

Owner regime = history-rewrite client (every turn a pool miss), 25 turns,
~11.6k->14.4k tok prompts, 100 tok/turn, owner serve env (128k floor,
MAX_SESSIONS=1, REUSE_POOL=1):

- BEFORE (`curve-before-miss.jsonl` / `server-before-miss.log`): 24 evict-retry
  cycles (one per turn), total 290.2s, per-turn wall 10.4->12.9s.
- AFTER (`curve-after-miss.jsonl` / `server-after-miss.log`): 1 evict-retry
  (the learning turn) + 23 pre-alloc evictions (no failed alloc), total 289.4s.
  Byte-identical text all 25 turns (text_sha; `exactness-check.txt`).
- The wall-time slope (~+2.4s over 22 turns, both runs) is prompt-growth prime
  cost (every turn re-primes ~12-14k tokens), NOT alloc churn — the alloc
  fail+retry itself is cheap on this driver stack. The owner's felt slowdown is
  dominated by the every-turn full re-prime, which is the resume-rate problem
  (session-id affinity follow-up), plus the removed churn.
- Control (literal-extension client, `curve-before.jsonl`): 26/29 turns resume
  from the parked session, 1.1s/turn flat — the pool works when the client
  extends literally.

Ladder path (5GB VRAM ballast attached post-load; `curve-ladder-miss.jsonl` /
`server-ladder-miss.log`): full-size 128k ask impossible; sessions land at
ctx 16384 of 131072 and serve spec bursts with text byte-identical to the
unballasted run (`ladder-exactness.txt`). Pre-fix binary in the same scenario
400s immediately (`server-ladder-prefix.log`, "cache alloc failed" turn 0) —
the ladder converts a hard failure into served-at-right-size.
Known remaining gap (pre-existing, out of F5 scope): a mid-generation OOM
inside step_session at extreme pressure still errors the request (turn-4
"step error" in the ballast scenario).

## Gates

- n=1200 deep greedy probe, owner config, pre-fix vs post-fix binary:
  BYTE-IDENTICAL (`probe1200-prefix.txt` == `probe1200-postfix.txt`, 1201 tok).
- 25-turn owner-regime curve: text_sha byte-identical pre/post all turns.
- serve-st-gate + kernel-check: see `gates.log`.

## Receipts

- `owner-session.log` — the owner's live session (178 lines, ctx 8.6k→21.9k).
- `curve-*.jsonl` + `server-*.log` — per-turn wall time + server-side evict
  classification per run tag; scripts `drive-session.py`, `run-curve.sh`,
  `run-ladder.sh`, `run-probe1200.sh`, `vram-ballast.py`.
