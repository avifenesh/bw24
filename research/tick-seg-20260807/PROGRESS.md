# lane/tick-seg — kill the SECOND prefill segmentation axis on step35 (serve's per-tick prime_cache calls)

**Mission:** the first axis (MEMRA_PRIME_CHUNK, inside one `prime_cache` call) died in
lane/step35-chunkfix (commit `c809181d`, merged `006aca75`): the SWA arm predicate is keyed on
`seq_end = cache.pos + t`, computed once per CALL. But serve primes a long prompt across SEVERAL
`prime_cache` calls — one per scheduler tick — and each call gets its own `cache.pos`, hence its
own `seq_end`. The predicate is fixed *within* a call and free *between* calls. This lane makes
`prime_cache` see the REQUEST-level seq_end regardless of tick splits.

**Defect receipt (prior lane's, do not re-derive):**
`research/step35-chunkfix-20260807/PROGRESS.md` §9 + `raw/tickinv35-20260807T022010Z.log`.
tickinv probe (concat_prime_probe.rs `"tickinv"` mode, ~line 1031): T=4883, budgets
1024/513 EXACT; 512/256/64 DIFFER maxdiff `1.813e0`, first_div_row 0, greedy step 6 — the SAME
signature as the original chunk defect (same mechanism, one level up). Control T=402 all EXACT.
Nested arm (budget x MEMRA_PRIME_CHUNK=64): inner axis invariant, outer not — axes independent.

**Exposure map (from §9):**
- Interactive lane: MEMRA_PREFILL_TICK=1024 > 512 => not exposed via budget.
- Dark lanes (judge/harvest): MEMRA_PREFILL_JUDGE/HARVEST=256 => EXPOSED; worse, worker.rs:2113
  caps dark budgets by live SLO headroom => LOAD-DEPENDENT segmentation.
- Interactive IS exposed via the prefix-cache LCP split: `snapshot_at = L` stops the first call
  exactly at L (worker.rs:3092-3110 region; prefill_tick bound_rem logic ~worker.rs:3488-3506);
  any LCP in [64, 512] reproduces the FA-prefix shape (PREFIX_CACHE_MIN_TOKENS=64, win=512).

**Upstream precedent (vLLM #51113, research/upstream-sweeps.md 08-07 section at train tip
9fae6a6c — NOT in this branch's history, read via `git show 9fae6a6c:research/upstream-sweeps.md`
lines 328-345, 531-537):** mamba align-chunking prefix-cache poisoning — mid-block chunk end
published as full block; single requests accidentally safe. Two laws the fix must satisfy:
(1) position-keyed state publishes only at grain-aligned ends — only the FINAL chunk of a request
may end unaligned; (2) unaligned STARTS (resume-from-cache at off-grid position) are a second
hole => this lane adds an off-grid-resume gate arm (prime from a prefix-cache hit at LCP in
[64,512], verify bit-identity vs monolithic).

---

## Code map (verified in this worktree at 006aca75)

- `prime_cache`: `crates/memra-engine/src/hybrid_forward.rs:407`. Computes
  `let seq_end = cache.pos + t;` at :465, threads to `prime_chunk(.., seq_end)` (:544) ->
  `full_attn_prime(.., seq_end)` (:1305, step35 divert at :1311) -> `step35_attn_prime` (:6960)
  -> `step35_attn_pre_wo` (:6829). Predicate at :6884-6891:
  `MEMRA_STEP35_SWA_TKV=1 ? t_kv > win : seq_end > win` (rollback seam, canary teeth).
- Cacheless path: `step35_attn` (:6944) passes `t` with
  `debug_assert_eq!(seq_end, t)` at :6921. `prime_chunk_captured` (:876) same.
- `prime_cache_batch` (:978): REFUSES step35 (`cfg.step35.is_some()` -> Err at :995). So the
  worker's batched-prime paths (worker.rs:2303, :2544) cannot carry step35 — per-tick single
  `prime_cache` is the only serve prefill for the SKU.
- Serve per-tick callers (the fix's targets):
  - `prefill_tick` worker.rs:3470-3532; prime call :3508. Budget from `LanePolicy::prefill_budget`
    (`crates/memra-lanes/src/lib.rs:118`); dark budgets SLO-capped at worker.rs:2113.
  - `step_session` prefill phase worker.rs:3941-3968; prime call :3954 (budget PREFILL_TICK_T).
  - LCP split: `snapshot_at` computed ~:2992-3046, bound_rem chunk-stop in prefill_tick
    :3488-3506, `prefix_insert_from_session` at :3523-3526.
- Engine-side non-tick callers (single call = whole request; seq_end already right):
  decode.rs:2105,:2349; spec.rs:3456; gemma_spec.rs:651; dflash.rs:553; ~15 bins incl.
  run_gen.rs, prime_batch_gate.rs (prefix+suffix two-call pattern — NOTE: that's a
  resume-style caller too), concat_prime_probe.rs (tickinv arm loop :1058 is the replica).
- tickinv probe mode: `crates/memra-engine/src/bin/concat_prime_probe.rs:1031-1105`. Replicates
  worker tick loop incl. PRIME_MIN_T tail merge; budget 0 = monolithic ref.
- Box scripts to reuse: `research/step35-chunkfix-20260807/{tickinv35,gate35,exact35,perf35,
  spec35,battery35}.sh`; box state `~/STATE-chunkfix.md` on `ubuntu@18.195.123.14`
  (key ~/.ssh/darklanes-g7e12.pem), artifact `~/step37/models/step-3.7-flash/IQ4_XS/`,
  flock /tmp/memra-gpu.lock, PP-2 = `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`.

## Fast-gate registration shape (from chunkinv35, models.tsv:96-97 + map.tsv:77)

`tools/fast-gate/models.tsv` `kind=cmd` rows; fast-gate.sh reads a `SKIP` word from the script's
output for missing artifacts (fast-gate.sh:174-180). chunkinv35 rows call
`tools/chunk-invariance-gate.sh --label step35-swa --prompts .../prompt-pp6257.txt
--chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24` (+ `--canary` twin).
map.tsv:77 routes `hybrid_forward.rs` (+decode/pp/etc) to the arm list. Plan: a new
`tools/tick-invariance-gate.sh` (same self-gating shape, resolves the step37 artifact per-label,
SKIPs clean when absent) wrapping the probe's `tickinv` mode with budgets `0,1024,513,512,256,64`
+ an off-grid-resume arm; registered as `tickinv35` (+ `tickinv35c` canary via the
MEMRA_STEP35_SWA_TKV... NO — the canary seam for the OUTER axis must break tick-splits, see Fix
plan below; seam choice recorded there). map.tsv must route BOTH hybrid_forward.rs and
memra-server/src/worker.rs to the arm.

## Fix plan (sketch, pre-implementation)

The per-call `seq_end = cache.pos + t` is correct only when the call IS the whole request. The
request-level quantity must survive tick splits. Options considered:

A. **Change `prime_cache` API**: `prime_cache(e, tokens, cache, seq_end: usize)` where seq_end =
   the REQUEST's absolute end. Every caller passes it; single-call callers pass
   `cache.pos + tokens.len()` (their request == call). Mechanical but touches ~60 call sites,
   most of which would pass the degenerate value — high churn, low information.
B. **Carry it on the Cache**: `cache.prime_target: Option<usize>` set once by the serve loop at
   request admission (cache.pos + full_prompt_len), consumed/validated by prime_cache; None =>
   per-call seq_end (today's behavior, correct for single-call users). Risk: stateful, silent
   if a caller forgets to clear it.
C. **API with explicit request extent, default-correct**: keep `prime_cache` but add
   `prime_cache_to(e, tokens, cache, req_end)` — thin wrapper both serve call sites use;
   `prime_cache` delegates with `req_end = cache.pos + t`. Only ~4 sites change (2 worker ticks +
   probe replica + prime_batch_gate resume pattern), every other caller stays byte-identical by
   construction.

The brief says "fix changes prime_cache's per-call API and every caller" — decision recorded
after reading how gemma4/e4b paths interact (they divert before seq_end exists today; their
gates prove byte-identity). Non-step35 arches ignore seq_end entirely (only
step35_attn_pre_wo reads it) => byte-identity elsewhere is by construction, gates confirm.

Canary/teeth for the new gate: the tick axis has no dedicated seam yet. The old-world behavior
(per-call seq_end) must be restorable to give the gate teeth: plan is
`MEMRA_STEP35_PRIME_CALLLOCAL=1` (name TBD at implementation) restoring per-call seq_end
inside prime_cache — chunk/tick-variant by construction. Flags doctrine: rollback seam,
documented in docs/FLAGS.md.

Off-grid-resume arm: two-call prime (call 1 = L tokens, call 2 = rest) with L in [64,512] —
exactly the prefix-cache LCP shape — must be bit-identical to monolithic. The tickinv probe's
budget loop already produces unaligned starts (call 2 starts at pos=budget); a dedicated
`--split L` arm (or reusing budgets that hit L) makes the resume hole explicit.

## Deliverables checklist

- [x] Commit 1: PROGRESS.md (ae8e3616) — context anchor against restart churn.
- [x] Commit 2: tickinv35/tickinv35c registered (bd7c2c30) — tools/tick-invariance-gate.sh,
      models.tsv rows, map.tsv routes hybrid_forward-class AND memra-server; probe grew
      --splits (off-grid-resume arms, rows `sp<L>`). RED at registration by construction.
- [x] Commit 3: the fix (6b535472) — `prime_cache(e, tokens, cache, queued_after)`;
      seq_end = cache.pos + t + queued_after computed once; worker prefill_tick + step_session
      pass prefill_queue.len() post-drain; all ~60 single-shot callers pass 0; probe's tickinv
      replica passes t-fed-take. Seam MEMRA_PRIME_CALLLOCAL=1 restores per-call value
      (docs/FLAGS.md cataloged). Workspace builds clean.
- [x] Commit 4: ppprime --budget (cbf4a7d0) — times the serve-shaped multi-call prime.
- [x] Box scripts (3544fba5): gate-tickseg.sh / exact-tickseg.sh / perf-tickseg.sh.
- [x] 5090 unaffected-arch control GREEN (9fe7c359,
      raw/unaffected-q9-q35-5090-20260807T114112Z.log): qwen chunkinv PASS + canary teeth;
      q9 tickinv budgets 64/32 + sp64 ALL EXACT (arch never reads seq_end); q9/q35 run-gen
      argmax MATCH at prior-receipt speeds.
- [~] Box gate battery (gate-tickseg.sh, dispatched 11:39Z, raw/gate-tickseg-20260807T113933Z.log):
      **tickinv35 GREEN** — T=4883, budgets 1024(5 calls)/513(10)/512(10)/256(20)/64(77) ALL
      EXACT (0.000e0, identical greedy), and sp64/sp256/sp512 off-grid-resume arms ALL EXACT.
      Pre-fix 512/256/64 DIFFERed 1.813e0.
      **tickinv35c canary HAS TEETH** — seam on: 1024/513 EXACT, 512/256/64 DIFFER 1.813e0 at
      greedy step 6 (the finding lane's receipt digit-for-digit — the seam is a faithful
      pre-fix restoration, hence a legitimate perf BEFORE arm), and the split arms give the
      FIRST MEASURED receipt of the LCP-split exposure (prior lane only enumerated it):
      sp64 DIFFER 1.735e0 @ step 10, sp256 1.594e0 @ step 6, sp512 1.813e0 @ step 6 — every
      LCP class in [64,512] steers the served text pre-fix.
      **chunkinv35 no-regress PASS + chunkinv35c teeth intact** — axis 1 undisturbed by the
      queued_after threading. Battery complete 4/4 rc=0, one flock window 11:40:01Z-12:06:58Z.
- [x] Box exactness (exact-tickseg.sh, raw/exact-tickseg-20260807T115356Z.log, rc=0):
      kernel-check FULL model-backed **ALL GREEN**; run-gen PP-2 argmax **6776 MATCH** (same
      argmax as the chunkfix lane's receipt) + batched-prime MATCH; ppn-gate **BIT-IDENTICAL
      serial + pipelined** (24 steps, n_vocab=128896, fence=[0,22,45]); run-spec K=1..8
      **8/8 SELF-CONSISTENCY PASS**, acceptance digit-identical to the chunkfix baseline
      (K=1 14/18=77.8%; K=2..8 all 15 accepted, 44.1%->11.0%) — the fix is inert on the
      single-call spec path, receipted not asserted.
- [~] Perf (perf-tickseg.sh, running since 12:14:24Z, raw/perf-tickseg-20260807T115358Z.log):
      N=5 interleaved, cells pp6257 budget=1024 (SHIPPED DEFAULT, 1% STOP bar), budget=256
      (dark-lane, where arms change), pp512 null control. Instrument: ppprime --budget (the
      serve-shaped multi-call prime — monolithic ppprime and run-gen's prefill line are blind
      to a multi-call change by construction).
- [~] Raw logs pulled back as they complete; PROGRESS.md finalized after perf.

## Why the fix is correct-by-construction (not a tolerance argument)

Identical in kind to the chunkfix lane's §2 argument, one level up. The request's absolute end
position is a property of the REQUEST; `queued_after` merely restores information the caller
always had and the engine structurally lacked. For any segmentation of the same request,
`cache.pos + t + queued_after` is the same constant at every call, so step35's arm predicate
evaluates identically wherever the tick boundaries fall — including the LCP-split boundary
(bound_rem only shortens a tick's take; the remainder stays in prefill_queue, so the worker's
post-drain `prefill_queue.len()` is exact at the split too) and an off-grid RESUME (call 2's
`cache.pos = L` plus its own t and remainder still sums to the same request end).

Why nothing else can move:
- Only `step35_attn_pre_wo` reads `seq_end`; every other arch's dispatch never sees it. The
  5090 control (q9/q35) and the box exactness battery prove this on silicon, not just by
  construction.
- Every single-call caller passes `queued_after=0`, making `seq_end = cache.pos + t` — the
  pre-fix expression exactly. Those paths are byte-identical by substitution: same value, same
  arithmetic (run-gen argmax 6776 unchanged from the chunkfix lane's receipt confirms it).
- SESSION CONTINUATIONS (a new user turn on a live cache) pass 0 deliberately: a new turn is a
  new request whose arithmetic keys to its own extent. This preserves the existing serve
  contract (spec-session turns, kv-reuse resumes) — the fix changes only how ONE request's
  own segmentation is expressed, never how requests compose.
- The smem ceiling argument transfers: the arm only flips FA->naive_w on chunks whose t_kv <=
  win+chunk (the windowed floor's t_kv = win-1+t <= 12287 constraint is per-CALL geometry,
  untouched by where seq_end comes from).

## Residual found while fixing (NOT fixed here, named for the record)

**step35 prefix-cache entries are EXTENT-CLASSED.** Keying the SWA arm on the request's extent
(seq_end) makes the arm — and therefore the numeric class of every hidden row, and therefore the
KV BYTES appended at layers > 0 — a function of the request's total length. Two requests sharing
a >= 64-token prefix but straddling the window (creator seq_end <= 512 = FA class, consumer
seq_end > 512 = windowed class, or vice versa) will produce different prefix KV bytes. A
prefix-cache RESUME therefore continues from bytes of the creator's class, and can differ from a
cold prime of the same prompt — exactly vLLM #51113's cache-entry-side law ("state snapshot keyed
by position may only be published at grain-aligned ends" — here the state is additionally keyed
by the creator's extent, which position-keying cannot see). This is currently INSIDE the
documented contract (worker.rs prefix-seed comment: "the entry stores whatever config ran"), and
the same class of cold-vs-resume divergence already exists engine-wide for session continuations.
The canonical fix is ONE numeric class for all step35 SWA prefill rows regardless of extent
(always-windowed; the mask is a causal no-op for seq_end <= win) — a default-path perf trade on
every short interactive prompt, i.e. its own measured lane, not a rider on this one. What THIS
lane guarantees: a single request is bit-identical under every segmentation serve can produce
(tick budgets, SLO-capped budgets, LCP splits, off-grid resumes of its own prefix).

## State log

- 2026-08-07: worktree clean at 006aca75 (lane/tick-seg). Three restart kills burned context on
  re-reads; this file is the anchor. Code map above verified by grep in THIS worktree.
- 11:39Z: box synced (files-only rsync, no deletions), BOX-COMMIT stamped cbf4a7d0,
  ~/STATE-tickseg.md written, gate-tickseg.sh dispatched (build 28s green, flock acquired
  11:40:01Z, cards 0 MiB).
- 11:41Z: 5090 control battery green in 31s under CPUQuota=1200%.
