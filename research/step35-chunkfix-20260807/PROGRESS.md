# lane/step35-chunkfix — the step35 chunk-dependence defect, fixed and gated

**Mission:** close the one receipted exactness hole on THE SKU (Step-3.7-Flash). step35 prefill was
CHUNK-DEPENDENT past its 512-token SWA window: `MEMRA_PRIME_CHUNK` — documented in `docs/FLAGS.md`
as a machine-config/OOM knob an operator is invited to set per rig — changed the prefill logits, the
hidden rows, and the generated text.

**Defect receipt (not this lane's work, read it first):**
`research/step37-p2-20260806/raw/chunkinv-step35-GAP2-CONFIRMED-20260807.txt`, commit `66a81371`,
merged `9971e7f8`. That lane found, measured, and reduced the defect to a closed form, then
deliberately did NOT fix it (a kernel-selection change on the launch SKU's served prefill needs
before/after numbers per `research/benchmarks.md`, not a bring-up commit) and deliberately did NOT
land the matching gate (it would have been a known-red check). This lane does both.

**Box:** 2x RTX PRO 6000 Blackwell Server 96GB, PP-2 (`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`),
`18.195.123.14`, artifact staged at `~/step37/models/step-3.7-flash/IQ4_XS/` (IQ4_XS, 3 shards).
Branch commit on box stamped in `~/step37/memra/BOX-COMMIT.txt`. Every GPU window under
`flock /tmp/memra-gpu.lock`. Box is SPOT; state carried in `~/STATE-chunkfix.md`.

---

## 1. The defect in one paragraph (from the receipt, not re-derived)

On SWA layers (33 of 45, `win=512`) step35's prime attention picked its kernel per chunk at
`hybrid_forward.rs:6820-6844`:

    off  = swa ? base_len.saturating_sub(win-1) : 0
    t_kv = base_len + t - off
    swa && t_kv > win  ->  sdpa_naive_w_quantized_view   (f32 windowed floor)
    else               ->  fa_prefill_view_ws            (hd128 dequant-once FA)

A chunk `[b,e)` with `b <= win-1` has `off = 0`, hence `t_kv = e`, hence it is FA iff `e <= win`;
every later chunk has `t_kv = t + (win-1) > win`. The FA rows were therefore a contiguous **prefix**
`[0,P)` with

    P = c * floor(win/c)   for c <= win ;   P = 0   for c > win

and the output depended **only** on `P`. The two kernels are not the same numeric class — swapping
them on the same rows moves the logits by ~1.8 — so the pre-fix comment "Same cache bytes, same
numeric class" was false as written. Measured at T=4883: `P(4096)=P(513)=0` mutually EXACT,
`P(512)=P(64)=512` mutually EXACT (10 chunks vs 77 — a reduction-order account forbids that) and
both DIFFER from the P=0 family by maxdiff `1.813e0` with greedy text diverging at step 6. `P`
differs across `{64,512,1024,2048,4096}` for 95.7% of prompt lengths under 12000, from T=513.

---

## 2. The fix: select on the REQUEST, not on the chunk

Commit `c809181d`. The arm predicate becomes `seq_end > win`, where `seq_end` is the **absolute end
position of the whole prime request** (`cache.pos + prompt_len`, computed **once** in `prime_cache`
before the chunk loop starts). Every chunk of a given request then evaluates the same predicate
whatever the chunk size, so `P` is identically 0 and `MEMRA_PRIME_CHUNK` is a pure memory knob again.

Threading: `prime_cache` -> `prime_chunk(..., seq_end)` -> `full_attn_prime(..., seq_end)` ->
`step35_attn_prime(..., seq_end)` -> `step35_attn_pre_wo(..., seq_end)`. Two paths pass `t` with a
`debug_assert` pinning why: `step35_attn` (cacheless prefill — `forward`/`forward_last`/t2probe, no
chunk loop exists there) and `prime_chunk_captured` (one unchunked bucket over a fresh cache). Those
asserts are the guard against a future chunked variant of either path silently re-opening the door.

**Why this is correct-by-construction, not a tolerance argument.** For `e <= win` the window mask is
a no-op under causal masking — that is precisely why the FA arm was legal on those rows in the first
place. The windowed kernel computes the same masked attention; it is now simply the only arm used
once the request passes the window. Nothing is being traded for invariance.

**Rollback seam / canary:** `MEMRA_STEP35_SWA_TKV=1` restores the pre-fix `t_kv` predicate. It is
chunk-variant by construction, which is exactly what gives the new gate teeth (see §4).

### 2.1 Enumeration: the shipped default cannot move

Before touching the box, the arm assignment was enumerated against the real loop
(`hybrid_forward.rs:461-477`, **including** the `PRIME_MIN_T=16` tail merge) pre- and post-fix:

| check | result |
|---|---|
| Post-fix `P` across `c` in `{0,2,16,32,64,128,256,384,512,513,600,768,1024,2048,4096,8192}`, for every T in [2,3000) plus {4096,4883,6257,8192,12000,16384,32768,120000} | **chunk-dependent at 0 values of T** |
| Pre-fix, same chunk set `{64,512,1024,2048,4096}`, T in [2,12000) | chunk-dependent at **11487 / 11998** values of T (matches the receipt's 95.7%) |
| `c=4096` (the shipped default) arm SEQUENCE, pre vs post, for every T in [2,20000) plus {32768,65536,120000,131072} | **IDENTICAL at every T — 0 differences** |
| smem ceiling: max `t_kv` among chunks that NEWLY take `naive_w`, over every (T,c) tested | **512** (= `win`), far under the `t_kv <= 12287` ceiling |
| pre-existing large-chunk ceiling cases (`c=12000`, T=131072: max naive `t_kv` = 12511) | **numerically unchanged** pre vs post — this lane neither creates nor fixes that |

Why the default is provably still: at `c=4096` on a `seq_end > win` prime, chunk 0 already had
`t_kv = min(chunk, seq_end) > win`, and every later chunk has `t_kv >= t + win - 1 > win` because
`PRIME_MIN_T` keeps `t >= 16`. A `seq_end <= win` prime keeps every chunk on FA exactly as before.
So the fix is a no-op on both default regimes and only removes the FA prefix at small chunk values.
**The perf measurement in §5 exists to confirm that prediction on silicon, not to discover it.**

---

## 3. Gate: `chunkinv35` / `chunkinv35c`

The finding lane named the assertion; this lane lands it, green, in the same commit as the fix.

    tools/chunk-invariance-gate.sh <step37.gguf> --label step35-swa \
        --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
        --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24

Registered as `chunkinv35` (naked, assert invariant) and `chunkinv35c` (canary) in
`tools/fast-gate/models.tsv`; `tools/fast-gate/map.tsv` routes `hybrid_forward.rs` to both.

**Why a second arm rather than widening the existing `chunkinv`.** Two independent reasons the qwen
arm was blind here, both from the receipt:
1. Its pinned prompts are 96 and 147 tokens — **below** step35's 512 window — so every chunk took
   the same kernel and the gate compared one kernel against itself (GAP 2).
2. Its canary seam `MEMRA_PRIME_F32CHUNK0` is read in `full_attn_prime_fa_dispatch`, which step35
   never reaches (`full_attn_prime` diverts at `:1289`). The canary was **inert** on this arch
   (GAP 1). `MEMRA_STEP35_SWA_TKV` is the seam that arch actually needed.

The chunk set is not arbitrary: it spans both sides of the closed form — `4096,513` gave `P=0`,
`512,256` gave `P=512`, and `64` gave `P=512` via 77 chunks instead of 10. Pre-fix those families
agreed *within* and disagreed *across*; post-fix all five must be byte-identical, which is a
strictly stronger assertion than "some chunk sizes agree".

Script changes (`tools/chunk-invariance-gate.sh`): `--prompts` / `--seam` / `--label` so the
arch-specific arms share one script; per-label artifact resolution for the box-staged step37 GGUF
(`MEMRA_STEP37_GGUF` or `~/step37/...`) with a clean SKIP when absent (fast-gate reads the script's
own SKIP word — the hole that once reported `chunkinv` as "PASS (0s)" on a rig with no artifact);
and the summary-table grep now keys off the actual `--chunks` values, since the hardcoded
`2048|64|32` printed nothing on any other chunk set.

---

## 4. RESULTS — gate battery

Raw: `raw/gate35-20260806T235547Z.log`. One flock window from 23:55:47Z, cards 0 MiB at acquire.

### 4.1 `chunkinv35` — GREEN (this is the deliverable)

    label=step35-swa assert=invariant seam=MEMRA_STEP35_SWA_TKV legacy-seam=off got=invariant
    canary=0 chunks=4096,513,512,256,64  T=4883
        513 | EXACT | -1 | 0.000e0 | identical
        512 | EXACT | -1 | 0.000e0 | identical
        256 | EXACT | -1 | 0.000e0 | identical
         64 | EXACT | -1 | 0.000e0 | identical
    chunkinv verdict: CHUNK-INVARIANT — prefill logits bit-identical at every chunk size
    chunk-invariance-gate: PASS

Pre-fix this same invocation returned CHUNK-DEPENDENT with 512/256/64 all diverging.

### 4.2 `chunkinv35c` — canary has teeth, and it reproduces the finding lane's numbers exactly

    legacy-seam=on got=variant canary=1
        513 | EXACT  |  -1 | 0.000e0 | identical
        512 | DIFFER |   0 | 1.813e0 | step 6
        256 | DIFFER |   0 | 1.813e0 | step 6
         64 | DIFFER |   0 | 1.813e0 | step 6
    chunkinv verdict: *** CHUNK-DEPENDENT ***
    chunk-invariance-gate: PASS (canary broke the assertion as required — gate has teeth)

Two things worth stating plainly. First, the canary changes the **world**, not the label — the trap
documented in the gate script's header (a label-only canary is perfectly correlated with the default
gate and proves nothing). Second, the seam reproduces the receipt's numbers to the digit: maxdiff
`1.813e0`, `first_div_pos = 0`, greedy divergence at step 6, and `513` EXACT while `512` DIFFERs —
the one-token knife edge. That is independent confirmation that `MEMRA_STEP35_SWA_TKV` is a faithful
restoration of the pre-fix arithmetic and therefore a legitimate BEFORE arm for §5's perf work.

It also newly shows `256 | DIFFER`, which the finding lane measured only against `ref=512` (where it
was EXACT, `P` matching). Against `ref=4096` the closed form predicts DIFFER (`P=512` vs `P=0`) and
it does — a 14th arm-pair consistent with the model.

---

## 5. RESULTS — the finding lane's own falsification battery, re-run post-fix

Raw: `raw/battery35-20260807T001751Z.log`. One flock window 00:17:51Z -> 00:38:44Z (21 min), cards
0 MiB at release. `battery35.sh` re-runs the bodies of the finding lane's committed
`chunkinv-long.sh` and `chunkinv-knife.sh` against the fixed build, plus a wider boundary sweep and
the below-window control. The finding lane pre-registered four falsification predictions and hit
4/4; every one of them is now dead:

| arm | pre-fix (the receipt) | post-fix |
|---|---|---|
| LONG `prompt-pp6257` T=4883, chunks 4096,2048,512,64 | 512 and 64 **DIFFER**, maxdiff `1.813e0`, greedy step 6 | **all EXACT** (`-1`, `0.000e0`, identical) |
| KNIFE PRED-1+2 ref=4096 vs 513,512 (the one-token flip) | 513 EXACT, 512 **DIFFER** | **both EXACT** |
| KNIFE PRED-3+4 ref=512 vs 384,256 | 384 **DIFFER** @row 384, 256 EXACT | **both EXACT** |
| BOUNDARY T=4883, chunks 4096,1024,600,128,32,16 | `P` in {0,0,512,384,512,512} -> mixed | **all EXACT** |
| CONTROL T=402 (below the 512 window), chunks 4096,512,64,32 | all EXACT (nothing to break) | **all EXACT** — unchanged |

Every arm returns `chunkinv verdict: CHUNK-INVARIANT — prefill logits bit-identical at every chunk
size`, `rc=0`. The KNIFE arms matter most: they were built to be the sharpest possible probe of the
closed form (a **one-token** change in chunk size flipping the verdict, because `P` jumps 512 -> 0
between `c=512` and `c=513`). A fix that merely moved the boundary would still show a flip
somewhere in 384/512/513/600; none does. The T=402 control is the guard against the trivial way to
pass this battery — breaking prefill so badly that everything agrees on garbage: it exercises the
same code with `seq_end <= win` and is byte-identical to its pre-fix self.

---

## 6. RESULTS — exactness battery (BAR-2)

Raw: `raw/exact35-20260807T004546Z.log`. One flock window 00:45:46Z -> 00:47:22Z, cards 0/0 MiB at
release. Same `c809181d` binaries the gate and battery ran on (`BOX-COMMIT.txt`).

| gate | result |
|---|---|
| `kernel-check` model-backed on the step35 IQ4_XS artifact, FULL (no `MEMRA_KC_FAST` / `MEMRA_KC_ONLY`) | **`ALL GREEN: kernels match CPU reference.`** exit 0 |
| `run-gen` argmax, PP-2, ngen=64 | `prefill argmax=6776 decode argmax=6776 ... MATCH` + `batched-prime argmax=6776 tokenwise argmax=6776 MATCH`, exit 0 |
| `ppn-gate` stages=2 (this is the pair-topology receipt) | **`ppn gate PASS [serial]`** and **`PASS [pipelined]`**: 24 steps (8 prime + 16 gen) **BIT-IDENTICAL** logits vs the door-OFF reference, `n_vocab=128896`, `fence=[0, 22, 45]`, exit 0 |

`ppn-gate` is the load-bearing one here, and worth being precise about what it does and does not
cover. It asserts the PP-2 split path is bit-identical to the unsplit walk over the same sharded
placement, i.e. the fix did not perturb anything at the stage boundary (stage 0 = layers 0-21,
stage 1 = 22-44 — both stages carry SWA layers, so both run the changed arm). It runs 8 prime
tokens, so it does not itself cross the window; the window-crossing assertion is §4/§5's job.

The `kernel-check` run is model-backed on the SKU's own bytes: `iq4xs-mmq
[Step-3.7-flash-IQ4_XS-00001-of-00003.gguf token_embd.weight]` at T=16/64/128/512 all OK
(`rel<=2.04e-4`). The many `KC-SKIP [section] <other model>.gguf: absent on this box` lines are
this box holding only the step artifact — they are pre-existing coverage gaps of this *box*, not of
this change, and the arms they gate (qwen/gemma/ornith NVFP4, Q4_0, Q8_0) are covered on the 5090
in §7. Recorded rather than glossed because a reader counting green lines would otherwise
overcount.

### 6.1 q9/q35 unaffected (BAR-4)

Raw: `raw/unaffected-q9-q35-5090-20260807.log`. Local RTX 5090 Laptop, under
`systemd-run --scope -p CPUQuota=1200% -p MemoryMax=48G` with `flock /tmp/memra-gpu.lock` held
(desktop stays responsive; no uncapped saturation).

| check | result |
|---|---|
| qwen `chunkinv` (the pre-existing arm, default label/seam/prompts) | PASS — CHUNK-INVARIANT on both pinned prompts |
| qwen `chunkinvc` canary | PASS — still has teeth (64 DIFFER `5.269e-1`, 32 DIFFER `6.375e-1`) |
| `run-gen` q9 (Qwen3.5-9B-NVFP4-MTP) | `prefill argmax=271 decode argmax=271 MATCH`, `batched-prime MATCH`, pp89 2486.5 tok/s, decode 134.79 tok/s |
| `run-gen` q35 (Qwen3.6-35B-A3B IQ4_XS) | `prefill argmax=271 decode argmax=271 MATCH`, `batched-prime MATCH`, pp89 1693.3 tok/s, decode 180.73 tok/s |

Two independent reasons the other arches cannot move, and the receipts above are the belt to that
braces: (1) the predicate change lives inside `step35_attn_pre_wo`, reachable only through
`full_attn_prime`'s `self.cfg.step35.is_some()` divert at `:1289` — every other arch takes the
`full_attn_prime_fa_dispatch` path, untouched; (2) the only shared-path edit is threading one
`usize` argument, which no other arch reads. The qwen `chunkinv` pair also confirms the generalized
`chunk-invariance-gate.sh` (new `--label` / `--prompts` / `--seam` flags, rewritten summary grep)
did not break its original arm — the script change is as load-bearing as the engine change here.

---

## 7. RESULTS — prefill perf, before vs after (BAR-3)

Raw: `raw/perf35-20260807T004722Z.log`. **One flock window** 00:47:22Z -> 02:12:39Z (85 min), 30 arm
invocations, cards 0/0 MiB at release. Thermal regime: warm steady-state throughout — GPU 0 held
36-39 C at 2392-2400 MHz and GPU 1 32 C at 2325 MHz across the whole window (per-rep
`nvidia-smi` samples in the log), so no arm ran on a cold or throttled card.

**Instrument.** `concat-prime-probe ppprime`, which times `prime_cache` — the path this fix changes.
`run-gen`'s "prefill tok/s" line times `forward_last`, the **cacheless monolithic** path where
`seq_end == t` by construction, so it cannot see this change at all. Recorded so nobody
re-measures the wrong thing and concludes "no effect" from an instrument that is blind by design.

**Arms.** Same binary, one process per arm, strictly alternating AFTER/BEFORE (the repo's
interleaved law — cross-run and cross-day comparisons are clock-drift-invalid, including for a
self-comparison). AFTER = naked default (`seq_end` predicate, the shipped path). BEFORE =
`MEMRA_STEP35_SWA_TKV=1`, the rollback seam. The seam is a legitimate BEFORE arm because §4.2
showed it reproduces the pre-fix receipt's arithmetic to the digit; using it avoids a second build
of `c809181d^` and therefore avoids comparing two different compilations.

Each printed median is itself the median of 3 timed reps after 1 warmup, so each cell is
5 interleaved arm-medians per side (N=5 of the quantity compared), 15 timed primes per side.

| cell | arm | N | median | tok/s | within-arm spread | delta |
|---|---|---|---|---|---|---|
| **pp6257 (T=4883) chunk=4096 — THE SHIPPED DEFAULT** | AFTER | 5 | 52.1910 s | **93.56** | 0.119% | **+0.009%** |
| | BEFORE | 5 | 52.1956 s | 93.55 | 0.181% | |
| pp6257 (T=4883) chunk=512 — where the fix changes an arm | AFTER | 5 | 45.4441 s | 107.45 | 0.268% | **-0.467%** |
| | BEFORE | 5 | 45.2320 s | 107.95 | 0.190% | |
| pp512 (T=402) chunk=4096 — null control, below the window | AFTER | 5 | 3.6498 s | 110.14 | 0.408% | **-0.093%** |
| | BEFORE | 5 | 3.6464 s | 110.25 | 0.255% | |

### Verdict against the BAR's stop condition

The bar was: *if the default moves >1%, STOP and report rather than ship.* **The default moved
+0.009%** — three orders of magnitude inside the threshold, and an order of magnitude below the
arms' own within-arm spread. The lane ships.

Read the null control first, because it calibrates everything above it. At T=402 the two arms are
**the same machine code taking the same branch** (`seq_end = t = 402 <= win`, so both predicates
evaluate false and select FA): the only honest expected delta is 0. It measured **-0.093%**. That
is this box's noise floor for this instrument, and it means a delta of a few tenths of a percent
carries no signal. The default cell's +0.009% is comfortably below even that.

The chunk=512 cell's -0.467% is the only delta larger than the null control, and it is the one cell
where the fix genuinely changes work: pre-fix, rows [0,512) took dequant-once FA; post-fix they take
the f32 windowed kernel, one chunk out of ten. It is a **non-default** configuration, it is still
half the STOP threshold, and it is only ~2.5x the null control's magnitude, so calling it "a real
0.47% cost" rather than "noise with a plausible story" would be over-reading three tenths of a
percent. What can be said without over-claiming: at the one chunk size where the fix does extra
work, the cost is bounded well under 1%, and it buys exactness.

§2.1's enumeration predicted the default's arm sequence is **identical** pre- and post-fix at every
T, hence a zero delta. That prediction is now confirmed on silicon rather than argued. The purpose
of a measurement whose result is known in advance is exactly this: it converts "the code should not
be able to move" into "the code did not move".

### Incidental finding, recorded not acted on

At T=4883 the **non-default** `MEMRA_PRIME_CHUNK=512` primes **14.8% faster** than the shipped
default 4096 (107.45 vs 93.56 tok/s, both AFTER arms, same window, same thermal state). That is a
prefill-tuning lead, not this lane's business: it is a *serving default* question that needs its own
sweep across T and its own memory-headroom accounting (the chunk default exists to bound per-layer
transient allocation — the long-ctx OOM fix), and changing it would be a board-moving change under
a different lane. Flagged here because the number was measured cleanly and would otherwise be lost;
this lane does not change the default.
