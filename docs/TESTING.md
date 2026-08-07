# Testing: the tiered gate structure

Two regimes, one rule: **the full battery gates every merge and tag, unchanged; fast-gate
accelerates the dev loop between battery points.** Nothing in this document weakens the
merge/tag bar — a fast-gate green is a *keep going* signal, never a *ship* signal.

## The tiers

| Tier | Wall (5090 rig, measured 2026-08-02) | What runs | When |
|---|---|---|---|
| 0 | seconds (~2 s kernel-check scoped + build) | workspace compile + kernel-check scoped to the touched sections | every edit-compile loop |
| 1 | ~1–2 min | tier 0 + golden-token argmax probe on ONE model per affected kernel class (+ one single-K spec probe when the diff touches the spec pipeline) | before every dev-loop commit |
| 2 | tens of minutes | the full battery: `tools/local-ci.sh` — kernel-check ALL GREEN (~4.5 min), prime-gate, run-gen argmax per model, VERIFY-GATE, `run-spec` K=1..8 self-consistency on the Qwen 35B target + external MTP draft (`MEMRA_CI_RUNSPEC=0` skips), Gemma-4 31B stream agreement 64/64, decode-batch-gate (config + Q8_0 strict — the serving tick's exactness, wired in 2026-08-05), graph-warmup stress (`tools/graph-warmup-stress-gate.sh` — pool-growth adversarial bit-identity behind the `MEMRA_GRAPH_WARMUPS=1` default, wired 2026-08-05), serve-smoke, serve-stress (`tools/serve-stress-gate.sh` — the c=64 concurrency contract behind the admission spec-headroom fix, wired 2026-08-06; `MEMRA_CI_STRESS=0` skips), accept-gate (`tools/accept-gate.sh` — exact served-spec acceptance counts + a 128-token text sha at the production drafter/K, wired 2026-08-06; smoke cell by default, `--full` for the 6-cell matrix, `MEMRA_CI_ACCEPT=0` skips) | **every merge, every tag** (unchanged) |

The docs-fit owner call is closed: tier 2 now runs the full `run-spec` K=1..8 sweep and requires
eight per-K PASS lines plus the final `SELF-CONSISTENCY PASS` marker. The raw run is logged before
parsing; a red quotes the failing K and `FIRST DIVERGENCE` index.

**One correction to how tier 2 is described elsewhere:** `--tier 2` does not run the perf stage.
`fast-gate.sh:68` is `exec tools/local-ci.sh` with no arguments, and local-ci.sh gates its perf
cells behind `--perf` / `--perf-quick`. So `--tier 2` is correctness-only; run
`tools/local-ci.sh --perf` directly for the cell battery.

Entry point:

```bash
tools/fast-gate/fast-gate.sh                       # tier 1 vs HEAD (uncommitted work)
tools/fast-gate/fast-gate.sh --tier 0              # compile + scoped kernel-check only
tools/fast-gate/fast-gate.sh --diff main           # scope = everything since main
tools/fast-gate/fast-gate.sh --tier 2              # execs tools/local-ci.sh (the real gate)
tools/fast-gate/fast-gate.sh --smoke               # add the perf tripwire (see below)
tools/fast-gate/fast-gate.sh --probes k27,amargin  # name the arms explicitly (see below)
```

**`--probes` is not optional convenience — it is how you gate a tree with no diff.** A clean tree
(release candidate, a fresh rsync onto another rig, a tree with no `.git` at all) has an empty
`CHANGED` set, and the diff-driven path exits 0 with "nothing to gate". That is a FALSE GREEN if
you meant to gate it: it reports success having run **zero** probes. Found on the v0.71.0 pod
battery, where the rsync'd tree had no `.git` and the k27 regression check "passed" without
executing (`fast-gate.sh:80-88`). Naming arms with `--probes` suppresses the short-circuit.

It is also the **only** way to reach four registered probes. `tools/fast-gate/map.tsv` dispatches
`accept, aw35, chunkinv, chunkinvc, chunkinv35, chunkinv35c, g12, g12d, g26, g31, gwstress,
isogap, isogapc, k27, o35, o9, q35, q35slru, q9, samp, sampt, sstress, tickinv35, tickinv35c`
(+ spec probes `q35spec, g31spec`), with `DEFAULT = g12,q9,q35,gwstress`.
Not in any row, including DEFAULT: **`amargin`, `amarginc`, `e4b`, `kat`**. They are registered in
`models.tsv` and run correctly when named, but nothing dispatches them automatically — so treat
the amargin gate below as an explicit-invocation gate, not a standing one.

## Change-scoped gating (tier 0)

`git diff --name-only <ref>` (plus untracked files) is mapped through
[`tools/fast-gate/map.tsv`](../tools/fast-gate/map.tsv) — an editable TSV that encodes the
dispatch structure: which kernels serve which model classes. Every matching row contributes
to the plan (union); an unmatched path falls back to the conservative full plan and prints a
warning (add a row when that happens).

kernel-check gained two loud diagnostics seams for this (see the `kc_model` header in
`crates/memra-engine/src/bin/kernel_check.rs`):

- `MEMRA_KC_FAST=1` — synthetic arms only, every weight-oracle section skipped (loudly).
  Measured: **~1.4 s** vs **~4.5 min** full (the model-backed GEMM oracles are >98% of the
  wall — 266 s of 268 s in the timed run,
  `research/fast-gate-20260802/kernel-check-full-timed.log`).
- `MEMRA_KC_ONLY=csv` — synthetic arms + only the weight-oracle sections whose name matches
  (`dtype5`, `nvfp4-gemm`, `q8mmq-gemm`, `q4_0-mmq`, `q4_0-sk-arm`, `iq4xs-mmq`,
  `f16g-kq-direct`, `nvfp4-27b-shape`, `nvfp4-mmvq`, `nvfp4-batched`, `a6-split-plane`,
  `d2-cache-bit-identity`, `fast-router-batch`).

Both seams print a `KC-SKIP` line per skipped section naming the env — a scoped run is never
silently narrower than it looks. They are diagnostics-class flags per the flags doctrine
(dev-loop scoping), not defaults: the battery runs kernel-check naked.

## Golden-token pinning (tier 1)

The battery's run-gen argmax gate re-derives its reference every run (prefill forward +
tokenwise decode + batched prime — three primes per invocation on a big model). fast-gate
pins the *output*: for each probe in
[`tools/fast-gate/models.tsv`](../tools/fast-gate/models.tsv), the greedy `tokens: [...]`
line at a battery-green commit is stored in `tools/fast-gate/goldens/<id>.tokens` with its
SHA and timestamp. A tier-1 probe then:

1. runs `run-gen` with `MEMRA_NGEN=20` (one model per affected kernel class),
2. requires the in-run gates green (prefill/decode argmax `MATCH`, no
   `MISMATCH-STRUCTURED` from the batched-prime gate; `FLIP-NEARTIE` stays reported,
   non-fatal, per the #46 contract),
3. byte-compares the `tokens:` line against the pinned golden — **any diverged token id is
   a FAIL**, with an instant verdict and no reference recompute.

Greedy decode is deterministic on this engine (run-to-run nondeterminism is itself a gated
bug class — see the ping-pong SSM-state fix in `cache.rs`), so token divergence == behavior
change. Exactness is clock-independent: goldens generated under any thermal/power regime
are bit-valid.

Spec-pipeline diffs add one single-K spec probe (`run-spec` + `MEMRA_SPEC_K` for the qwen
MTP family; `gemma-gate` + `MEMRA_SPEC` stream-agreement for the gemma drafter family).
The K=1..8 sweep stays tier-2.

`kind=cmd` probes (models.tsv) are self-gating commands — host unit tests or GPU oracle
gates like `sample-check` — whose gate is exit 0. They pin no golden and exist for code
the greedy token goldens structurally cannot see (the sampler chain). Three landed
2026-08-05: `chunkinv` (chunked-prefill byte-identity across `MEMRA_PRIME_CHUNK` values,
naked env — the grain-free default's contract; note its **coverage is per-architecture and
prompt-length-bounded** — the pinned probe prompts are short, which is precisely why they could
not reach the `step35` chunk-dependence defect that lives past a 512-token SWA window. The
long-prompt arm for that arch is `chunkinv35`/`chunkinv35c` (landed green with the fix,
`research/step35-chunkfix-20260807/`), and the axis one level up — serve splitting a prompt
across several `prime_cache` CALLS (per-tick budgets + the prefix-cache LCP split) — is
`tickinv35`/`tickinv35c` (`tools/tick-invariance-gate.sh`, landed with the request-level
`seq_end` fix, `research/tick-seg-20260807/`; its `--splits` arms pin the off-grid-resume hole,
vLLM #51113's second law). Standalone, the probe is
`tools/chunk-invariance-gate.sh [<model.gguf>] [--chunks 2048,64,32] [--steps 48]
[--expect-invariant|--expect-variant] [--canary]` — `--expect-variant` is how you assert a known
chunk-DEPENDENT arch stays detected rather than silently passing, and `--chunks` is how you widen
past the default triple), `chunkinvc` (its canary:
injects the
`MEMRA_PRIME_F32CHUNK0=1` legacy arithmetic and must FAIL, proving the gate detects the
mechanism), and `gwstress` (the graph-warmup pool-growth stress gate behind the
`MEMRA_GRAPH_WARMUPS=1` default). A fourth landed 2026-08-06: `sstress`
(`tools/serve-stress-gate.sh` — 64 staggered streaming clients, asserting every stream
completes well-formed with a live worker and no OOM lines; it is the *concurrency*
contract, which no exactness golden can see, and the regression proof for the admission
spec-headroom fix). Its own teeth: `--teeth` forces the admission reserve to 16 MB and the
verdict must invert. It also closed a map hole where `crates/memra-server/` diffs mapped to
no gate at all.

`isogap` / `isogapc` landed 2026-08-07 (`tools/iso-gap-gate.sh`, lane/iso-gap task #91): the
**staggered-depth serve isolation** contract at the engine tick — a session's logits must be
bit-identical solo-vs-coresident **across a `fa_split_keys` ladder-rung boundary**, the shape
both the equal-depth serve gate and the kernel-check seqs-vs-loop pin (whose depths all sit
inside one rung) were structurally blind to. The straddle is placed **per-rig** (`iso-gap-probe
--auto` scans the SM-keyed ladder: the 82-SM 5090's first boundary is t_kv=513, a 188-SM pod's
is 2049), so the arm has teeth on both rigs instead of straddling nothing off-rig. One run
covers same-rung batched (seqs arm), the straddling per-seq fallback window, and both
transitions. `isogapc` injects a wrong token into the co-resident arm's feed (changes the
world, not the label) and must be caught. Note: the serve-level solo-vs-loaded byte drift
(`spec-gate` REF/REF_LOAD) is **not** this class — it is the `b_n==1` fused-trunk↔batched-body
config flip at the co-residence boundary (`research/iso-gap-20260807/`); this arm pins the
within-config isolation that any fix for that flip relies on.

`amargin` / `amarginc` landed 2026-08-06 (`tools/argmax-margin-gate.sh`, + its `--canary` teeth):
run-gen's prefill-vs-decode argmax assert calibrated against the **top-2 margin at the deciding
position**, because a near-tie flip and a real cache bug are the same red until you measure the
margin (see `research/q8-argmax-20260806/`). Flags:
`tools/argmax-margin-gate.sh [<model.gguf>] [--prompt f] [--window N] [--max-flips N]
[--margin-floor F] [--canary] [--logdir D]`. **Effective `--window` default is 12**, not the 24
the probe binary advertises in its own usage line — the wrapper always passes its own
`WINDOW=12` (`argmax-margin-gate.sh:70,111`). Worth knowing, since window width *is* this gate's
coverage. And per the dispatch note above, `amargin` fires only when named with `--probes`.

Also 2026-08-06: `accept` (`tools/accept-gate.sh` — the **served-spec acceptance +
long-text** assertion). It exists because the battery was *provably* blind to a class:
`research/f8f4-flip-20260806/` receipted a kernel arm that moved served greedy text in 4 of 6
regime cells at temperature 0 and moved spec acceptance up to −9.5pp while **every gate above
stayed green in both arms**. Three structural reasons: the token goldens stop at 20 tokens and
both divergences landed at generated index 22 and 38; `--refresh-goldens` after such a change
would have silently re-pinned the new arm; and nothing compared accepted-draft *counts*, which
are spec throughput. `run-spec` self-consistency cannot see it either — it asserts spec == plain
*within one arm*, which both arms satisfy.

So `accept` asserts, at the **production serve config** (the artifact's real regime drafter
attached via `MEMRA_MODELS "+draft"`, its real serve K, driven through the server): exact
`(rounds, drafted, accepted)` integers — temperature 0 makes drafting deterministic, so these are
hard numbers, not a band — plus the full generated text sha256 to `ngen=128`, 6.4× past the
golden window. The drafter is load-bearing: the acceptance sign follows (model × drafter ×
prompt) and *inverted* between the GGUF's embedded MTP head and the regime drafter on the same
models the same day, so a bare-head number is not evidence about a served config. References
live in `tools/fast-gate/accept-refs/<cell>.{ref,text}`; cells in
`tools/fast-gate/accept-cells.tsv`. `--pin` is the only writer and **refuses on a dirty
`crates/`. There is deliberately no `--force`** — that is the gate's central law, not a default
you can override (`tools/accept-gate.sh:120` says so verbatim), because pinning references beside
an uncommitted kernel change is exactly the receipted failure mode: the new arm's numbers become
the reference and the gate then defends the regression (`research/f8f4-flip-20260806`). A dirty
tree OUTSIDE `crates/` is allowed and merely noted — engine code is what must be committed.
`--pin` has a **second** guard for the same trap wearing a different hat: it refuses when any of
`MEMRA_MMQ_F8F4`, `MEMRA_MMQ_F8F4_PLAIN`, `MEMRA_MMQ_FP8BLK_PLAIN`, `MEMRA_FAST`, or
`MEMRA_PRIME_F32CHUNK0` is set in the environment, since references must describe the NAKED
default build (flags doctrine: winners are defaults). Its teeth: `tools/accept-gate.sh --teeth` sets `MEMRA_MMQ_F8F4=1` and
the verdict must invert (proven both directions, `research/accept-gate-20260806/`); `--control`
re-measures in a second independent server boot, which is what licenses the single-shot read.

The `k27` argmax probe pins `MEMRA_FA_SPLIT=8` in its
env column so its golden is rig-portable across the 82-vs-188-SM `fa_split_keys` rung
(lane/k27-divergence — a near-tie flip class, not a defect; `k27div-probe` is the
cross-rig teacher-forced localizer).

### Golden refresh protocol

Goldens refresh **only at full-battery green points**, never mid-dev:

```bash
tools/local-ci.sh                                  # must be ALL GREEN first
tools/fast-gate/fast-gate.sh --refresh-goldens     # refuses a dirty tree (--force overrides, loudly)
git add tools/fast-gate/goldens && git commit ...  # goldens are checked in, SHA-stamped
```

If a legitimate behavior change moves tokens (new kernel numeric config promoted through the
full battery), the refresh happens in the same commit that lands the change *after* its
battery run — the golden diff in review is the visible record that tokens moved.

## Perf smoke (`--smoke`) — tripwire, not evidence

Each tier-1 probe already times its decode window; `--smoke` compares that **single rep**
against the tok/s recorded at the golden point (`goldens/<id>.perf`): WARN at >10% drop,
FAIL at >25%. This exists to catch catastrophic regressions (a kernel fell off its fast
path) inside the dev loop — it is explicitly **not** a publishable number and never moves a
board: publishable performance stays N≥5 interleaved same-session medians per
[`research/benchmarks.md`](../research/benchmarks.md), and drift detection at fine grain
stays `tools/local-ci.sh --perf`. A smoke WARN/FAIL means "re-measure with the real
protocol", nothing more.

## The probe-regime laws (learned by breaking kernels on purpose)

The catch demonstrations below exposed three ways a probe can be green while the touched
code is broken. The mapping table encodes the fixes; keep them in mind when adding rows:

1. **The probe must EXERCISE the touched dispatch class, not just the touched model
   family.** On a 24GB rig every daily MoE model loads RESIDENT — the SLRU cache, staged
   `moe_cached_gemm*`, and spill dispatch never run under a naked probe, and a deliberate
   gate/up weight swap there passed all four default probes. `q35slru`
   (`MEMRA_MOE_RESIDENT=0` + `MEMRA_MOE_SLOTS=1024`) forces that regime (68.5% hit rate,
   185k misses in its pin log) and caught the same break instantly.
2. **Depth is a dispatch axis.** The short probes decode at t_kv below/near the FA vec
   floor and windows; the gemma fp8-KV g-module arms (hd512 tb512 staging, windowed SWA)
   only execute at depth. `g12d` (the battery's 1736-id depth prompt) caught a K
   element-permutation break in the live tb512 staging arm that every short probe missed.
   (Its golden is 16 ids, not 20: token 17 of the g12 depth continuation is a real
   run-to-run near-tie flip — `g12-depth-nondeterminism.log` — and a 20-id golden
   false-fails ~1/8 runs. q9/q35/g31 deep continuations measured deterministic x5-x8.)
3. **Greedy goldens route around the sampler entirely.** temp=0 collapses to argmax, so a
   broken gumbel/softmax-gather kernel or a backwards top-k is invisible to every token
   golden. `samp` (the `sample-check` GPU oracle) and `sampt` (`cargo test -p
   memra-sampling`) are `kind=cmd` probes — self-gating commands, exit 0 = PASS, no golden.

### Catch demonstrations (all breaks reverted; diffs + consoles in receipts)

| Break (deliberate) | Caught by | Receipt |
|---|---|---|
| MoE staged dispatch: up-projection reads GATE weights (`moe_cached_gemm_q8`) | tier-1 `q35slru` (run-gen argmax gate, exit 101); plain `q35` was BLIND (resident regime) | `break-moe-staged-*` |
| FA v4 K-scale skew x1.001 (default hd256 staging arm) | tier-0 kernel-check synthetic arms — 46 bit-identity FAILs (`fa_decode_rows`/`seqs_v4`) | `break-fa-v4-*` |
| FA hd512 tb512 K element permutation (live gemma fp8 global arm) | tier-1 `g12d` depth probe (prefill/decode argmax MISMATCH, exit 101) | `break-fa-tb512-perm-*` |
| MMQ Q8_0 wrong-block scale (index mixup, `load_tiles_q8_0`) | tier-0 `MEMRA_KC_ONLY=q8mmq-gemm` — rel=2.4e-1 vs the f32 oracle, 8 FAILs | `break-mmq-q8-idx-*` |
| f16g IQ4_XS dequant off-by-one (`ls-31`) | tier-0 `MEMRA_KC_ONLY=f16g-kq-direct` — byte-identity maxdiff 1.15e2, 8 FAILs | `break-f16g-iq4xs-*` |
| device sampler acceptance-prob skew x1.001 (`softmax_gather_f32`) | tier-1 `samp` (sample-check vs CPU softmax, exit 1) | `break-sampling2-*` |
| host sampler top-k keeps WORST k (ascending sort) | tier-1 `sampt` (memra-sampling unit tests, exit 101) | `break-sampler-host-*` |

### Demonstrated coverage gaps (documented honestly, not closed)

- **Default-dead rollback seams**: kernels/helpers only reachable through non-default env
  seams are invisible to naked probes *by construction*. Verified twice: a `dq_K_lane`
  q8_0-branch lane swap (only live under `MEMRA_NO_FA_VEC`/v4-off arms at hd256, smem twin,
  `MEMRA_GEMMA_GKV=0` globals) passed everything, and so did an fp8 `dq_K_lane` lane swap
  (`break-fa-decode2-*`, `break-fa-fp8k-*`) and a kd requant skew (`break-fa-tb512-*` —
  int8 requant made the x1.001/126-vs-127 skews vanish at the __float2int_rn rounding,
  a magnitude-tolerant class, while the permutation break in the same loop was caught).
  Scale-skew breaks below the requant rounding step need value-exact oracles, not probes;
  the bit-identity kernel-check arms are the teeth there — extend those when adding arms.
- **A subtle *uniform* K-scale skew (x1.001) on an arm with no kernel-check bit-identity
  twin was caught by NO tier** (`break-fa-decode-*` — attention renormalizes softmax, so a
  uniform score scale barely moves greedy tokens at short depth). Wide-margin numeric skews
  are a tier-2/battery class; fast-gate's teeth are structural breaks (index/lane/element
  mixups), which it demonstrably catches.
- **Sampled serving path** (temp>0 end-to-end): `samp` oracles the kernels, but no probe
  runs a sampled generation stream; distribution-level drift stays a battery/eval concern.

## The perf stage's tok/s verdict is a tripwire, not evidence

`tools/local-ci.sh --perf` verdicts each cell against a **rolling median of that cell's prior
rows** — rows measured on earlier days. A tok/s FAIL there is therefore a *cross-day*
comparison, exactly the form [`research/benchmarks.md`](../research/benchmarks.md) forbids as
proof: clock, thermal and power state drift under numerator and denominator alike. It answers
"did something move?", never "did this commit regress?" — and it is **not** by itself a
merge/tag blocker.

When it goes red, settle it and record the settle:

1. build the last-green commit's binary for that cell,
2. run the cell **interleaved A/B/A/B, N≥5 each, in ONE thermal window under one exclusive
   lock hold** (harness: `research/v071-prep-20260806/battery-logs/perf-ab.sh`),
3. compare medians *within that window only*.

The v0.71.0 release battery is the worked example: 10/10 cells reported FAIL at −8.31% to
−24.75% with correctness fully green, and the interleaved A/B measured the **last-green
baseline binary at 37.87 tok/s against the candidate's 37.87 (+0.00%)** — the drop was machine
state, and no code had regressed. A uniform drop across many unrelated cells with correctness
green is that signature, not many simultaneous regressions.

Two holes in this stage were closed by that same red (2026-08-06):

- **The reps now run under `/tmp/gpu5090.lock`.** `window_free_now()` samples only *between*
  reps, so a neighbor lane that started and finished inside a rep was invisible — and its
  poisoned rows still recorded `window_clean:true`. Every other GPU consumer in the repo
  already took the lock; the one stage whose entire output is a timing number did not.
- **A tok/s FAIL now prints the settle protocol** instead of only a percentage, so the next
  reader does not have to re-derive why the number alone cannot convict a commit.

## What fast-gate does NOT cover

- **Serving surface** (`crates/memra-server/`): run `tools/serve-smoke.sh` (fast-gate prints
  the pointer when the diff touches it). Three more serving gates exist and are **not** wired
  into fast-gate or `local-ci.sh` — invoke them by hand for any diff in their area:
  - `tools/serve-st-gate.sh [st_dir]` — an HF **safetensors dir** served end-to-end: `/models`
    lists it, `/v1/chat/completions` returns coherent text through the checkpoint's *own* chat
    template, and the CLI-vs-server exactness contract (same checkpoint, same prompt, same
    template, greedy → `run-gen`'s ST-dir tokenwise branch and the server's batched-prime +
    serving decode must produce IDENTICAL id streams). Runs `MEMRA_SERVE_SPEC=0` because with
    spec on a Token event carries one id per flush.
  - `tools/apikeys-gate.sh [model] [out_dir]` — auth refusals (401/403), single-key
    back-compat, the two-tenant **cache-isolation proof** via a cache-hit oracle, per-tenant
    rate-limit headers, the batch-class lane law, and hot revoke.
  - `tools/serve-stress-gate.sh [--teeth] [model [draft [n_clients]]]` — the c=64 concurrency
    contract (this one *is* in local-ci.sh; `MEMRA_CI_STRESS=0` skips it).
- **Acceptance drift**: invisible to every *exactness* gate by construction (decode and verify
  shift together, so spec still equals plain). Two things catch it, and they are not
  interchangeable: the tier-2 perf battery's per-cell acceptance verdicts (a rolling-median
  tripwire), and `accept-gate` (an **exact-integer assertion** against a pinned reference at the
  production serve config, tier 2 + `kind=cmd`). Acceptance is a ratio and therefore
  clock-independent: an acceptance FAIL is real evidence, unlike a tok/s FAIL (below).
  fast-gate's tier-1 probes still do not see it — a spec-pipeline or NVFP4-prefill diff maps
  `accept`, but running it costs a server boot, so it lands at tier 2.
- **H100/sm_90a lane**: `tools/validate-h100.sh [--quick]` on an H100, per its own laws. Its
  contents are worth naming here, because three of them exist in no other battery: kernel-check
  with config pins, `decode-batch-gate --mode config` (B=8) and `--mode strict` (B=4 equalized),
  then the **graph lane** — `decode-dc-gate`, `graph-decode-gate`, `graph-session-gate`. The
  script's own header explains why they live there: `graph-decode-gate` "rotted OUTSIDE this
  battery for weeks" (an emission off-by-one in the gate masqueraded as 171/256 stream
  corruption), which is the origin of law 3 — anything guarding a live lane belongs inside the
  battery.
- **Cross-model blast radius**: tier 1 probes one model per kernel class; the full per-model
  matrix runs at tier 2.
- **Multi-GPU (PP-N) exactness**: needs 2+ cards, so it is neither in fast-gate nor in
  `local-ci.sh` — it runs on the multi-card box for any diff touching `pp.rs`, the stage-split
  dispatch, or a decode path a split reaches. See below.

## Multi-GPU (PP-N) exactness gates — run on the multi-card box

These are not in `tools/local-ci.sh`: the single owned rig has one GPU, so a green local
battery says nothing about them. Any change to `pp.rs`, the stage-split dispatch, or a decode
path a split walks needs these re-run on a 2+ card box.

| gate | invocation | what it proves |
|---|---|---|
| `ppn-gate` | `MEMRA_PP_DEVICES=0,1 ppn-gate <model.gguf> [stages=2] [P=16] [N=32]` | the eager stage-split decode (`decode_step_h_ppn`) is bit-identical to the unsplit walk, serial and pipelined arms |
| `decode-batch-gate --mode pp` | `decode-batch-gate <model.gguf> --mode pp [--batch 1,4,8] [--stages N] [--reps R]` | the **batched** stage split (`decode_step_batch_ppn`) is bit-identical per row per step. Honours `MEMRA_PP_DEVICES` / `MEMRA_PP_SPLITS` / `MEMRA_PP_SHARD` from the caller |
| `decode-batch-gate --mode ppspec` | `decode-batch-gate <model.gguf> --mode ppspec [--ts 2,5,9] [--stages N] [--reps R]` | the **spec-verify** stage split (`decode_step_t_core_ppn`, T=K+1) is bit-identical per logit column per round, plus the `h_seed` column the drafter is re-seeded from. `--ts 2,5,9` = K=1,4,8 |
| `pp2-gate` | `MEMRA_PP_DEVICES=0,1 pp2-gate <model.gguf> [P=16] [N=32] [split=n_layers/2]` | the N=2 spelling of the eager gate, with M1 binary semantics. It **owns the door** — it resets `MEMRA_PP_STAGES`/`SPLITS` itself regardless of the caller's environment, while the increment-2 knobs (`STREAMS`/`OVERLAP`/`DEVICES`) deliberately pass through. Keep it alongside `ppn-gate`: it is the gate the gemma4 N=2 arm is validated by |
| `pp-transport-smoke` | `pp-transport-smoke` | the peer boundary transport primitive alone |

Three properties of these gates are load-bearing and were learned by measurement:

1. **`--reps` defaults to 2 because the class they must catch was a 35% FLAKE.** The
   shared-`Engine` scratch race (`fa_part_pool` / `argmax_partials` / `fa_vf16_scratch` are
   stable-pointer pools, single-stream-safe by design) surfaced as an intermittent failure on
   2026-08-02. One green replay is not evidence of absence — always run reps.
2. **The door must open BEFORE load**, because weight sharding is a load-time decision. The
   gates set `MEMRA_PP_STAGES` themselves for exactly this reason; a battery that opened it
   after load would be measuring the wrong placement.
3. **Two arms make these localizers rather than coin flips.** The `unsplit@ppncache` arm
   replays the unsplit walk over the *same* stage-owned caches, holding cache placement
   constant so only the walk varies — a red split arm then points at the stage split and
   nothing else. The `epilogue` arm runs mixed per-row metas and checks the lean
   `last_logits_dev` park through UVA from the primary context, the same read the server's
   retire path does.

Arm 4 of the pp battery deserves its own note: the B=1 fast path's exactness bar is
bit-identity **to the eager split** (`decode_step_h`), not to the batched body — against which
it carries the accepted m=1 fusion FP gap by design. That is why arms 1-3 pin
`set_b1_fast(false)`, and why arm 4 also asserts per-step `pos` equality, so a double-advance
cannot hide behind matching logits.

Receipts: [`research/pp2-batch-20260806/`](../research/pp2-batch-20260806/),
[`research/pp2-spec-20260806/`](../research/pp2-spec-20260806/),
[`research/pp2-hardening-20260806/`](../research/pp2-hardening-20260806/) (the 20-arm
fail-closed guard battery), [`research/m2-pp8-20260802/`](../research/m2-pp8-20260802/) (N=2/4/8
on an 8xH100 box).

## Serve-path exactness — mode switches, and the law they are pinned under

`local-ci.sh` covers the serve surface's *shape* (`serve-smoke`, `serve-st-gate`,
`serve-stress-gate`, `apikeys-gate`). What it does not cover is a session **changing execution
mode mid-stream**, which the concurrency-gated spec scheduler does by design
(`MEMRA_SPEC_GATE`, default ON — see [SERVING.md](SERVING.md) and [FLAGS.md §1](FLAGS.md)): a
demoted session's stream must be byte-identical to one batched from the start. That harness is
`research/spec-gate-20260806/exactness.py` — 5 arms, one server boot per arm, greedy, 768-token
budget — and it is **not** wired into any battery. Re-run it by hand for any change to the
scheduler's phase order, the demotion handoff, or `Session.device_next`.

Two things about it are load-bearing, and both are the kind of thing a later reader re-breaks:

1. **Load-triggered demotion can never be a clean exactness test**, so the harness does not try.
   Both the arrival timing and the batch composition are nondeterministic, and — with the spec
   path OFF and none of the scheduler's code involved — the same greedy request already diverges
   between a solo run and one sharing batched decode with concurrent rows. Two runs put the first
   divergence at byte **2379** and byte **1347**; the byte *moving between runs* is the proof.
   That is the engine's own documented behavior (`fa_decode_batch_seqs_v4` carries a single
   `split_keys` for sessions at different depths — the ladder-rung straddle law — and the
   batched-linear tier selection changes with B), not a new bug. So the handoff is pinned at a
   **fixed batch shape** through a diagnostics-only door, `MEMRA_SPEC_DEMOTE_AT=N`, which forces
   demotion at a pinned generated-token count with no load at all, holding B=1 across the
   boundary. Never set it in production. The generalizable rule: **when the property under test
   sits inside a nondeterministic configuration, pin the configuration and force the transition
   — do not try to provoke it under load and diff the result.**
2. **Three of its own arms exist to stop a false green**, each recorded because it produced one:
   a *vacuous pass* (q9 is a thinking model — every token lands in `message.reasoning` and
   `content` is empty, so the first version compared 0 bytes on three arms and called it PASS;
   now it compares both fields and hard-fails a near-empty stream), a *wrong session* (load fired
   after the target had finished, so a background filler took the spec slot — the verdict now
   requires the demote line to prove it fired on the target), and a *wrong reference*, whose
   discriminator arm `REF_LOAD` is what surfaced the batch-vs-solo finding above.

## Receipts

Timings, the deliberate-break catch demonstrations (diffs, consoles, per-probe raw logs),
and the depth-determinism sweeps: [`research/fast-gate-20260802/`](../research/fast-gate-20260802/).
The serve-path mode-switch exactness harness and its verdicts:
[`research/spec-gate-20260806/`](../research/spec-gate-20260806/) (`RESULTS.md` §2, `exactness.py`).
