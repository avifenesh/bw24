# v0.71.0 release notes — FINAL (tag day, 2026-08-06, train @ 4cbf5e39)

Re-cut from the merged tree: `bash tools/changelog.sh v0.70.0` → `changelog-raw-final.txt`.
Folded onto `CHANGELOG-DRAFT.md` (prep, @ a85135ae) — the draft was floor, not ceiling.
**Delta since prep:** three admit-oom fix commits (the c=64 red), the serving-density
receipts (prefix-sharing dead verdict + the max_tokens config finding), the fp4-act-scoping
door-shut brief, and the prefill/v3 closures.
**docs(biz) leak-check: CLEAN** — zero product-layer commits in the v0.70.0..HEAD public
prefixes (re-grepped on the merged tree; the only `darklane` string is inside the runbook's
own sync-note description, not a product-layer change).

Version argument: MINOR. v0.71 flips three defaults (grain-free chunk invariance,
block-128 FP8 native residency, `MEMRA_GRAPH_WARMUPS=1`), changes the felt-latency serving
behavior (round-cadence SSE + admission yield), changes *admission* behavior (spec headroom
reserve + step-OOM park), and removes a documented flag door. Not patch-class.

**The headline five:** chunk-invariance by default (one canonical greedy output per prompt)
· the felt-latency arc (solo first text 0.41 → 0.12 s, contended 1.60 → 0.15 s) · 64
concurrent speculative clients survive a 24GB card, gated (was 0/64) · block-128 FP8 native
residency by default (3.8 day-one, no flags) · graph warmups default 1 (recapture −41%).

---

Changes since v0.70.0:

### Exactness contract

- **Chunked prefill is chunk-size-invariant by default** — the grain-free fix drops the
  `base_len == 0` f32 special case, so chunk 0 attends the quantized KV cache exactly
  like every later chunk (quantize-then-attend, one numeric class for every row).
  Prefill logits are bit-identical across `MEMRA_PRIME_CHUNK` values naked, reclaiming
  "one canonical greedy output per prompt" as the shipping contract. Quality: NLL never
  worse, 27B improves 1.1% (0.8407 vs 0.8504); the contract change vs the old f32
  first-chunk arithmetic is a quantified near-tie class (11/1024 + 16/1024 teacher-forced
  flips). Perf-free (+0.10%/+0.00%, N=5). `MEMRA_PRIME_F32CHUNK0=1` is the rollback seam
  and the gate canary's injection; the interim `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN`
  pin-the-boundary door is superseded and REMOVED this release per the flags doctrine.
- **The k27 cross-rig divergence is named and closed as FLIP-NEARTIE**: the
  `fa_split_keys` SM rung picks split 8 on 82 SM vs 16 on ≥128 SM — a legal near-tie flip
  at exactly the FA vec floor, not a numeric defect (at matched split, cross-rig
  teacher-forced logits are byte-identical to every digit). Contract wording: one
  canonical output per FA-split config. The k27 fast-gate row pins `MEMRA_FA_SPLIT=8`;
  `k27div-probe` is the new cross-rig teacher-forced localizer.

### Performance

- **Block-128 FP8 native residency is now the default** (`MEMRA_ST_E4M3_BLK`, on) — the
  class Qwen3.6-FP8 actually ships (`weight_block_size [128,128]`; 208 projections /
  6880 MiB on the 27B). Decode +1.69% via `qmatvec_e4m3_blk_mmvq` (per-block-128 scale
  lookup, host-reference-gated over all 254 e4m3 codes), prefill +0.83% once the class
  routes through the per-block MMQ tile (`MEMRA_FP8_MMQ`, on for the native-resident
  source), and 430 MiB freed at single residency. NaN/ragged checkpoints fail SAFE to the
  dequant arm. This completes the FP8-ST program: loader, exact floor, native per-tensor,
  native block-128, spec-on-ST — a block-128 checkpoint now needs NO flags.
- **`MEMRA_GRAPH_WARMUPS` default 2 → 1** — recapture −41.4% q27 / −39.2% q9, decode
  +1.07/+1.11%, capture+prime −13 ms, SM-count-invariant (pod re-mint agrees). The flip
  was earned by an adversarial stress gate, not just the win: pool-growth cycles both
  directions x10 + forced mid-stream recaptures + a two-live-graphs overlap arm, with
  per-token bit-identity vs eager as arbiter and a canary proving the comparator's teeth.
  `MEMRA_GRAPH_WARMUPS=2` is the rollback seam.
- **MoE SLRU `on_hit` goes O(1)** — intrusive doubly-linked segments (9 B/slot arena)
  replace the per-hit VecDeque scan; policy matched exactly against a verbatim
  transcription of the old code (200k-op randomized soak: full segment-order equality
  every op, victim identity every eviction). Spill-regime e2e: +3.7% decode on a forced
  15k-slot 35B run, hit/miss counters byte-identical.
- **PrefixCache eviction goes O(log E)** — recency-index timestamp-LRU, policy-identical
  (same victims, same pattern), with equal-Instant ties strictly determinized.

### Serving

- **Round-cadence SSE streaming** — spec-burst sessions flush text at every spec-round
  commit instead of once per burst (same detokenize-tail/utf8-cursor/EOS rules; content
  byte-identical, only chunk boundaries move). Solo first text 0.41 → **0.12 s** and
  inter-chunk gap p50 299 → 27 ms at ANY burst size. `MEMRA_SSE_PER_BURST=1` rollback.
- **Admission yield + cold-first ordering** — a request arriving mid-burst ends the
  in-flight spec burst at the next round boundary, and not-yet-emitted sessions burst
  before mid-generation peers. Contended first text 1.60 → **0.152 s** at B128
  (0.54 → 0.123 s at B32) — the solo class at any burst size, where it used to scale with
  `MEMRA_SPEC_BURST`. Content byte-identical on/off, solo AND contended. The measured
  cost lives at c=8 saturation only: −3.4% agg tok/s for 3.8x better p50.
  `MEMRA_ADMIT_YIELD=0` rollback (both pieces).
- **Loud, resettable graph fallbacks** — a session silently dropping off CUDA-graph
  decode now warns exactly once per flip and resets on pool resume; real graph-step
  failures surface as stream errors instead of truncating generation silently.
- With both felt-latency fixes in, `MEMRA_SPEC_BURST=128` is now a documented
  throughput-tier setting (+8.4% c=1 / +8.5% c=8) one 29 ms cadence-quantum behind the
  B32 default on contended first text — an owner call, no longer a cliff. Default holds
  at 32.
- **Ops note: send an explicit `max_tokens`.** Admission sizes each session's KV ladder
  from the request's own bound; omitting it falls back to the context ceiling, stranding a
  measured 6.3% (c=16) / 12.6% (c=32) of a 96GB card in ladder slack at
  `MEMRA_CTX=32768`. Right-sized requests strand ~0%. Recommended in serve configs and
  client defaults (docs/SERVING.md).

### Fixes

- **64 concurrent speculative clients no longer OOM a 24GB card** — at
  `MEMRA_MAX_SESSIONS=64` with spec ON, admission's cost model under-charged the live
  burst and **all 64 streams died** with a quoted
  `step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` (0/64 well-formed x3; the worker
  itself survived, so it was never a hang or a panic). Three separate errors, all fixed:
  (1) the parked-session delta understated the live cost 1.49x *and* a ~1.3 GiB
  draft-graph capture arena is constant, not per-session, so no headroom multiple could
  cover it — admission now charges a flat `SPEC_SHRINK_RESERVE` on **spec-capable models
  only** (the plain path is untolled and passed c=64 unaided); (2) retires returned KV to
  the pinned async pool where driver `free` cannot see it, so the gate now reads
  `free + pool_cached` (deferrals 36 → 5, 59 sessions active sustained); (3) a spec step
  that OOMs despite admission now rebuilds and re-queues at the FRONT
  (`MEMRA_STEP_OOM_RETRIES`, default 3) instead of erroring the stream — bounded, and only
  for a session that has emitted nothing, only on a quoted CUDA OOM, so a streamed prefix
  is never replayed. **Result: 64/64 x3, peak 23.1 of 24.5 GB.** The c=8 no-regression
  control is behaviorally identical (+0.49% agg, zero defer/park events).
  `MEMRA_ADMIT_RESERVE_MB` is the teeth/diagnostics door, not a tuning knob.
- 64-bit `offset_dst` in all 11 vendored MMQ launchers + quantizer thread-id widening
  (audit Q7) — kernel-check ALL GREEN, zero bit change.

### Gates

Four new battery arms, all wired into `tools/local-ci.sh` / fast-gate (gates outside the
battery rot silently — the H100-lane law):

- `chunkinv` / `chunkinvc` — chunked-prefill byte-identity across chunk sizes, naked env,
  plus the canary arm that injects the legacy arithmetic and must FAIL.
- `gwstress` — the graph-warmup pool-growth adversarial gate behind the
  `MEMRA_GRAPH_WARMUPS=1` default.
- `sstress` (`tools/serve-stress-gate.sh`) — 64 staggered streaming clients, asserting all
  complete + well-formed + worker alive + no OOM lines. Has teeth: `--teeth` forces the
  admission reserve to 16 MB and the verdict must invert (it does: 11/64). A CI hole where
  `memra-server` mapped to no gate is closed alongside it.
- the k27 fast-gate row pins `MEMRA_FA_SPLIT=8` so its golden is rig-portable.

Three gate bugs were found *by this release's own battery* and fixed in it:

- **fast-gate reported a FALSE GREEN on any clean tree.** The empty-diff early exit ran
  before `--probes` was applied, so an explicitly-named plan on a clean tree (a release
  candidate, a fresh rsync onto another rig, a tree with no `.git` at all) printed
  "nothing to gate" and exited 0 having run **zero** probes. Caught on the pod battery,
  where the runbook's named k27 regression check "passed" without executing. Only the
  diff-driven path may short-circuit now.
- **The perf stage ran its reps without the GPU lock.** `window_free_now()` samples only
  between reps, so a neighbor lane that ran inside a rep was invisible and its poisoned rows
  still recorded `window_clean:true`. The reps now hold `/tmp/gpu5090.lock`.
- **A perf tok/s FAIL now states what it is.** It verdicts against a cross-day rolling
  median, which the measurement law forbids as proof, so it is a drift tripwire — the red
  now prints the interleaved-A/B settle protocol instead of an unqualified percentage.

### Configuration

- The `MEMRA_PRIME_INVARIANT` / `MEMRA_PRIME_GRAIN` door is REMOVED (superseded by the
  grain-free fix; the research record keeps its history). `MEMRA_PRIME_CHUNK` is a pure
  memory/transient knob again.

### Documentation

- SERVING.md: the felt-TTFT arc section (both fixes + the B32/B128 owner call), the
  64-client robustness story under Admission, the `max_tokens` config recommendation, and
  the chunked-prefill section corrected to its FIXED state.
- PERFORMANCE.md: serve-board TTFT caveat (rows predate the felt-latency fixes), the
  frozen head-to-head amendment (memra-side latency stack moved; the 0.53-vs-0.19 s row
  stays frozen — competitor benching remains stopped), and the locked-clock law (locked
  and free-clock numbers never mix in one comparison).
- README: serving paragraph carries the felt-latency arc and the gated 64-client property;
  the exactness claim now includes chunk-size-invariance; the known-gaps TTFT bullet
  updated honestly.
- TESTING.md / CONTRIBUTING.md battery lists name the new gates.
- FLAGS.md: `MEMRA_STEP_OOM_RETRIES` and `MEMRA_ADMIT_RESERVE_MB` cataloged with their
  re-verified teeth; `MEMRA_FA_SPLIT` documents the SM rung + FLIP-NEARTIE verdict.

### Research closures (no shipped code)

- **Sealed-prefix sharing: RECEIPTED-DEAD at the agent-trace shape.** Prefix duplication
  measures 0.85–7.69% of a 96GB card at c=16/32 with 4–8k shared prefixes — below the 10%
  bar. Revive at ~22k+ sealed prefixes (a RAG/repo-context shape, not the coding-agent
  trace). The larger stranding turned out to be config, not duplication — see the
  `max_tokens` note above.
- **FP4 activations (W4A4): door KEEP SHUT, decision brief receipted.** The prior rescue
  attempt's correction overfit its tuning corpus (5/5 → 4/10 widened, adversarial fork at
  8/9 k-depths); the best published W4A4 method still costs +0.4 PPL / 2.9 pt zero-shot on
  7B and *no* rotation method publishes runtime latency, so the nominal 2.4x ceiling
  plausibly collapses to 0.9–1.8x net. W4A8 is flagged as the pragmatic alternative (rides
  the FP8-ST infra). Resurrection bar written down.
- The PREFILL-GEMM-REBUILD plan is marked superseded — its target kernel does not run.

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full
experiment log in research/tune-data/

---

## Release battery — both rigs (2026-08-06)

**Local 5090 (`eea5a9ed`) — correctness ALL GREEN.** kernel-check GREEN; prime-gate 8/8 MATCH
(0 FLIP-NEARTIE, 0 STRUCTURED, 0 det_fails); run-gen argmax MATCH 31B + 12B depth;
VERIFY-GATE K=7 PASS both; spec self-consistency 64/64 PASS (31B); decode-batch config B=8 +
strict B=4 equalized ALL GREEN on 9B NVFP4 *and* 9B Q8_0; graph-warmup stress 10 cycles x 4
arms + overlap bit-identical; serve-smoke 0 failed; **serve-stress 64/64 complete** (wall
p50 53.2 s, ttfb p50 0.89 s). Explicit v0.71 fast-gate arms: chunkinv PASS, chunkinvc PASS,
gwstress PASS, k27 PASS (golden token-identical at the pinned split) — 0 fail. serve-st-gate
0 failed, apikeys-gate 18/18. `run-spec` K=1..8 on the NVFP4-MTP arm: **8/8
SELF-CONSISTENCY PASS**, every K exit 0.

**Pod, 188 SM (`/root/bw24-v071`) — GREEN, no lane disturbed.** kernel-check ALL GREEN;
run-gen argmax MATCH (27B NVFP4-MTP); run-spec K=1..3 PASS (acceptance 80%); k27 PASS at its
pinned `MEMRA_FA_SPLIT=8` — the regression check the runbook names, genuinely executed after
the fast-gate false-green fix; serve-smoke 0 failed. The 122B lane's processes were left
untouched throughout (every arm gated on a free-GPU poll).

### The two reds, and how each was settled

**RED 1 — pod 27B Q8_0 run-gen MISMATCH: pre-existing, NOT a v0.71 regression.** One prompt
(`board-2048`) on one artifact pair reports `prefill argmax=332 decode argmax=485 logit
maxdiff=4.659e-1 MISMATCH` (panic at `run_gen.rs:896`, exit 101). Isolation: 7 of 8 prompts
MATCH; 27B NVFP4-MTP and 9B-Q8_0 both MATCH; reproduces on a fully clean GPU 3/3 (not
contention); FA_SPLIT-independent (8/16/1 identical). Decisive arm — the same prompt and
artifact against four binaries in one window:

```
v071-again : prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
v070-q1    : prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
v069-bw24  : prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
ctrl-0804  : prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
```

Byte-identical on **v0.70.0 and v0.69.0**, both tagged and shipped — the defect predates this
release and is not created by it. The standing battery form on this pair is green and stays
green (`MEMRA_NGEN=128 MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt` → `prefill
argmax=198 decode argmax=198 MATCH` + batched-prime `argmax=198` MATCH), which is why no gate
saw it: the local rig holds no 27B Q8_0 artifact, so local-ci never exercises the pair. Filed
as a standing investigation, not a v0.71 blocker. Receipt:
`battery-logs/pod-q8-regression-or-not.log`.

**RED 2 — local perf 10/10 cells FAIL: machine state, zero code regression.** Reported −8.31%
to −24.75% against the rolling medians with correctness fully green. Ruled out in order:
not the new sstress heat-soak and not thermal (a 300 s-cooldown re-run with the stress/serve
arms disabled reproduced the numbers to within 0.2% — 37.90 vs 37.98), full power confirmed
applied (`nv_dynamic_boost=25/25`, `nv_temp_target=87/87`, profile `performance`, power limit
175 W). Two invalidating facts about the verdict itself: the last green row is `fcfe3837`,
which **predates every v0.71 default flip**, and the comparison is cross-day, which the
measurement law forbids as proof. Settled the only legal way — the last-green binary rebuilt
and run interleaved against the candidate, N=5 each, one thermal window, one exclusive lock
hold:

```
A fcfe3837 median: 37.87 tok/s   [37.94 37.87 37.85 37.86 37.93]
B eea5a9ed median: 37.87 tok/s   [37.87 37.85 37.86 37.93 37.93]
B vs A: +0.00%   => NO code regression
```

The last-green baseline binary reads the same 37.87 as the candidate on the same card in the
same window: the shipped code is exonerated and the rolling median is the invalid side of the
comparison. A concurrent lane (`research/rp-on-st-20260806`) held `run-gen` on this card from
07:27Z, overlapping both perf runs — which is how the unlocked-reps hole above was found.
Board numbers are untouched by this (no tracked cell moved; `update-perf-board.py --check`
green). Receipts: `battery-logs/perf-ab-31b-interleaved.log`, `battery-logs/perf-ab.sh`.
