# v0.71.0 release notes — DRAFT (prep-lane, 2026-08-06, lane/v071-prep @ train a85135ae)

Raw generator output: `changelog-raw.txt` (`bash tools/changelog.sh v0.70.0` on this lane).
If more lanes merge before the tag, re-run the generator on the merged tree and fold —
draft is floor, not ceiling. **docs(biz) leak-check: CLEAN** — zero product-layer commits
in the v0.70.0..HEAD public prefixes (grep receipt in the prep-lane transcript); nothing
to delete from the workflow draft this time.

Version argument: MINOR. v0.71 flips three defaults (grain-free chunk invariance,
block-128 FP8 native residency, `MEMRA_GRAPH_WARMUPS=1`), changes the felt-latency
serving behavior (round-cadence SSE + admission yield), and removes a documented flag
door. Not patch-class.

**The headline five:** chunk-invariance by default (one canonical greedy output per
prompt) · the felt-latency arc (solo first text 0.41 → 0.12 s, contended 1.60 → 0.15 s)
· block-128 FP8 native residency by default (3.8 day-one, no flags) · graph warmups
default 1 (recapture −41%) · three new adversarial gates in the battery.

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

### Fixes

- 64-bit `offset_dst` in all 11 vendored MMQ launchers + quantizer thread-id widening
  (audit Q7) — kernel-check ALL GREEN, zero bit change.

### Gates

Three new battery arms, all wired into `tools/local-ci.sh` / fast-gate (gates outside the
battery rot silently — the H100-lane law):

- `chunkinv` / `chunkinvc` — chunked-prefill byte-identity across chunk sizes, naked env,
  plus the canary arm that injects the legacy arithmetic and must FAIL.
- `gwstress` — the graph-warmup pool-growth adversarial gate behind the
  `MEMRA_GRAPH_WARMUPS=1` default.
- the k27 fast-gate row pins `MEMRA_FA_SPLIT=8` so its golden is rig-portable.

### Configuration

- The `MEMRA_PRIME_INVARIANT` / `MEMRA_PRIME_GRAIN` door is REMOVED (superseded by the
  grain-free fix; the research record keeps its history). `MEMRA_PRIME_CHUNK` is a pure
  memory/transient knob again.

### Documentation

- SERVING.md: the felt-TTFT arc section (both fixes + the B32/B128 owner call), the
  chunked-prefill section corrected to its FIXED state.
- PERFORMANCE.md: serve-board TTFT caveat (rows predate the felt-latency fixes), the
  frozen head-to-head amendment (memra-side latency stack moved; the 0.53-vs-0.19 s row
  stays frozen — competitor benching remains stopped), and the locked-clock law (locked
  and free-clock numbers never mix in one comparison).
- README: serving paragraph carries the felt-latency arc; the exactness claim now
  includes chunk-size-invariance; the known-gaps TTFT bullet updated honestly.
- TESTING.md / CONTRIBUTING.md battery lists name the new gates.

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full
experiment log in research/tune-data/
