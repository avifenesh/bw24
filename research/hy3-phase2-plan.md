# Hy3-REAP50 phase 2 — spill-tier tuning plan

Phase 1 (2026-07-09, rig5090.jsonl rows): first-light forward ALL GATES GREEN, 0.15 tok/s plain
decode. 81.5 GB checkpoint against 19.2 GB VRAM in use; SLRU hit-rate warms 39.9→56.9% over four
runs. Diagnosis: expert spill dominates, and the middle tier is underused — host RAM sat at
~23/60 GB while experts streamed from NVMe.

Goal: maximize tokens/s by making the tier pyramid honest — VRAM holds the hot set, RAM holds the
warm majority, NVMe only the cold tail. REAP50 was chosen exactly for this shape (majority
resident, real NVMe minority exercising the spill machinery).

## Levers, in measurement order

1. **RAM-tier budget** (biggest untouched lever). ~30 GB of MemAvailable unused during phase-1
   runs. Knobs: `BW24_SPILL_PINNED_FRAC` (default 0.60 of MemAvailable), `BW24_MOE_RESIDENT_GB`.
   Sweep pinned-frac {0.60, 0.75, 0.85} with everything else fixed; watch `free -g` and page-cache
   pressure (the ST_PINNED lesson: pinning past the page cache's working set regressed 30x on the
   M3 — the sweep must include a cache-thrash check, cold-run after each arm).
2. **Expert-cache VRAM fraction**: `BW24_MOE_VRAM_FRAC` default 0.85 was tuned on the 35B
   (fits-RAM regime). Hy3's non-expert working set differs (80 layers, shared MLP always hot);
   re-sweep {0.70, 0.85, 0.92} — more VRAM slots only pay if hit-rate climbs past what RAM-tier
   latency already covers.
3. **Routing locality → prewarm policy**: capture `BW24_MOE_TRACE` on the fixed prompt set,
   run `research/scripts/moe_trace_analyze.py`. Two questions: (a) per-layer expert reuse run
   length (does SLRU depth match the reuse horizon — `BW24_MOE_SLOTS` override if not);
   (b) cross-token co-activation clusters (prewarm the cluster on first member hit, extending
   `BW24_MOE_PREWARM` from layer-granularity to cluster-granularity — code change, only if (a)
   shows the miss tail is clustered).
4. **NVMe read path**: phase-1 streams via mmap page faults on the `.bw24-repack` cache. Measure
   effective NVMe throughput during decode (iostat) vs the drive's sequential ceiling. If well
   under: batch expert fetches (the CSR batching already groups per-layer — extend the disk tier
   to issue one readahead per layer's miss-set instead of faulting per expert).
5. **Shared-path residency audit**: layer-0 dense + per-layer shared MLP + router/bias tensors
   must never occupy SLRU slots. Verify via `BW24_MOE_STATS` that misses are expert-only.

Gate per arm: argmax MATCH ×3 prompts + determinism pair, per phase-1 protocol (sigmoid-router
near-tie logit diffs are expected; token identity is the contract).

Exit criteria: hit-rate curve saturated (>90% steady-state) OR NVMe at sequential ceiling —
whichever wall is real decides whether phase 3 is worth it (MTP head is transcoded and waiting;
spec only pays once base decode is out of the 0.1x regime).
