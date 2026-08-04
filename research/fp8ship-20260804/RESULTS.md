# #68 root cause — ST serve-spec divergence (fp8-ship lane, 2026-08-04)

Branch `lane/fp8-ship` off `restructure/public-split` @ 3b98ca63. Rig: local RTX 5090
laptop, GPU work under `flock /tmp/gpu5090.lock`.

## Verdict

**ROOT CAUSE: the per-session persistent draft graph (2026-08-01) replays with DANGLING
POOL ADDRESSES.** Two compounding bugs, both fixed:

1. **`capture_graph` (non-retained) freed the capture-body transients.** The captured
   draft graph bakes the pool addresses of every transient the capture body allocated
   (e_norm/h_norm/concat/matmul outputs...). `Engine::capture_graph` drops them at exit —
   fine for one-shot `generate_spec` (pool layout never shifts between replays inside one
   call), WRONG once the graph persists across bursts on the SpecSession: burst-boundary
   work (prime, mtp_kv_fill, commit passes) reallocates those addresses and replay
   read/writes tear both sides.
   FIX: capture with `capture_graph_retained` and carry the keeper in `DraftGraphCtx`
   (`keeper`/`keeper_s`), exactly the gemma_spec/decode.rs/prime_graph pattern. The
   2026-07-13 keeper comment already called this "the draft-graph root cause" — the
   2026-08-01 persistence change reintroduced it by extending graph lifetime without
   extending keeper lifetime (one-shot calls kept a live borrow chain alive; sessions
   did not).

2. **`fa_part_pool` grow FREED the old buffers the captured graph baked.** The pooled
   fa-decode split partials (part_o/part_m/part_l) lazily GROW as trunk t_kv crosses
   split boundaries. The old (smaller) buffers were dropped on grow — but the persistent
   draft graph's `fa_decode_dc` node holds their addresses. After the drop, the async
   pool hands those addresses to live allocations (KV views, logits, hiddens) and every
   subsequent draft-graph replay memsets + writes fa partials over them.
   FIX: RETIRE-on-grow — old generations move to `Engine::fa_part_retired` (kept for the
   Engine's lifetime), growth doubles so total retired VRAM < final size. All 7 realloc
   sites patched.

## Why the receipts looked the way they did

- **run-spec CLI passed K=1..8 on the same checkpoints**: one `generate_spec` call = the
  graph is captured and replayed inside a single call; nothing reshuffles the pool between
  replays, and the K=1..8 loop recaptures per call. Both bugs need a PERSISTED graph plus
  pool churn between bursts — only the worker's session-burst pattern produced that.
- **"ST-only" was a red herring**: with a session-burst harness (`spec-st-probe`, added
  this lane) the SAME corruption reproduces on the 9B NVFP4 **GGUF** at n≥600
  (`probe-9bgguf-n1200-pre-fix.log`: acceptance pins at 25/27 while output goes garbage
  from tok 553). The 4B ST checkpoint merely hit the fa-pool grow boundary sooner (bf16
  ST → Q8_0 re-encode has different transient sizing). GGUF serve-spec "MATCH" receipts
  from 2026-08-03 were short (400-token) windows on the 9B — under the corruption onset.
- **acceptance collapse fingerprint**: pre-fix 4B server logs show cum acceptance decaying
  0.70 → 0.45 across bursts as replays read stale/garbage seed hiddens, then output
  corrupts outright once verified tokens themselves land on clobbered partials.

## Elimination table (all receipted in this dir)

| hypothesis | receipt | verdict |
|---|---|---|
| worker prime path vs CLI prime | probe uses engine-only session bursts, no server — still corrupts (`probe-4b-burst32-pre-fix.log`) | ELIMINATED |
| dtype/layout of dir-loaded draft weights | GGUF repro at n=1200 (`probe-9bgguf-n1200-pre-fix.log`) | ELIMINATED (not ST-specific) |
| burst-boundary state machine (pending-carry/next_pred) | `MEMRA_SPEC_NOGRAPH=1` session bursts MATCH 400/400 (`probe-4b-nograph.log`) | ELIMINATED |
| graph persistence across bursts | per-burst recapture diagnostic arm → MATCH; parked graph → DIVERGE | CONFIRMED trigger |
| capture-retain alone sufficient | fix 1 alone: still diverged once fa-pool grew (probe ctx=8192 run) | INSUFFICIENT — led to fix 2 |
| fa_part_pool grow-free during replays | `MEMRA_DEBUG_FAPOOL=1` trace: divergence burst == first burst after the o≥524288 grow (`probe-4b-fapool-trace.log`) | CONFIRMED (smoking gun) |

## Post-fix verification (this dir, all on the fixed build)

- probe 4B ST, ctx=8192 (worker floor), burst=32, n=400: **MATCH** (was DIVERGE at 266)
- probe 4B ST n=1200: **MATCH**; probe 9B ST n=1200: **MATCH**
- probe 9B NVFP4 GGUF n=1200: **MATCH** (was DIVERGE at 553)
- run-spec K=1..8 self-consistency: **PASS** on 4B ST dir, 9B ST dir, 9B NVFP4 GGUF
- memra-server 4B ST: spec-vs-plain text IDENTICAL on default K3/graph, K=2, burst=64,
  K=1 nograph (all four previously-diverging arms)
- memra-server 9B ST: spec text == the run-gen CLI tokenwise oracle for the full 400-token
  window (longest-common-prefix 1518/1523 chars; remainder = the CLI window ending).
  NOTE: the PLAIN serve arm ("questions" @ char 551) is the outlier vs CLI ("queries"),
  not the spec arm — a near-tie logits-delta in the batched-prefill+batched-decode plain
  path, a DIFFERENT (pre-existing, accepted FP-composition) class, not #68. Receipt:
  `9b-plain-vs-spec-vs-cli.txt`.

## The 9B "near-tie flip at K=1 nograph" from 2026-08-03 — reclassified

The serve-side spec arm reproduces the CLI tokenwise decode exactly. The plain serve arm
(prime_cache batched prefill + decode_step_batch) sits on the other side of a near-tie at
char 551. Spec-vs-plain equality on that checkpoint is therefore gated by the known
decode-config near-tie class (decode-batch-gate's calibrated dice), not by spec-session
state. The serve-st gate's spec arm should compare serve-spec to the CLI ORACLE (run-gen),
which is the exactness contract spec actually guarantees.
