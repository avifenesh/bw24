# Pre-fix probe outputs (verbatim session captures, build = 3b98ca63 + probe binary only)

The `spec-st-probe` harness was added in this lane; these runs used it against the
UNFIXED spec.rs/lib.rs (capture-retain + retire-on-grow not yet applied). Verbatim:

## 4B ST dir, burst=32 k=3 graph (n=400)
```
  burst 1: 32 tok, acc 3/87        <- acceptance already collapsed (stale-address reads)
  burst 2: 32 tok, acc 4/84
  burst 3: 32 tok, acc 2/90
  burst 4: 32 tok, acc 6/78
DIVERGE at tok 107: ref=[256, 12193, 424, 4087, 279] sess=[95865, 198, 248069, 198, 25]
```

## 4B ST dir, burst=32 k=3 NOGRAPH (n=400)
```
MATCH (400 tok)         <- eliminates burst-boundary state; graph arm only
```

## per-burst divergence localization (n=200, graph)
```
  burst 1: 32 tok, acc 21/30
  burst 2: 35 tok, acc 24/33
  burst 3: 33 tok, acc 22/33
  burst 4: 32 tok, acc 7/75  <-- FIRST DIVERGENCE at burst-local tok 6 (global 106)
  burst 5: 32 tok, acc 6/78  <-- FIRST DIVERGENCE at burst-local tok 0
```

## MEMRA_SPEC_RECAP=1 diagnostic (drop parked graph, recapture per burst; n=400)
```
MATCH (400 tok)         <- confirms cross-burst graph persistence as the trigger
```

## 9B NVFP4 GGUF (NOT ST), burst=32 k=3 graph, n=1200 — the "ST-only" refutation
```
  burst 17: 34 tok, acc 25/27  <-- FIRST DIVERGENCE at burst-local tok 22 (global 553)
  ... (every later burst diverged at tok 0)
DIVERGE at tok 553: ref=[248044, 248045, 846, ...] sess=[248045, 846, 198, ...]
```

## After fix 1 (capture-retain) ONLY — still diverges once fa-pool grows
4B ST, ctx=8192 (worker floor), n=400:
```
  burst 9: 32 tok, acc 1/93  <-- FIRST DIVERGENCE at burst-local tok 2 (global 266)
DIVERGE at tok 266: ref=[314, 9338, 369, 11751, 1149] sess=[1, 314, 11316, 314, 1]
```

## MEMRA_DEBUG_FAPOOL=1 trace on that build — the smoking gun
```
  burst 7: 34 tok, acc 22/36
[fa-pool] REALLOC o 524288 -> 540672 ml 2048 -> 2112     <- trunk t_kv crosses split key
[fa-pool] REALLOC o 540672 -> 557056 ml 2112 -> 2176
[fa-pool] REALLOC o 557056 -> 573440 ml 2176 -> 2240
[fa-pool] REALLOC o 573440 -> 589824 ml 2240 -> 2304
  burst 8: 32 tok, acc 16/48
[fa-pool] REALLOC o 589824 -> 606208 ml 2304 -> 2368
...
  burst 9: 32 tok, acc 1/93  <-- FIRST DIVERGENCE (the burst after the growth run began)
```
Old pool buffers freed on grow -> async pool re-hands their addresses to live buffers ->
the persisted draft graph's fa_decode_dc node writes partials over them on every replay.

## Pre-fix serve arms (memra-server @3b98ca63, 4B ST): see pre-fix-serve-4b-*.txt
plain-vs-specgraph: DIVERGE (corrupt tail: `is = is is is ... ::::: [ (` — 1573 vs 1083+ chars)
plain-vs-burst64:   DIVERGE from byte 1
plain-vs-K=2:       DIVERGE at byte 1304
plain-vs-K=1:       MATCH (single-K windows stayed under the grow boundary)
plain-vs-burst200/400: MATCH (fewer bursts => graph captured late/never grew past)
Determinism: every arm rep1==rep2 byte-identical across server restarts (deterministic
address-reuse, not a race).
