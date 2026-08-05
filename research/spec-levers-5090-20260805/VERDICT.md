# Spec-lever validation on the local 5090 — verdict (2026-08-05)

Re-mint of the pod's phase-2 spec-serve levers (research/q27-deepdive-p2-20260805: nv +14.0%,
q8 +20.0% via K + BURST=128 + PMIN=0.3+PMIN0) on the deployment target: RTX 5090 Laptop,
**82 SM**, 24GB, ~150W sustained. Tree 759083cc, built locally. GPU shared with three other
lanes via `flock /tmp/gpu5090.lock`; every round = one hold with all arms interleaved inside
it (same thermal window), order alternated per round. All medians state N; thermal 80-85C
sustained across holds (laptop board — relative deltas are the evidence, like the pod's).

## Headline: the pod stack does NOT transfer as-is — only the burst lever survives 82 SM

| lever | pod (188 SM) | 5090 (82 SM) | verdict |
|---|---|---|---|
| serve K | K=5 optimum, any burst | **K=3 optimum** (K=5: −4.9% at B32, −1.9% at B128) | NO FLIP — shipped K=3 default confirmed |
| BURST 32→128 | +7.5% c=1, +7.6% c=8 | **+5.4–7.0% c=1, +6.9% c=8, +9.2% q9** | WINS throughput everywhere, but costs streaming cadence — context-split, default stays 32 |
| PMIN 0.3+PMIN0 | +6.1% on top (at K=5) | **flat-to-negative at K=3** (−0.6%), acceptance +4–15pp reproduces | NO FLIP — the pod's pmin win was a K=5-chain effect |

The stacked pod recipe (K5/B128/pmin) on the 5090: 90.3 tok/s = **+5.6%** vs the shipped
default — real, but the naive local re-tune (K3/B128, no pmin) does better: 92.3 tok/s =
**+8.0%** at 128 tok, +7.0% at 512 tok. The 82-SM verify tier is not cheap the way 188 SM's
is (same asymmetry family as the vt-fixes T=1 glue tax that vanished on the pod).

## Cells (all in RESULTS.jsonl; raw rows logs/points.jsonl, driver.log has per-pass GPU state)

- **Core 5-arm, nv, c=1, N=10/arm**: default K3/B32 85.5 → K5/B32 81.3 (−4.9%) → K5/B64 81.0
  → K5/B128 90.1 (+5.4%) → K5/B128+pmin 90.3 (+5.6%).
- **K re-check at B128, N=6/arm**: K3 92.3 / K3+pmin 91.8 / K4+pmin 91.3 / K5+pmin 90.6 /
  K6+pmin 91.3 — optimum K=3, pmin flat.
- **Burst ladder at K=3, 512 tok, N=6/arm**: B32 88.8 → B64 91.2 (+2.7%) → **B128 94.9
  (+7.0%)** → B256 93.1 (+4.9%). 128 is a genuine local optimum at 82 SM (curve turns down).
- **Guards**: c=8 def 91.1 vs win 97.4 (+6.9%, p50 improves 11.27→10.51s, N=6); 512-tok
  long-gen +8.0% (N=4) — not a shape artifact.
- **q9 transfer, N=6/arm**: 213.6 → B128 233.3 (+9.2%) → +pmin 233.9. Lever is bigger on the
  faster model; pmin again flat at K=3.
- **Streaming cadence (the flip-blocker)**: stream:true 256 tok — B32: first chunk 0.41s,
  8 chunks; B128: first chunk **1.15s**, 2 chunks. One SSE event per burst (worker.rs), so
  B128 makes the felt path ~2.8x worse to first text. The dogfood head-to-head already has
  memra losing felt-TTFT — a global 128 default would regress the owner's daily driver.
- **Exactness, levers ON**: run-spec K=1..8 self-consistency 9 PASS rc=0 × 3 configs
  (nv-embedded, nv+draft, q9+draft) with PMIN+BURST set; greedy stream BYTE-IDENTICAL
  lever-on/off (nv B128, nv B128+pmin, q9 B128+pmin — real 523/508-byte captures; the first
  identity pass compared empty files from a wrong JSON field and was re-run); serve-smoke
  rc=0, 0 FAIL. serve-st-gate: N/A, no ST model in this lane.

## Decision (flags doctrine)

**No default flips.** K=3 stays (confirmed winner here). PMIN stays an env door (its win is
K≥5-specific). BURST stays 32 as the shipped latency-safe default because the throughput win
and the streaming-cadence loss are the same knob — "winner" is context-dependent, and the
interactive daily regime is the default's audience. BURST=128 is now the *documented,
5090-validated* throughput-tier setting (c≥2 / batch / judge-harvest serve configs), and the
pod rows stay the PRO 6000 deploy config (K5/B128/pmin there — both re-minted, both real,
genuinely different optima at 82 vs 188 SM). No per-SM code gate needed while the K/pmin
defaults stay put; if a future lane wants burst auto-selection, the graph-key-48
`sm_count() >= 180` precedent (decode.rs) is the pattern.

FLAGS.md updated: MEMRA_SPEC_BURST and MEMRA_SPEC_PMIN rows carry the per-SM optima and the
cadence tradeoff.
