# ST board pairing — 2026-07-09 (cold-start-paired, same-session)

Measurement lane run. Both NVFP4 safetensors checkpoints vs llama.cpp at its serve-best
config on this rig, all cells measured interleaved in ONE session with cold-start pairing:
fresh process per run, GPU idle-clock gate (<1000 MHz, observed 180–540 MHz) before every
run, engines alternated, N=2 per engine per cell, medians reported. gpu-full-power on.
Raw per-run logs (incl. per-run clocks/temp/power): `research/tune-data/st-pairing-logs/`.

- bw24: main@0f4ca26, `run-gen` / `run-spec` / `decode-bench` release binaries (built 13:14).
- llama.cpp: b9743 build at `/home/avifenesh/projects/llama.cpp/build/bin`, `GGML_CUDA_GRAPH_OPT=1`.
- Spec timing method = `research/e2e/run-e2e.sh` exactly: e2e prompts p1/p2/p3, NGEN=256,
  greedy, llama via `/completion` timings with `cache_prompt:false`.
- Board policy note: ST checkpoints have no GGUF twin — llama comparator is its own best
  GGUF checkpoint (9B NVFP4 GGUF; 27B NVFP4+Q4_K_M GGUF + MTP draft `mtp-Qwen3.6-27B-Q4_K_M.gguf`,
  `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2→0.1 per run-e2e.sh`, KV q8_0/q5_1).
  llama 9B serve-best is RAW (no 9B MTP draft exists on disk).

## Gates (run first, this session — all numbers below count)

| model | argmax gate | spec self-consistency |
|---|---|---|
| 9B ST (qwen35-9b-nvfp4-st-modelopt) | MATCH (268, logit maxdiff 0.000e0) | K=1..8 ALL PASS |
| 27B ST (nvidia-qwen36-27b-nvfp4) | MATCH (1178, logit maxdiff 0.000e0) | K=1..8 ALL PASS |

## Plain decode — tg128 @ d512 (`decode-bench <dir> 512 128 eager`, llama-bench `-n 128 -d 512`)

| cell | bw24 median (runs) | llama median (runs) | ratio |
|---|---|---|---|
| 9B plain | **129.05** (129.1 / 129.0) | 123.69 (123.66 / 123.72) | **1.04x** |
| 27B plain (NV_W4=1) | 40.8 (40.8 / 40.8) | **44.92** (44.32 / 45.52) | **0.91x** |

## Spec — p1/p2/p3, NGEN=256 greedy (bw24 established configs vs llama serve-best)

bw24 9B ST: `BW24_SPEC_K=2 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM=.../frspec-9bst-modelopt-32768.gguf`
bw24 27B ST: `BW24_SPEC_K=3 BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.4 BW24_SPEC_NV_W4=1 BW24_FRSPEC_TRIM=.../frspec-corpus-32768.gguf`

| cell | bw24 median (runs) | llama median (runs) | ratio |
|---|---|---|---|
| 9B p1 short-code | **203.9** (202.9 / 204.9) | 122.9 (123.0 / 122.9) | **1.66x** |
| 9B p2 medium-code | **192.5** (191.4 / 193.6) | 122.2 (122.4 / 122.1) | **1.57x** |
| 9B p3 agentic-long | **223.7** (223.7 / 223.8) | 118.2 (118.4 / 118.1) | **1.89x** |
| 27B p1 short-code | **89.3** (89.6 / 89.0) | 86.0 (86.4 / 85.5) | **1.04x** |
| 27B p2 medium-code | 85.4 (85.6 / 85.2) | **91.2** (91.7 / 90.7) | **0.94x** |
| 27B p3 agentic-long | 71.6 (71.7 / 71.5) | **76.7** (77.1 / 76.2) | **0.93x** |

Inline plain-gen references from the same spec runs ([generate] line, N=2):
9B 129.7 / 126.1 / 124.3 — 27B 40.4 / 39.8 / 39.2 (consistent with decode-bench cells).

## Thermal / regime notes

- Every run launched from a verified idle GPU (<60C, 180–540 MHz). No stale `-lgc` lock
  observed at any point (idle clocks always dropped back to 180–540 MHz).
- Under load: bw24 1785–2167 MHz, 61–73C, ~170 W cap; llama 1725–1905 MHz, 61–73C.
  Both engines rode the same ~170 W wall — no clock-regime asymmetry between paired runs.
- Within-pair spread <1% on every bw24 cell and <1.5% on every llama cell; the pairing
  discipline (idle-gate before each run) is doing its job.

## Anomalies (flagged, not investigated — per lane protocol)

1. **27B ST plain 40.8 vs lane-session 47.5 (−14%, >10% flag).** Same NV_W4=1 config class.
   Candidates: lane-session number was a warm/session measurement, or main moved between
   sessions. Today's N=2 is internally exact (40.8/40.8) and inline [generate] agrees (40.4).
2. **27B ST spec p2 85.4 vs lane-session 64.0 (+33%).** Large positive shift vs the
   2026-07-09 lane row; today's two rounds agree to 0.5%.
3. 9B ST spec today (203.9/192.5/223.7) sits ~7% ABOVE the lane's own cold-start N=2
   (190.5/188.5/217.7) — direction consistent with stricter idle-gating.
4. One bw24 27B p3 (round 2) run was killed by a harness shell timeout mid-run; GPU state
   captured, cell cleanly re-run from cold (see `27bst-spec-round2.log`). No engine error.

## Verdict line

- **9B ST: above llama everywhere.** Plain 1.04x; spec 1.66x / 1.57x / 1.89x (llama has no
  9B draft — its serve-best is raw).
- **27B ST: mixed.** Spec p1 above (1.04x), p2/p3 6–7% behind llama's MTP serve; plain 0.91x
  (the GGUF-27B plain parity does not yet transfer to the ST NV_W4 decode path).

---

## CORRECTION (same day, tag `27bst-board-pairing-FIX`)

Every bw24 27B cell above ran with `BW24_SPEC_NV_W4=1` — a nonexistent flag (real: `BW24_NV_W4`);
the attention requant never engaged. Additionally the afternoon regime drifted ~8% on both engines
(llama plain 44.9 → 41.2 between hours), so cross-hour ratios were invalid. All 27B cells
(bw24 AND llama) re-measured in one hour-regime; raw logs: `st-pairing-logs/27bst-*-fix-*`,
`27b-llama-spec-fix-*`.

| cell | bw24 | llama | ratio |
|---|---|---|---|
| 27B plain tg128@d512 (BW24_NV_W4=1) | 45.2 (46.3/45.2/45.0) | 41.2 (41.28/41.14) | **1.10x** |
| 27B spec p1 | 92.9 (95.6/90.3) | 79.7 (80.8/78.6) | **1.17x** |
| 27B spec p2 | 81.3 (81.7/81.0) | 84.7 (85.9/83.5) | 0.96x |
| 27B spec p3 | 84.6 (84.8/84.3) | 71.3 (72.5/70.1) | **1.19x** |

9B cells above remain valid (internally paired, correct flags). 9B addition: per-content K
(`9bst-k3-repro`) — p3 at K=3 = 256.0 (acceptance 100%), replacing K=2's 223.7 on the board.
