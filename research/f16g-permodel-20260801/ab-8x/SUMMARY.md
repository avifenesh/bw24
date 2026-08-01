# f16g default arbitration — 8×H100 capacity-block box (darklanes-bench, 18.117.231.105)

Date: 2026-08-01. Runs launched ~10:12Z via `f16g-ab.sh` (this dir), one model per GPU on
GPUs 1-4; all four logs reached `AB_DONE` before 10:17Z collection. Box hard-terminates
11:30Z (capacity block cr-0a6421a10d8e2ea94).

Protocol: naked default (Hopper `MEMRA_MOE_F16G=1`) vs `MEMRA_MOE_F16G=0`, interleaved x5
pairs, board-2048 prompt, `MEMRA_NGEN=32`, prefill + decode captured per run. Same box,
same session — cross-run clock-drift rules satisfied. Medians of N=5 per arm, from the raw
logs committed alongside (`ab-*.log`).

Model tags → GGUFs (from `~/models` inventory): q27 = Qwen3.6-27B-Q4_K_M,
g31 = gemma-4-31B_q4_0-it, q35 = Qwen3.6-35B-A3B-UD-IQ4_XS, g26 = gemma-4-26B_q4_0-it.

## Verdicts (prefill tok/s, median of 5 per arm)

| model | def median | off median | delta | verdict |
|-------|-----------:|-----------:|------:|---------|
| q27 | 4843.6 | 4833.9 | +0.2% | TIE — first-ever default-arm measurement; default safe, no win (N=5/arm) |
| g31 | 4718.5 | 4719.0 | -0.0% | TIE — first-ever default-arm measurement; default safe, no win (N=5/arm) |
| q35 | 7901.9 | 5442.3 | +45.2% | HELPS — re-verifies the established +53% win; direction confirmed, +45.2% on this box (N=5/arm) |
| g26 | 10588.3 | 11261.6 | -6.0% | HURTS — cross-box confirm of the established -8.3% regression; -6.0% here (N=5/arm) |

Notes:
- q27 and g31 were never measured under the Hopper f16g default before — these close the
  missing default-flip verdicts (stale-verdict law). Both are ties well inside run scatter.
- g26 def arm is noisy (range 7420.2–11779.1) while the off arm is tight (11162.9–11318.7);
  the regression direction matches the established finding regardless.
- q35 def pair4 (7013.3) is a low outlier; even min(def)/max(off) is +28.8%.

## Decode deltas (median of 5, should be ~0 — f16g is prefill-only)

q27 +1.4% (92.62 vs 91.31), g31 +0.2% (81.27 vs 81.14), q35 -0.3% (183.61 vs 184.11),
g26 -1.2% (207.96 vs 210.42). Per-arm ranges overlap in all four cases; no decode effect
claimed. q27 (+1.4%) and g26 (-1.2%) are the two largest and are flagged here for the
record, but both are within interleave scatter.

## Box wind-down state (at collection, 10:17–10:23Z)

- GPUs 1-4: all four A/B runs complete (`AB_DONE` in every log); no compute processes.
- GPUs 5-7: v0.60.0 fleet (6× `memra-server`, ~20 GB each on 3 GPUs) still serving —
  belongs to the completed fleet validation whose receipts are already committed on
  lane/fleet-v060 (5e68200b; box SUMMARY.md md5-matches the committed copy). Left as-is.
- No shutdown action taken; the capacity block terminates the box at 11:30Z.

## box-sweep/ — unreceipted evidence recovered from the box

Sweep scope: `/tmp` files newer than 09:00Z, `~/fleet-v060/research/`, `ls ~/` stray dirs.
`~/fleet-v060` and `~/arc-sk` are plain rsync'd trees, not git repos — nothing on the box
was committed anywhere, so everything was cross-checked against the local repo:

- fleet-v060-20260801 validation receipts: already committed (lane/fleet-v060 5e68200b,
  md5-verified) — not duplicated here.
- darklane-serving-20260801: already committed (lane/serving-v1 9963cc1b et al.) —
  file-for-file present, not duplicated here.
- m0-nccl receipts: already committed (lane/m0-nccl, merged) — not duplicated here.
- sk-vs-cublas (lane/sk-ncu d22159d2): the TEXT receipts are committed, but the driver
  scripts, driver stdout, build log, and the binary ncu/nsys profiles existed only in
  `/tmp` on the terminating box. Recovered into `box-sweep/sk-vs-cublas/`:
  `skbench.sh/.out`, `skncu2-5.sh/.out`, `build-sk.log`, `ncu-sk.ncu-rep`,
  `ncu-cublas.ncu-rep`, `nsys-f16g1.nsys-rep`, `nsys-f16g2.nsys-rep`. The `.sqlite`
  exports were skipped (regenerable from the `.nsys-rep` files).

This directory complements the Mumbai arbitration receipts landing under
`research/f16g-permodel-20260801/` on a separate branch; everything here stays under
`ab-8x/` to avoid path collisions.
