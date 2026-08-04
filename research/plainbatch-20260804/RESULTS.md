# plainbatch probe — batched-PLAIN serve arm vs CLI tokenwise oracle (2026-08-04)

Branch `lane/plainbatch-probe` off `restructure/public-split` @ ac99e675. Rig: local RTX
5090 laptop, GPU work under `flock /tmp/gpu5090.lock`. Mission: receipt (or refute) the
#68 lane's reclassification of the batched-plain near-tie divergence as the accepted
decode-config FP class — with the n-scaling check #68 taught us to run.

## Verdict: **FP NEAR-TIE CLASS CONFIRMED, n-STABLE — no new bug, no code change.**

All three receipt criteria hold:

1. **Every flip is near-tie class.** 16/16 independent flips (13 q9long + 3 9bst) have
   oracle-vs-served logit gaps 0.0009–0.1332 — ALL sub-0.2. Step-margin median at the
   oracle: **3.736** (q9long) — a dead match to the established near-tie profile
   ("sub-0.2 vs median ~3.7"). 9bst France-prompt median 10.632 (peakier distribution,
   same sub-0.2 flips).
2. **Independent flip rate does NOT grow with depth.** Naive stream-diff rate DOES grow
   (0.565 → 0.739 → 0.869 on 9bst) — that is post-flip sequence separation (after one
   flip the two streams condition on different prefixes), NOT accumulating error. The
   teacher-forced probe (`plainbatch-margin-probe`: force the SERVED stream through the
   CLI-oracle config, count positions where oracle-greedy != served) is the correct
   metric: per-400-token segment flip counts are q9long 2/5/3/3 and 9bst 3/0/0/0 —
   fluctuating around a constant (q9long) or *declining to zero* (9bst). No onset, no
   growth. Contrast #68's fingerprint: clean until ~553 then total corruption with
   every burst diverging at tok 0.
3. **Determinism + #68-fix invariance.** Every cell rep1==rep2==rep3 byte-identical
   across server restarts; served streams are exact prefixes across n=400/800/1600.
   The 9bst n=400 plain text is **byte-identical (1514/1514 chars)** to the post-fix
   receipt `research/fp8ship-20260804/post-fix-serve-9bplain.txt`, and shows the SAME
   flip ("questions" @ char 551 / tok 152, ids 4602-vs-18959) that the PRE-fix receipts
   documented (serve-st-20260803 RESULTS: 'near-tie "questions" vs "queries", char 551').
   The fa_part_retired fix changed nothing on the plain arm — expected: the plain batched
   path never replays a persisted graph, so freed-buffer retirement has no seam here.

## n-scaling table (N=3 per cell, all reps identical — single value shown)

Naive stream diff (first_div + rate = differing positions / window):

| arm | n=400 | n=800 | n=1600 | first_div |
|---|---|---|---|---|
| q9 France (GGUF) | MATCH (EOS caps window at 208) | MATCH | MATCH | — |
| q9long essay (GGUF) | 0.598 | 0.798 | 0.890 | tok 161 |
| 9bst France (NVFP4 ST) | 0.565 | 0.739 | 0.869 | tok 152 |

Independent flips (teacher-forced oracle over the served 1600-tok stream):

| arm | flips 0-400 | 400-800 | 800-1200 | 1200-1600 | total | max gap | median step margin |
|---|---|---|---|---|---|---|---|
| q9long | 2 | 5 | 3 | 3 | 13 (0.81/100tok) | 0.1332 | 3.736 |
| 9bst | 3 | 0 | 0 | 0 | 3 (0.19/100tok) | 0.0643 | 10.632 |

Flip gap list (q9long): 0.0352 0.1332 0.0362 0.0249 0.0072 0.0235 0.0618 0.0025 0.0137
0.0837 0.0067 0.0120 0.0009. (9bst): 0.0300 0.0643 0.0043.

## Method

- `probe.sh`: black-box per-cell arm pair (server-launch + native /v1/completions raw-id
  request pattern copied from tools/serve-st-gate.sh). CLI arm = `run-gen --prompt`
  MEMRA_CHAT=1 tokenwise decode; serve arm = fresh `memra-server MEMRA_SERVE_SPEC=0`
  per rep (no prefix-cache carryover), default batched tick. `battery.sh` = France
  prompt x {q9, 9bst} x n{400,800,1600} x N=3; `battery-q9long.sh` = essay prompt
  (the France q9 run EOSes at 209 — the short-window trap in the other direction).
- `plainbatch-margin-probe` (new bin, crates/memra-engine/src/bin/plainbatch_margin_probe.rs):
  teacher-forces a served stream through the oracle config (batched prime_cache +
  tokenwise decode_step), prints each flip's logit gap + running top1-top2 margins.
- Control: `MEMRA_SERVE_GS=0` cell (graph-session promotion disabled) reproduces the
  9bst flip exactly (first_div=152, 226/400) — the divergence is the batched-prefill/
  batched-decode config itself, not the GS promotion arm.
- Raw receipts in this dir: `table.jsonl` (all 30 cells), `battery*.log`, per-cell
  `cli-*/server-*/srv-*` logs, `tokens-{cli,srv}-*.txt` streams,
  `margin-*-servedforced.log`.

## Jurisdiction

This class is decode-batch-gate's calibrated dice (config mode: near-tie argmax flips
across FP compositions are the accepted WARN class; bit-strength gates cover plumbing).
serve-st-gate item 4 already compares serve-spec to the tokenwise serve oracle for
exactly this reason. No gate change needed: the q9 France MATCH cells show the gate
windows aren't hiding an onset — there is no onset to hide.
