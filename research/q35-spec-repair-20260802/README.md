# q35-spec-repair — 2026-08-02

Board q35 speculative row re-measured under the new sm_120a naked default (AUTO-KQUANT
f16g mode 3, merge cf8a9358), exact published-row protocol. Rig: RTX 5090 Laptop,
`performance` platform profile, `gpu-full-power on`, idle-gated (<1200 MHz, zero
non-embedding compute procs), every GPU run under `flock /tmp/gpu5090.lock`. Co-resident
`llama-server --embedding` (port 8181, -ngl 0, 332 MiB) untouched throughout.

## Protocol (provenance: rig5090.jsonl:345 2026-07-18 + rebaseline2-2026-07-09.md)

- memra: `run-spec`, `MEMRA_MTP_DRAFT=draft-35b-owntrim-nvfp4head-q4blk.gguf`,
  `MEMRA_SPEC_K=2`, `MEMRA_NGEN=256`, naked otherwise (fixed-K is the qwen default).
  p1 `p1-code-short.txt` greedy (28 tok) / p2 `p2-code-medium.txt` greedy (1845 tok) /
  p3 `p3-agentic-long-v3.txt` `MEMRA_CHAT=1` (5420 tok) SAMPLED `MEMRA_SPEC_TEMP=0.7
  MEMRA_SEED=42`. Metric: `[generate_spec K=2]` gen-only tok/s.
- llama (build 9837 c73069749): llama-server self-MTP (embedded NextN, NO `-md`),
  `--spec-type draft-mtp --spec-draft-p-min 0.1`, `-ngl 999 -fa on -c 16384 --parallel 1
  -ctk q8_0 -ctv q5_1`, `GGML_CUDA_GRAPH_OPT=1`. p1 `/completion` temp 0
  `ignore_eos:true` (raw p1 EOSes at 1 tok, 2026-07-08 row) / p2 `/completion` temp 0 /
  p3 `/v1/chat/completions` temperature 0.7 seed 42 max_tokens 256. `cache_prompt:false`.
  Metric: `timings.predicted_per_second`, `predicted_n=256` on every run, MTP engagement
  verified via `draft_n`/`draft_n_accepted`.
- `--spec-draft-n-max` re-swept per class same-session on the current llama build
  (`llama-nmax-sweep.out`): p1 optimum 3 (240.9@2 / 251.1@3 / 249.7@4), p2 optimum 2
  (217.5@2 / 178.7@3 / 193.2@4), p3 optimum 4 (231.3@2 / 241.5@3 / 243.5@4 / 227.2@6).
  Published llama columns use the per-class optimum (gemma-row precedent).
- N=3 medians per arm per cell, both engines interleaved in one session
  (`run-row-repair.sh`, 23:4x–00:07 UTC + best-config finals 00:2x–00:4x UTC same
  window); memra rep4 confirmation runs land in-band against the later llama finals.

## Gates (before measuring, naked build @ cf8a9358)

- `gate-run-gen.log`: prefill/decode argmax **MATCH** (maxdiff 1.376e0, exit 0).
  batched-prime line reads FLIP-NEARTIE — the documented non-fatal cross-config
  near-tie class (run_gen.rs gap #46; q35 pp512 probe, research/prime-gate-coverage-20260802).
- `gate-run-spec-k2.log`: K=2 self-consistency **PASS** (identical to generate).
- Every greedy measurement run: self-consistency PASS. Every sampled p3 run:
  PASS (seeded rerun identical), acceptance 84.2% stable.

## Results (tok/s, gen-only spec decode, N=3 medians)

| col | memra runs | med | old | llama runs (best n-max) | med | old | ratio old→new |
|---|---|---|---|---|---|---|---|
| p1 short-code | 305.28/305.38/305.68 (+rep4 306.37) | **305.4** | 280.6 | 244.98/251.37/251.37 @3 | **251.4** | 236.5 | 1.19x → 1.21x |
| p2 medium-code | 241.65/241.88/241.90 (+rep4 241.63) | **241.9** | 259.6 | 216.53/217.46/217.86 @2 | **217.5** | 174.6 | 1.49x → 1.11x |
| p3 long-agentic | 275.61/275.62/275.99 (+rep4 275.93) | **275.6** | 258.0 | 242.94/246.45/246.85 @4 | **246.5** | 173.5 | 1.49x → 1.12x |

Spreads: memra ≤0.1% every cell; llama 2.6%/0.6%/1.6%. Thermal regime: GPU 55–71 °C
across all runs (temps stamped per row in `row-repair.jsonl`).

Old-row spread (recovered from rig5090.jsonl:345, N=2): p1 278.4/282.7 (1.5%),
p2 259.8/259.4 (0.2%), p3 257.1/258.9 (0.7%).

## Verdict: every column MOVED

- memra p1 **+8.8%**, p3 **+6.8%** — outside both the old N=2 and new N=3 spreads.
- memra p2 **−6.8%** — MOVED, but pre-dates the AUTO-KQUANT flip: the 2026-07-30
  fullboard (pre-flip, same greedy p2 protocol) already read 240.8
  (`research/tune-data/fullboard-20260730.jsonl`) vs today's 241.9 — 07-18→07-30 code/window
  drift, not the flip.
- llama +6.3% / +24.6% / +42.1% — their build moved since 07-18 (b9837 MTP path faster)
  plus the per-class n-max re-sweep (old row ran n-max 2 everywhere).
- Board republished on today's same-session pairs: ratios 1.19/1.49/1.49 → 1.21/1.11/1.12;
  35B keeps all three cells above the 1.05 bold threshold.

## Files

- `run-row-repair.sh` — the harness (protocol encoded in header).
- `row-repair.jsonl` — per-run rows (ts, git, llama build, temp_c, profile, acceptance).
- `console.log` — battery console; `memra-*-rep{1..4}.log` raw run-spec logs
  (`memra-p3-agentic-long-rep4-GREEDY-MISFIRE.log` = discarded mis-invoked rep, kept for the record).
- `llama-server-rep*.log`, `llama-{p1,p2,p3}-rep*.json` (full responses),
  `llama-completion-rep*.out` — llama N=3 @ n-max 2.
- `llama-nmax-sweep.out`, `llama-best-final.out`, `llama-server-nmax*-{probe,final}.log`
  — the n-max re-sweep and the winning-config N=3 finals.
- `gate-run-gen.log`, `gate-run-spec-k2.log` — pre-measurement gates.
