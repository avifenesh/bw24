# Ornith serve bench + vs-llama board cells — 2026-08-01

Lane: `lane/ornith-serve-bench` (from `restructure/public-split`, commit `b94adf5d` — the
v0.62 tree with the exact-16 decode tier). Rig: RTX 5090 Laptop 24 GB, platform_profile
`performance`, `gpu-full-power on`. Every GPU run under `flock /tmp/gpu5090.lock`;
the co-resident `llama-server --embedding` (allowlisted) was left alone. llama.cpp arm:
local build `c818263f2 (9833)`. Models = the three onboarded in
`research/onboard-ornith-20260801/` (sha256-verified there; zero engine code change —
this lane is measurement-only, no code touched).

Thermal regimes (stated per battery below; `thermal.log`):

- **Serve battery** (14:07–14:45Z): rig otherwise free, session warmed 57→87 °C across the
  38-min window — the N=3 passes are INTERLEAVED (pass loop outside the config loop) so every
  config's reps span the same window.
- **Board cells** (15:00–17:00Z): a co-resident drafters lane (`frspec-owngen`, the
  ornith-drafters worktree) occupied the GPU continuously (busy=1, 76–87 °C, clocks
  1700–2450 MHz). `wait_idle` never opened; **flock serialized every arm**, llama/memra pairs
  stayed adjacent, so both engines saw the same hot co-loaded regime. These cells are
  therefore NOT comparable to the README board's cold-start rebaseline rows — same-session
  ratios are the deliverable, absolute numbers are regime-bound.

## 1. Serve points (memra-server single replica, tools/load-serve.py)

Protocol: c ∈ {1, 8, 16}, requests = 4×c (min 8), max_tokens 128, ~200-token prompt,
temperature 0.7 + per-request seed (divergent sequences), 1 warmup per point. N=3 interleaved
passes per point; medians below, all three reps shown. Raw: `serve-points.jsonl` (36 points),
`serve-per-request.jsonl`, `server-*-rep*.log`, console `serve-points-run.log`. Zero errors
across all points.

| config | c | reqs | N | agg tok/s (median) | reps | p50 lat s | p95 lat s |
|---|---|---|---|---|---|---|---|
| o9b-naked  | 1 | 8 | 3 | **77.8** | 78.2, 75.1, 77.8 | 1.65 | 1.65 |
| o9b-naked  | 8 | 32 | 3 | **368.9** | 376.8, 362.9, 368.9 | 2.77 | 2.88 |
| o9b-naked  | 16 | 64 | 3 | **364.8** | 371.9, 360.6, 364.8 | 5.64 | 5.76 |
| o9b-q8rp (MEMRA_Q8RP=1) | 1 | 8 | 3 | **76.1** | 76.1, 75.6, 78.9 | 1.68 | 1.70 |
| o9b-q8rp (MEMRA_Q8RP=1) | 8 | 32 | 3 | **390.2** | 390.2, 376.1, 417.2 | 2.61 | 2.65 |
| o9b-q8rp (MEMRA_Q8RP=1) | 16 | 64 | 3 | **483.0** | 483.0, 442.0, 503.8 | 4.23 | 4.58 |
| o35b-naked | 1 | 8 | 3 | **80.0** | 80.0, 79.5, 81.4 | 1.56 | 1.99 |
| o35b-naked | 8 | 32 | 3 | **152.7** | 152.7, 147.8, 154.4 | 6.72 | 7.83 |
| o35b-naked | 16 | 64 | 3 | **151.0** | 150.8, 151.0, 153.8 | 13.56 | 16.05 |
| kat-naked  | 1 | 8 | 3 | **50.0** | 52.6, 50.0, 50.0 | 2.45 | 2.67 |
| kat-naked  | 8 | 32 | 3 | **63.5** | 63.6, 63.5, 62.7 | 16.24 | 16.46 |
| kat-naked  | 16 | 64 | 3 | **63.1** | 63.2, 62.9, 63.1 | 32.60 | 34.55 |

(o9b = Ornith-1.0-9B Q8_0 · o35b = Ornith-1.0-35B Q4_K_M · kat = KAT-Coder-V2.5-Dev IQ4_XS)

### Exact-16 chunk-tier engagement (server-log `decode chunk cap` lines, `chunk-cap.log`)

| model | naked | knob | evidence |
|---|---|---|---|
| Ornith-9B Q8_0 | **chunk 8** (Q8_0 has no rp4 mirror naked on the 5090 — mirror default is hopper-only) | `MEMRA_Q8RP=1` → **chunk 16 (exact-16 tier)**, "[q8rp] split-plane decode mirrors built: 249 tensors" | server-o9b-*-rep*.log |
| Ornith-35B Q4_K_M | **chunk 8** | none applicable | server-o35b-naked-rep*.log |
| KAT-Coder IQ4_XS | **chunk 8** | none applicable | server-kat-naked-rep*.log |

**Correction to the mission brief**: Q4_K_M / IQ4_XS do NOT qualify natively. The admission
check (`crates/memra-engine/src/decode_batch.rs::decode_batch_exact16_ok`) admits only
Q4_0 / Q6_K / F8_E4M3 / Q8_0-with-rp4 matmuls and hard-excludes `Ffn::Moe` (and `Mixer::Mla`)
— both 35B-class models are MoE, so they are chunk-8 regardless of quant, and neither Q4_K
nor IQ4_XS is in the b16-exact kernel class anyway.

What the tier is worth where it engages: o9b c=16 **+32%** aggregate (483.0 vs 364.8) plus a
c=8 gain from the rp4 mirrors themselves (390.2 vs 368.9). Cost: mirrored trunk VRAM
(~16.8 GiB total vs ~10.5 GiB naked). c=1 is mirror-neutral (76.1 vs 77.8, within rep spread).
Scaling shape: o9b-naked saturates at c=8 (chunk-8 ceiling: c=16 = two chunk-8 ticks);
o35b saturates at c=8 (~152, MoE expert-stream bound); KAT barely scales (63 — prefill-bound,
see §4).

## 2. Serve-level isolation (greedy c=1 vs c=16, check-batch-exact.py, 16 prompts, 96 tok)

Phase A: 16 distinct greedy prompts sequentially (c=1). Phase B: same 16 concurrently (c=16,
batched to the chunk cap). Byte-identity per prompt. Raw: `greedy-hash-*.{jsonl,log}`,
refs `greedy-refs-*.json`, servers `server-hash-*.log`.

| config | verdict | match |
|---|---|---|
| o9b-naked (chunk 8) | **PASS** | 16/16 |
| o9b-q8rp (chunk 16, exact-16 tier) | **PASS** | 16/16 |
| o35b-naked | **FAIL** | 10/16 (1 divergence at char 0) |
| kat-naked | **FAIL** | 8/16 |
| ctrl-q35 (supported Qwen3.6-35B UD-IQ4_XS, naked) | **PASS** | 16/16 |
| o35b + MEMRA_DECODE_BATCH_CAP=1 (decode batching OFF) | **FAIL** | 8/16 |
| o35b + MEMRA_PRIME_BATCH=1 (prime batching OFF) | **PASS** | 16/16 |

**Attribution (differentials above): the divergence is the batch-prime concat prefill path,
NOT batched decode.** With decode batching disabled the failure persists; with prime batching
disabled it vanishes. Batched decode holds its exactness contract on all three models
(and at chunk 16 for o9b-q8rp — first serve-level exact-16 receipts on a second Q8_0 model).
The batch-prime gate ("per-seq argmax + decode-stream equality", worker.rs task #13) was
green on supported models — the supported control still passes 16/16 here — but the concat
GEMM's m=ΣT numerics flip greedy argmax on the Ornith/KAT post-trains at real rates
(6–8 of 16 prompts within 96 tokens). All serve points in §1 ran the naked (batch-prime ON)
config. **Deployment caveat: greedy determinism under concurrent load is not
isolated-identical for Ornith-35B/KAT unless `MEMRA_PRIME_BATCH=1` (its throughput cost was
not measured in this lane).** This is prefill argmax sensitivity, not a wrong-bytes bug: all
three models hold run-gen argmax MATCH (prefill==decode) in every board arm below.

## 3. Board-protocol cells vs llama.cpp (all three models + control)

Reference recipe = the supported q35-plain cell of `tools/full-board-bench.sh`: llama-bench
`-ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 512 -n 128 -r 3` vs memra `run-gen` naked
(`MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt`, `MEMRA_NGEN=128`), llama→memra
interleaved per pair, N=3 pairs per cell, same session. Raw: `board-cells.jsonl` (54 rows),
`board-*-rep*.log`, consoles `board-cells-console-*.log`. All memra arms: argmax MATCH,
[stop: MaxNew] (full 128 tokens) — except the ctrl pp512.txt arms, see note.

| cell | metric | memra median (reps) | llama median (reps) | ratio |
|---|---|---|---|---|
| Ornith-9B Q8_0 | decode/tg128 | 81.1 (79.5, 81.1, 87.8) | 83.8 (80.4, 83.8, 84.0) | **0.97** |
| Ornith-9B Q8_0 | prefill/pp512 | 4933.4 (4870.9, 4933.4, 5255.6) | 5027.3 (4927.1, 5027.3, 5100.9) | **0.98** |
| Ornith-35B Q4_K_M | decode/tg128 | 137.3 (134.8, 137.3, 137.5) | 191.9 (190.1, 191.9, 192.2) | **0.72** |
| Ornith-35B Q4_K_M | prefill/pp512 | 492.2 (491.1, 492.2, 493.0) | 3385.6 (3359.3, 3385.6, 3513.4) | **0.15** |
| KAT-Coder IQ4_XS | decode/tg128 | 105.9 (105.6, 105.9, 106.5) | 192.9 (192.9, 192.9, 194.0) | **0.55** |
| KAT-Coder IQ4_XS | prefill/pp512 | 223.9 (214.9, 223.9, 225.9) | 4167.9 (4159.3, 4167.9, 4206.9) | **0.05** |
| ctrl Qwen3.6-35B UD-IQ4_XS | decode/tg128 | 191.5* (191.3, 191.5, 191.7) | 169.9 (165.1, 169.9, 170.2) | **1.13** |
| ctrl Qwen3.6-35B UD-IQ4_XS | prefill/pp512 | 2395.9* (2394.6, 2395.9, 2404.7) | 4171.5 (3056.0, 4171.5, 4176.4) | **0.57** |

\* ctrl memra readings from `ctrl-prose-512.txt` (511 tok, same depth class): on the ctrl
model the board pp512.txt prompt hits Eos after 2 tokens (`[stop: Eos]`, reps logged in
`board-ctrl-q35-memra-rep*.log` — a 2-token reading is not tg128-class and was discarded).
The prose reruns are N=3, same session, ~10 min after the ctrl llama arms; llama-bench tg128
is prompt-independent and pp512 is length-defined, so the denominator carries. The ctrl cell
is ATTRIBUTION evidence (same-session family anchor), not a board row.

e2e proxy (512 prefill + 128 decode wall from cell medians):

| cell | memra s | llama s | memra advantage |
|---|---|---|---|
| Ornith-9B Q8_0 | 1.682 | 1.629 | 0.97x |
| Ornith-35B Q4_K_M | 1.972 | 0.818 | 0.42x |
| KAT-Coder IQ4_XS | 3.495 | 0.786 | 0.23x |
| ctrl Qwen3.6-35B | 0.882 | 0.876 | 0.99x |

Protocol notes: o9b pair 2 was split ~35 min (the harness stopped the first runner between
arms; resumed by `run-board-cells-resume.sh` — llama rep2 sits within its rep1/rep3 spread,
medians unaffected). The hot co-loaded regime pins clocks high: the same-protocol ctrl decode
rows (memra 191.5 / llama 169.9) sit above their cold-start board values (178.2 / 167.8) —
another reason these absolute numbers must not be pasted into the README board.

## 4. Anomaly analysis (mission item 3: flag, don't tune)

The supported family anchor holds **same-session**: ctrl decode ratio 1.13x memra-favored.
The new models' deficits are therefore **trunk-quant-mix-specific, not arch-specific** —
all four are the same qwen35/qwen35moe stack. GGUF tensor-mix dumps (header parse):

| model | attn/shexp/linear trunk | experts | memra decode ratio | memra prefill ratio |
|---|---|---|---|---|
| ctrl UD-IQ4_XS | **Q8_0** (attn 104×Q8_0, shexp 123×Q8_0) | IQ3_S 78 + IQ4_XS 39 | 1.13 | 0.57 |
| Ornith-35B Q4_K_M | **Q4_K** (attn 80, shexp 100, linear 90) | Q4_K 100 + Q6_K 20 | 0.72 | 0.15 |
| KAT IQ4_XS | **IQ4_XS** (attn 65) + Q8_0 shexp 60 | IQ4_XS 120 | 0.55 | 0.05 |
| Ornith-9B Q8_0 (dense) | Q8_0 | — | 0.97 | 0.98 |

The supported "IQ4_XS 35B" win is really a *Q8_0-trunk + compact-experts* win; Ornith-35B
exercises the Q4_K trunk path and KAT the IQ4_XS trunk path, both far off memra's tuned
kernels while llama runs all three mixes at full speed (llama is FASTER on Q4_K_M than on
UD-IQ4_XS: 191.9 vs 169.9 tg128). Flags for follow-up lanes (NOT touched here):

1. **Q4_K trunk decode + prefill** (Ornith-35B): m≤16 Q4_K mmvq and MoE Q4_K prefill.
   `MEMRA_KQRP` (q4_K/q6_K split-plane mirrors, 2026-08-01 H100 coalescing fix,
   default = hopper lane) already exists as a seam — unswept on the 5090, left untested per
   the no-tuning rule.
2. **IQ4_XS trunk path** (KAT): decode 0.55x and prefill 224 tok/s (0.05x) — the single
   largest gap measured; it also explains KAT's serve saturation at ~63 agg tok/s
   (c=8 p50 latency 16 s is prefill queueing; onboarding saw the same slow pp22/pp302).
3. **MoE pp512-class prefill generally**: even the winning ctrl is 0.57x on prefill —
   consistent with the known prefill-GEMM gap on the dense 9B lane; the MoE variant of that
   rebuild is unowned.
4. Serve-level batch-prime argmax instability on the Ornith/KAT post-trains (§2) — an
   exactness-policy question (bit-exact concat prime, or per-model gating), not a perf knob.

## 5. Verdicts (owner bar: beat llama ≥1.1x e2e before deploy)

| model | verdict | detail |
|---|---|---|
| Ornith-1.0-9B Q8_0 | **TRAILS (parity-class)** | 0.97x decode / 0.98x prefill / 0.97x e2e — dead heat, below the 1.1x bar. Open lane: Q8_0 dense-trunk decode on the 5090 (the q8rp mirrors that pay +32% at serve c=16 cost ~2x weight VRAM and don't move c=1). |
| Ornith-1.0-35B Q4_K_M | **TRAILS** | 0.72x decode / 0.15x prefill / 0.42x e2e. Open lane: Q4_K trunk kernels (flag #1). Same-arch control beats llama 1.13x same-session, so the gap is the quant path, not the model. |
| KAT-Coder-V2.5-Dev IQ4_XS | **TRAILS** | 0.55x decode / 0.05x prefill / 0.23x e2e. Open lane: IQ4_XS trunk kernels (flag #2). Worst of the three; also weakest serve scaling. |

**Publication: all three stay "onboarded, pre-deployment" — nothing here qualifies for the
README generated board.** Protocol-wise the cells are interleaved same-session N=3 with raw
receipts (evidence-grade for this verdict), but (a) every model TRAILS, (b) the regime is
hot co-loaded (frspec co-lane), not the board's cold-start rebaseline, and (c) the board
publishes supported/winning configurations. `current-board.json` untouched; these numbers
live here as prose/receipts.
