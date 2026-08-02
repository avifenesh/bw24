# Qwen-AgentWorld-35B-A3B — onboarding + drafter + deployment bar (2026-08-02)

Lane `lane/agentworld` (from `restructure/public-split`, 0983c4ba). Rig: RTX 5090 Laptop
24.5GiB. Every GPU run under `flock /tmp/gpu5090.lock` (one co-lane shares the rig; the
co-resident `llama-server --embedding` at 332 MiB is allowlisted and untouched). Build:
`nice cargo build --release`, sm_120a auto-detected.

Context: support-by-construction was verified at onboarding
(`research/onboard-ornith-20260801/` — header byte-identical qwen35moe stack) but the
artifact was never downloaded, never gated, never benched, and had no drafter. This lane
runs the complete bar pipeline. Deployment bar (owner): beat llama >=1.1x e2e
best-vs-best + a gated own-gen trimmed drafter before publishing as supported.

## Stage 1 — artifact

| field | value |
|---|---|
| HF repo | unsloth/Qwen-AgentWorld-35B-A3B-GGUF |
| revision (pinned at onboarding) | `3a305abf5cfd119ee999dfe929c433746edd8d63` |
| file | `Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf` (Q4_K_M class, mirrors the Ornith-35B pick) |
| size | 22,134,529,280 bytes |
| sha256 | `e7a8eafdd8013443b6bcc4b6fb47b2d2025f772d359650b9ceb7d75971e22cad` — **verified vs the HF LFS pointer** (`sha-verify.log`) |
| local path | `/data/ai-ml/hf-models/agentworld-35b-gguf/` |

Download transcript: `downloads.log`. Disk pre-check: /data 733G free.

## Stage 2 — onboarding gates (all logs in `gates/`)

| gate | result | log |
|---|---|---|
| kernel-check (once per branch build) | **ALL GREEN** | `kernel-check.log` |
| run-gen argmax pp22 (MEMRA_CHAT=1) | **MATCH** (prefill==decode 90700; batched-prime MATCH) | `gates/agentworld-q4km-argmax-pp22.log` |
| run-gen argmax pp302-class depth | **MATCH** (90700; batched-prime MATCH) | `gates/agentworld-q4km-chat-sanity.log` |
| chat sanity (MEMRA_CHAT=1, NGEN=250, real prompt) | **clean** — correct ChatML + `<think>` tail (`<|im_end|>\n<|im_start|>assistant\n<think>\n`, token 248068 present — the Bonsai template-bug class is absent), coherent structured on-topic output, no looping | same log |
| resident-if-fits decision | `[moe] resident-experts decision: experts 19.57GB + trunk 2.56GB vs free 23.92GB (expert budget 19.37GB) -> SLRU cache` — the bank misses residency by 0.20GB with the 332MiB co-resident embedding server on the card | both gate logs |

Gate-run speed readings (SINGLE RUNS, cold SLRU, NOT board numbers): pp22 148.6 tok/s,
pp302 128.2 tok/s, decode 69.9 tok/s.

## Stage 3 — own-gen trimmed drafter (donor-block regime, `docs/DRAFT-REGIME.md`)

AgentWorld ships NO NextN/MTP head (40 blocks / 733 tensors — metadata receipts at
onboarding), so the drafter is the donor-block variant, Ornith-35B recipe 1:1
(`research/ornith-drafters-20260801/RECIPE.md`): donor = Qwen3.6-35B-A3B-UD-IQ4_XS
blk.40 (byte-verbatim, law 2), ranks = AgentWorld's OWN generations (law 1, 32768
protocol, canonical 254-prompt pack, chat template ON, bounded 64-prompt flock chunks),
quantize AFTER trim (NVFP4 head + Q4_K_M block, law 3).

- corpus: 254/254 prompts, **129,578 generated tokens** (4 bounded 64-prompt flock
  chunks, greedy ≡ single-run; small-corpus warning at the same level the supported and
  Ornith builds accepted — Ornith-35B ran 128,617). Log: `corpus/agentworld-owngen.log`,
  ids manifest `corpus/agentworld-owngen-ids.txt` (kept on /data next to the model).
- ranks: `owngen-ranks-32768.gguf(.txt)`, ranks.txt sha256 `fd937bf5...`; drafter:
  `draft-agentworld-owntrim-nvfp4head-q4blk.gguf` (890 MiB), sha256 `e3ee8c8b...`
  (`build-agentworld-draft.log`).
- run-spec K=1..8 self-consistency (p1, ngen 128): **PASS 8/8, acceptance>0 every K**
  (`gates/drafter/gate-k1-8.log`): K1 91.0% K2 74.5% K3 62.9% K4 50.6% K5 43.5%
  K6 37.6% K7 33.5% K8 29.3%.
- acceptance table (greedy, ngen 256, board prompts; single runs per cell — greedy
  acceptance is deterministic per (prompt,K)), vs the Ornith-35B donor-block reference:

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | **73.8% / 1.10x** | **78.8% / 1.08x** | **88.6% / 1.12x** |
| 3 | 58.4% / 0.96x | 67.5% / 0.99x | 74.7% / 1.04x |
| 4 | 48.0% / 0.83x | 57.7% / 0.89x | 60.3% / 0.90x |

Ornith-35B reference (same donor, same recipe, 2026-08-01): K2 65.9%/1.39x,
63.8%/1.11x, 63.8%/1.00x. AgentWorld ACCEPTANCE is higher on every cell (+7.9 to
+24.8 pts — the AgentWorld post-train sits closer to the Qwen3.6-35B donor's
distribution), but the spec/plain RATIO is lower on p1 (1.10x vs 1.39x): AgentWorld's
plain decode base runs the 19.57GB expert bank through the SLRU spill cache (bank
misses residency by 0.2GB), and each spec verify round widens the per-step expert
working set — the spill path compresses the speedup that the higher acceptance would
otherwise buy. Per-class best K = **2** everywhere (the q35-family serving K).

## Stage 4 — bar cells (N=3 medians, interleaved same-session, temps 65–84 °C per-row in `aw-cell.jsonl`)

Harness `run-bar-cell.sh` (o9b-cell shape: llama→memra pairs, rep loop outside class
loop, every GPU run under the flock). llama per-class best on this NextN-less GGUF =
plain (`llama-completion`, `-ngl 999 -fa on -ctk q8_0 -ctv q5_1`, greedy, --ignore-eos,
256 new tokens; build 9839 bb090d1f1) — its draftless spec doors are structurally broken
on the qwen35 M-RoPE arch (screen receipts
`research/ornith-bar-20260802/llama-spec-doors-screen.md`), and llama has no AgentWorld
draft artifact. memra best = the gated drafter at swept K=2, naked otherwise. memra
plain rides the same run-spec invocation's in-process `[generate]`. e2e = prime/prompt-
eval wall + 256/decode (the o9b-cell convention). Raw: `aw-cell.jsonl` (117 rows),
`aw-{llama-plain,memra-spec}-*-rep{1..3}.log`, console `aw-cell-console.log`, medians
`summary-output.txt`.

| class | metric | memra plain | memra best (spec K=2) | llama best (plain) | plain ratio | best ratio | 1.1x bar |
|---|---|---|---|---|---|---|---|
| p1-code-short (27 tok) | decode tok/s | 72.08 | **79.61** (acc 73.8%) | 155.83 | 0.46x | 0.51x | |
| | prefill tok/s | 117.9 | 120.5 | 130.9 | 0.90x | 0.92x | |
| | e2e s (256 tok) | 3.781 | 3.440 | **1.910** | 0.51x | **0.56x** | **FAIL** |
| p2-code-medium (1845 tok) | decode tok/s | 63.28 | **68.32** (acc 78.8%) | 153.91 | 0.41x | 0.44x | |
| | prefill tok/s | 123.5 | 122.6 | 2296.4 | 0.05x | 0.05x | |
| | e2e s | 18.981 | 18.819 | **2.393** | 0.13x | **0.13x** | **FAIL** |
| p3-agentic-long (6257 tok) | decode tok/s | 58.72 | **65.51** (acc 88.6%) | 148.21 | 0.40x | 0.44x | |
| | prefill tok/s | 118.3 | 118.0 | 2951.0 | 0.04x | 0.04x | |
| | e2e s | 57.207 | 56.920 | **3.837** | 0.07x | **0.07x** | **FAIL** |

memra self-consistency 9/9 PASS (spec ≡ plain tokens every run). Rep spreads: memra spec
within ±2.8% (p1) / ±3.0% (p2) / ±0.4% (p3); llama within ±2.7% every class.

## Root cause — Q5_K expert projections have no fast kernels (+ residency miss)

The gap is NOT the spill path alone and NOT the new mode-2 default. Probes (single runs,
logs committed):

1. **Residency lever** (`probe-resident-p2.log`): `MEMRA_MOE_RESIDENT_HEADROOM_GB=1.7`
   (the measured board-shape headroom, docs/FLAGS.md) flips the decision line to
   RESIDENT — budget 19.67 > bank 19.57 — and runs clean with the drafter loaded (no
   OOM, p2 spec K=2, acceptance 79.3%, PASS). Decode moves 63.3 → **93.8** plain /
   68.3 → **104.5** spec (+48/53%). **Prefill does not move: 126 tok/s resident vs 123
   spilled.** The naked decision spills because AgentWorld's trunk is 0.9GB fatter than
   Ornith-35B's (2.56 vs ~1.7GB) — the bank misses the naked budget by 0.20GB.
2. **Mode seam** (`probe-mode3-slru-p2.log`): `MEMRA_MOE_F16G=3` (the pre-2026-08-02
   AUTO-KQUANT default) reads pp1845 = 115.2 — identical to mode 2's ~120. The mode-2
   flip is exonerated. (Mode 3 + forced-resident OOMs — captured
   `CUDA_ERROR_OUT_OF_MEMORY`, `probe-mode3-resident-p2.log` — the AUTO-KQUANT f16
   workspace does not fit beside a fully-resident 19.57GB bank; not a viable config.)
3. **The actual wall** (`aw-tensor-dump.txt` + code): the UD-Q4_K_M artifact quantizes
   `ffn_down_exps` at **Q5_K in 37 of 40 layers** (3 Q6_K; gate/up all Q4_K). Every
   fast expert path admits Q3_K/Q4_K/Q6_K/IQ/NVFP4 but **not Q5_K**:
   `f16g_proj_ok` (grouped-f16 prefill door, `hybrid_forward.rs:233`),
   `q8_expert_supported` (fused q8 decode arm, `:216`), `expert_dp4a_supported`
   (`:195`), and the kquant direct-tile visitor loaders
   (`research/kquant-tile-loaders-20260802/`, Q4_K/Q6_K only). So 37/40 down
   projections ride the slow f32/staged arms in BOTH prefill and decode. Cross-check:
   Ornith-1.0-35B Q4_K_M (gate/up Q4_K + down Q6_K — fully covered) on this exact
   build reads pp512 3155 / decode 209 resident vs AgentWorld's 120 / 94 — same arch,
   same bank size, the only structural delta is the down-projection qtype.

## Stage 5 — verdict: **HOLD** (do not publish as supported)

- Drafter track: **gated and adopted-grade** — K=1..8 PASS, acceptance above the
  Ornith-35B donor-block reference on every cell, spec e2e ≥ plain on every class.
  The drafter is not the blocker.
- Bar track: best-vs-best e2e **0.56x / 0.13x / 0.07x** vs the 1.1x bar — FAIL on all
  three classes.

What publishing as supported needs (in priority order):

1. **Q5_K expert kernel coverage** — extend the kq direct-tile loaders + q8 decode arm
   to Q5_K (the exact work rounds 49 + kquant-tile-loaders did for Q4_K/Q6_K; Q5_K
   superblocks are the same 256-value k-quant family). Ornith-35B numbers on this build
   show the ceiling: decode 209 (1.08x llama), pp2048 1.26x llama — with the AgentWorld
   drafter's 74-89% acceptance on top, p2/p3-class 1.1x is plausible but must be
   re-measured.
2. **Residency**: naked resident-if-fits misses by 0.20GB on this rig with the 332MiB
   co-resident. `MEMRA_MOE_RESIDENT_HEADROOM_GB=1.7` is measured-safe here (no OOM incl
   drafter at p2; p3-depth unprobed) — a machine-config seam, not a default change.
3. **Or re-pick the artifact**: the pinned repo also publishes `UD-IQ4_XS` (17.9GB
   class, mirrors the supported q35 daily). IQ4_XS experts are fully covered by
   today's fast paths (f16g door, q8 arm, iq_fast direct loaders) AND its ~17GB bank
   fits naked residency. If the demand data doesn't force Q4_K_M, the IQ4_XS artifact
   inherits the entire q35 fast stack by construction — likely the shortest path to the
   bar, at the cost of re-running stages 1-4 on the new file.

README In-bring-up → Supported move: NOT this lane (rides the next docs pass, and only
after the bar is green).

## Files

- `downloads.log`, `sha-verify.log` — stage 1; `run-gates.sh`, `gates/`,
  `gates-console.log` — stage 2; `gen-corpus-chunk.sh`, `corpus/`, `build-draft.sh`,
  `build-agentworld-draft.log`, `gates-drafter.sh`, `gates/drafter/`,
  `summarize-gates.py` — stage 3; `run-bar-cell.sh`, `aw-cell.jsonl`,
  `aw-*-rep*.log`, `summarize-cell.py`, `summary-output.txt` — stage 4;
  `probe-*.log`, `aw-tensor-dump.txt` — root-cause probes.
- Model-side artifacts (on /data, never committed): the gguf, `owngen-ranks-32768.gguf(.txt)`,
  `draft-agentworld-owntrim-nvfp4head-q4blk.gguf`, `corpus ids` manifest.
