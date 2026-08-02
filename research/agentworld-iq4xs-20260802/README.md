# Qwen-AgentWorld-35B-A3B UD-IQ4_XS — the re-pick: onboarding + drafter re-gate + deployment bar (2026-08-02)

Lane `lane/agentworld-iq4xs` (from `restructure/public-split`, 5819215e). Rig: RTX 5090
Laptop 24.5GiB. Every GPU run under `flock /tmp/gpu5090.lock` (two co-lanes share the
rig; the co-resident `llama-server --embedding` at 332 MiB is allowlisted and
untouched). Build: `nice cargo build --release`, sm_120a auto-detected.

Context: `research/agentworld-20260802/` HOLDs the UD-Q4_K_M artifact on a Q5_K
expert-coverage hole (ffn_down_exps Q5_K in 37/40 layers, outside every fast expert
path) plus a 0.20GB residency miss. This lane executes that README's option 3: re-pick
**UD-IQ4_XS from the same pinned repo/revision** — fully covered by today's fast paths
and small enough for naked residency. The drafter is artifact-independent at the ranks
level (built vs the Q4_K_M target's own-gen ranks; same model weights) — its gates
re-ran against the new artifact below.

## Stage 1 — artifact

| field | value |
|---|---|
| HF repo | unsloth/Qwen-AgentWorld-35B-A3B-GGUF |
| revision (same pin as the Q4_K_M lane) | `3a305abf5cfd119ee999dfe929c433746edd8d63` |
| file | `Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf` (mirrors the supported q35 daily pick) |
| size | 17,785,036,032 bytes |
| sha256 | `ff4201b0c163950dc96aeaca033398543a1d62513ddc1c4030f9b94823764e06` — **verified vs the HF LFS pointer** (`sha-verify.log`) |
| local path | `/data/ai-ml/hf-models/agentworld-35b-gguf/` |

Download transcript: `downloads.log`. Disk pre-check: /data 711G free.

Coverage receipt (`aw-tensor-dump.txt`): expert mix is gate/up **IQ3_S x39 + IQ4_XS x1**,
down **IQ4_XS x37 + Q6_K x3** — byte-for-byte the q35 daily UD shape that round 49
covered. Every projection is admitted by `f16g_proj_ok` (IQ4_XS/IQ3_S/Q3_K/Q4_K/Q6_K),
`q8_expert_supported`, `expert_dp4a_supported`, and the direct tile loaders
(Q4_K/Q6_K/IQ4_XS/IQ3_S). No Q5_K anywhere. Full coverage by construction, as priced.

## Stage 2 — onboarding gates (logs in `gates/`, console `gates-console.log`)

| gate | result | log |
|---|---|---|
| kernel-check (once per branch build) | **ALL GREEN** | `kernel-check.log` |
| run-gen argmax pp22 (MEMRA_CHAT=1) | **MATCH** (prefill==decode 90700; batched-prime MATCH) | `gates/agentworld-iq4xs-argmax-pp22.log` |
| run-gen argmax pp302-class depth | **MATCH** (90700; batched-prime MATCH) | `gates/agentworld-iq4xs-chat-sanity.log` |
| chat sanity (MEMRA_CHAT=1, NGEN=250, real prompt) | **clean** — correct ChatML + `<think>` tail (token 248068 present; Bonsai-class template bug absent), coherent structured on-topic review, no looping | same log |
| resident-if-fits decision | `[moe] resident-experts decision: experts 15.22GB + trunk 2.56GB vs free 23.93GB (expert budget 19.37GB) -> RESIDENT` — **naked residency**, 4.15GB under budget, with the 332MiB co-resident embedding server on the card | both gate logs |

Gate-run speed readings (SINGLE RUNS, resident, NOT board numbers): pp302 2780.7 tok/s,
decode 180.96 tok/s — vs the Q4_K_M lane's 128.2 / 69.9 on the same rig same day.

## Stage 3 — drafter re-gate vs the new artifact (`gates/drafter/`)

Drafter unchanged: `draft-agentworld-owntrim-nvfp4head-q4blk.gguf` (sha256 `e3ee8c8b...`,
built 2026-08-02 vs the Q4_K_M target's own-gen ranks — same weights, ranks transfer).

- run-spec K=1..8 self-consistency (p1, ngen 128): **PASS 8/8, acceptance>0 every K**
  (`gates/drafter/gate-k1-8.log`): K1 84.1% K2 69.8% K3 60.7% K4 51.2% K5 42.4%
  K6 38.5% K7 33.0% K8 27.8%.
- acceptance table (greedy, ngen 256, board prompts; single runs per cell — greedy
  acceptance is deterministic per (prompt,K)), vs the Q4_K_M-artifact reference:

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | **65.9% / 1.44x** | **80.6% / 1.58x** | **87.1% / 1.59x** |
| 3 | 60.1% / 1.51x | 65.9% / 1.54x | 77.1% / 1.62x |
| 4 | 50.9% / 1.36x | 58.1% / 1.45x | 66.1% / 1.51x |

Q4_K_M reference at K=2 (same weights, same drafter): 73.8% / 78.8% / 88.6% —
acceptance transfers as expected (-7.9 / +1.8 / -1.5 pts; the quantization delta moves
individual greedy token paths, not the distribution match). The spec/plain RATIO jumps
1.10/1.08/1.12x -> 1.44/1.58/1.59x: the resident bank returns the speedup that the
Q4_K_M spill path compressed. Serving K stays **2** (the q35-family K; K=3 reads
+1..5% on p1/p3 in single runs — within/near single-run spread, not adopted).

## Stage 4 — bar cells (N=3 medians, interleaved same-session, temps 60–74 °C per-row in `aw-cell.jsonl`)

Same harness shape as the Q4_K_M lane (`run-bar-cell.sh`, o9b-cell convention: llama→
memra pairs, rep loop outside class loop, every GPU run under the flock; e2e =
prime/prompt-eval wall + 256/decode). llama per-class best on this NextN-less GGUF =
plain (`llama-completion`, `-ngl 999 -fa on -ctk q8_0 -ctv q5_1`, greedy, --ignore-eos,
256 new tokens; build 9839 bb090d1f1). memra best = the gated drafter at K=2, naked
otherwise; memra plain rides the same run-spec invocation's in-process `[generate]`.
Raw: `aw-cell.jsonl` (126 rows), `aw-{llama-plain,memra-spec}-*-rep{1..3}.log`, console
`aw-cell-console.log`, medians `summary-output.txt`.

| class | metric | memra plain | memra best (spec K=2) | llama best (plain) | plain ratio | best ratio | 1.1x bar |
|---|---|---|---|---|---|---|---|
| p1-code-short (27 tok) | decode tok/s | 180.96 | **259.25** (acc 65.9%) | 163.30 | 1.11x | 1.59x | |
| | prefill tok/s | 771.4 | 794.1 | 230.5 | 3.35x | 3.45x | |
| | e2e s (256 tok) | 1.450 | **1.021** | 1.713 | 1.18x | **1.68x** | **PASS** |
| p2-code-medium (1845 tok) | decode tok/s | 174.02 | **273.67** (acc 80.6%) | 161.61 | 1.08x | 1.69x | |
| | prefill tok/s | 5507.5 | 5379.0 | 2877.7 | 1.91x | 1.87x | |
| | e2e s | 1.806 | **1.278** | 2.244 | 1.24x | **1.76x** | **PASS** |
| p3-agentic-long (6257 tok) | decode tok/s | 173.18 | **275.66** (acc 87.1%) | 150.64 | 1.15x | 1.83x | |
| | prefill tok/s | 5384.7 | 5389.3 | 3238.0 | 1.66x | 1.66x | |
| | e2e s | 2.640 | **2.090** | 3.652 | 1.38x | **1.75x** | **PASS** |

memra self-consistency 9/9 PASS (spec ≡ plain tokens every run). Rep spreads: memra
spec within ±0.9% (p1) / +2.9/−0.4% (p2) / ±0.6% (p3); memra plain within ±1.0%
every class; llama within ±2.6% every class. Ornith-1.0-35B same-build reference
(209 decode / 3155 pp512): AgentWorld IQ4_XS lands in the ceiling class as priced —
decode 181 plain (IQ3_S gate/up vs Ornith's Q4_K), prefill 5.4k at pp1845/pp6257 depth.

## Stage 5 — verdict: **DEPLOY** (1.1x bar green on all three classes)

- Best-vs-best e2e: **1.68x / 1.76x / 1.75x** vs llama — clears the 1.1x bar with
  margin on every class. PLAIN e2e alone also clears it (1.18x / 1.24x / 1.38x).
- Gates: kernel-check ALL GREEN, argmax MATCH both depths, chat sanity clean, drafter
  K=1..8 PASS, bar-cell self-consistency 9/9.
- Root-cause closure: the Q4_K_M HOLD attributed the loss to (a) the Q5_K coverage
  hole and (b) the 0.20GB residency miss. This artifact removes both and the same
  build/harness flips FAIL 0.56/0.13/0.07x -> PASS 1.68/1.76/1.75x. Attribution
  confirmed; no kernel work was needed.
- README In-bring-up → Supported move: rides the **next docs pass** (not this lane).
  The UD-Q4_K_M artifact stays receipted as the not-recommended quant — model-table
  line: "use IQ4_XS; the UD-Q4_K_M mix hits a Q5_K expert path outside current
  coverage."

## Files

- `downloads.log`, `sha-verify.log`, `aw-tensor-dump.txt` — stage 1; `run-gates.sh`,
  `gates/`, `gates-console.log`, `kernel-check.log` — stage 2; `gates-drafter.sh`,
  `gates/drafter/`, `summarize-gates.py` — stage 3; `run-bar-cell.sh`, `aw-cell.jsonl`,
  `aw-*-rep*.log`, `summarize-cell.py`, `summary-output.txt` — stage 4.
- Model-side artifacts (on /data, never committed): the IQ4_XS gguf, the shared
  drafter `draft-agentworld-owntrim-nvfp4head-q4blk.gguf` + ranks (Q4_K_M-lane builds).
