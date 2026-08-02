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

## Stage 4 — bar cells

PENDING (`run-bar-cell.sh`, o9b-cell shape: interleaved same-session llama/memra pairs,
N=3, rep loop outside class loop; llama per-class best on this NextN-less GGUF = plain —
its draftless spec doors are structurally broken on the qwen35 M-RoPE arch, screen
receipts `research/ornith-bar-20260802/llama-spec-doors-screen.md`).

## Verdict

PENDING
