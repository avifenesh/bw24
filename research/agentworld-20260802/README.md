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

- corpus: `corpus/agentworld-owngen.log` + ids manifest — PENDING
- ranks + drafter shas: PENDING
- run-spec K=1..8 self-consistency: PENDING
- acceptance table K=2..4 vs the Ornith-35B reference: PENDING

## Stage 4 — bar cells

PENDING (`run-bar-cell.sh`, o9b-cell shape: interleaved same-session llama/memra pairs,
N=3, rep loop outside class loop; llama per-class best on this NextN-less GGUF = plain —
its draftless spec doors are structurally broken on the qwen35 M-RoPE arch, screen
receipts `research/ornith-bar-20260802/llama-spec-doors-screen.md`).

## Verdict

PENDING
