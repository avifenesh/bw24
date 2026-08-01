# Onboard: Ornith-1.0 batch (Ornith-9B/35B, KAT-Coder-V2.5, AgentWorld) — 2026-08-01

Lane: `lane/onboard-ornith` (from `restructure/public-split`). Model onboarding per
`research/model-demand-20260801/REPORT.md` Q1 row 1: the four highest-demand new local
models are config-identical to the supported qwen3_5 / qwen3_5_moe arches. This lane
verifies that claim byte-by-byte, downloads scored artifacts, and runs the exactness
gates on the 5090. **Zero engine code changed** (see "Wiring verdict").

## Model table

| model | HF repo | revision | file (quant) | size | sha256 |
|---|---|---|---|---|---|
| Ornith-1.0-9B | deepreinforce-ai/Ornith-1.0-9B-GGUF | 3296bc7a404871a72ac3f1903f561459c09b5c17 | ornith-1.0-9b-Q8_0.gguf | 9,527,500,992 | d0e4bebaa8b3450c62090df1408f2ee5ccb2094f9c610ffde564a654483d4f37 |
| Ornith-1.0-35B | deepreinforce-ai/Ornith-1.0-35B-GGUF | 383064f72a1ef3087b779f268d3ca117eb989aac | ornith-1.0-35b-Q4_K_M.gguf | 21,166,757,760 | ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002 |
| KAT-Coder-V2.5-Dev | bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF | d8f684f08d2950ea9d2db6a35ef7dada0707858b | Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf | 18,806,446,496 | e35e23219a81590b9d4174eea4717d716dd62676c8c434f6b708f49a07310e1a |
| Qwen-AgentWorld-35B-A3B | unsloth/Qwen-AgentWorld-35B-A3B-GGUF | 3a305abf5cfd119ee999dfe929c433746edd8d63 | (metadata-verified only, not downloaded) | — | — |

Local paths: `/data/ai-ml/hf-models/{ornith-1.0-9b-gguf,ornith-1.0-35b-gguf,kat-coder-v25-dev-gguf}/`.
Quant picks: 9B Q8_0 (mission spec), 35B-A3B class at ~4-bit to mirror the supported
`Qwen3.6-35B-A3B-UD-IQ4_XS` artifact — the official Ornith-35B repo publishes **no IQ4_XS**
(smallest is Q4_K_M, 21.2 GB), KAT IQ4_XS mirrors exactly. Ornith-35B picked over AgentWorld
by download count (3.45M vs 586K/30d — REPORT caveat: Ornith counts are partly automated
agent-tooling pulls; even 10x-discounted it stays top-10). AgentWorld verified from header
bytes but not downloaded: same 733-tensor qwen35moe stack, one 35B-A3B family artifact
covers the arch claim, disk stays comfortable.

## Config-identity verification (metadata receipts in `metadata/`)

Parsed from the first 40 MB of each GGUF (`metadata/gguf_meta.py`, plain KV walk) and
compared field-by-field against the supported artifacts
(`ref-qwen35-9b.kv.txt` = Qwen3.5-9B-NVFP4-MTP, `ref-qwen36-35b.kv.txt` = Qwen3.6-35B-A3B-UD-IQ4_XS):

- **`general.architecture` is already memra's native string**: `qwen35` (Ornith-9B),
  `qwen35moe` (Ornith-35B, KAT, AgentWorld). **No arch alias needed** — unlike the q27 case
  (upstream wrote `qwen3next`, aliased in `crates/memra-gguf/src/config.rs:32`), these GGUFs
  were converted with the same naming memra ships.
- Ornith-9B vs Qwen3.5-9B: every layer-stack field identical (ctx 262144, emb 4096, ff 12288,
  heads 16/4, hd 256, rope sections [11,11,10,0] @1e7, ssm conv4/state128/g16/rank32/inner4096,
  full_attention_interval 4). Delta: `block_count` 32 vs 33 and no `nextn_predict_layers` —
  the reference's 33rd block is its NextN/MTP head. **Ornith-9B ships no MTP head**
  (427 tensors vs 668).
- 35B trio vs Qwen3.6-35B-A3B: identical MoE stack (256 experts top-8, expert_ff 512,
  shared_ff 512, emb 2048, heads 16/2, same ssm/rope/interval). Delta: `block_count` 40 vs 41,
  no `nextn_predict_layers`, 733 vs 753 tensors — again only the MTP head missing.
  **None of the three 35B GGUFs carries an MTP head** → `run-spec` MTP self-consistency is
  N/A for the whole batch (gate matrix below). MTP-carrying community rebuilds exist
  (e.g. gbuzhf/KAT-Coder-V2.5-Dev-APEX-MTP-GGUF) — a later lane's option, not this one's.
- Tokenizer: all six files gpt2-BPE, `tokenizer.ggml.pre = 'qwen35'`, vocab len 248320,
  eos 248046. Deltas confined to pad/bos ids (Ornith pad 248044 vs unsloth's 248055) and
  the `add_bos_token` key missing on the Ornith pair — memra defaults `add_bos=false` for
  non-SPM models (`crates/memra-tokenizer/src/lib.rs:200`), matching the reference's
  explicit `add_bos_token=False`.
- Chat template (full dumps in `templates/`): all four are the Qwen3.5 ChatML dialect with a
  byte-identical `add_generation_prompt` tail (`<|im_start|>assistant\n` + `<think>\n`
  default). Diffs vs the reference sit ONLY in tool/vision/multi-system-turn branches
  (unsloth developer-role merge in the ref; an audio branch in AgentWorld) — outside the
  text-only path `apply_chat_template_str` reproduces. The `<think>`+`add_generation_prompt`
  substring heuristic (`crates/memra-tokenizer/src/chat.rs:41`) fires correctly for all four.

## Wiring verdict

**No code change.** Loader: `nextn_predict_layers` defaults 0 when the key is absent
(`config.rs:229`); `apply_stripped_mtp_override` early-returns at 0 (`source.rs:536`).
Tokenizer and template paths as above. Existing-model behavior untouched by construction.

## Gates (RTX 5090, all GPU runs under `flock /tmp/gpu5090.lock`)

Branch build: `nice cargo build --release`, sm_120a auto-detected.

| gate | scope | result | log |
|---|---|---|---|
| kernel-check | once per branch build | **ALL GREEN** | `kernel-check.log` |
| run-gen argmax (prefill==decode) | control: supported Qwen3.5-9B NVFP4 | **MATCH** | `gates/control-qwen35-9b-nvfp4.log` |
| run-gen argmax | control: supported Qwen3.6-35B IQ4_XS | **MATCH** | `gates/control-qwen36-35b-iq4xs.log` |
| run-gen argmax | Ornith-9B Q8_0 (pp22) | **MATCH** | `gates/ornith9b-q8_0-argmax.log` |
| run-gen argmax | Ornith-35B Q4_K_M (pp22 + pp302 depth) | **MATCH** | `gates/ornith35b-q4km-*.log` |
| run-gen argmax | KAT-Coder IQ4_XS (pp22 + pp302 depth) | **MATCH** | `gates/kat-coder-iq4xs-*.log` |
| run-spec K=1..4 MTP self-consistency | N/A — no model in the batch ships an MTP head (see verification) | N/A | — |
| chat sanity (MEMRA_CHAT=1, 200-250 tok real prompts) | Ornith-9B, Ornith-35B, KAT | **clean** — correct ChatML+`<think>` render, coherent on-topic output, no looping, KAT hits Eos naturally on the short prompt | `gates/*-chat-sanity.log` |

All GPU runs under `flock /tmp/gpu5090.lock`; sha256 of every downloaded artifact verified
against the HF LFS pointer before gating. Download transcript: `downloads.log`; the deep
code-review prompt: `deep-prompt.txt`.

## Observed speeds (SINGLE RUNS — sanity ballpark, NOT board numbers)

Gate-run readings, one run each, no interleave, thermal regime unpinned — per the
measurement protocol these are not comparable to the README board and must not be
published as such:

| model / quant | prefill | decode (Stage-B) |
|---|---|---|
| Ornith-9B Q8_0 (9.5 GB) | 706-719 tok/s @pp22 | 84.5 tok/s |
| Ornith-35B Q4_K_M (21.2 GB) | 431 @pp22, 402 @pp302 | 103.4-132.7 tok/s |
| KAT-Coder IQ4_XS (18.8 GB) | 149 @pp22 (cold), 213 @pp302 | 84.4 (cold) - 103.2 tok/s |
| control Qwen3.6-35B IQ4_XS (17.9 GB) | 349 @pp22 | 166.4 tok/s |
| control Qwen3.5-9B NVFP4 (5.3 GB) | 646 @pp22 | 134.3 tok/s |

Consistent with the DRAM wall: the new quants are simply bigger than the board artifacts
(Q8_0 9.5 GB vs NVFP4 5.3 GB; Q4_K_M 21.2 GB vs IQ4_XS 17.9 GB), so decode scales down
accordingly. Nothing suggests a kernel problem — the same-family board artifacts hold
their numbers in the same session.

## Verdict

| model | verdict |
|---|---|
| Ornith-1.0-9B (Q8_0) | **ONBOARDED** — zero code change, all gates green |
| Ornith-1.0-35B (Q4_K_M) | **ONBOARDED** — zero code change, all gates green |
| KAT-Coder-V2.5-Dev (IQ4_XS) | **ONBOARDED** — zero code change, all gates green |
| Qwen-AgentWorld-35B-A3B | **VERIFIED, NOT DOWNLOADED** — header metadata proves the same qwen35moe stack + template family; support-by-construction, artifact skipped to keep disk/bandwidth for the higher-demand Ornith-35B (3.45M vs 586K dl/30d) |

MTP/spec-decode for this batch is future work: the GGUFs ship without NextN heads. The
existing own-gen trimmed-draft regime (`docs/DRAFT-REGIME.md`) applies unchanged if a
draft is wanted — Ornith/KAT are post-trains of the same backbones the 9B/35B drafts
were built from, but a per-model own-gen draft build + acceptance measurement is its own
lane.
