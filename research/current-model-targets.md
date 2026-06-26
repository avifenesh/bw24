# Current Model Targets (verified on-disk 2026-06-26)

Prior ARCHITECTURE.md assumed Qwen3-8B dense + 30B-A3B MoE. STALE. Real current local models:

## Primary targets — user's stated daily drivers (priority order)

> User (2026-06-26): "my main target used daily is the dense 27b and the judge 9b, but working with
> other. not plain transformer, not only moe."

### Design stance: MULTI-ARCH, do NOT lock on hybrid
> User (2026-06-26): "but dont lock on hybrid, i use a lot of different models, and do a lot of work
> with models. Yes have to support mtp and eagle."

The engine is a **model-agnostic multi-arch runtime** with a pluggable ARCH REGISTRY. It must run:
- **plain dense transformers** (qwen3, llama, gemma, … — vanilla softmax attention),
- **MoE** (qwen3moe, gpt-oss, mixtral, gemma-MoE),
- **hybrid linear-attn** (qwen35/qwen3-next: gated-deltanet + periodic full-attn).
Plus **speculative decoding as a first-class cross-cutting feature: MTP (built-in NextN) AND EAGLE
(target EAGLE 3.1 — the current version, per user 2026-06-26; local EAGLE3 drafts exist for qwen35/36).** No single arch is "the spine" — the spine is GGUF-load +
arch-dispatch + KV/state cache + scheduler + sampler; arch-specific forward graphs plug in.

The qwen35 HYBRID path is the FIRST hard arch to implement because the daily-driver 27B + 9B use it,
but it is implemented as one registry entry behind a common Forward trait, alongside a vanilla-dense
entry and a MoE entry. Build order favors a vanilla-dense arch first (simplest, proves the spine),
then hybrid, then MoE.

| Priority | Model | Quant on disk | Arch | Fits 24GB? |
|---|---|---|---|---|
| **P0 daily** | **Qwen3.6-27B dense** | **NVFP4 5.6G**, Q4_K_M 15G | qwen35 HYBRID | yes (resident) |
| **P0 daily** | **Qwen3.5-9B judge** | Q8_0 8.9G, **NVFP4 5.3G**, f16 17G | qwen35 HYBRID | yes (resident) |
| P1 working | Qwen3.6-35B-A3B MoE | IQ4_XS 17G, Q6_K_XL 31G (SPILLS) | qwen35moe HYBRID+MoE | partial → spill |
| bring-up only | Qwen3-1.7B / 0.6B | HF | qwen3 vanilla dense | trivial |
| spec-decode | EAGLE3-qwen35/36 + built-in MTP | present | — | — |

NVFP4 GGUFs ALREADY EXIST for BOTH daily targets → direct fuel for the FP4 weapon. Both fit 24GB
resident at NVFP4, so the daily path is **fully-resident single-stream decode of a hybrid model** —
that is THE thing to make fastest. MoE/spilling is the secondary capability, not the headline.

## CRITICAL: Qwen3.5/3.6 is a HYBRID linear-attention arch (Qwen3-Next lineage), NOT dense

Verified from the Q8_0 model metadata + llama.cpp `src/models/qwen35.cpp`:
- 32 blocks, `qwen35.full_attention_interval = 4`.
- **8 FULL-ATTENTION layers** (every 4th, i.e. il where (il+1)%4==0): tensors `attn_q [4096,8192]`,
  `attn_k [4096,1024]`, `attn_v [4096,1024]`, `attn_output`, `attn_q_norm [256]`, `attn_k_norm [256]`.
  GQA 16 heads / 4 KV, head_dim (key/value_length) = 256, RoPE NEOX (rope.dimension_count 64 + 4 sections).
- **24 LINEAR-ATTENTION layers** (Gated DeltaNet / SSM): tensors `attn_qkv [4096,8192]`, `attn_gate [4096,4096]`,
  `ssm_conv1d [4,8192]`, `ssm_a [32]`, `ssm_alpha.weight [4096,32]`, `ssm_beta.weight [4096,32]`,
  `ssm_dt.bias [32]`, `ssm_norm [128]`, `ssm_out [4096,4096]`.
  SSM params: conv_kernel 4, state_size 128, group_count 16, inner_size 4096, time_step_rank 32.
- Every block also has `attn_norm` + `post_attention_norm` (F32) and a standard SwiGLU FFN
  (`ffn_gate/up [4096,12288]`, `ffn_down [12288,4096]`).
- **MTP / NextN** layers present (multi-token-predict; built-in speculative decoding) — `nextn.*` tensors.
- Tokenizer: gpt2 BPE, pre="qwen35", vocab 248320, 247587 merges, eos 248046. Tied embeddings possible
  (output.weight may fall back to token_embd).

### Engine implications (big)
1. **KV cache grows for ONLY 8/32 layers.** Linear-attn layers keep a fixed-size recurrent state
   (conv state + SSM state), independent of context length → enormous 24GB / long-context advantage IF
   we implement the hybrid correctly. Competitors that treat it well (llama.cpp已支持) set the bar.
2. **Must build a Gated DeltaNet (gated delta rule) scan kernel** + a causal conv1d, in addition to
   flash attention. This is a NEW critical-path component — needs its own research vs llama.cpp
   `src/models/qwen35.cpp` forward, flash-linear-attention (fla), vLLM/SGLang Qwen3-Next support.
3. Full-attention layers still need FA-2-style mma.sync attention (per sm_120 findings) + QK-norm + GQA.
4. MTP gives built-in speculative decoding — a latency lever competitors may not exploit on this exact model.

### Bring-up plan adjustment
- Phase 1a: validate the GGUF→forward→logits spine on a **pure-dense** current model (Qwen3-1.7B,
  vanilla qwen3 arch) — softmax attention only, simplest correct path.
- Phase 1b: add the hybrid path (conv1d + gated-delta scan + interleaved full-attn) for the real
  Qwen3.5-9B target, logit-matched against llama.cpp.
- MoE (Qwen3.6-35B-A3B) and spilling come after the hybrid single-stream path is correct.

Source of truth for the forward graph: `~/projects/llama.cpp/src/models/qwen35.cpp` + `qwen35moe.cpp`.
