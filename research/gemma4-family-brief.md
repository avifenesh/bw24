# Gemma 4 family brief + bw24 port plan — 2026-07-10

## The family (released 2026-04-02, Gemini-3 lineage)

| model | class | active/total | ctx | notes |
|---|---|---|---|---|
| E2B | edge dense + PLE | ~2B eff | 128K | per-layer embeddings, multimodal incl audio |
| E4B | edge dense + PLE | ~4B eff | 128K | llama.cpp's ONLY merged MTP targets |
| 12B | dense, encoder-free multimodal | 12B | 256K | linear input projections replace encoders |
| 31B | dense flagship | 31B | 256K | server-grade-local |
| 26B-A4B | MoE 128-expert top-8 | 4B/26B | 256K | scoped in gemma4-nvfp4-port-scope.md |

**Shared architecture class:** hybrid attention interleaving sliding-window (1024) with global
layers 5:1, final layer always global; GLOBAL layers use unified K=V (no v_proj) + p-RoPE
(partial_rotary 0.25) + fewer KV heads at doubled head_dim; q/k norms everywhere; softmax-renorm
MoE gating (qwen3moe recipe) on the 26B.

**Checkpoints (all official Google, HF):** QAT Q4_0 GGUF for EVERY size *including the drafter
(MTP) models*; also qat-q4_0-unquantized (bf16-weights QAT) and w4a16 compressed-tensors; the
NVFP4 26B-A4B safetensors (nvidia) already on disk. QAT = Google-endorsed 4-bit quality — the
quality question is answered by the vendor, not by us.

## The competitive opening (SOTA serving state)

- Google's MTP drafters claim up to 3x, integrated-drafter design (our Qwen MTP class exactly).
- **llama.cpp MTP for Gemma 4 is immature**: merged only for E2B/E4B (PR #24282, 2026-06-08),
  llama-server-only (llama-speculative + llama-bench BROKEN for E-models), and the 26B MoE gets
  only **1.2-1.3x** from their impl. Their small-model spec multiplier history on our board
  (9B: 0.98x!) suggests the same weakness class here.
- bw24 brings: tuned MTP machinery (persistent draft KV, HPOST, frspec trims, PMIN0, batched
  rows verify, FA-v4), resident-MoE dev path (128-expert CSR dedup applies), and the whole
  measurement discipline.

## Port targets (24GB rig, board discipline: llama floor, >=1.1x bar per cell)

1. **26B-A4B qat-q4_0 GGUF + drafter** (~15GB, resident-experts fits): known gap list in
   gemma4-nvfp4-port-scope.md — SWA KV rings, dual attn geometry (25xSWA hd256/8kv +
   5xglobal hd512/2kv K=V), p-RoPE, router prologue (attn_out-based + per_expert_scale),
   parallel shared-MLP + MoE block. MoE machinery (CSR, router kernel, resident slab) ports.
2. **31B dense qat-q4_0 + drafter** (~17.4GB fits): dense flagship, likely the flagship cells.
3. **E4B qat-q4_0 + drafter** (small cell; the one place llama's MTP actually works = the
   honest fight).

Sources: ai.google.dev/gemma/docs/core, blog.google (gemma-4, mtp drafters), huggingface.co
google/gemma-4-*-qat-q4_0-gguf, ggml-org/llama.cpp PR #24282 + discussion #21975.

## P0 census (gemma-4-26B_q4_0-it.gguf, 2026-07-10)

- arch `gemma4`, 30L, n_embd 2816, ctx 262144, softcap 30, 128 experts top-8 (exp_ff 704) +
  parallel shared FFN (ff 2112), layer_output_scale per layer.
- **Per-layer attention arrays in metadata**: head_count_kv = [8,8,8,8,8,2]x5 (5:1 SWA:global),
  key/value_length 512 global / 256 SWA, sliding_window 1024, rope base 1M/10k + dims 512/256
  per layer type, **rope_freqs.weight tensor SHIPPED** (R9 simplifies: load, don't compute).
- attn_v on 25 layers only (K=V globals confirmed, R7).
- **Quant layout: Q4_0 x 265 (everything hot) + F32 x 392 (norms/scales) + Q6_K x 1** —
  the new qmatvec_q4_0_mmvq is THE kernel for this file.
- **ffn_gate_up_exps FUSED** (GGUF gap #1 returns on this route — expert loader must split or
  handle fused); ffn_gate_inp x2/layer (router proj + per_expert_scale), ffn_down_exps x2/layer.
- tokenizer.ggml.model = "gemma4" (neither gpt2 nor llama — N1 tokenizer arm confirmed, format
  to be derived from the gguf token arrays).
- Drafter = SEPARATE repo (main repo carries model + mmproj only) — locate + download next.
