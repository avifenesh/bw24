# gemma-4 E4B arch map (fork lane, 2026-07-11)

Files on disk (downloaded this session, 524G free on /data):
- main: `/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf` (5.2GB, QAT Q4_0,
  from `google/gemma-4-E4B-it-qat-q4_0-gguf`; mmproj skipped — no vision in scope)
- drafter: `.../drafter/MTP/gemma-4-E4B-it-assistant.{F16,Q8_0}.gguf` (174MB/100MB, from
  `AtomicChat/gemma-4-E4B-it-assistant-GGUF`, arch `gemma4_assistant`, requires_target gemma4)

## Shape (from GGUF metadata + llama.cpp src/models/gemma4.cpp — E4B is the SAME
## llama_model_gemma4 as 26B/31B; NO altup/laurel/matformer, those are gemma3n)

- 42 layers (LLM_TYPE_E4B), n_embd 2560, n_vocab 262144, DENSE ffn 10240 (GELU_PAR — the 31B arm)
- SWA pattern 5:1 `[T,T,T,T,T,F]*7`, sliding_window 512 (NOT 1024 like 26B)
- head dims: SWA hd 256 (8 q heads), global hd 512 (4 q heads); head_count_kv=2 (llama derives
  per-layer: SWA k/v out 512 = 2x256; global k/v out 512 — verify 1x512 vs 2x256 at bring-up)
- K != V everywhere kv exists (attn_v.weight present on all 24 kv layers — NO wv:=wk here)
- rope: 1e6 global + rope_freqs[256] freq-factors, 1e4 swa; softcap 30; tied Q6_K head; eps 1e-6
- layer_output_scale per layer (same as 26B/31B)

## NEW piece 1: KV SHARING (shared_kv_layers = 18)
- Layers 0..23 have own KV (attn_k/attn_v exist, 24x tensors); layers 24..41 have NO k/v
  weights — Q-only attention (wq + q_norm + rope per layer) over an EARLIER layer's cache:
  map (llama-model.cpp:2139): il >= 24 -> cache of layer `24 - (is_swa(il) ? 2 : 1)`
  i.e. shared SWA layers read layer 22's cache, shared globals read layer 23's.
- bw24 already has this MACHINERY CLASS: the 26B MTP drafter is Q-only attention over the
  main model's KV (gemma_spec draft_trunk: SWA->main L28, global->L29). Port = Cache carries
  Option<usize> share_target per layer; append skips shared layers; verify/decode attention
  reads the target layer's kvl (len bookkeeping shared automatically).

## NEW piece 2: PER-LAYER EMBEDDINGS (n_embd_per_layer = 256)
Model tensors: per_layer_token_embd [10752=256*42, 262144] Q6_K; per_layer_model_proj
[2560, 10752] F16; per_layer_proj_norm [256] F32.
Prologue (once per forward, gemma4.cpp build_inp_per_layer + project_per_layer_inputs):
  a = gather(per_layer_token_embd, tok) reshaped [256, 42] * sqrt(256)
  b = rms_norm(per_layer_model_proj^T . x_embd_scaled * (1/sqrt(2560)), per_layer_proj_norm)
      (reshaped [256, 42])
  inp_pl[il] = (a + b) * (1/sqrt(2))          # [256] per token per layer
Per-layer tail (AFTER the ffn residual, BEFORE layer_output_scale):
  g   = gelu(inp_gate^T . cur)                # blk.N.inp_gate [2560,256]
  y   = proj^T . (g * inp_pl[il])             # blk.N.proj [256,2560]
  cur = cur + rms_norm(y, blk.N.post_norm)    # blk.N.post_norm [2560]
All small (256-wide) — mmvq/matmul_pre cover it; ONE new gather kernel for the per-layer
token embd (row gather of 10752 Q6_K cols — embed_gather machinery exists, wider row).

## REUSABLE from the landed 26B/31B work
- dense gemma forward arm (31B), SWA/global fa incl. hd512 rows_dpl16 + i2 walk, parity-law
  device-len attention, graphs + rung buckets, q4rp split-plane mirrors (trunk is Q4_0!),
  tokenizer (arch string 'gemma4', same 262144 vocab), chat template, adaptive drafting.

## Drafter deltas vs the 26B drafter (defer to MTP phase)
gemma4_assistant, 4 layers, n_embd 256, backbone 2560, swa [T,T,T,F] window 512,
shared_kv_layers=4 (all four over MAIN kv, same as 26B wiring), K!=V (k_eq_v=false),
NEW: n_centroids 2048 / centroid_top_k 32 + use_ordered_embeddings — a centroid-gated
head (llama gemma4-assistant.cpp has the reference; NOT in the 26B drafter).

## Hardest parts, ranked
1. KV-share plumbing through Cache/verify/graph paths (bookkeeping, rewind on spec rounds).
2. Per-layer-embed prologue (gather + F16 matmul + reshape discipline; prefill T-wide).
3. Per-layer tail block (mechanical; small matmuls).
4. Drafter centroid head (new sampling machinery — MTP phase only).
5. Window=512 + hd512-global-single-kv-head geometry checks in fa dims (likely free).
