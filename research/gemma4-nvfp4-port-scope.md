# Gemma-4-26B-A4B NVFP4 (safetensors) port scope — 2026-07-07

Scope of porting `nvidia/Gemma-4-26B-A4B-NVFP4` through the NEW bw24 safetensors loader
path (the one that carried MiniMax-M3 and NVIDIA Qwen3.6-27B). Checkpoint at
`/data/ai-ml/hf-models/gemma4-26b-a4b-nvfp4/` (the HF-cache dir
`models--nvidia--Gemma-4-26B-A4B-NVFP4` is refs-only; the materialized snapshot lives in
the flat dir). Census = headers only (index + per-shard safetensors JSON headers), no
tensor data loaded, CPU-only.

Prior art: `research/tune-data/rig5090.jsonl` row `gemma4-26b-moe-gap-list` (2026-07-05,
g7e jsonl) — 9 hard gaps for the GGUF q4km route. This doc answers: which of those the
ST+NVFP4 route dissolves.

## 1. Checkpoint census

47,033 tensors, 18.78 GB, 2 shards. `total_parameters` 14.4B stored (25.2B logical — the
128-expert NVFP4 packing halves stored element count).

**Quant layout** (`hf_quant_config.json`, modelopt 0.43): `quant_algo: NVFP4`, group_size
16, kv_cache FP8 (informational — bw24 owns its KV quant). The `ignore` list excludes
`lm_head` (absent anyway — tied), every layer's `mlp*` / `router*` / `self_attn*`, and the
vision tower. **Net: ONLY the 3,840 routed-expert Linears are NVFP4; everything else is
BF16.** Exactly the modelopt encoding bw24 already imports:

| class | tensors | dtype | shape | bytes |
|---|---|---|---|---|
| expert gate/up (30L x 128E x 2) | 7,680 | U8 packed e2m1 | [704, 1408] (in_f 2816) | 11.42 GB total U8 |
| expert down (30L x 128E) | 3,840 | U8 packed e2m1 | [2816, 352] (in_f 704) | (incl above) |
| per-16 scales (x3 projections) | 11,520 | F8_E4M3 | [out, in/16] | 1.43 GB |
| `weight_scale_2` + `input_scale` | 23,040 | F32 scalar | () | ~0 |
| attn q/k/o (+v on 25 layers) | 115 | BF16 | per-layer, see below | 5.94 GB BF16 total |
| shared MLP gate/up/down | 90 | BF16 | [2112, 2816] / [2816, 2112] | (incl) |
| router proj/scale/per_expert_scale | 90 | BF16 | [128, 2816] / [2816] / [128] | (incl) |
| norms x7-8 per layer + layer_scalar | ~250 | BF16 | [2816] / [1] | (incl) |
| embed_tokens (tied lm_head) | 1 | BF16 | [262144, 2816] | 1.48 GB |
| vision tower + embed_vision | ~380 | BF16 | — | ~1.1 GB (SKIP: text-only port) |

NVFP4 alignment check: expert in_f 2816 = 64x44, down in_f 704 = 64x11 — both pass the
repack's in_f % 64 == 0 requirement. `weight_scale_2` per expert tensor → rides the
existing `ffn_act_scaled` macro-scale fold (M3 path). `input_scale` (activation scale) is
unused by the W4A16 path — same as NV-27B.

**Attention geometry** (config + shard headers agree):
- 30 layers, pattern `[5x sliding, 1x full] x5` — global layers = {5, 11, 17, 23, 29}.
- SWA layers (25): 16 Q-heads x hd 256 (q [4096,2816]), 8 KV-heads (k/v [2048,2816]),
  window 1024, rope theta 10k full-rot, q/k_norm [256].
- Global layers (5): 16 Q-heads x hd **512** (q [8192,2816]), **2** KV-heads
  (k [1024,2816]), **NO v_proj** (`attention_k_eq_v: true` — V IS the K projection,
  llama.cpp gemma4.cpp:246-249: `Vcur = wv ? mm(wv,cur) : Kcur` then weightless
  `ggml_rms_norm(Vcur)`), rope theta 1M "proportional" p-RoPE with
  `partial_rotary_factor 0.25` (128 of 512 dims), q/k_norm [512], o [2816,8192].
- `num_kv_shared_layers: 0` (no cross-layer KV reuse in this config — the llama.cpp
  `has_kv` machinery is for other gemma4 sizes).

**MoE**: 128 experts, top-8, expert_ff 704, softmax gating with weight renorm
(llama.cpp: `GATING_FUNC_SOFTMAX`, norm_w=true — the exact qwen3moe recipe bw24's host
router already implements), NO selection bias, NO sigmoid. Router prologue is custom:
`logits = router.proj @ (rms_norm(attn_out) * 1/sqrt(2816) * router.scale)` — operates on
attn_out, not the branch-normed input. `router.per_expert_scale` [128] scales expert
contributions. Every one of the 30 layers is MoE (`enable_moe_block: true`), and every
layer ALSO has a dense shared-MLP branch (n_ff 2112) summed in parallel — the "shared
expert" is a full parallel FFN branch, not a qwen35moe-style shexp.

**Block wiring** (llama.cpp gemma4.cpp:180-405): embed x sqrt(n_embd) → per layer:
attn → post_attention_layernorm → +residual = attn_out →
{pre_feedforward_layernorm → shared MLP (GELU-par) → post_feedforward_layernorm_1} +
{pre_feedforward_layernorm_2 → MoE (GELU) → post_feedforward_layernorm_2} → sum →
post_feedforward_layernorm → +attn_out residual → x layer_scalar → next layer.
Final: output_norm → lm_head(tied embed) → softcap 30 (scale, tanh, scale).

**Activation**: `gelu_pytorch_tanh`, parallel form `gelu(gate) * up` (LLM_FFN_GELU +
LLM_FFN_PAR) for BOTH shared MLP and experts.

**Tokenizer**: `tokenizer.json` present (32MB), model type **BPE** — but it's a
sentencepiece-style serialization: normalizer `Replace(" "→"▁")`, pre_tokenizer
`Split(" ", MergedWithPrevious)` (NOT ByteLevel), decoder `Replace(▁→" ") + ByteFallback
+ Fuse`. 262144 vocab, 24 added tokens. `chat_template.jinja` present.
bw24-tokenizer (lib.rs:254) hard-errors on non-ByteLevel pre_tokenizers.

## 2. Old GGUF gap list (9) — dissolved vs remaining

| # | GGUF gap | ST+NVFP4 status |
|---|---|---|
| 1 | fused `ffn_gate_up_exps` single tensor | **DISSOLVED** — ST stores separate `experts.{e}.{gate,up,down}_proj`, exactly the `hf_expert_name` gather path (needs only the name-pattern tweak, N3) |
| 2 | scale sidecars (`ffn_down_exps.scale`[128] + `ffn_gate_inp.scale`) | **REMAINS, smaller** — they are `router.per_expert_scale` (R3, trivial) + `router.scale` in the router prologue (R2, small); no new dequant epilogue (the GGUF framing was misleading — these are router-side, not dequant-side) |
| 3 | per-layer head_count_kv array + SWA pattern + dual rope | **REMAINS** — R5 (hard) + R6 (hard) + R9 (small) |
| 4 | V-less global layers (V=K + rms_norm) | **REMAINS** — R7 (small) |
| 5 | block wiring (dual post-norms, parallel branches, layer_scalar, embd scale) | **REMAINS** — R8 (hard-ish: new graph, zero new kernels) |
| 6 | final_logit_softcapping 30 | **REMAINS** — R4 (small) |
| 7 | GELU activation | **REMAINS** — R1 (small; the `ffn_act` seam is exactly swigluoai-shaped) |
| 8 | Q5_0 dequant | **DISSOLVED** — no GGUF qtypes anywhere; experts are modelopt NVFP4 (native import incl `weight_scale_2` → `ffn_act_scaled`, zero new work), rest BF16 |
| 9 | sentencepiece tokenizer | **MOSTLY DISSOLVED** — tokenizer.json BPE exists, no spm proto parsing; replaced by the much smaller metaspace/ByteFallback arm (N1) |

Score: 2 fully dissolved, 1 mostly dissolved, 1 shrunk, 5 remain.

## 3. Full gap list (what bw24 is missing NOW), tagged

Remaining from the GGUF list:

- **R1 [small]** `gelu_pytorch_tanh` parallel activation: one `gelu_tanh_mul(_scaled)`
  kernel + a `FfnAct` dispatch in `ffn_act_scaled` (hybrid_forward.rs:1038). Same seam
  swigluoai used; the `_scaled` variant carries the expert `weight_scale_2` folds.
- **R2 [small]** router prologue: `rms_norm(attn_out) * 1/sqrt(n_embd) * router.scale`
  before the router matmul. All primitives exist (rms_norm, scale, elementwise mul);
  host-router path first, fused-router kernel later.
- **R3 [trivial]** `router.per_expert_scale`[128]: fold into the selected routing weights
  post-renorm (`w[i] *= pes[sel[i]]`) in the host router.
- **R4 [small]** logit softcap 30: `logits = 30 * tanh(logits/30)` epilogue after lm_head.
  bw24 has no tanh op — one tiny kernel or host-side epilogue (logits already come host-side
  for sampling).
- **R5 [hard]** per-layer attention geometry: `n_head_kv`, `head_dim` become per-layer
  (8kv x hd256 SWA vs 2kv x hd512 global; q/k/o shapes differ per layer; q/k_norm sizes
  differ). Touches `ModelConfig` (scalars → per-layer accessors), KV-cache allocation
  (per-layer sizing already exists structurally for hybrid's mixed layers — extend to
  per-layer dims), attention kernel launches (head_dim 512 path must be checked against
  the FA/attention kernels' dim assumptions).
- **R6 [hard]** SWA masking: 25 layers, window 1024. v0 = full KV cache + windowed mask
  in the attention kernel (correctness, small); real ring-buffer SWA cache is the perf
  follow-up (matters at long ctx: 25/30 layers capped at 1024 entries is a huge KV win).
- **R7 [small]** V-less global layers: `load_mixer` makes `wv` optional (`load_opt`),
  forward uses K-projection output as V pre-rope, plus a WEIGHTLESS rms_norm on V
  (llama.cpp applies it on all kv-layers of gemma4 — verify vs HF modeling code whether
  SWA layers' V is also rms-normed; gemma4.cpp:255 says yes, unconditional).
- **R8 [hard]** block wiring: new gemma4 layer graph (post-attn norm, parallel
  shared-MLP + MoE branches each pre/post-normed, branch sum, final ffn post-norm,
  layer_scalar multiply, embed x sqrt(n_embd) prologue). No new kernels — rms_norm,
  matmul, add, scale, mul all exist — but it's a new `forward` arm, not a config switch
  on the qwen/M3 graph.
- **R9 [small]** per-layer-type rope: SWA theta 10k full-rot(256) vs global theta 1M
  partial-rot 128/512 "proportional" p-RoPE. bw24 has partial rope (M3 rotary_dim) and
  single freq_base; needs per-layer (base, n_rot) + the proportional freq-factor formula
  (GGUF route shipped a `rope_freqs` tensor; ST must compute it — llama.cpp
  `get_rope_freq_base` + freq_factors path is the reference).

New, ST-route-specific (all bounded):

- **N1 [small]** tokenizer metaspace arm: `Replace(" "→"▁")` normalizer, `Split`
  pre-tokenizer, `ByteFallback + Fuse` decoder in bw24-tokenizer. The BPE merge engine
  is reusable; this is a second pre-tok/decoder mode next to ByteLevel (~150-250 lines
  + tests against HF tokenizers output).
- **N2 [small]** config.json parsing: `Arch::Gemma4` + `model_type: gemma4/gemma4_text`,
  `layer_types[]` array, nested `rope_parameters` object (per layer-type), `global_head_dim`,
  `num_global_key_value_heads`, `top_k_experts`, `final_logit_softcapping`,
  `hidden_activation`, `sliding_window`, `attention_k_eq_v`. The flat JsonObj parser
  handles nested objects (text_config precedent); `layer_types` needs a string-array
  reader (u32_array precedent).
- **N3 [trivial]** hf_mapping gemma4 arm: experts live at `layers.{il}.experts.{e}.*`
  (directly under the layer, NOT `mlp.experts`); `router.proj` → ffn_gate_inp;
  `router.scale` / `router.per_expert_scale` / `layer_scalar` / the 5 extra norm names
  (`post_attention_layernorm` here is a true post-norm, unlike qwen's pre-FFN-norm
  aliasing — map carefully); tied lm_head already falls back to token_embd.
  The `model.language_model.` prefix fallback in `source.rs lookup()` already covers the
  wrapper namespace.
- **N4 [trivial]** gemma (1+w) norm fold: reuse `TransformKind::NormPlusOne` gated on the
  gemma4 arch (M3 `use_gemma_norm` precedent). VERIFY against HF gemma4 modeling code that
  all 8 norm classes (incl the weightless V-norm — N/A — and q/k_norm) are zero-centered.
- **N5 [trivial]** BF16 large-matrix re-encode: attn q/k/v/o + shared MLP are BF16 (~4.4GB
  text-side). The BF16→Q8_0 (or →NVFP4 under `BW24_NV_W4`) re-encode arm in source.rs:315
  is currently name-gated to `mtp.*`/embed — widen the gate (or add gemma4 names). Embed
  [262144,2816] BF16 1.48GB → Q8_0 0.79GB rides the existing embed arm.

Non-gaps (confirmed supported): NVFP4 expert import incl macro-scale (M3-proven,
`find_nvfp4_native` header-only `has`), 128-expert top-8 softmax+renorm routing
(qwen3moe-identical recipe), disk-tier ST experts (`st_dir` repack cache), tied
embeddings fallback, BF16 dequant, expert gather via per-expert names, chat template.
Vision tower: ignored by construction (engine only requests text ggml names).

## 4. Recommended port order

1. **Foundations (CPU-verifiable, no GPU):** N2 config + N3 mapping + N4 norm fold + N1
   tokenizer. Gate: a `minimax_name_mapping_against_index`-style test over the real shard
   index + tokenizer parity vs HF tokenizers on a text battery.
2. **Small seams:** R1 gelu_tanh, R2 router prologue, R3 per_expert_scale, R4 softcap,
   embed-scale + layer_scalar (part of R8's plumbing but trivial standalone ops), N5
   BF16 gate widening.
3. **Block graph (R8):** new gemma4 layer forward wiring the seams from (2). Gate:
   argmax oracle per-layer vs HF transformers CPU reference (the M3 bring-up recipe).
4. **Attention core (R5 + R7 + R9):** per-layer geometry, V=K global layers, dual rope.
   The hard center of the port.
5. **SWA (R6):** v0 windowed mask on full cache (correctness), ring cache as the perf arc.
6. Bench vs llama.cpp same-box reference (gemma4 q4km tg128 224.3, pp512 11234 from the
   g7e row; re-baseline on rig5090).

## 5. Verdict

**GO, staged.** The ST+NVFP4 route dissolves the two worst mechanical gaps (Q5_0, fused
gate_up) and shrinks the tokenizer gap from a sentencepiece port to a pre-tok/decoder arm.
The quant story is the best possible: 100% of the NVFP4 surface (all 11.4GB of experts) is
the already-proven modelopt import, and everything else is plain BF16. What remains is a
genuine new-arch port concentrated in three hard items — per-layer attention geometry,
SWA, and the parallel-branch block graph — plus ~11 trivial/small items on proven seams.
That's smaller than the MiniMax-M3 bring-up was (which added routing semantics + a new
activation + disk-tier all at once) and reuses its playbook end-to-end.

## 6. SCOPE REFRESH (2026-07-10) — post-format-decision, post-sampled-protocol

Everything above was scoped 2026-07-07. Changes since that affect this port:

**A. THE FORK (owner decision needed before work starts):** GGUF is now bw24's primary format
(FORMAT DECISION, 2026-07-10 — chosen on long-context serving reliability of the QWEN
checkpoints). The Gemma-4 NVFP4 official checkpoint is ST-ONLY (this doc's route). Options:
  (1) Port via ST anyway — the checkpoint class this route was scoped for; lands in the
      best-effort tier by current policy, but it is the only NVFP4-official Gemma. The Qwen
      ST long-context degradation was a CHECKPOINT property, not an ST-loader property — Gemma's
      trunk may not share it (unknown until the loop matrix runs on it).
  (2) Port the community GGUF (q4km) — primary-tier format, but re-inherits the 9-gap list
      incl Q5_0 dequant and the fused-expert tensor work this doc showed the ST route dissolves,
      AND a community quant lineage instead of NVIDIA's calibrated one.
  (3) Both, ST first (cheaper per this doc), GGUF after — the Qwen pattern.
  The port-order recommendation (§4) assumed ST; it survives under (1) or (3).

**B. Landed since scoping — costs DOWN:**
- Sampled-spec protocol (v0.10-v0.12): Gemma has NO MTP head — spec is N/A entirely; the
  plain-decode + pp cells are the whole board for it. Simplifies gating (no K=1..8, no
  acceptance; argmax + text-audit + loop-matrix only).
- BW24_MMQ_F8F4 exists: Gemma's 3,840 NVFP4 expert Linears could ride it (prefill class) —
  but the acceptance-shift law is moot (no spec), so f8f4 adoption is FREE of its usual risk:
  gate on argmax+text only. Potential prefill lever from day one.
- The trim/lineage laws, loop-matrix protocol, evidence-discipline harness rules: all apply
  directly; the loop matrix (2 prompts x 3 seeds) is the mandatory long-context gate given
  the vocab (262k) and sliding-window attention are new territory.

**C. Unchanged hard items:** R5 (per-layer kv-head array + SWA window-KV — THE headline kernel
work: windowed KV ring + masking in FA prefill/decode), R6 (dual rope), R8 (block wiring).
Estimate unchanged from §4-5; the fastest true path remains (1)/(3) ST-first.
