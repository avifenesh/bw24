# GLM-5.2 MLA bring-up — increment 1 receipts (ground truth, fetched 2026-08-01)

Every fact below was fetched live on 2026-08-01 (no training knowledge). Pinned copies live in
this directory; upstream commits are recorded so line references stay resolvable.

## 1. Primary sources

| # | Source | Pin |
|---|--------|-----|
| S1 | `zai-org/GLM-5.2` HF `config.json` | `glm-5.2-config.pinned.json` (sha256 `185f93ee…d859a`), repo revision `b4734de4facf877f85769a911abafc5283eab3d9` (HF API `sha`, lastModified 2026-07-02) |
| S2 | GLM-5 tech report, arXiv **2602.15763** ("GLM-5: from Vibe Coding to Agentic Engineering", v2 HTML) | §2.1 Architecture, §2.1.1 DSA, Appendix A Table 10 |
| S3 | HF API `https://huggingface.co/api/models/zai-org/GLM-5.2` | `safetensors: {"parameters": {"BF16": 753329921024, "F32": 19456}, "total": 753329940480}` |
| S4 | llama.cpp @ `ddd4ec1428a6201e18975ea52b07c71e0f9aef26` (master, 2026-08-01) | `src/models/glm-dsa.cpp` (pinned: `llamacpp-glm-dsa.cpp.pinned`, sha256 `7c3e910a…3db5dec`), `src/llama-arch.cpp`, `conversion/glm.py`, `conversion/deepseek.py`, `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` |
| S5 | vLLM @ `3986b967b249468fc1bd052f89a1eeeada6377b4` (main, 2026-08-01) | `vllm/model_executor/layers/attention/mla_attention.py` (the canonical absorbed-vs-naive writeup, ex-`mla/common.py`); sparse backends in `vllm/v1/attention/backends/mla/` |
| S6 | vLLM recipes page `https://recipes.vllm.ai/zai-org/GLM-5.2` | serving configs, param claim |
| S7 | GLM-5.2 model card `https://huggingface.co/zai-org/GLM-5.2` (README.md raw) | IndexShare claim, framework support list |
| S8 | arXiv **2603.12201** — "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" | the paper behind the model card's "IndexShare" link (note the name mismatch: README says IndexShare, paper title says IndexCache) |
| S9 | llama.cpp PR #19460 (`ngxson`) "model: support GLM MoE DSA arch" | initial GGUF arch; indexer initially unused, since implemented (S4 has a full DSA path) |

## 2. Pinned MLA dims (S1, confirmed by S2 Table 10)

| Param | Value | config.json key |
|---|---|---|
| hidden_size | 6144 | `hidden_size` |
| n_layers | 78 (+1 MTP/NextN layer, `num_nextn_predict_layers: 1`) | `num_hidden_layers` |
| n_heads | 64 (`num_key_value_heads` also says 64 — nominal; MLA decode is effectively MQA) | `num_attention_heads` |
| q_lora_rank | 2048 | `q_lora_rank` |
| kv_lora_rank | 512 | `kv_lora_rank` |
| qk_nope_head_dim | 192 | `qk_nope_head_dim` |
| qk_rope_head_dim | 64 | `qk_rope_head_dim` |
| qk_head_dim | 256 (= 192 + 64) | `qk_head_dim` |
| v_head_dim | 256 | `v_head_dim` |
| latent KV row | **576** = kv_lora_rank + qk_rope_head_dim (paper: "576-dimension latent KV-cache") | derived |
| rope | theta 8,000,000, `rope_type: "default"` (no yarn/scaling), **`rope_interleave: true`**, partial: 64 of 256 q dims | `rope_parameters`, `rope_interleave` |
| max ctx | 1,048,576 | `max_position_embeddings` |
| vocab | 154,880 | `vocab_size` |
| MoE | 256 routed / 8 used / 1 shared, moe_ffn 2048, dense_ffn 12288, first 3 layers dense, sigmoid + `noaux_tc`, routed_scaling 2.5, router f32 | `n_routed_experts` … |
| head_dim | 192 (present in config; NOT the attention head dim — matches `index_head_dim`-adjacent legacy field; qk_head_dim/v_head_dim are authoritative) | `head_dim` |

DSA (DeepSeek Sparse Attention) indexer (S1):

| Param | Value |
|---|---|
| index_n_heads | 32 |
| index_head_dim | 128 (64 rope + 64 nope inside each indexer head; `indexer_rope_interleave: true`) |
| index_topk | 2048 |
| indexer_types | 21 "full" layers: **0, 1, 2, then every 4th from 6 to 74**; remaining 57 "shared" (reuse previous full layer's top-k) — IndexShare (S7/S8) |
| index_topk_freq / index_skip_topk_offset | 4 / 3 (consistent with the full-layer stride/offset) |
| index_share_for_mtp_iteration | true |

## 3. GLM-5 paper attention facts (S2)

- §2.1 "Multi-latent Attention": MLA-576 latent cache; Muon Split (per-head orthogonalization)
  needed to match GQA-8 quality; **MLA-256**: "we increase the head dimension from 192 to 256 and
  decrease the number of attention heads by 1/3 … decreasing the decoding computation" (the H800
  roofline argument; DeepSeek-V3's 128 heads×192 was deemed wrong for other hardware).
- §2.1.1: DSA adopted via continued pre-training (dense warm-up 1000 steps training only the
  indexer, then ~20B-token sparse adaptation). "DSA reduces the attention computation by roughly
  1.5-2x for long sequences."
- Table 10 (Appendix A): GLM-5 = 3 dense + 75 MoE + 1 MTP layers, hidden 6144, QK head dim 192
  (nope), V head dim 256, Q LoRA 2048, KV LoRA 512, 64 heads, indexer 32 heads × 128,
  256 experts, vocab 154880 — **matches S1 field-for-field**.
- MTP: 1 shared MTP layer, re-applied iteratively for multi-step drafting (§2.1 "Multi-token
  Prediction with Parameter Sharing"); GLM-5 accept length 2.76 vs DeepSeek-V3.2's 2.55 (Table 2).

## 4. Parameter-count discrepancy (flagged per mission)

Conflicting published numbers — all recorded, none silently reconciled:

| Claim | Source | Quote/value |
|---|---|---|
| **744B total / 40B active** | S2 Table 10 + §2.1 ("This results in a 744B parameter model (40B active parameters)") | Counting rule stated in Table 10 caption: "we include the parameters of MTP layers but **not word embeddings and the output layer**" |
| **753,329,940,480 total** | S3 (HF safetensors metadata — counts every stored tensor) | 753.33B |
| **~743B total / 39B active** | S6 (vLLM recipe: "a ~743B-parameter MoE (39B active)") | conflicts with both above |
| **32B active** | Lambda inference page (via model-selection sweep 2, `research/model-selection-20260801/`) | wrong — 32B is GLM-4.5's active count per S2 Table 10 |

Config-derived arithmetic (from S1, this lane's own count): total ≈ 753.4B — embeddings+lm_head
1.90B + 3 dense layers ≈ 1.18B + 75 MoE layers ≈ 740.4B (9.87B/layer: attn 165M + 257 experts ×
37.75M + router) + 21 indexer layers ≈ 0.2B + 1 MTP layer ≈ 10B — **matches S3 (753.33B)**, so the
HF number is the full-tensor count and the paper's 744B is its stated exclusion rule (which still
under-counts by ~7B vs config arithmetic; the paper number is approximate). Active per token ≈
41.6B incl. embeddings/lm_head, ≈ 39.7B excluding — both "40B" (paper) and "39B" (vLLM) are
defensible roundings of the same model. **Resolution for memra planning: use config-derived
values; quote "744B-A40B (paper convention), 753.3B stored tensors" in any published claim.**
This resolves the open item flagged in `research/model-selection-20260801/synthesis-v2-8x-corrected.txt`
("active params conflict 32B/39B/40B — resolve in week 1").

## 5. llama.cpp ground truth (S4)

- Arch name: **`glm-dsa`** (`LLM_ARCH_GLM_DSA`, `src/llama-arch.cpp:84`); HF `GlmMoeDsaForCausalLM`
  registered in `conversion/glm.py:214` as `GlmMoeDsaModel(DeepseekV2Model)`. Size enum:
  `LLM_TYPE_744B_A40B` (glm-dsa.cpp `load_arch_hparams`, layer count 78/79).
- Rope type: **`LLAMA_ROPE_TYPE_NORM`** (`llama-model.cpp:2548` block) — interleaved adjacent-pair
  rotation, matching `rope_interleave: true`. NOT neox.
- KV cache: dedicated `llama-kv-cache-dsa.h`; main cache row = `[kv_cmpr(512) | k_pe(64)]`
  (Kcur concat, glm-dsa.cpp:~470), V = first-512 view of the same row. Indexer keys get their own
  cache ("lid"), stored post-rope + post-Hadamard.
- Absorbed decode (glm-dsa.cpp "MLA attention" block): `q_nope_absorbed = wk_b · q_nope` per head →
  Q = [absorbed(512) | q_pe(64)] = 576; "MLA with the absorption optimization converts into MQA";
  `wv_b` passed into `build_attn` for post-attention decompression (512 → 256 per head).
- Softmax scale: `kq_scale = mscale²/sqrt(n_embd_head_k)` with n_embd_head_k = 256 and mscale = 1
  (no yarn, freq_scale 1) → **1/sqrt(256) = 1/16** — the ORIGINAL qk head dim, not 576.
- DSA indexer graph: `indexer_q = indexer.attn_q_b(q_a_norm(wq_a(x)))` (32×128 from the q latent),
  `indexer_k = k_norm(indexer.attn_k(x))` (128, single head), both split rope(64,@offset 0)/nope(64),
  roped `LLAMA_ROPE_TYPE_NORM`, concat(pe,nope), Hadamard-rotated; `indexer_weights = indexer.proj(x)`
  scaled 1/sqrt(128·32); score = Σ_heads w_h·ReLU(q_h·k_t), masked, `ggml_top_k` 2048 → sparse index
  into `build_attn`. Shared layers assert-reuse `prev_top_k`. Comment: "Difference vs Deepseek 3.2:
  shared indexer layers reuse the top_k from the previous full indexer layers."
- MTP context: NextN layer runs **dense MLA (no DSA indexer)** with a plain KV cache
  (llama-model.cpp:2091 block).
- GGUF metadata written by the converter (conversion/deepseek.py `set_gguf_parameters`, inherited +
  conversion/glm.py additions):
  `attention.key_length = kv_lora_rank + qk_rope_head_dim = 576`, `attention.value_length = 512`,
  `attention.key_length_mla = 256`, `attention.value_length_mla = 256`,
  `attention.q_lora_rank = 2048`, `attention.kv_lora_rank = 512`, `rope.dimension_count = 64`,
  `attention.indexer.head_count = 32`, `attention.indexer.key_length = 128`,
  `attention.indexer.top_k = 2048`, `attention.indexer.types = [bool; 78]` (true = full),
  `nextn_predict_layers = 1`, plus the deepseek-style MoE keys.
- `kv_b_proj` is **split at conversion** (conversion/deepseek.py:415-431): view
  `(n_head, nope+v, kv_lora)` → `attn_k_b` = nope slice **transposed** to (n_head, kv_lora, nope)
  and `attn_v_b` = v slice (n_head, v, kv_lora); comment "MLA with the absorption optimization,
  needs these two split and k_b_proj transposed". `attn_kv_b` (unsplit) also kept in the arch
  tensor list.
- GLM-5.2 GGUF availability: unsloth/GLM-5.2-GGUF exists (model card S7 lists llama.cpp-family
  runners); community REAP-pruned GGUFs exist (0xSero/GLM-5.2-REAP-504B-GGUF,
  pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF). Early stock llama.cpp could not load GLM-5.2 because
  the loader required indexer tensors on every layer while 5.2 ships them only on "full" layers —
  since fixed upstream (S4 has `indexer_types` metadata + BC defaults table
  `GLM_5_2_DEFAULT_INDEXER_TYPES`). NO weights were downloaded in this lane (410GB+ class).

## 6. vLLM ground truth (S5)

`mla_attention.py` header (the definitional reference for both forms; DSV3 dims in comments):

- "Compute Friendly Approach (forward_mha)" — decompress `k_nope = kv_c @ W_UK`, `v = kv_c @ W_UV`,
  run MHA at qk head dim P+R. Used for prefill (Sq/Skv near 1).
- "Data-Movement Friendly Approach (forward_mqa)" — `ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)`
  then **MQA with QK headdim = Lkv + R**, V headdim = Lkv, and
  `o = einsum("snl,lnv->snv", sdpa_o, W_UV)` after attention. Used for decode.
  Quote: "computes the same outputs … but is more data-movement friendly since its MQA vs MHA".
- Sparse MLA backends exist per-vendor: `flashmla_sparse.py`, `flashinfer_mla_sparse.py`,
  `flashinfer_mla_sparse_sm120.py` (sm_120 sparse MLA exists upstream — relevant to the 5090 rig
  later), `indexer.py` (the DSA lightning indexer).
- Recipe (S6): FP8 checkpoint targets 8xH200/8xH20; full 1M ctx needs 8xB200 with fp8 KV. MTP
  extended to **5 draft tokens** (`--speculative-config.num_speculative_tokens 5`). H100 absent
  from every official recipe (confirms the model-selection moat argument).

## 7. Known-good conflicts / open items

1. Param counts: see §4 — recorded, resolved for planning purposes.
2. "IndexShare" (model card) vs "IndexCache" (arXiv 2603.12201 title) — same mechanism, two names.
3. `head_dim: 192` in config.json is a red herring vs `qk_head_dim: 256`/`v_head_dim: 256`
   (transformers' generic `head_dim` field; the MLA-specific keys are authoritative — llama.cpp
   ignores it and uses key_length_mla/value_length_mla).
4. z.ai blog (https://z.ai/blog/glm-5.2) is JS-rendered and returned empty to readability
   extraction — not used as a source.
5. GLM-4.7-Flash (per S2 §2.1.2, an MLA model used for small-scale DSA ablations) may be a
   small-scale MLA bring-up vehicle; size/arch unverified — verify before relying on it.
   DeepSeek-V2-Lite (15.7B, kv_lora 512, deepseek2 GGUFs abundant) is the fallback dev vehicle.
