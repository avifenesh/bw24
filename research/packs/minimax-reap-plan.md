# MiniMax-M3 REAP + quant plan (bw24 big-MoE pipeline)

Status: 2026-07-04. Downloads in flight, recon done. This is the working plan for turning
MiniMax-M3 into a spill-serveable artifact for the 24GB rig.

## 1. Target

**MiniMaxAI/MiniMax-M3** (HF, released ~2026-06, arXiv 2606.13392) — the current MiniMax MoE.
There is no "MiniMax-3" repo; the M-series is the model line (M2 → M2.5 → M2.7 → M3). M3 is the
latest and the biggest: ~428B total / ~23B active. Note this SUPERSEDES shared-delta-moe's target
(M2.7, 230B/256-expert); M3 is a different architecture generation (fewer, fatter experts + sparse
attention + native multimodal).

### Architecture (from config.json + tensor index, verified)

| field | value |
|---|---|
| layers | 60 (3 dense-FFN + 57 MoE, `moe_layer_freq=[0,0,0,1...]`) |
| experts | **128/layer, top-4**, +1 shared expert per MoE layer |
| routing | **sigmoid** scoring + `e_score_correction_bias` (DeepSeek-V3 style) + `routed_scaling_factor=2.0` |
| hidden / expert-FF / shared-FF / dense-FF | 6144 / 3072 / 3072 / 12288 |
| attention | GQA 64 heads / 4 KV, head_dim 128, **QK-norm per-head**, partial RoPE (rotary_dim 64 = 0.5) |
| **MSA sparse attention** | per-GQA-group indexer (index_q/k_proj + norms), top-16 blocks × 128 = 2048-KV budget/query, layers 3..59; first 3 layers full attention |
| norm | `use_gemma_norm: true` (the (1+w) RMSNorm variant) |
| activation | `swigluoai` (clamped swiglu, alpha=1.702, limit=7.0 — GPT-OSS style) |
| vocab / context | 200064 / 1M |
| MTP | config says `num_mtp_modules: 7` but **no MTP tensors in released ckpt** (verified in index; llama.cpp PR 24908 confirms) |
| vision | CLIP-style tower, 515 tensors, ~1.3GB — **droppable for text-only serving** |
| expert tensor names | `block_sparse_moe.experts.{e}.w1/w2/w3` (Mixtral-style; w1=gate, w2=down, w3=up) |

### Checkpoints and sizes

| repo | format | size | note |
|---|---|---|---|
| MiniMaxAI/MiniMax-M3 | bf16 | **854 GB** | too big for our disk alongside anything else |
| **MiniMaxAI/MiniMax-M3-MXFP8** | FP8 (weight + weight_scale_inv) | **444 GB** | official quant — our DIY-REAP base. DOWNLOADING → `/data/ai-ml/hf-models/minimax-m3-mxfp8` |
| **sparkarena/Minimax-M3-v0-NVFP4-REAP50** | NVFP4, 64 experts/layer | **129 GB** | community REAP50+NVFP4 — instant baseline. DOWNLOADING → `/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50` |
| nvidia/MiniMax-M3-NVFP4 | NVFP4, full 128 experts | 250 GB | fallback quant base if MXFP8 handling is painful |
| unsloth/MiniMax-M3-GGUF | GGUF Q2..Q8/BF16/MXFP4 | UD-Q4_K_M ≈ 243 GB | made against the open llama.cpp M3 PRs; fastest "runs today" route for calibration passes |
| Inferact/MiniMax-M3-EAGLE3 | draft model | small | future spec-dec bonus |

Param math (from config): non-expert "always-resident" params (attn+norms+shared-experts+dense-FFN+embed+lm_head)
= **12.8B → ~7.7GB NVFP4**. Expert pool = 57 layers × 128 × 56.6M = **413B → ~248GB NVFP4**.
Prune ladder (total artifact @ NVFP4):

| prune | experts left | total params | NVFP4 artifact |
|---|---|---|---|
| 0% | 128 | 426B | ~250 GB |
| 50% | 64 | 219B | **~130 GB** |
| 62.5% | 48 | 168B | **~101 GB** |
| 75% | 32 | 116B | **~70 GB** |
| 87.5% | 16 | 64B | ~40 GB |

## 2. Loader verdict (the biggest open) — NEW LOADER ARC NEEDED, but bounded

The bw24 qwen35moe/olmoe loader **cannot** load M3 as-is. Honest gap list, ordered by size:

1. **MSA sparse attention — the one real arc.** New tensors (`index_q/k_proj`, `index_q/k_norm`)
   and a new attention algorithm (block score → top-k gather → attend over 2048 selected KV).
   MITIGATION: for total context ≤ 16×128 = 2048 tokens, top-16-of-≤16-blocks selects
   *everything* → plain full attention is **bit-equivalent**. So a v0 bw24 path can run M3 with
   the existing full-attention kernels, exactly, for ≤2K contexts, and defer MSA. (llama.cpp
   PR 24523 ships exactly this "no sparse attention yet" shape; PR 24908 has the full MSA impl
   to crib from — both open, actively updated.)
2. **Sigmoid routing + bias correction** — `moe_route` (hybrid_forward.rs:661) does softmax
   top-k. M3 needs: sigmoid(logits) + e_score_correction_bias for *selection*, plain sigmoid
   scores for *weights*, normalize, × routed_scaling_factor. Small, isolated change.
3. **Gemma-norm** — (1+w) RMSNorm variant; small kernel/epilogue tweak.
4. **swigluoai** — clamped SwiGLU (alpha 1.702, limit 7.0); new activation epilogue next to
   `silu_mul`. Small.
5. **HF name mapping** — `hf_mapping.rs` needs the `block_sparse_moe.experts.{e}.w1/w2/w3`
   branch (the comment at hf_mapping.rs:68 already anticipates a Mixtral-style arch) +
   `e_score_correction_bias` + shared_experts + index_* names + `language_model.` prefix strip.
   Mechanical.
6. **FP8 safetensors** — safetensors.rs *panics* on F8_E4M3 by design. Either dequant FP8+scale_inv
   on load (new, small), or sidestep by consuming GGUF (bw24 speaks GGUF natively; unsloth's
   GGUF uses arch string `minimax-m3` — config.rs Arch::parse needs the new arm either way).

Already covered by existing bw24 infra: partial RoPE (`rope_dim_count`), QK-norm, GQA, shared
expert (qwen35moe shexp path), dense-FFN layers, stacked-expert spill/SLRU cache, NVFP4 fast path.

**Verdict: no showstopper.** Items 2-6 are days-scale plumbing; MSA (item 1) is the real work but
has an exact short-context fallback, so the pipeline is not blocked on it. The 1M-context story
needs MSA; the "does the pruned artifact run on the rig" story does not.

## 3. Two paths to the artifact — run BOTH, compare

### Path A (shortcut): community REAP50 baseline — sparkarena NVFP4-REAP50, 129GB
- Already pruned 128→64 experts/layer + NVFP4-quantized. Zero GPU work to obtain.
- CAVEATS (from their card): "v0" calibration is admitted work-in-progress; NVFP4 uses
  nonstandard w1/w3 scale normalization (needs their sglang patch or a renormalization pass when
  we convert to GGUF); coherence tested, quality NOT quantified. Unknown REAP calibration corpus
  (likely generic, not our workload).
- Role: instant eval baseline + spill-bringup artifact. If its quality gates pass, it may simply
  BE the artifact and Path B becomes an optimization.

### Path B (DIY): REAP from MXFP8 with our recipe — full control
Reuse the shared-delta-moe CONCLUDED recipe (JOURNEY.md phases 59-63, prune_reap.py,
prune_quant_qwen.py), adapted to M3 shapes:

1. **Saliency**: REAP saliency = mean over routed tokens of `gate_value_e(x) × ||expert_e(x)||`,
   collected with router hooks. M3 gotchas vs the qwen script: sigmoid+bias routing (recompute
   top-k *inside* the hook — the "naive logit-mask is a no-op" bug from Phase 60 applies),
   per-expert ModuleList `w1/w2/w3` (NOT fused like Qwen3MoeExperts — simpler; the Phase-63 fused
   -tensor grad-waste OOM does not apply), shared expert is untouchable (never prune it).
2. **Calibration corpus**: ~500-1000 seqs × 2-4K tokens, mixed to our workload: code +
   agentic/tool-use + reasoning + general web (the model is agent/coding-tuned; datasets already
   in /data/ai-ml/hf-models: the-stack-smol, gsm8k, fineweb-edu, plus atlas theme corpora).
   ≤2K seqs keeps the MSA-vs-full-attention question moot during calibration (exact regime).
3. **Prune**: physically slice experts (drop rows in the ModuleList + shrink router + bias) —
   NOT gate-masking (Phase 63 lesson: masking leaves 4x memory/grad waste; slicing is the real
   artifact anyway).
4. **Heal**: FULL-params heal (NOT router-only — Phase 62: router-only heal is a dead lever,
   full heal took prune50 ABOVE baseline). Adafactor or PagedAdamW8bit, ~200-600 steps, lr 1e-5,
   diverse streamed corpus (heal_corpus.py recipe — never repeat data, Phase 29 lesson).
5. **Quant**: NVFP4 via the imatrix-aware quantizer (llama.cpp PR 25153, already on our local
   branch f143e48b4) with imatrix from the same calibration corpus. Q4_K_M as fallback ftype.
   Phase 63 result: NVFP4 PTQ loss on healed weights is additive + tiny (-.014 arc), not
   compounding — expect the same.

### Prune-ratio candidates and the honest 24GB math
Evidence (Qwen3-30B, 128-exp, our harness): 50% near-lossless one-shot; 62% -0.07 arc; 75% -0.08
arc *after* heal (graceful, not lossless). M3 is 14x bigger → redundancy-scales says these are
conservative bounds.

- **REAP50 → 130GB**: quality-safe (validated depth). Rig serving: ~8GB resident + hot experts in
  VRAM/pinned (~35-40GB hot tier on this 60GB-RAM rig) + cold via BW24_SPILL_DISK mmap. Works —
  this is exactly the 35B-A3B spill pattern, one size class up. Expected to be disk-bound on
  cold-expert faults; top-4-of-64 routing + SLRU should keep hit-rate high.
- **REAP62-75 → 101-70GB**: the likely sweet spot — smaller cold tier, measurably better tok/s,
  quality gated by heal. Decide by eval, not upfront.
- The stated "30-40GB total" target = REAP~87 (16 experts/layer). That is PAST the validated
  cliff (75% was already -0.08 healed on a 14x-smaller model; 87% on OLMoE was ppl 2025). Flagged
  honestly: treat 30-40GB as an *experimental* artifact (REAP80-87 + heal), not the primary. The
  primary artifact target is **70-130GB NVFP4 + tiered spill**, which the rig's spill infra was
  built for. Fewer experts also = fewer PCIe fetches/token, so prune depth buys speed even while
  spilled (Phase 63 note).

### Where each stage runs
- **Local CPU/disk (now)**: downloads, tensor inventory, tokenizer sanity, saliency/prune
  *plumbing* (structure-only dry-run on the index — no weights needed), GGUF conversion prep.
- **G7e 96GB (when its MoE agent frees)**: everything with forward passes. NOTE: even the 129GB
  REAP50 does NOT fit 96GB — calibration must be **layer-streamed** (embed→layer-by-layer with
  activation checkpoints on disk; REAP saliency only needs per-layer router logits + expert
  output norms, so this is clean) or accelerate CPU-offloaded (simpler, slower). Heal at REAP50
  needs multi-GPU or aggressive offload — budget the G7e request accordingly, or heal only the
  deeper-prune variants (≤70GB weights + Adafactor state fits 96GB with room).
- **Local GPU (kernel agent owns it)**: nothing until free; then bw24 loader-arc bringup +
  spill-serving validation of the final GGUF.

## 4. Immediate next actions
1. [running] MXFP8 download (~444GB, ~75MB/s ≈ 100+ min remaining) — `minimax-m3-mxfp8-dl.log`
2. [running] REAP50 download (~129GB) — `minimax-m3-nvfp4-reap50-dl.log`
3. [when REAP50 lands] config/tokenizer sanity + full tensor inventory vs this plan +
   structure-only prune dry-run (verify slicing plan against real shapes, CPU only)
4. [G7e queue] layer-streamed calibration pass → REAP saliency on our corpus → compare against
   sparkarena's pruning choices (their kept-expert sets are visible in their ckpt — cheap diff)
5. [local, anytime] bw24 `Arch::MinimaxM3` + name-mapping + sigmoid-router + gemma-norm +
   swigluoai plumbing (items 2-6 above) against the GGUF, full-attention v0 (≤2K exact)
6. [decision gate] Path A vs Path B on eval numbers (arc/arc_e/gsm8k + ppl trend, per
   shared-delta-moe eval harness), then quant the winner with PR-25153 NVFP4 + our imatrix
