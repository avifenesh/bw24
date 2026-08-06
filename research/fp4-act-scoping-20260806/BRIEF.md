# FP4-Activation Accuracy Scoping — Decision Brief

**Date:** 2026-08-06  
**Question:** Can FP4 (e2m1) activation quantization be recovered to memra's shipping bar, and at what engineering cost?

**Context:** The mxf4nvf4 k64 tensor-core instruction (m16n8k64, e2m1 activations) is real on sm_120a, measuring **1.9685x end-to-end prefill** (2.4x ceiling if fully realized, PREFILL-GEMM-REBUILD.md §4) with zero data repack needed and already implemented behind `MEMRA_MMQ=1` (cu/mmq_fp4.cu). It is **structurally blocked** because the k64 instruction class only accepts e2m1 (FP4) activations, and quantizing activations to FP4 loses accuracy. This brief evaluates whether that accuracy can be recovered to production standards and at what cost.

---

## 1. Prior-Art Baseline (w4a4-rescue-20260803)

### 1.1 Measured Quality Deltas

The w4a4-rescue lane attempted FP4-activation (W4A4) quantization via a residual correction kernel approach. Baseline divergence measurements (cross-model NLL deltas and argmax changes):

**Baseline W4A4 quality (no correction):**

| Model | Prompt | Verdict | cross_prefill_maxdiff | first_divergent_pos |
|-------|--------|---------|----------------------|---------------------|
| q9 (9B) | p2-code-medium (1845 tok) | DIVERGENT | 1.511 | token 26/48 |
| q27 (27B) | p2-code-medium | IDENTICAL | 2.495 | — |
| q9 | p3-agentic-long (6257 tok) | DIVERGENT | 2.133 | token 17/48 |
| q27 | p3-agentic-long | DIVERGENT | 4.792 | token 3/48 |
| q9 | p4-16k (23476 tok) | DIVERGENT | 1.677 | token 5/48 |
| q27 | p4-16k | OOM (24GiB capacity) | — | — |

**Summary:** 4/5 cells diverged at baseline W4A4. Cross-prefill maxdiff ranged from 1.51 to 4.79 (ref: memra's exactness gate is typically maxdiff <1e-3 for bit-exact paths; looser for lossy quantization but this is order-of-magnitude higher). Argmax divergences occurred within 3-26 tokens of generation start.

### 1.2 Attempted Mitigation: Residual Correction (k=32)

**Mechanism:** Ported llama.cpp's NVFP4 quantizer path + implemented a residual correction kernel that computed FP32 residuals between W4A4 and higher-precision reference outputs, accumulated per-token corrections over a calibration depth k (e.g., k=32 matmul layers), and applied the correction at generation time.

**Results on tuning corpus (5 cells):**
- k=32 residual correction: **5/5 cells IDENTICAL**, full battery green (kernel-check, run-gen argmax match q9+q27, run-spec K=1..8 PASS)
- Performance: 1.341-1.426x prefill speedup (vs uncorrected W4A4 1.53-1.64x, which was never exact)
- Kernel optimization trajectory: 0.840x → 1.341x via compile-time-K templating, 2D grid expansion (16→464 CTAs), wider token staging

**Generalization failure (widened corpus, 10 prompts):**
- k=32 dropped to **4/10 IDENTICAL** (corpus was originally 3/11 prompts, widened to 10)
- k=64: **0/5 IDENTICAL** (plateau, not a trend)
- One adversarial cell (q27/board-2048): diverged at 8 of 9 k-depths **including k=0** (token 0 fork) → a "W4A4-fails-this-prompt" cell, not a k-defect

**Key finding:** Group-of-16 rounding was load-bearing — switching to pure f32 chain accumulation flipped **3/5 cells DIVERGENT** (residual-k32-puresum.jsonl). The correction method **overfitted its tuning corpus** and did not generalize.

**Lane verdict (commit 9461c7bf, 2026-08-03):** NEGATIVE. W4A4 arm stays OFF by default. Residual correction insufficient for production.

### 1.3 What Any New Method Must Beat

- **Accuracy bar:** Must generalize beyond tuning corpus to diverse prompts. At minimum, achieve <1% perplexity delta and <1 in 100 token argmax drift on a representative eval set (e.g., 50+ diverse prompts spanning code, chat, reasoning, long context). memra's typical exactness gate for lossy formats: cross-run maxdiff <0.5, argmax match on reference prompts.
- **Coverage bar:** Must not have "fails-this-prompt" cells where the method structurally cannot recover accuracy (i.e., the q27/board-2048 signature).
- **No ad-hoc fixes:** A calibration-tuned correction kernel that works on 5/5 prompts but drops to 4/10 on a wider set is insufficient. The method must be principled (e.g., rotation-based outlier removal, learned transforms) rather than prompt-specific fitting.

---

## 2. Method Landscape for Low-Bit Activation Quantization Quality Recovery

The following methods target W4A4 (4-bit weights, 4-bit activations) or FP4 activation quantization specifically. All dates and results from published literature (2022-2025).

### 2.1 QuaRot (Rotation-Based Outlier Removal)

**Publication:** April 2024 (arXiv:2404.00456, ETH Zurich/SPCL)

**Mechanism:** Applies computational-invariance-preserving rotations (Hadamard transforms) to remove activation outliers without changing model outputs. Rotations applied to: (1) hidden states/residual stream, (2) feed-forward activations, (3) attention mechanism components, (4) KV cache. Random rotations decorrelate channels and suppress outliers, enabling uniform 4-bit quantization across all matrix multiplications with no mixed-precision escape hatches. This is a **one-time offline transformation** of model weights followed by **online per-layer Hadamard application** during inference.

**Published Accuracy (W4A4):**
- LLaMA-2 70B: 0.29 WikiText-2 perplexity loss vs full precision, retains 99% zero-shot performance
- LLaMA-2 7B/13B/30B: Specific numbers not in accessible docs (repository supports only LLaMA-2 models currently)

**Runtime Overhead:**
- Online Hadamard transforms required per layer (fast O(n log n) but adds latency)
- Exact overhead NOT published in paper or repository — no tok/s measurements
- Repository includes CUDA kernel implementations but no benchmarked latency impact

**Format Fit for memra (mxf4 m16n8k64):**
- The mxf4 instruction takes **e2m1 activations with UE4M3 scales per 16 elements** (finer scale granularity than typical per-tensor/per-channel)
- QuaRot uses per-channel scaling by default; would require adaptation to per-16-element scale blocks
- The Hadamard transform itself is scale-agnostic (operates on values), but quantizer must respect the per-16 scale structure

**memra Integration Points:**
- **Offline:** Rotate model weights once during quantization/conversion (pre-GGUF or post-load transform)
- **Online (activation quant path in MMQ):** Apply Hadamard to activation tiles before FP4 quantization in cu/mmq_fp4.cu (insertion point: before `mma_mxf4_m16n8k64`, after activation fetch)
- **Calibration:** Requires calibration data to tune rotation parameters — memra doctrine allows calibration from non-public traces only (acceptable)

**Risks:**
- Rotation overhead may consume significant portion of 2.4x ceiling (unmeasured)
- LLaMA-2-only current support; Qwen architecture may have different outlier patterns
- Per-16 scale granularity interaction unclear

---

### 2.2 SpinQuant (Learned Rotations)

**Publication:** May 2024, accepted ICLR 2025 (arXiv:2405.16406, Meta FAIR)

**Mechanism:** Recognizes that rotation choice dramatically affects quantization quality (up to 13-point spread in zero-shot performance). Instead of random Hadamard like QuaRot, **learns optimal rotation matrices** via Cayley optimization on the Stiefel manifold. Learned rotations minimize quantization error while preserving full-precision outputs. At inference, rotations applied using fast Hadamard transforms with the learned parameters.

**Published Accuracy (W4A4KV16, WikiText-2 PPL / zero-shot avg %):**

| Model | Full Precision | SpinQuant W4A4KV16 | QuaRot W4A4KV16 | Gap to FP |
|-------|----------------|---------------------|------------------|-----------|
| LLaMA-2 7B | 5.5 / 66.9 | 5.9 / 64.1 | 6.1 / 63.5 | 2.9 pts |
| LLaMA-2 13B | 5.0 / 68.3 | 5.2 / 67.2 | 5.4 / 66.7 | 1.1 pts |
| LLaMA-3 8B | — / — | (45.1% gap reduction vs QuaRot) | — | — |

**W4A4KV4 (stricter, 4-bit KV cache):**
- LLaMA-2 7B: 5.9 PPL / 64.0% (vs QuaRot 6.4 / 62.5)
- LLaMA-2 13B: 5.3 / 66.9 (vs QuaRot 5.4 / 66.2)

**Beats prior W4A4 SoTA:** +25.0 points over SmoothQuant, +19.1 over LLM-QAT (on 7B zero-shot)

**Runtime Overhead:**
- **Calibration (one-time offline):** 10-16 hours on single A100-40G per model
- **Inference overhead:** NOT published — same "apply learned rotation per layer" as QuaRot, but no tok/s measurements
- Requires `fast-hadamard-transform` package

**Format Fit:**
- Supports per-channel scaling; per-16 element scales would require custom integration
- Learned rotations are model-specific (must re-calibrate per Qwen checkpoint)

**memra Integration:**
- **Offline:** Learn rotations on calibration set (10-16hr on A100 acceptable for one-time setup), store rotation matrices with model
- **Online:** Apply learned Hadamard per layer (same insertion point as QuaRot)
- **Calibration:** Non-public traces acceptable per memra doctrine

**Risks:**
- 2.9-point zero-shot gap on 7B (best published W4A4 result) — is this acceptable for production? memra's bar is user-perceived quality, not academic benchmarks
- Inference overhead unmeasured (could erase 2.4x gain)
- Requires per-model calibration (operational burden for new checkpoints)

**Advantage over QuaRot:** Learned > random rotations, demonstrably better accuracy

---

### 2.3 Online Hadamard Transforms (TensorRT-LLM / vLLM Reference)

**Status:** Referenced in frameworks but **not documented in detail**

**vLLM:** Supports MXFP4/NVFP4/INT4 formats but documentation does NOT describe W4A4 activation quantization methods. No explicit QuaRot/SpinQuant integration. Focus: weight-only quantization (W4A16) and W8A8.

**TensorRT-LLM:** References FP4 models (DeepSeek-R1-FP4 via NVIDIA Model Optimizer) but public docs lack W4A4 pipeline details. NVFP4 format targets Blackwell (sm_90+).

**What Hadamard Transforms Do (general principle):**
- Act as pseudo-random orthogonal transformations that **decorrelate activation channels** and **spread outlier values** across multiple elements
- Makes uniform quantization viable by reducing dynamic range within quantization groups
- Applied online per layer before quantization

**Overhead (from PyTorch blog on INT4 KV cache):**
- Without kernel fusion, **dequantization overhead can eliminate performance gains entirely**
- Implication: Hadamard must be fused with quantization and matmul to avoid roundtrip overhead

**memra Applicability:**
- Principle is sound (decorrelate → quantize), but no published implementation for mxf4 k64
- Would need custom CUDA kernel fusing Hadamard + per-16 scale quantization + mma_mxf4_m16n8k64

---

### 2.4 SmoothQuant (Activation-to-Weight Difficulty Migration)

**Publication:** November 2022, ICML 2023 (arXiv:2211.10438, MIT)

**Mechanism:** Exploits asymmetry that "weights are easy to quantize while activations are not." **Offline** migrates quantization difficulty from activations to weights via mathematically equivalent per-channel rescaling transformations: `Y = (Xdiag(s)^-1)(diag(s)W) = X'W'`. An alpha parameter (0.6-0.9) controls migration strength. This smooths activation distributions, enabling W8A8 quantization.

**Published Accuracy (W8A8, WikiText-2 PPL):**

| Model | FP16 | SmoothQuant W8A8 | Alpha |
|-------|------|------------------|-------|
| LLaMA-2 7B | 5.474 | 5.515 | 0.85 |
| LLaMA-2 13B | 4.950 | 4.929 | 0.85 |
| LLaMA-3 8B | 6.138 | 6.258 | 0.85 |
| Mistral 7B | 5.253 | 5.277 | 0.8 |

(Delta: +0.04 to +0.12 PPL, negligible)

**W4A4 Capability:** SmoothQuant is **designed for W8A8**. Extending to W4A4 significantly degrades accuracy (SpinQuant paper shows **25-point zero-shot gap** vs full precision at W4A4 with SmoothQuant).

**Runtime Overhead (W8A8):**
- 1.56× speedup and 2× memory reduction vs FP16 (with real INT8 GEMM kernels)
- Scaling applied offline to weights; no inference-time overhead

**memra Applicability for W4A4:**
- **NOT viable** — method targets 8-bit, known to fail at 4-bit
- Could be combined with rotation methods (smooth THEN rotate) but no published evidence this recovers W4A4 accuracy

---

### 2.5 NVIDIA NVFP4 Inference (TensorRT Model Optimizer)

**Status:** Production deployment format for Blackwell (sm_90+), publicly shipping (DeepSeek-R1-FP4)

**Mechanism (inferred from model cards, NOT detailed in public docs):**
- Post-training quantization to FP4 (E2M1 format) using NVIDIA Model Optimizer (nvidia-modelopt v0.23+)
- Quantizes weights and activations of linear operators within transformer blocks
- Attention/embedding layers remain higher precision (mixed dtype safetensors show F32, BF16, F8_E4M3, U8)
- Uses calibration dataset (e.g., CNN/DailyMail) during PTQ
- Likely incorporates smoothing or scaling techniques internally (NOT documented)

**Published Accuracy (DeepSeek-R1-FP4 vs FP8 base, 671B model):**
- MMLU: 90.7 vs 90.8 (−0.1)
- GSM8K: 96.1 vs 96.3 (−0.2)
- Minimal degradation FP8→FP4 (but baseline is already FP8, not FP16)

**Runtime Overhead:** NOT published. Format targets Blackwell-optimized kernels.

**Scale Granularity:** Not specified; safetensors show mixed dtypes suggesting heterogeneous quantization (not uniform per-16 scales).

**memra Applicability:**
- NVIDIA's approach is **proprietary black-box** — no published method details
- DeepSeek-R1-FP4 exists as proof-of-concept that W4A4 CAN ship at scale (671B model), but path to achieve it is undocumented
- memra's mxf4 instruction on sm_120a is the SAME hardware primitive NVIDIA uses (e2m1 + UE4M3 block scales) — so the instruction itself is viable
- **Cannot replicate NVIDIA's method without reverse-engineering their quantizer**

**Key Insight:** NVIDIA ships W4A4 in production (DeepSeek-R1-FP4) → proves it's solvable at massive scale, but method is not open

---

### 2.6 OmniQuant (Learnable Clipping + Equivalent Transformation)

**Publication:** August 2023, ICLR 2024 (arXiv:2308.13137, OpenGVLab)

**Mechanism (two learnable components for W4A4):**
1. **LWC (Learnable Weight Clipping):** Optimizes clipping thresholds for extreme weight values
2. **LET (Learnable Equivalent Transformation):** Shifts quantization difficulty from activations to weights via learned channel-wise rescaling (similar spirit to SmoothQuant but learned, not alpha-tuned)

Optimized via block-wise error minimization, 128 calibration samples, 20 epochs (PTQ-level efficiency).

**Published Accuracy:** Claims "SoTA performance" in weight-activation quantization. Exact W4A4 numbers for 7B-30B models presented only in paper figures (not extracted text). Provides pre-quantized checkpoints for LLaMA 7B/13B/30B/65B and LLaMA-2 7B/13B/70B.

**Runtime Cost:**
- **Calibration:** 1-16 hours on single A100-40G depending on model size
- **Inference with `--real_quant`:** Weight-only modes achieve memory reduction but **slower inference** without proper kernel support (requires AutoGPTQ or MLC-LLM deployment for speedup)
- No published tok/s for W4A4

**Scale Granularity:** Supports group-wise quantization (g128 common); per-channel activation scales.

**memra Applicability:**
- Learned transformation approach (not rotation-based) — different structural fit than QuaRot/SpinQuant
- No evidence of per-16 scale support
- Inference speedup NOT demonstrated (repo warns of slowdown without kernel support)

---

### 2.7 QServe (W4A8 with SmoothAttention)

**Publication:** January 2024 (MIT Han Lab, related to FP6-LLM work)

**Mechanism (W4A8KV4, not W4A4):**
- Progressive quantization for low dequantization overhead
- Compute-aware weight reordering
- Register-level parallelism
- SmoothAttention for KV4 accuracy (not Hadamard transforms)

**Published Accuracy (W4A8KV4, per-channel, WikiText-2 PPL):**
- LLaMA-2 7B: 5.75 (vs FP16 5.5) — +0.25 PPL
- LLaMA-2 13B: 5.12 (vs 5.0) — +0.12
- LLaMA-3 8B: 6.89 (vs QuaRot W4A4 8.33) — **BETTER than W4A4**

**Runtime Performance:**
- 1.2-1.4× higher throughput than TensorRT-LLM on LLaMA-3-8B
- 2.4-3.5× on Qwen1.5-72B
- Existing INT4 methods suffer 20-90% overhead from dequantization; QServe optimizes this

**Key Insight:** **W4A8 is more practical than W4A4 for production** — better accuracy/speed tradeoff. W4A4 struggles with 2.9-point zero-shot gaps; W4A8 is near-lossless (+0.25 PPL).

**memra Applicability:**
- QServe is W4A8, not W4A4 — does NOT use mxf4 k64 (which requires FP4 activations)
- If mxf4 k64 is blocked on activation precision, **falling back to int8 or FP8 activations** (different instruction) may be the pragmatic path
- memra already has FP8 activation paths (fp8st lanes) — W4A8 would be a quantization target, not a kernel change

---

## 3. Format Fit: Per-16 Scale Granularity Analysis

**The mxf4 m16n8k64 instruction constraint:** Takes **e2m1 activations with UE4M3 scales per 16 elements**. This is **finer scale granularity** than most published methods assume (typical: per-tensor, per-channel, or per-128 group-wise).

### Does Per-16 Scaling Alone Recover Meaningful Accuracy?

**Quantization Error Arithmetic:**

For uniform quantization of a tensor X to b bits with scale s:
- Quantization error per element: `ε = (X - round(X/s)*s)`
- RMS error scales as: `σ_ε ∝ σ_X / 2^b` (where σ_X is input standard deviation)
- **Key:** If σ_X varies across the tensor, a single global scale s under-scales high-variance regions (clipping) and over-scales low-variance regions (precision waste)

**Effect of finer scale granularity (per-tensor → per-channel → per-16):**
- **Per-tensor:** One scale for entire activation tensor (e.g., [batch, seq_len, hidden_dim]). If hidden_dim has outlier channels, quantization error dominated by outliers.
- **Per-channel:** One scale per channel (hidden_dim axis). Removes inter-channel variance but not intra-channel variance (e.g., across tokens).
- **Per-128 group:** Commonly used in weight quantization (e.g., GPTQ, AWQ). Removes both inter-channel and coarse intra-sequence variance.
- **Per-16 element (mxf4):** FINEST published scale granularity for matmul instructions. Each 16-element block gets its own scale.

**Quantization SNR scaling:**
- SNR ∝ 2^(2b) / Var(X/s_local)
- With per-16 scales, Var(X/s_local) is the **within-block variance** (after normalizing by per-block scale) — much smaller than per-tensor or per-channel variance
- For b=4 (FP4), this is critical: 4 bits gives only 16 representable values (e2m1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6, -∞, +∞, NaN}). A coarse scale leads to most values clipping to ±6 or quantizing to same bin.

**Measured Evidence (from w4a4-rescue baseline):**
- Cross-prefill maxdiff ranged 1.51–4.79 at baseline W4A4
- memra's NVFP4 port (which DOES use per-16 scales) shows **uncorrected W4A4 never reached exactness** (verdict: divergence on 4/5 cells)
- This suggests **per-16 scaling alone is insufficient** — outliers within 16-element blocks still corrupt quantization

**Theoretical Limit:**
- Activation tensors in transformers have **heavy-tailed distributions** with rare large outliers (the core motivation for QuaRot/SpinQuant)
- Per-16 scaling reduces the dynamic range from global to local, but if a SINGLE outlier exists in a 16-element block, the scale must accommodate it → other 15 elements lose precision
- Example: Block contains [0.1, 0.12, ..., 0.15, 8.3]. Scale must cover 8.3 → small values all quantize to same bin.

**Conclusion:**
Per-16 scaling is **necessary but not sufficient**. It is significantly better than per-tensor/per-channel (removes coarse variance), but without outlier suppression (rotation/transform), the within-block outliers still dominate quantization error. The w4a4-rescue measurements empirically confirm this: per-16 scales were used, yet accuracy was unacceptable without correction.

**Implication for method selection:** Any viable method must **suppress outliers** (QuaRot/SpinQuant via rotation, or OmniQuant via learned transforms) in addition to leveraging per-16 scales. Per-16 scaling sets the hardware capability; outlier suppression is the algorithmic requirement.

---

## 4. Decision Table

| Method | Projected Net Speedup | Accuracy Risk Class | Engineering Days Estimate | Calibration Needs | Recommendation |
|--------|----------------------|---------------------|--------------------------|-------------------|----------------|
| **QuaRot** | 0.9–1.8x (2.4x ceiling minus 20-60% overhead, unmeasured) | **MEDIUM** — 0.29 PPL loss on 70B (good), but LLaMA-2 only, Qwen untested | **30-45 days** — Port rotation logic, fuse Hadamard with per-16 quant, implement CUDA kernel for online Hadamard application, integrate with MMQ path, validate on Qwen | Non-public calibration traces (acceptable per memra doctrine) | **Viable, but high risk/effort** |
| **SpinQuant** | 0.9–1.8x (same overhead class as QuaRot) | **MEDIUM-HIGH** — Best published W4A4 (2.9pt gap on 7B), but 2.9pt zero-shot loss may not meet memra's user-quality bar | **45-60 days** — Port Cayley optimization, 10-16hr calibration per model, same CUDA work as QuaRot, per-model recalibration burden | 10-16hr A100 calibration per checkpoint (operational burden) | **Highest accuracy of rotation methods, but still 2.9pt gap + high eng cost** |
| **Hadamard (generic)** | 0.9–1.8x (same kernel fusion requirement) | **MEDIUM** — Principle sound (decorrelate channels), but no published mxf4 implementation or Qwen results | **30-40 days** — Implement Hadamard kernel, fuse with per-16 quant, no learned params (simpler than SpinQuant) | Minimal (may not require calibration) | **Lower accuracy ceiling than SpinQuant, similar eng cost** |
| **SmoothQuant** | N/A (offline transform, no inference overhead) | **HIGH** — Known to fail at W4A4 (25pt gap), designed for W8A8 | **15-20 days** (but pointless — method incompatible with W4A4) | 512 calibration samples | **REJECT — incompatible with W4A4** |
| **OmniQuant** | <1.0x (repo warns of slowdown without kernel support) | **MEDIUM** — Claims SoTA W4A4 but no hard numbers for Qwen-class models | **40-50 days** — Port LWC+LET, implement custom kernels (no existing fast path), 1-16hr calibration | 128 calibration samples, 20 epochs per model | **Unproven speedup, no existing kernel support** |
| **NVIDIA NVFP4 (black box)** | Unknown (proprietary) | **LOW** — DeepSeek-R1-FP4 ships at 671B scale (proof-of-concept) | **N/A — proprietary, cannot replicate** | Unknown | **REJECT — not replicable** |
| **Fallback: Keep k64 door closed, pursue W4A8** | 0x (no mxf4 k64 gain) | **LOW** — QServe W4A8 shows +0.25 PPL (near-lossless), proven speedup 1.2-3.5x | **20-30 days** — Implement W4A8 quantization for memra (different instruction, not mxf4 k64), leverage existing FP8 paths | Standard PTQ calibration | **PRAGMATIC — proven accuracy, proven speed, lower risk** |

**Speedup Calculation Notes:**
- 2.4x ceiling from PREFILL-GEMM-REBUILD.md §4 (mxf4 k64 if fully realized)
- Rotation methods require **online per-layer Hadamard** — no published overhead, but PyTorch blog warns unfused dequant can eliminate gains entirely
- Conservative estimate: 20-60% overhead → net 0.9-1.8x (wide range due to lack of measurements)
- If Hadamard fusion is highly optimized (10-20% overhead), could approach 2.0-2.2x
- If poorly optimized (60%+ overhead), could fall below 1.0x (slower than baseline)

**Accuracy Risk Classes:**
- **LOW:** <0.5 PPL delta, <1 in 1000 argmax drift (production-safe)
- **MEDIUM:** 0.5-1.5 PPL delta, occasional argmax changes (may be acceptable depending on use case)
- **HIGH:** >1.5 PPL delta or >1 in 100 argmax drift (user-perceivable quality loss)

SpinQuant's 2.9-point zero-shot gap translates to ~0.4-0.6 PPL delta (based on LLaMA-2 7B: 5.5→5.9 PPL) → borderline MEDIUM/HIGH. For memra's production bar (user-critical infra, owner's default engine per MEMORY.md), this may be unacceptable.

---

## 5. Recommendation

### **KEEP THE DOOR SHUT** (do not fund an FP4-activation implementation lane at this time)

**Rationale:**

1. **Accuracy gap too large for memra's bar:** Best published W4A4 result (SpinQuant) shows 2.9-point zero-shot loss and +0.4 PPL on 7B models. memra is "owner-critical infra" and "dogfood findings feed serve-compat" (MEMORY.md) — a 2.9-point quality drop is likely user-perceivable and violates the "beat llama on every scenario" doctrine. The w4a4-rescue lane already demonstrated that even achieving exactness on a tuning corpus (5/5 cells) did not generalize (4/10 widened corpus), and residual correction failed at production scale.

2. **Runtime overhead is unmeasured and high-risk:** All rotation-based methods (QuaRot, SpinQuant, Hadamard) require online per-layer transforms. Zero published latency measurements exist. The 2.4x prefill ceiling could easily collapse to <1.2x or even <1.0x with naive Hadamard implementation (PyTorch blog precedent: unfused dequant eliminated INT4 KV cache gains). Conservative estimate 0.9-1.8x net speedup is a WIDE uncertainty band — not a confident bet for 45-60 days of eng work.

3. **Engineering cost is high (45-60 days for SpinQuant):** Port Cayley optimization, implement CUDA kernel fusing Hadamard + per-16 quant + mma_mxf4_m16n8k64, calibrate per model (10-16hr A100), validate on Qwen (untested architecture for rotation methods), integrate with MMQ path. This is a 1.5-2 month full-time lane with no guarantee of production-acceptable accuracy or speedup.

4. **Fallback path is lower-risk and proven:** W4A8 (QServe) shows near-lossless accuracy (+0.25 PPL on 7B, better than W4A4's +0.4-0.6) and proven speedup (1.2-3.5x vs TensorRT-LLM). memra already has FP8 activation paths (fp8st lanes merged 2026-08-03/04). Pursuing W4A8 quantization leverages existing infra, avoids the unmeasured Hadamard overhead, and targets a format with published production results. Engineering estimate: 20-30 days (vs 45-60 for SpinQuant).

5. **NVIDIA's proof-of-concept is not replicable:** DeepSeek-R1-FP4 (671B) proves W4A4 CAN ship at massive scale, but NVIDIA's method is proprietary black-box. Without access to Model Optimizer internals, memra cannot replicate their approach. The existence of a production W4A4 system does not provide a path to build one.

### **If Accuracy Requirements Relax or New Evidence Emerges:**

**Conditions to revisit this decision:**
1. **Published latency measurements for rotation methods:** If QuaRot/SpinQuant publish tok/s benchmarks showing <20% overhead (net >1.9x from 2.4x ceiling), the speedup case strengthens significantly.
2. **Accuracy improvement:** If a new method achieves <0.2 PPL delta and <1 in 500 argmax drift on Qwen-class models at W4A4, the quality bar may be met.
3. **Use-case pivot:** If memra targets a batch-serving or throughput-bound scenario where 2.9-point quality loss is acceptable (e.g., research-only, not production chat), W4A4 becomes viable.
4. **NVIDIA publishes method details:** If Model Optimizer's W4A4 quantization pipeline is documented (rotation params, calibration, scale granularity), replication becomes feasible.

**First quality gate if funded:** Implement QuaRot (simpler than SpinQuant, no learned params) on a SINGLE Qwen 7B checkpoint. Measure:
- WikiText-2 PPL delta (target: <0.3)
- Argmax match rate on 50 diverse prompts (target: >99%)
- Prefill tok/s with online Hadamard (target: >1.5x vs baseline, proving <38% overhead)

If this gate passes, proceed to SpinQuant (learned rotations for accuracy gain). If it fails on accuracy OR speed, abort the lane.

---

## 6. Summary for Owner

**Baseline:** w4a4-rescue lane (2026-08-03) measured 4/5 cells divergent at W4A4, attempted residual correction achieving 5/5 exact on tuning corpus but failing generalization (4/10 widened corpus). Verdict: NEGATIVE, arm stays OFF.

**Best published method:** SpinQuant (Meta FAIR, ICLR 2025) achieves 2.9-point zero-shot loss and +0.4 PPL on LLaMA-2 7B at W4A4 via learned rotations. Requires 10-16hr calibration per model, online per-layer Hadamard (overhead unmeasured), and 45-60 days engineering (port Cayley optimization, fuse CUDA kernel, validate on Qwen).

**The 2.4x ceiling problem:** mxf4 k64 measures 1.9685x e2e prefill with 2.4x ceiling (PREFILL-GEMM-REBUILD.md §4). Rotation methods require online Hadamard transforms per layer — zero published latency measurements exist. Conservative estimate: 20-60% overhead → 0.9-1.8x net (wide uncertainty). If overhead is 60%+, net speedup <1.0x (slower than baseline).

**Per-16 scaling alone insufficient:** mxf4's UE4M3 per-16-element scales are finer than typical methods (per-tensor/per-channel/per-128) but empirically insufficient (w4a4-rescue baseline used per-16 scales, still diverged). Outlier suppression (rotation/transform) is required in addition to fine-grained scaling.

**Recommendation:** **KEEP DOOR SHUT**. Accuracy gap (2.9pt / +0.4 PPL) likely violates memra's production bar ("owner-critical infra", "beat llama every scenario"). Runtime overhead unmeasured and high-risk (could collapse 2.4x → <1.2x). Engineering cost high (45-60 days) with no guarantee of success. **Pursue W4A8 instead** (QServe: +0.25 PPL, 1.2-3.5x proven speedup, 20-30 days eng, leverages existing fp8st infra) — lower risk, better accuracy, proven speed.

**Resurrection bar:** Publish rotation method latency showing <20% overhead (>1.9x net) AND new method achieving <0.3 PPL delta on Qwen at W4A4. Until then, the mxf4 k64 door stays closed, FP4-activation accuracy is the blocker (not kernel engineering).

---

**Sources:**
- QuaRot: arXiv:2404.00456 (April 2024, ETH Zurich)
- SpinQuant: arXiv:2405.16406 (May 2024, Meta FAIR, ICLR 2025)
- SmoothQuant: arXiv:2211.10438 (November 2022, MIT, ICML 2023)
- OmniQuant: arXiv:2308.13137 (August 2023, OpenGVLab, ICLR 2024)
- QServe: Han Lab MIT (January 2024)
- TensorRT-LLM NVFP4: DeepSeek-R1-FP4 model card (January 2025)
- PyTorch INT4 KV cache blog (overhead precedent)
- memra w4a4-rescue-20260803: commit 9461c7bf (August 3, 2026)
- memra PREFILL-GEMM-REBUILD.md: mxf4 k64 measurements (August 2026)
