# Step-3.7-Flash bring-up — Phase 1: research + onboarding groundwork

Date: 2026-08-02. Lane: `lane/step37-bringup` (base: `restructure/public-split`).
Context: the sku-repick verdict (`research/sku-repick-20260802/REPORT.md`) made Step-3.7-Flash
the flagship listing for the 4x 5090 box. This lane opens the bring-up. Phase 1 = research and
onboarding groundwork only — no GPU work, no large downloads, no performance claims. Every
external fact below is cited with source + fetch date; raw receipts in `raw/`.

---

## 0. Kill criteria — checked up front, none tripped, four downgrades

No kill. But four findings make the SKU **less good than the sku-repick framing**, stated first:

1. **The "differentiation window" is gone: Step-3.7 support is UPSTREAM llama.cpp, merged
   2026-06-02.** sku-repick assumed "llama.cpp via StepFun's fork, upstream merge unverified →
   differentiation window." Verified 2026-08-02 (receipt `raw/llamacpp-pr-status-20260802.json`):
   - ggml-org/llama.cpp PR #23845 "Model: Support Step3.7-Flash" — **merged 2026-06-02**
     (https://github.com/ggml-org/llama.cpp/pull/23845)
   - PR #23274 "StepFun 3.5 MTP" — **merged 2026-06-02** (MTP spec decode is upstream too)
   - PR #19283 "Support Step3.5-Flash" (the shared `step35` GGUF arch) — merged 2026-02-06.
   Every llama.cpp-based provider can serve this model today, with MTP. Our edge must come from
   measured speed + serving surface (tools/cache/metering), not exclusivity.
2. **The "95.3 GB IQ4_XS" artifact is not an honest IQ4_XS** (§3.2): it is unsloth's UD-IQ4_XS
   at **3.87 effective bpw** with 2/3 of the expert bank at IQ3_S. The honest uniform-expert
   IQ4_XS is StepFun's official GGUF at **105.0 GB** — still a 4-card fit, with less headroom.
3. **256K context does not fit under memra's current KV allocator** (§3.4): SWA layers are
   allocated at max_ctx today, so 256K FP16 KV = 45 GiB > all headroom. Serving the advertised
   256K needs the SWA ring-buffer allocation work item (or FP8-KV + the smaller artifact).
   Until that lands, the honest listing context is 128K.
4. **The incumbents serve FP8; any 4-bit listing from us is the lowest-quality-tagged endpoint
   on the page** (§5). StepFun and Novita both disclose `fp8` on OpenRouter; DeepInfra is
   `unknown` (fresh pull 2026-08-02). StepFun publishes **zero** eval deltas for its own NVFP4
   or GGUF quants vs BF16/FP8 — the quality cost of 4-bit Step is unmeasured by anyone,
   including the vendor. Listing gate: our own eval-parity + tool-call receipts (§5e), with a
   plainly stated SKU re-arb trigger if they fail.

Also inherited from sku-repick, still true: we serve **text-only** — Step-3.7 is a
vision-language model and its OR page includes an unmeasurable image-traffic share that we
forfeit. China-centric demand skew and held-price fragility as previously reported.

## 1. Weights availability (all fetched 2026-08-02)

| Repo | Contents | Size (receipts: `raw/hf-files-*-20260802.json`) |
|---|---|---|
| `stepfun-ai/Step-3.7-Flash` | BF16 safetensors source (24 text shards incl. MTP shard `model-00024`, 2 ViT shards) | text 398.8 GB + vision 3.96 GB = **402.7 GB** |
| `stepfun-ai/Step-3.7-Flash-FP8` | FP8 safetensors (MTP shard stays BF16) | **212.5 GB** |
| `stepfun-ai/Step-3.7-Flash-NVFP4` | NVFP4 (modelopt) safetensors | **129.2 GB** |
| `stepfun-ai/Step-3.7-Flash-GGUF` | **Official GGUF**: BF16 394.0 / Q8_0 209.4 / Q4_K_S 111.5 / **IQ4_XS 104.99** / Q3_K_L 102.5 / Q3_K_M 93.8 / IQ3_XXS 75.8 GB + **standalone MTP GGUFs** (BF16 6.97 GB, Q8_0 3.71 GB) + mmproj F16 3.97 GB + **imatrix (0.47 GB) + agentmix calibration text** | see left |
| `unsloth/Step-3.7-Flash-GGUF` | UD quants IQ1_M..Q8_K_XL; **UD-IQ4_XS = 95,336,010,208 B = 95.34 GB**; no MTP file | see left |

Model facts (model card, https://huggingface.co/stepfun-ai/Step-3.7-Flash, fetched 2026-08-02):
198B total = 196B language + 1.8B vision encoder, ~11B active/token, 256K context, three
reasoning levels. HF reports 201.4B params (incl. ViT + MTP). Our shape math from config.json:
**197.0B main text model** (cross-checked exactly: official BF16 GGUF 394.0 GB / 2 B-per-param)
+ **3.49B MTP** (3 layers + re-shipped embeddings/head; matches mtp-BF16.gguf 6.97 GB / 2).
Receipt: `raw/sizes-math-20260802.out`, script `sizes-math.py`.

**License: Apache-2.0 — commercial serving allowed, no use restrictions.** Confirmed in three
places 2026-08-02: HF repo metadata tag, model card §License, and inside the GGUF header itself
(`general.license = apache-2.0`, receipt `raw/gguf-header-unsloth-ud-iq4xs-shard1-20260802.txt`).
Verdict: **clean**. (Check for a NOTICE file at download time; none observed in the file listing.)

**MTP/drafter head: YES, ships officially — but NOT inside the main GGUF.**
- HF checkpoint: `model-00024.safetensors` (6.97 GB) holds 3 MTP layers;
  `num_nextn_predict_layers: 3` in config.json; vLLM card recipe uses
  `{"method": "mtp", "num_speculative_tokens": 3}` (model card, 2026-08-02).
- Main GGUFs (official AND unsloth) contain **45 blocks, 754 tensors — MTP excluded**
  (parsed headers, receipts in `raw/`). The official repo ships MTP as standalone GGUFs
  (`Step3.7-flash-mtp-{BF16,Q8_0}.gguf`, arch `step35`, `step35.nextn_predict_layers = 3`,
  48-block numbering). unsloth ships none — if we serve the unsloth artifact we pair it with
  StepFun's MTP file (same base weights; the acceptance-rate gate validates the pairing).
- **Exact MTP tensor names** (parsed from mtp-Q8_0 header, receipt
  `raw/gguf-header-stepfun-mtp-q8-20260802.txt`): per k in {45,46,47}:
  `blk.k.nextn.{eh_proj,enorm,hnorm,shared_head_norm,shared_head_head}.weight` + a full
  96-head SWA attention block (`attn_q [4096,12288]`, `attn_k/v [4096,1024]`,
  `attn_gate [4096,96]`, `attn_output [12288,4096]`, q/k norms) + dense FFN 11264; plus
  file-level `token_embd`, `output`, `output_norm`, `rope_freqs.weight [64]`.
  HF-side names: `model.layers.{45,46,47}.{eh_proj,enorm,hnorm,transformer.shared_head.*}`.

**StepFun's llama.cpp fork exists**: github.com/stepfun-ai/llama.cpp (fork of ggml-org),
branch `step3.7` @ 8f34864def3d, last push 2026-06-21 (gh API 2026-08-02). The model card still
points users at the fork, but the arch + MTP are upstream (§0.1) — for the head-to-head we
benchmark **both** the fork branch and upstream master, interleaved on-box.
Known upstream bug: issue #24257 "Crash with Step 3.7 flash" (open, Vulkan backend — not our
CUDA path, monitor anyway).

## 2. Architecture map → memra seams

From `raw/config.json` (fetched 2026-08-02) and the parsed GGUF headers. GGUF arch name:
**`step35`** (same arch as Step-3.5-Flash; 3.7 is a bigger sibling). config.json arrays are
48-long = 45 main + 3 MTP layers.

| Feature | Value (receipts) | memra seam | Status |
|---|---|---|---|
| Layers | 45 main (3 dense + 42 MoE) + 3 MTP | `leading_dense_block_count=3` ≙ `first_k_dense_replace` (Hy3/MLA seam, `crates/memra-gguf/src/config.rs`) | EXISTS |
| SWA pattern | 3:1 — 12 full + 33 SWA(512); GGUF `step35.attention.sliding_window_pattern` bool-array + `sliding_window=512` | gemma-4 seam: `Gemma4Config.swa_pattern` + per-layer geometry (`hybrid_forward.rs:4599`), same GGUF key shape | EXISTS |
| Dual rope base | full 5e6 / SWA 1e4 (`rope.freq_base` + `rope.freq_base_swa`) | gemma-4 `rope_base_global`/`rope_base_swa` pair — exactly this | EXISTS |
| Rope scaling | llama3-type, factor 2.0, 131072→262144, **full-attention layers only**; GGUF carries `rope_freqs.weight [64]`, no explicit scaling KVs | `rope_freqs.weight` loads (`hybrid.rs:1185`) + `rope_neox_ff` kernel exists; **generation/wiring per layer-type does not** | **GAP (C)** |
| Partial rotary | factor 0.5 on full-attn layers (64 of 128 dims), 1.0 on SWA — from config.json; **not in GGUF KVs**, baked into upstream `build_step35` | `rope_dim_count` exists but is **global** (M3 uses 64/128) | **GAP (C)** — lift exact semantics from upstream/fork source at onboarding |
| GQA dims | 8 KV heads x 128 head_dim, all layers | standard | EXISTS |
| **Per-layer Q heads** | full layers 64, SWA layers 96 (GGUF `attention.head_count` is an **array**) | only precedent is gemma4-E4B shape-derived heads; base `ModelConfig.n_head` is global | **GAP (B)** |
| QK-norm | `attn_q_norm`/`attn_k_norm` [128] per layer | qwen3-family seam | EXISTS |
| **Head-wise attn gate** | `use_head_wise_attn_gate`; separate tensor `blk.N.attn_gate.weight [4096, n_head]` — one sigmoid scalar per head | memra's gate seam assumes gate **fused into wq** (qwen35 form, `q_gate_split` kernels). Separate-tensor per-head-scalar form is new. **Footgun**: `attn_out_gate()` (config.rs:565) is a *negative* predicate — a new arch defaults it to true and mis-splits wq | **GAP (A)** |
| MoE | 288 experts, top-8, expert_ff 1280, **sigmoid router** + `exp_probs_b.bias` + weight-norm + scale 3.0 (`expert_gating_func=2`) | `moe_route_sigmoid_host` (hybrid_forward.rs:3074) is this recipe **verbatim** (DeepSeek/noaux, Hy3 seam); `exp_probs_b.bias` loader exists; 288 experts within all dispatch-arm limits (in_f 1280 %256≠0 — but 1280 is fine as checked; expert idx u16) | EXISTS — but device router kernel is **softmax-only**; sigmoid = host round-trip today → **perf GAP (G)** |
| Shared expert | 1 per MoE layer, ff 1280 (`ffn_*_shexp`) | qwen-MoE shexp seam (`MoeWeights.*_shexp`) | EXISTS |
| SwiGLU clamp | per-layer arrays `swiglu_clamp_exp`/`_shexp` — nonzero only layers 43-44 (7.0 / 16.0) | `swigluoai_mul_scaled` kernel exists but is gated on `cfg.m3` and **not wired into grouped-MoE epilogues or batched decode**; clamp-array semantics vs swiglu_oai must be verified against upstream `build_step35` | **GAP (H)** |
| MTP | 3 nextn layers (SWA-type), DeepSeek-style eh_proj/enorm/hnorm/shared_head | `MtpHead` seam loads **exactly 1** nextn block; external-draft-file loader (`MEMRA_MTP_DRAFT=<path.gguf>`) fits the standalone official MTP GGUF; EAGLE3 + own-gen fallbacks exist | EXISTS for 1 layer; 3-layer chain = **GAP (D)** (optional, phase 2 of spec integration) |
| Attention sinks | `sink: false` | none needed | N/A |
| Vocab / tokenizer | 128896; GGUF `tokenizer.ggml.model=gpt2`, **`pre=deepseek-v3`**; HF `LlamaTokenizerFast`, 818 added tokens; bos 0 `<｜begin▁of▁sentence｜>`, eos 128007 `<|im_end|>` (config eos set [1,2,128007]) | gpt2 BPE path exists; **verify the deepseek-v3 pretokenizer regex** in memra-tokenizer — the byte-check gate decides | VERIFY (J) |
| Chat template | ChatML-family: `<|im_start|>role`, forced-open `<think>\n` on generation, `Reasoning: low/med/high` system line, `<tool_call><function=X><parameter=Y>` tool format, `tool_response` role, `<im_patch>` image slot (ignored, text-only) | chat.rs is hand-written per-dialect renderers — **new StepFun arm needed** | **GAP (I)** (small, template in `raw/chat_template.jinja`) |
| PP-4 | 95-105 GB weights → 4 cards | `pp.rs` is **hard-coded 2-stage** (`MEMRA_PP_STAGES=2` only, fixed `[usize;2]` devices); plain eager decode only | **GAP (E)** — known from sku-repick |
| KV alloc | 33 of 45 layers are SWA-512 | memra allocates SWA layers at **max_ctx** (window is a mask, not a smaller buffer) — `memra-kv/src/lib.rs:230-290`; per-layer FP8-for-globals KV selection already exists | **GAP (F)** for 256K (see §3.4) |

**Schedule-risk list (the NO-seam items), in build order:**
A. attn_gate separate-tensor loader arm + per-head sigmoid epilogue (+ fix the `attn_out_gate()`
   negative predicate before anything else). Small kernel, high blast radius if skipped.
B. Per-layer Q-head-count plumbing (config vector → forward/decode/spec/graph arms; KV shapes
   are unaffected — KV heads are uniform 8).
C. Rope: llama3-scaling via the `rope_freqs` tensor path + per-layer-type partial rotary
   (64-dim on full layers). Reference semantics = upstream `build_step35` (PR #23845).
D. MTP 3-layer chain (after single-layer MTP works via the existing seam).
E. PP-4 generalization of pp.rs (+ stage-owned KV for 4 stages, bit-identical gate).
F. SWA ring-buffer KV allocation (required for 256K listing; 128K works without it).
G. Device-side sigmoid router kernel (host path is correct but is a dtoh round-trip per MoE
   layer x 42 layers — measure first, then decide).
H. SwiGLU clamp in grouped-MoE/batched epilogues (2 layers only; verify exact formula).
I. StepFun chat-template dialect arm. J. deepseek-v3 pretokenizer verification.

None of these is a new kernel *class* (no Mamba/MLA/new-format) — the sku-repick "no new kernel
classes" call stands, but items A-C are genuine per-layer-shape plumbing that its "none new"
row under-sold. The 2.5-4-week listing-grade estimate stands with 4 weeks as the honest center
if 256K (F) + device router (G) + PP-4 (E) are all in scope; first-tokens bring-up
(A/B/C + loader + template) is the 1-1.5-week head of it.

## 3. Artifact path

### 3.1 Conversion route
**No conversion needed for bring-up.** Ranked:
1. **Official prequantized IQ4_XS** (105.0 GB) — uniform-expert IQ4_XS built by StepFun with
   their agentmix imatrix (4868 chunks; header receipt). The honest-quant serving artifact.
2. **Custom quant later**: official **BF16 GGUF** (394.0 GB) + official imatrix + calibration
   text are all published → `llama-quantize` (upstream ≥ 2026-06-02 or fork) produces any mix
   we want without touching safetensors. This is the route to a memra-tuned mix (e.g. Q5_K
   attention + IQ4_XS experts) if quality gates want it.
3. `convert_hf_to_gguf.py` from safetensors (402.7 GB): supported upstream since PR #23845;
   only needed if we want tensor-level control the BF16 GGUF doesn't give. Not planned.
MTP: use official `Step3.7-flash-mtp-Q8_0.gguf` (3.71 GB) via the external-draft seam
(BF16 variant 6.97 GB held in reserve for an acceptance A/B).

### 3.2 The "IQ4_XS ≈ 95.3 GB" claim — verdict: number real, label wrong
- 95.3 GB traces **exactly** to unsloth UD-IQ4_XS: 95,336,010,208 B (95.34 GB = 88.79 GiB),
  file listing 2026-08-02.
- Parsed per-tensor receipt (shard 2+3 headers): routed-expert **gate/up = IQ3_S (3.44 bpw)**,
  only down_proj = IQ4_XS; attention/shexp Q8_0, output Q6_K. Effective **3.87 bpw** on a model
  that is 96.6% expert weights. Under our own quant discipline ("Q2/Q3 is not an honest serving
  quant for a flagship endpoint" — sku-repick §1), a bank that is 2/3 three-bit is gray-zone.
- The honest uniform IQ4_XS is the **official 104,993,562,624 B = 105.0 GB = 97.8 GiB**
  (per-tensor receipt: all expert projections IQ4_XS at 4.26 effective bpw). Shape math
  predicts 105.1 GB from tensor dims x bpw — matches to 0.1% (`raw/sizes-math-20260802.out`).
- **Both fit 4x 5090** (128 GiB): official + MTP Q8_0 = 101.2 GiB → 26.8 GiB headroom;
  unsloth + MTP = 92.2 GiB → 35.8 GiB. The sku-repick "+32.7 GB headroom" line mixed the
  unsloth bytes with the official label; correct statement: **honest-quant headroom is
  26.8 GiB**, and the 95.3 GB artifact buys 9 GiB more at a quant-quality cost that the
  five-arm-style quality gate must price before we'd serve it.

### 3.3 Which artifact do we list?
Default: **official IQ4_XS** (honest quant, OR label `int4`). The unsloth artifact is the
fallback iff 256K-at-launch is mandatory before the SWA-ring work lands AND its quality gate
(perplexity + task probes vs official IQ4_XS, same prompts) passes. Decision point: end of
stage 4 below. NVFP4 (129.2 GB safetensors) does not fit 4 cards as-is and its GGUF path is
unproven — parked.

### 3.4 KV-cache budget at target context (8 KV heads x 128, K+V; receipts §4 of sizes-math)
Per token: full-attn layers 48 KiB (FP16) / 24 KiB (FP8) across the 12 full layers;
SWA layers are window-bounded (33 x 512 tokens = 66 MiB FP16 total) **if ring-allocated**.

| Allocator | KV dtype | 128K ctx | 256K ctx | Fits official-IQ4_XS headroom (26.8 GiB)? |
|---|---|---|---|---|
| memra today (SWA at max_ctx) | FP16 | 22.5 GiB | 45.0 GiB | 128K marginal; **256K NO** |
| memra today | FP8 (seam exists) | 11.25 GiB | 22.5 GiB | 128K yes; 256K only 4.3 GiB slack — no |
| + SWA ring (work item F) | FP16 | 6.1 GiB | 12.1 GiB | yes / yes (~14 GiB slack) |
| + SWA ring | FP8 | 3.0 GiB | 6.0 GiB | comfortable |

Honest listing plan: bring up at **128K** (works today with FP8-KV globals), advertise 256K
only after (F) lands and the battery re-runs at 256K. Activation/graph-pool overhead is
unknown until PP-4 shakeout — measured there, no number claimed here.

## 4. Disk budget (receipt `raw/df-20260802.txt`, 2026-08-02T14:01Z)
```
/data  1.9T  used 1.2T  avail 696G
/      1.9T  used 281G  avail 1.5T   (/home is on /)
```
- Phase-2 download set → **`/data/models/step-3.7-flash/`**: official IQ4_XS 105.0 +
  MTP Q8_0 3.7 + MTP BF16 7.0 + imatrix 0.5 + unsloth UD-IQ4_XS 95.3 (quality-gate comparator)
  = **211.5 GB → fits /data with ~485 GB to spare. VERDICT: fits, land on /data.**
- BF16 GGUF for custom quants (394.0 GB) also fits *today* (605.5 GB total, ~90 GB margin) but
  is deferred until a quality gate actually demands a custom mix — do not fetch by default.
- The 402.7 GB safetensors source is **not needed** (BF16 GGUF supersedes it for our pipeline).
- **No download started in this phase** (per lane rules). Verify sha256 against HF ETags at
  download time and record the manifest next to the artifact.

## 5. Serving quantization standard (owner kill-criteria question — decisive)

### 5a. What the incumbents actually disclose (fresh OR endpoints pull, 2026-08-02, receipt `raw/or-endpoints-step37-20260802.json`)

| Provider | `quantization` field | Ctx | Uptime 30m | $/M in / out |
|---|---|---|---|---|
| StepFun (first-party) | **fp8** | 256,000 | 99.71 | 0.20 / 1.15 |
| Novita | **fp8** | 262,144 | 100 | 0.20 / 1.15 |
| DeepInfra | **unknown** | 262,144 | 100 | 0.20 / 1.15 |

**Two of three incumbents — including the model owner — serve and disclose FP8; none discloses
below FP8.** If we list IQ4_XS (`int4`) or NVFP4 (`nvfp4`), we are the **lowest-quality-tagged
endpoint on the page**, one full precision class below the first-party endpoint. That is the
honest framing and it goes on the record here: our 4-bit listing competes as "cheaper and
faster than FP8, disclosed as 4-bit", never as a peer-precision endpoint.

### 5b. OpenRouter's disclosure/filtering mechanics (from committed `research/or-provider-20260802/REPORT.md`, citations therein, fetched 2026-08-02)

- Quantization is a **required disclosed enum** in the provider `/v1/models` schema:
  {int4, int8, fp4, mxfp4, nvfp4, fp6, fp8, mxfp8, fp16, bf16, fp32} (OR for-providers doc).
  `int4` (IQ4_XS/GGUF 4-bit class) and `nvfp4` are both valid, honest labels.
- Users can route with the `quantizations` filter; quality-sensitive users filter
  `["fp8","bf16"]` — **a sub-fp8 endpoint is invisible to that segment by construction**
  (or-provider report §3, same conclusion drawn for hy3).
- Undisclosed/`unknown` quant is a community trust penalty (r/LocalLLaMA "Be careful in
  selecting providers on openrouter" 2025-08-07; OR blog response 2026-06-12). DeepInfra's
  `unknown` on this page shows the market tolerates it — we still declare honestly.
- **Auto Exacto runs on every tool-calling request** and benchmarks endpoint accuracy against
  a fixed baseline (median − 2σ of the model's first ~21 days — i.e., *the FP8 incumbents set
  the accuracy baseline*). OR's own blog: "our own benchmark analysis found aggressively
  quantized endpoints that match full-precision competitors" — the mechanism cuts both ways: a
  4-bit endpoint that measurably degrades on their benchmark or on tool-call success gets
  deprioritized on exactly the agentic traffic that is this page's revenue (93.5:1 in:out).
  A 4-bit endpoint that holds parity is fine per OR's own published position.

### 5c. StepFun's official NVFP4 checkpoint (fetched 2026-08-02)

- **Exists**: `stepfun-ai/Step-3.7-Flash-NVFP4`, official StepFun org, modelopt (TensorRT
  Model Optimizer) format, **129.2 GB** safetensors (14 shards incl. `model-mtp-bf16` — MTP
  head kept BF16; scales/norms F32/BF16/FP8 per repo metadata). Receipts:
  `raw/hf-files-stepfun-ai-Step-3p7-Flash-NVFP4-20260802.json`,
  `raw/hf-readme-nvfp4-20260802.md`. Apache-2.0.
- **Published eval deltas vs BF16/FP8: NONE.** The NVFP4 model card carries only the generic
  Step-3.7 benchmark table (identical text across the BF16/FP8/NVFP4 cards — diff-checked);
  the StepFun blog page has no per-precision numbers (fetched 2026-08-02, receipt
  `raw/stepfun-blog-step37-20260802.html`); the vLLM recipe page has serving commands and
  hardware (4xB200 for NVFP4) but no lm_eval/accuracy section (recipes.vllm.ai, 2026-08-02).
  The official GGUF card describes IQ4_XS only as "imatrix-calibrated… comparable quality
  [to Q4_K_S]" — marketing language, not a measurement.
- What official status buys us: "the model owner's own 4-bit artifact" is a real provenance
  argument (their imatrix, their calibration set, their release QA) and is categorically
  stronger than community re-quants — but **provenance is not parity evidence**. Nobody,
  vendor included, has published what 4-bit costs this model. §5e's receipts are therefore
  not optional paperwork; they are the only quality evidence that will exist.
- Practical note: NVFP4 safetensors are 129.2 GB = 120.4 GiB → weights alone leave 7.6 GiB
  across 4 cards; with KV + runtime it is a 5-card artifact as-is (`raw/sizes-math` §6), and
  its GGUF path is unproven. The 4-card 4-bit artifact remains the official **GGUF IQ4_XS**
  (105.0 GB) — which shares provenance (StepFun-built, StepFun imatrix) with the NVFP4 repo.

### 5d. FP8 fit math — why FP8-Step is out of the envelope (receipt `raw/sizes-math-20260802.out` §6)

| Artifact | Bytes | GiB | Cards (32 GiB) weights-only | + 128K FP8-KV + ~7 GB runtime |
|---|---|---|---|---|
| FP8 safetensors (official) | 212.5 GB | 197.9 | **7** | 216 GiB → 7 cards (8 for headroom) |
| Q8_0 official GGUF | 209.4 GB | 195.0 | **7** | 213 GiB → 7 cards (8 for headroom) |
| NVFP4 safetensors | 129.2 GB | 120.4 | 4 (7.6 GiB slack) | 138 GiB → 5 cards |

Serving Step at the incumbents' FP8/8-bit standard needs **7-8 RTX 5090s — outside the 2-4
card envelope** that defines this box (sku-repick's premise). There is no 8-bit Step SKU for
us; the choice is 4-bit Step on 4 cards, or no Step. That is the decision this section prices.

### 5e. Honest verdict

**A 4-bit Step listing is *conditionally* defensible — defensible only with receipts that do
not yet exist anywhere, and with two pre-registered kill conditions.**

The case for: the artifact is the **model owner's own** quant lineage (official GGUF, official
imatrix + calibration text); OR's disclosure system is built for exactly this (declare `int4`,
compete on price/speed, let Auto Exacto verify); OR states aggressively-quantized endpoints
*can* match full-precision competitors; and the page's buyers at `:floor` are price-first.
The case against, stated with equal weight: **both disclosing incumbents are FP8, the
first-party endpoint sets the Auto Exacto accuracy baseline at FP8, the quality-filtered
segment (`quantizations:["fp8","bf16"]`) never sees us, and zero published evidence says
4-bit Step holds quality.** Under the owner rule "never deliver a weak running model", absence
of degradation evidence is not evidence of absence — we generate the evidence or we don't list.

**Required receipts before any OR application (all committed to the repo, N stated, raw runs
per evidence discipline):**
1. **Eval parity**: official IQ4_XS vs FP8 reference on an agentic-weighted battery (tool-use
   + code + reasoning probes; reference = StepFun's own API endpoint (fp8) or local Q8_0 GGUF
   spot-runs on rented HW), same prompts/template/settings, temp 0 where applicable. Gate:
   within noise of the FP8 reference on the battery aggregate; any single-suite drop > a
   pre-registered margin fails the gate.
2. **Tool-call success rate** vs the same reference on multi-turn tool trajectories (this is
   what Auto Exacto meters on 93.5:1 traffic). Gate: parity; this is the revenue-critical one.
3. Stage-4 artifact honesty gate (§6): official IQ4_XS vs unsloth UD-IQ4_XS — we do not serve
   the 3.87-bpw artifact unless it passes the same battery.
4. Listing declares `int4` honestly; published eval receipts linked from our provider page.

**Pre-registered kill/re-arb conditions (plain statement):** incumbents ARE FP8 (5a — the
condition is live, not hypothetical). Therefore: **if the 4-bit artifact shows real degradation
on receipts 1-2, that is the SKU re-arb trigger** — do not ship a degraded flagship; fall back
to the sku-repick bench (MiniMax-M2.7 PP-4, or Laguna-S/XS positioning play) or hold Step until
a 5th card makes NVFP4-class serving honest. Likewise, if Auto Exacto deprioritizes the live
endpoint below the incumbents on accuracy/tool-success after listing, the same trigger fires:
delist rather than run a weak endpoint.

### 5f. Launch lane is unaffected

The launch SKU (Qwen3.6-35B-A3B on 2 cards) lists at **Q8_0 — an 8-bit (`int8`) endpoint that
meets the incumbents' disclosure standard outright** (35B @ Q8_0 ≈ 37 GB fits 2x32 GiB with
full KV headroom). One honesty note so nothing hides: the current board rows (178.2/302 tok/s)
were measured on the UD-IQ4_XS artifact (Q8_0 trunk + 4-bit experts,
`research/verify-economics-20260802/RESULTS.md`); the Q8_0-artifact numbers need their own
rows before the listing quotes speeds — a re-measure, not a bring-up. This whole section gates
only the Step flagship; nothing here delays the launch listing.

## 6. Staged checklist with gates

Hardware note (sequencing dependency): PP-4 and the head-to-head need a **4-card** box; per
sku-repick §7 the box is at 2 cards until grown. Stages 1-5 run on 2 cards via PP-2 + the
Hy3-lane expert-spill paths (correctness-only, low ctx); stages 6-7 gate on card 4.

| # | Stage | Work | Gate (hard) |
|---|---|---|---|
| 0 | Download + stage | §4 set → /data; manifest hashes | sha256 match, manifest committed |
| 1 | Onboarding | `Arch::parse` += `step35`; config plumbing (head-count **arrays**, swa pattern, freq_base_swa, gating_func=2, leading_dense=3, clamp arrays, nextn KV); loader arms (separate `attn_gate`, MTP-external); **fix `attn_out_gate()` predicate first**; risk items A-C | loader dry-run maps all 754 tensors, zero silent fallbacks; kernel-check GREEN on step35 shapes |
| 2 | Template/tokenizer byte-check | StepFun dialect arm in chat.rs (reasoning_effort, forced `<think>`, `<function=>` tools); deepseek-v3 pre-regex check | token-ids byte-identical vs HF `apply_chat_template` on a battery incl. tools + all 3 reasoning levels |
| 3 | SWA+MoE forward | gemma4-seam wiring for 3:1 SWA-512; sigmoid-router host path; shexp; clamp on 43-44 | run-gen **argmax MATCH** vs llama.cpp upstream master, same GGUF, temp 0, fixed prompts (2-card PP-2/spill, ctx-limited); `MEMRA_FAST=0` oracle parity |
| 4 | Quality gate + artifact pick | official IQ4_XS vs unsloth UD-IQ4_XS: perplexity + task probes, same prompts/settings; **plus the §5e serving-standard receipts**: eval parity + tool-call success vs an FP8 reference (StepFun API or rented-HW Q8_0 spot-runs) | pick recorded with receipts (§3.3); §5e gates pass or the **SKU re-arb trigger** fires (fall back per §5e); no public-eval-driven tuning |
| 5 | MTP integration | official MTP Q8_0 via `MEMRA_MTP_DRAFT` single-layer seam (blk.45); then optional 3-layer chain (D) | run-spec K=1..8 self-consistency PASS; acceptance ≥ own-gen baseline else ship plain |
| 6 | PP-4 shakeout | pp.rs 4-stage (E); SWA-ring KV (F) if 256K in scope; device sigmoid router (G) if stage-3 profiling says host path is the bottleneck | PP-4 logits **bit-identical** to PP-2/spill reference (m1-pp2 method); memory fit measured at 128K (and 256K if F landed) |
| 7 | Battery + head-to-head | full battery on the 4-card rig; **N=5 interleaved** vs StepFun fork branch `step3.7` AND upstream master, same box/same day (clock-drift law); **prefill measured** at pp512-class + long-prefill (agentic 93.5:1 shape — prefill IS the revenue; no pricing before this number exists) | kernel-check ALL GREEN; run-gen argmax MATCH; run-spec K=1..8 PASS; ≥1.1x e2e deployment bar; prefill receipt + §5e quant-standard receipts in repo before any OR application |

Per project rules: raw sweep JSONL committed next to every summary row; failure causes quoted,
never inferred; every median states N + thermal regime; no `--no-verify`.

## Source index
Fetched 2026-08-02 unless noted: HF model card + file listings via HF API `?blobs=true`
(receipts `raw/hf-files-*.json`) for stepfun-ai/{Step-3.7-Flash,-FP8,-NVFP4,-GGUF} and
unsloth/Step-3.7-Flash-GGUF; config.json + chat_template.jinja + tokenizer_config.json from
stepfun-ai/Step-3.7-Flash; GGUF headers parsed locally (`raw/parse_gguf.py`) from the unsloth
UD-IQ4_XS metadata shard (full download, 5.2 MB) and range-requests of official IQ4_XS shard 1
+ official mtp-Q8_0 (receipts `raw/gguf-header-*.txt`); github.com/ggml-org/llama.cpp PRs
#23845/#23274/#19283/#19271 + issue #24257 + stepfun-ai/llama.cpp fork state via gh API
(receipt `raw/llamacpp-pr-status-20260802.json`); disk `raw/df-20260802.txt`.
§5 additions, fetched 2026-08-02: fresh OR endpoints pull `openrouter.ai/api/v1/models/
stepfun/step-3.7-flash/endpoints` (receipt `raw/or-endpoints-step37-20260802.json`); HF repo
READMEs for -NVFP4/-FP8/-GGUF (receipts `raw/hf-readme-{nvfp4,fp8,gguf}-20260802.md` — the
three cards share identical benchmark text, no per-precision evals); StepFun blog page
(receipt `raw/stepfun-blog-step37-20260802.html`, zero fp8/fp4 eval mentions); vLLM recipe
recipes.vllm.ai/stepfun-ai/Step-3.7-Flash (no accuracy section; 4xB200 for NVFP4).
Committed data: `research/sku-repick-20260802/REPORT.md` (demand/economics, in:out 93.5:1 —
not re-verified here); `research/or-provider-20260802/REPORT.md` (OR disclosure enum,
`quantizations` filter, Auto Exacto mechanics, community-trust precedent — citations therein).
memra seam locations: repo state at branch point 86b22e85, paths cited inline in §2.
