
## Sweep 2026-07-15T07:30:04Z (since 2026-07-15T00:00:00Z)

### llama.cpp commits (decode-relevant, CUDA)
- (none)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-07-27T06:17:01Z (since 2026-07-15T07:30:04Z)

### llama.cpp commits (decode-relevant, CUDA)
- cohere2 moe template parser: enforce JSON schema for text responses if a response schema is provided (#26018)
- common : auto-download dflash- and eagle3- HF sidecars (#25811)
- conversion: fix non-MoE NomicBert GGUF conversion error (#25996)
- convert : fix dflash target tokenizer mismatch during conversion (#25733)
- cuda: add sqrt_softplus in topk-moe for dsv4 (#25896)
- cuda : CUDA GGML_OP_LIGHTNING_INDEXER implementation (generic vector kernel + wmma kernel) (#25545)
- CUDA: dedup MoE gate/up activation quantization (#25441)
- cuda: extract Q1_0 elements via __byte_perm (#25628)
- CUDA: fix external compilation of q1_0 MMQ (#25778)
- cuda: GET_ROWS quants (#25962)
- CUDA: Improve NVFP4 W4A4 activation quantization (#25730)
- cuda : relax tensor contiguity requirements for quantized concat (#25678)
- CUDA: Support CUDA Virtual Devices (#25228)
- CUDA: tighter MMQ src1 buffer size for native fp4 (#25613)
- CUDA: vectorize same-type get_rows with int4 copy (#25929)
- DeepseekV4: Add fused hyper-connection ops (#25585)
- DeepseekV4: reduce graph splits (#25702)
- Enable CUDA graphs on volta+turing (#25749)
- metal: fuse snake activation (mul, sin, sqr, mul, add) (#25459)
- model: rotate injected K/V cache for DFlash (#25823)
- mtmd : use align_corners for qwen3vl vision position embedding interpolation (#25781)

### vllm-project/vllm releases
#### v0.26.0 (2026-07-27T01:06:58Z)
- * **New Inkling model family** with a full support stack: base modeling (#48799), piecewise CUDA graph support (#48822), Hopper FA4 relative attention (#48858), MTP=1 speculative decoding (#48869), LoRA (#48884), and standard ModelOpt NVFP4 quantization (#48990).
- * **DeepSeek-V4 performance push** across vendors: a specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` (1.5–2x kernel, #47463), and redundant repeat/copy removal (1.8% E2E TPOT, #48137), plus ROCm two-stage compressor for HCA prefill (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), and DSpark speculative decoding on AMD (#47419) and XPU (#47677).
- * **Flexible attention backends**: the attention backend can now be selected per KV-cache group (#48012), and sliding-window support is now an explicit backend capability (#48011) — improving support for hybrid models.
- * **KV offloading & tiered secondary storage** matured substantially: offloading metrics (#45958, #47666, #47679), tier-owned event handling (#46544, #47923), object-store secondary tier with workload identity (#47063, #47274, #48150), DP-replica-aware tiering (#47987), and encoder-cache (EC) connectors including CPU offloading (#42433, #47423).
- * New models: Inkling family (#48799, #48822, #48858, #48869, #48884, #48990), BertForMaskedLM (#48463), RobertaForTokenClassification / XLMRobertaForTokenClassification (#47991), LongCat-Flash-Lite n-gram embedding (#47857), Cosmos3 Edge Reasoner (#48291) and Cosmos3-Super registration (#48211), TranslateGemma-12b-it (#41599).
- * GLM5.2: migrate MoE sequence-parallel support to the non-torch-compiled path (#47881).
- * LoRA: FlashInfer MoE LoRA for BF16 models (#48632), LoRA for tower/connector in LlavaNextVideo (#48594), fp32 `lm_head` on the LoRA path (#48525), optimized `TrtLlmLoRAExperts` (#48759).
- * fp32 `lm_head` for generation models via `head_dtype` (#48390); lower memory for capturing large CUDA graph sizes (#48483); opt-in persistence and reuse of the memory-profiling result across boots (#47388); improved InstantTensor loading (#46868).
- * Attention: select a different attention backend per KV-cache group (#48012); sliding-window as an explicit backend capability (#48011); KV-cache layout refactor packing K/V into the content dim across backends (#44455); MRV2 virtual-batch PCP for MLA (#46570).
- * Speculative decoding: runtime draft weight update (#46725), hybrid (SWA + full attention) DFlash drafters (#47914), SWA support for qwen-eagle3 (#47568), Gemma4-12B DSpark draft model (#47216), DSv4 DSpark on AMD (#47419), separate `kv_cache_dtype` for `speculative_config` (#48787).
- * KV offloading: basic offloading metrics (#45958), split CPU cache usage into read/write gauges (#47666) and tiering-lookup-delay into sync/async histograms (#47679), tier-owned event handling and BlockStored events (#46544, #47923), object-store secondary tier with workload identity (#47063, #47274, #48150), DP-replica-aware tiering (#47987), `blocks_per_chunk` config for heterogeneous KV groups (#48878), P2P default host/port env vars (#47636).
- * DeepSeek-V4: specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` 1.5–2x (#47463), redundant repeat/copy removal (1.8% TPOT, #48137).
- * MoE router GEMMs: BF16x3 router GEMM (#47973), FP32 router GEMV (#48335), generic CuteDSL LL BF16 router GEMM (#42562); TRTLLM BF16 MoE modular kernel (#45182); write FlashInfer combine into final output (#47156).
- * Qwen: fuse more RMSNorm + all-reduce in Qwen3.5 (#46998), replace MoE all-reduce with reduce-scatter (#47006), Qwen3.5 H20 optimization (#48350), expand Triton warmup coverage (#47546).
- * MLA: dense MHA path for short sparse-MLA sequences (#47327); MiniMax-M3 long-context decode indexer on sm100 (#48582).
- * Kernels: CUDA kernel for ReLUSquaredActivation / relu^2 (#39058), Helion kernel lazy registration (#48264), vectorize `_copy_mamba_state_block` to uint64 (#48110), stop upcasting logits to fp32 in the sampler (#48641).
- * ROCm: fp32 `head_dtype` `torch.mm` fast path (#48688), DSv4 two-stage compressor kernel (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), DSv3.2 sparse MLA KV-split heuristic (#46832) and MTP CUDA-graph mode (#45149), MXFP8 GEMM for MiniMax-M3 (#46117), AITER sparse paged attention + spec decode for MiniMax-M3 (#47287, #47984), MiniMax-M2 fused QK-norm + all-reduce via AITER (#44849), HybridW4A16 linear kernel (#40977), Qwen3-30B-A3B QK-Norm+RoPE+KV runtime fusion (#42749).
- * XPU: batch-invariant kernels (#41934), HND KV layout support (#47975), DSpark spec decode for DSv4 (#47677), nightly/release image publishing (#47880, #48126).
- * CPU: DFlash speculative decoding for GDN models on CPU (#46090), s390x NUMA topology (#40714), native macOS arm64 CPU wheel builds (#48289); POWER VSX math function optimization (#47321) and IBM Power docker builds using prebuilt wheels (#46017).
- * Distributed fusion: FlashInfer MNNVL all-reduce RMS quant fusion (#48064).
- * Build/autotune: arm64 Blackwell SM10x/SM110 image builds (#48041); skip CuTeDSL fp4_gemm autotuning by default (#48268).
- * Decode Context Parallel (DCP): hybrid attention support (#40996), DCP + Eagle for Tokenspeed MLA backends (#48180).
- * Humming w[2-7]a[4,8] weight-only inference with compressed-tensors (#46390); int4 quantization for the emulation MoE backend (#48451); INT2 XPU weight-only quant linear (#47521).
- * NVFP4/MXFP4: `nvfp4_per_token` online MoE quantization (#48538), CuTe-DSL FlashInfer MXFP4 quantization (#48417); bounded peak memory when repacking FP4 MoE weights for Marlin (#47851) and for NVFP4 MoE weight loading (#46276).
- * MLA: `kv_cache_dtype_skip_layers` support (#47309).
- * Transformers 5.13.0 (#47867), FlashInfer 0.6.14 (#47669), NIXL 1.3.1 (#47559), tpu-inference v0.24.0 (#47835), nvidia-cutlass-dsl 4.6.0 (#47442), vllm_xpu_kernels v0.1.11.1 (#48942).
- * FlashAttention 3 pinned to the torch stable-ABI commit (#47995); ABI-stable FlashMLA build (#48174).

### sgl-project/sglang releases
#### v0.5.16 (2026-07-25T00:13:18Z)
- **DSpark: confidence-driven speculative decoding**: A new speculative algorithm. It drafts semi-autoregressively in blocks, then sizes each verify window from the draft's own confidence instead of a fixed draft length. Reaches **383.7 tok/s at accept length ~5** on DeepSeek-V4-Pro, TP8 on B300 (bs=1). Enable with `--speculative-algorithm DSPARK` and `SGLANG_RAGGED_VERIFY_MODE=compact`; tune the block with `--speculative-dspark-block-size` ([#30261](https://github.com/sgl-project/sglang/pull/30261), [#31434](https://github.com/sgl-project/sglang/pull/31434), [blog](https://www.lmsys.org/blog/2026-07-06-dspark-sglang)).
- **Inkling support**: A 975B-parameter multimodal MoE with a 1M-token context. It mixes sliding-window, full and Mamba2 linear attention, and adds an NVFP4 MoE, optional vision/audio towers and native MTP. On Blackwell it reaches up to **71.7k tok/s input** and **171.0 tok/s per-user decode**. Verified on Blackwell TP4/TP8, H200 and AMD MI350X / MI355X ([#31681](https://github.com/sgl-project/sglang/pull/31681), [blog](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support), [cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling)).
- **Other new models added**: [LongCat 2.0 FP8](https://docs.sglang.io/cookbook/autoregressive/Meituan/LongCat-2.0), JetBrains Mellum v2, [Pi0.5](https://docs.sglang.io/cookbook/vla/OpenPI/Pi0.5), plus diffusion support for [LongLive 2.0](https://docs.sglang.io/cookbook/diffusion/LongLive/LongLive-2.0).
- **GLM-5.2 DSA cache layer split under prefill CP**: KV and indexer cache layers are sharded across CP ranks. Each rank owns a disjoint layer range instead of all layers. That cuts per-rank KV memory by **~74%** (0.77 to 0.20 GB/rank) at 8192 tokens on GLM-5.2-FP8, 78 layers, cp_size=4. Enable with `--enable-dsa-cache-layer-split`, which needs `--enable-prefill-cp --cp-strategy interleave` ([#29421](https://github.com/sgl-project/sglang/pull/29421)).
- **ReplaySSM Ring Spec-Verify (GDN)**: Drops the per-draft SSM snapshot. Speculative scratch goes from **11.5 GB to 1.8 GB per GPU (6.4x smaller)** on Qwen3.5-35B-A3B at TP1, at accuracy and throughput parity. Opt in with `--enable-gdn-replayssm-spec` (default off; GDN with a linear draft chain only, `--speculative-eagle-topk` in {None, 1}), and tune the ring via `--linear-replayssm-cache-len` ([#28695](https://github.com/sgl-project/sglang/pull/28695)).
- **Linear attention on Blackwell (SM100)**: The first correct KDA MTP path. Its `recurrent_kda` decode kernel runs at **29.6 us vs 36.8 us** for Triton (ncu, B=64). The full decode path reaches parity by B=128 and **1.35x at B=256**, and is slower below that ([#30113](https://github.com/sgl-project/sglang/pull/30113)). Separately, GDN/KDA CuteDSL prefill fuses state I/O into the chunk-h kernel ([#30169](https://github.com/sgl-project/sglang/pull/30169)).
- **QServe and FBGEMM FP8 quantization are removed**: the experimental QServe (QoQ) W4A8 and FBGEMM FP8 paths are gone. `--fp4-gemm-backend cutlass` goes too, along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer ([#31109](https://github.com/sgl-project/sglang/pull/31109), [#30448](https://github.com/sgl-project/sglang/pull/30448)).
- **Dependencies**: flashinfer 0.6.14 ([#29910](https://github.com/sgl-project/sglang/pull/29910)), CuTe DSL 4.6.0 ([#31714](https://github.com/sgl-project/sglang/pull/31714)), sgl-kernel 0.4.5 ([#31496](https://github.com/sgl-project/sglang/pull/31496)), llguidance 1.7.6 ([#31484](https://github.com/sgl-project/sglang/pull/31484)).
- * **`--fp4-gemm-backend cutlass` is removed** along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer. Use `auto`, which picks `flashinfer_cutedsl` on SM100 and `flashinfer_cutlass` on SM120: [#30448](https://github.com/sgl-project/sglang/pull/30448)
- * **The SGLang-Diffusion post-training rollout endpoint now returns `application/msgpack`** instead of JSON, with tensors as raw msgpack bytes rather than base64 (`tensor_to_base64` / `base64_to_tensor` become `tensor_to_bytes` / `bytes_to_tensor`), so RL rollout consumers must be upgraded in lockstep with the server: [#31565](https://github.com/sgl-project/sglang/pull/31565)
- * **Temperature-0 nondeterminism under DP attention with breakable prefill CUDA graph.** On the DSV4-Flash FP4 recipe, the idle-rank dummy extend introduced by [#30898](https://github.com/sgl-project/sglang/pull/30898) perturbs real requests' logits, so identical temperature-0 requests can diverge. The guarding determinism test is disabled as a stopgap rather than fixed ([#31125](https://github.com/sgl-project/sglang/pull/31125)); not enabling breakable prefill CUDA graph avoids the path.
- * A bump to **flashinfer 0.6.15** was landed and reverted this cycle; this release pins **0.6.14** ([#31502](https://github.com/sgl-project/sglang/pull/31502), [#31625](https://github.com/sgl-project/sglang/pull/31625)).
- * **CPU AMX optimizations for diffusion** were reverted ([#28527](https://github.com/sgl-project/sglang/pull/28527), [#30716](https://github.com/sgl-project/sglang/pull/30716)).
- | **LongLive 2.0** | diffusion | [#27639](https://github.com/sgl-project/sglang/pull/27639) | [link](https://docs.sglang.io/cookbook/diffusion/LongLive/LongLive-2.0) |
- * [Docs] Inkling cookbook: LoRA cells require --disable-prefill-cuda-graph: [#31418](https://github.com/sgl-project/sglang/pull/31418)
- * [Spec] fix inkling multi layer mtp draft extend cuda graph: [#32254](https://github.com/sgl-project/sglang/pull/32254) (cherry-picked as [#32260](https://github.com/sgl-project/sglang/pull/32260))
- * [Fix] Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay: [#30839](https://github.com/sgl-project/sglang/pull/30839)
- * [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly: [#28416](https://github.com/sgl-project/sglang/pull/28416)
- * Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing): [#31001](https://github.com/sgl-project/sglang/pull/31001)
- * [DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill into the DSA Backend (DeepSeek-V3.2): [#30514](https://github.com/sgl-project/sglang/pull/30514)
- * Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2: [#30272](https://github.com/sgl-project/sglang/pull/30272)
- * [DSA] Fix top-k v2 emitting invalid indices under tie overflow / inf scores (IMA in FA3 sparse decode): [#30645](https://github.com/sgl-project/sglang/pull/30645)
- * [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph: [#31705](https://github.com/sgl-project/sglang/pull/31705)
- * Fix nvfp4 online scale with pcg: [#32246](https://github.com/sgl-project/sglang/pull/32246) (cherry-picked as [#32259](https://github.com/sgl-project/sglang/pull/32259))
- * Fix stale flashinfer-MLA fallback poisoning spec verify capture (trtllm_mla + tc_piecewise): [#32288](https://github.com/sgl-project/sglang/pull/32288) (cherry-picked as [#32346](https://github.com/sgl-project/sglang/pull/32346))
- * flashmla: sync-free spec via device-side draft-extend: [#31090](https://github.com/sgl-project/sglang/pull/31090)
- * [Spec] DFlash: remove per-step host syncs so the CPU runs a full step ahead (spec-v2 overlap): [#31468](https://github.com/sgl-project/sglang/pull/31468)
## Piecewise & Breakable CUDA Graph
- * Enable breakable prefill CUDA graph for DP attention: [#30898](https://github.com/sgl-project/sglang/pull/30898)
- * feat: enable piecewise prefill graph for Kimi K2.5/K2.7: [#30889](https://github.com/sgl-project/sglang/pull/30889)
- * [Diffusion] Enable breakable CUDA graph (BCG) for diffusion DiTs: [#27436](https://github.com/sgl-project/sglang/pull/27436)
- * [KDA] Add FlashInfer SM100 KDA decode + MTP (target_verify) backend: [#30113](https://github.com/sgl-project/sglang/pull/30113) ⭐
- * [GDN/KDA] Fuse SM100 CuteDSL prefill state I/O into the chunk h kernel: [#30169](https://github.com/sgl-project/sglang/pull/30169) ⭐
- * [GDN] Auto-select FlashInfer GDN prefill on validated SM100 configs: [#29734](https://github.com/sgl-project/sglang/pull/29734)
- * [Feature] Add FP4 KV Cache Design and support SM120 GPUs: [#21601](https://github.com/sgl-project/sglang/pull/21601)
- * Fuse the preprocess kernels of trtllm-gen attention: [#29690](https://github.com/sgl-project/sglang/pull/29690)
## MoE & Expert Parallelism
- * Support Waterfill with MegaMoE backend: [#27350](https://github.com/sgl-project/sglang/pull/27350)
- * Support Flashinfer one-sided A2A + CuteDSL MoE for Nemotron Ultra: [#28309](https://github.com/sgl-project/sglang/pull/28309)
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-03T21:57:39Z (since 2026-07-27T06:17:01Z)

### llama.cpp commits (decode-relevant, CUDA)
- add rdna3.5, and 3 to mmq configs so they can be tuned independently. (#26199)
- chat : add qwen3 specialized parser (#26252)
- conversion: fix Qwen2.5-Omni mmproj conversion regression (#26262)
- CUDA: Add backend sampler for penalties sampler (#25262)
- CUDA: add Q2_0 support (#25707)
- cuda: extract Q2_0 elements via __byte_perm (#25603)
- CUDA: Fix data-races when reusing SMEM in block_reduce (#26385)
- DeepseekV4 MTP + DSpark (#25784)
- ggml-cuda: add chunked SSD matmul for Mamba-2 prefill acceleration (#22675)
- ggml-cuda: Allow transpose-free gemmv computation (#26171)
- ggml-cuda : disable MMQ on devices with less than 48 KiB shared memory (#26141)
- ggml: use dynamic allocation for split graph inputs (#22789)
- graph : fix unused input tensors in minimax m3 graph (#26519)
- model: MTP support for Qwen3-Next (#25589)
- model : support MTP in GLM-4.7-Flash (#24868)
- Remove custom cpu op from the M3 graph, express with stock ops (#26297)
- Support rotated kv cache quant (#26180)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-05T12:20:53Z (since 2026-08-03T21:57:39Z)

_Frame: self-competition (llama benching stopped 2026-08-03); upstreams are idea mines._
_Filter: sm_120/Blackwell, FP8/FP4 (FP8-ST program = priority 1), MTP/spec-decode_
_(MTP-p2 lane), chunked-prefill determinism (open chunk-order reduction-stability_
_finding), session/prefix affinity, batch-invariance._

### llama.cpp commits (decode-relevant, CUDA)
- #26605 — fit: include NextN/MTP layers in auto VRAM fitting (alloc crash with
  `--n-cpu-moe=0`). CARE: MTP-p2 must account for draft-layer VRAM in memra's budget
  fitting; check our accounting before the lane merges. lever-queue: no (checklist item).
- #26389 — server: spec-decode counters on /metrics, incl. per-draft-position accepted
  counter (vLLM schema). CARE: dogfood found short-ctx sampled acceptance 0.55 vs 0.73 —
  per-position acceptance telemetry in memra-serve /metrics turns that from a posthoc dig
  into a live gauge. lever-queue: YES (serve, small).
- #26510 — speculative: refactor common_speculative_init config dedup. SKIP: code hygiene only.
- #26524 — samplers: drop "full-context window" from history-based penalizers (backend
  sampler init-order constraint). SKIP for the mechanism; NOTE: llama continues moving
  samplers GPU-side (#25262 family) — same direction as our lsampler.
- #26531/#26577 — loader: allow tensor reshape during load + dflash wo_a fix. SKIP: loader flexibility.
- #7bd8282c3 #26510 / #26618 / #26254 / mtmd batch — SKIP: conversion, TTS, multimodal.
- PR #24364 (OPEN, updated 2026-08-05) — force NVFP4 W4A8 path for W4A16_NVFP4 layers on
  Blackwell via new `GGML_HINT_NO_QUANT_SRC1` hint (GGUF metadata-driven, mul_mat +
  mul_mat_id); measured quality gain vs native W4A4, cites W4A4-hard-for-small-LLMs.
  CARE: converges exactly with FP8-ST — FP4 weights x FP8 activations as the
  quality-safe alternative to W4A4; bears directly on nvfp4-strict lane arm design.
  lever-queue: YES (priority — track the PR, steal the W4A8 framing).

### vllm-project/vllm commits
- #40372 (merged) — batch-invariant NVFP4 MoE via cutlass: invariance pinned with static
  asserts in the .cu + a dedicated bs1-vs-bsN test so nobody breaks it silently. CARE:
  the *pattern* — treat batch-invariance as a gated property with asserts + tests, not a
  posthoc observation. Adopt for memra's chunk-order stability work. lever-queue: YES.
- #38561 (closed for rebase, not rejected) — batch-invariant chunked prefill for
  mamba/hybrid: force chunk splits ONLY at mamba chunk boundaries + align prefix-cache
  entries to the same grain. CARE: DIRECT hit on our open chunk-order
  reduction-stability finding — vLLM independently hit reduction-order instability
  across chunk sizes and fixed it by constraining split points to a fixed grain.
  lever-queue: YES (top candidate).
- #45683 (open) — deterministic MoE combine under VLLM_BATCH_INVARIANT: fixed-root
  reduce+scatter instead of routing-dependent reduce_scatterv. CARE: more evidence
  reduction-order instability is a live upstream problem (cross-rank flavor); memra is
  single-GPU so the fix itself is N/A. lever-queue: no (evidence note).
- #50992 — KV-offload ARC batch eviction was quadratic (rescan from head per evicted
  block); monotonic iterators + deferred mutations = 16-81x on eviction batches. CARE:
  dogfood found F5 evict-realloc on 12/12 requests — audit memra's host-cache/SLRU
  eviction for the same rescan shape while fixing F5. lever-queue: YES (pairs with F5).
- #48048 — first-class request-level `session_id` (HTTP X-Session-ID + engine plumbing),
  explicitly built for KV/prefix affinity consumers. CARE: session/prefix affinity for
  memra-serve (owner dogfoods multi-turn agentic); a session key is the cheap
  prerequisite for prefix-cache pinning. lever-queue: YES (serve, medium).
- #49969 — DSpark top-k Markov projection: select top-k from base logits once, apply the
  sequential per-step draft bias only to those candidates, scatter into dense row,
  sampling path unchanged. CARE: draft-cost pattern for MTP-p2 — restrict per-step
  draft-side correction to a candidate set instead of full vocab. lever-queue: YES.
- #50911 — fused non-causal TokenSpeed MLA for DSpark verify. SKIP: MLA-specific.
- #49792 — SM100 CuTeDSL fused query kernel (DSA fused_q). SKIP: model/arch-specific.
- #48861 — NVFP4 quant out_dtype must match model dtype, not torch default. SKIP: their-stack bug.
- #50323 — CI: fail evals when NaNs appear in logits. NOTE: cheap evidence-discipline
  gate, matches our battery philosophy. lever-queue: no.

### sgl-project/sglang commits
- #33063 — trtllm_mha decode: stop allocating per-layer scratch inside the decode CUDA
  graph — 3 fill launches/layer baked into every replay, AND they sat between PDL-linked
  kernels so the PDL dependent-launch chain signalled a FillFunctor instead of the real
  consumer. CARE: memra graph-decode + PDL on sm_120 — audit capture for allocs/fills
  inside the graph and for PDL chain breakage. lever-queue: YES (top candidate).
- #33306 — avoid TRTLLM prefill output copy: kernel writes directly into the
  preallocated piecewise-graph output buffer via `out=`. CARE: same audit for memra
  prefill outputs — any epilogue copy into a graph buffer is free meat. lever-queue: yes (small).
- #32575 — build empty-prefix last_loc sentinel on-device: inline `torch.tensor([-1])`
  per fresh request = pageable H2D + stream sync draining the in-flight forward. CARE:
  audit memra-serve batch-prep for host-materialized scalars/masks; the F5/prime-chunk
  work touches exactly this region. lever-queue: yes (audit item).
- #33545 — optimistic prefill with L2 hierarchical cache + write-back. NOTE: prefix-cache
  tiering direction; memra single-box, low priority. lever-queue: no.
- #16072-adjacent SGLang #33427/#33598 — post-capture KV sizing + clearer reservation
  logs. SKIP: parity, memra sizes before capture.
- #30206 (merged this window) — capture legal multi-request prefill CUDA graph batches.
  SKIP: their piecewise-prefill bug.
- #81c7a54ec — sm_100f (family) instead of sm_100a for sgl-kernel. NOTE: family-arch
  flag now load-bearing upstream; memra stays sm_120a per doctrine. SKIP.

### flashinfer-ai/flashinfer commits
- #4210 — remove NVFP4 TMA input padding copy: TMA tensor map describes physical M rows,
  G2S TMA out-of-bounds zero-fill covers padded scale-layout tiles — kills the
  non-128-aligned-M copy cliff (212µs -> 59µs at M~32k, B200). CARE: FP8-ST/NVFP4
  prefill activation-quant path + the prime-chunk/prefill wall — odd-M chunks are
  exactly where padded staging copies bite; the OOB-zero-fill trick is HW-portable to
  sm_120 TMA. lever-queue: YES (top candidate).
- #4263 — 64-bit row addressing in per-token NVFP4 quantizer (int32 overflow at large
  M*K). CARE: one-line audit of memra quantizer index math at 27B prefill shapes. lever-queue: audit.
- #3523 — sm120 groupwise GEMM called cudaGetDeviceProperties (~1.7ms, synchronous) per
  invocation with dead results. CARE: grep memra host launch paths for per-call device
  queries. lever-queue: audit.
- #4202 — fp8 e5m2 output in rmsnorm_quant / fused_add_rmsnorm_quant. NOTE: fused
  norm->FP8-quant epilogue is the FP8-ST activation-side lever; they now cover both
  e4m3/e5m2. lever-queue: yes (FP8-ST reference impl).
- cake_kda / delta-rule / MxInt4 MoE family — SKIP: model- or SM100-specific.

### NVIDIA/TensorRT-LLM commits
- #17234 — opt-in pinned staging for weight-load H2D: pageable H2D degrades to a ~260µs
  per-work-item driver polling crawl on GB300 (36min-4h loads). PARITY: memra spill
  doctrine already mandates bounded pinned host buffers — this is the receipt for why.
  lever-queue: no (doctrine confirmed).
- #16072 — pre-allocate CUDA-graph padding dummies during warmup; lazy alloc under KV
  saturation silently dropped graph coverage for the process lifetime. CARE: verify
  memra graph-decode preallocates padding resources at warmup, and that a failed graph
  path is loud, not a silent eager fallback. lever-queue: yes (gate item).
- #17159 — reject NaN top_p/min_p/temperature in SamplingParams. CARE: fold NaN
  validation into the lsampler top_p/min_p truncation bugfix already queued from
  dogfood. lever-queue: yes (rides the existing fix).
- #17172 — per-request seed in TorchSampler. PARITY: memra sampled-regime gates already
  seeded. SKIP.
- #16957 — TriAttention KV-cache compression (training-free decode-time eviction, ICML
  2026): score evictable region every beta tokens, keep budget, compact in place. NOTE:
  research file for long-context serve; quality-affecting, not a near lever. lever-queue: no.
- #17106 — enable KV block reuse for flashinfer backend. SKIP: parity.
- #16558 — DSA indexer k-cache only for owning layers. SKIP: model-specific.

### Ranked shortlist (lever candidates, with receipts)
1. **Chunk-boundary-pinned prefill invariance** (vLLM #38561 + #45683 + #40372): two
   independent vLLM lanes hit reduction-order instability (chunked prefill across chunk
   sizes; MoE combine across routing) and both fixes are the same shape — constrain the
   reduction segmentation to a fixed grain, then pin the property with static asserts +
   a bs/chunk-invariance test. Direct answer to our open chunk-order
   reduction-stability finding: align memra chunk split points to a fixed grain and add
   the invariance gate to the battery.
2. **Decode-graph scratch + PDL-chain audit** (SGLang #33063 + TRT-LLM #16072): fill
   kernels allocated inside graph capture replay forever AND break PDL dependent-launch
   chains; padding dummies allocated lazily silently kill graph coverage under load.
   memra runs graph-decode + PDL on sm_120 — one afternoon audit, high odds of found
   money on the felt decode path (the dogfood deficit).
3. **TMA OOB-zero-fill for odd-M prefill quant** (FlashInfer #4210): 3.6x on
   non-128-aligned M by deleting the padded staging copy and letting G2S TMA zero-fill.
   Priority-1 filter match (FP8/FP4 prefill); attacks the prime-chunk/prefill wall
   named in the dogfood head-to-head.
4. **W4A8 as the quality-safe NVFP4 activation path** (llama.cpp PR #24364, open): FP4
   weights x FP8 activations beats native W4A4 on quality with receipts; converges with
   FP8-ST (we already have e4m3 activation kernels) and reframes an nvfp4-strict arm.
5. **Spec-decode acceptance telemetry + draft-cost top-k** (llama.cpp #26389 + vLLM
   #49969): per-position acceptance counters on /metrics make the 0.55-vs-0.73 short-ctx
   acceptance finding a live gauge; DSpark's restrict-draft-correction-to-top-k pattern
   is portable to MTP-p2's per-step draft cost.

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._
