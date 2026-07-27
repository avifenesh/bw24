
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
