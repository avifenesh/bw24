# bw24 Unified Inference Feature Map (all engines)

This confirms the critical details. EDGE-1 is genuinely dynamic per-token selective staging (the loop re-derives top-8 per token at line 234-282, stages only those experts). Notably:
- **No SLRU / hot-expert cache exists yet** (the bw24 survey claims an "SLRU runtime hot-expert cache" but the code re-stages every expert every token — no caching across tokens). This is a real gap to flag.
- **Safetensors reader exists** but MoE path is "NOT yet wired into the loader" — PARTIAL, not HAVE.
- **No CUDA-graph capture** in the code, and **no async pinned H2D pipeline** wired (the stage_expert is a plain copy). Both flagged as planned-only in the survey.

I have enough verified ground truth. Producing the map now.

---

# bw24 Unified Inference Feature Map

**Target config:** single-stream (batch=1), one RTX 5090 Laptop (Blackwell sm_120, 24 GB, 82 SMs, ~896 GB/s, ~175 W thermal-bound), GGUF resident-quant edge inference. CPU is Intel Core Ultra 9 275HX — **AVX2 + AVX_VNNI only, NO AMX, NO AVX-512** (verified: `/proc/cpuinfo`). Goal: beat vLLM/SGLang/llama.cpp on *this* config.

**Honesty rule applied throughout:** A technique's "Single-stream-edge value" is rated on what it does for **one sequential request on one GPU**. The single most important filter is: *is this a multi-request server win or a single-stream win?* Continuous batching, PagedAttention, radix/prefix sharing, disaggregated PD, in-flight batching, LoRA multiplexing, request routing — these are **near-zero value at batch=1** no matter how famous they are.

---

## 1. MASTER TABLE (deduped across engines)

| # | Technique | Category | What / Mechanism (terse) | Helps | Engines that have it | bw24 status | Single-stream-edge value |
|---|-----------|----------|--------------------------|-------|----------------------|-------------|--------------------------|
| 1 | **Flash-Decoding split-KV** | attention | Split KV axis into chunks, each emits partial O + LSE, merge via online-softmax. Fills idle SMs at batch=1 long-ctx. | decode, latency | Stanford/FA2.2+, TRT-LLM (multi-block), bw24 | **HAVE** (flash_attn.cu:568-690, `fa_decode_f32`+`fa_decode_combine_f32`) | **HIGH** — at batch=1 this is THE thing keeping >1 of 82 SMs busy |
| 2 | **FlashAttention kernel fusion (FA-2 base)** | attention | QKV→softmax→PV fused, O(N) HBM. bw24 has hand-written m16n8k16 bf16 mma. | prefill, decode, mem | vLLM, SGLang, llama.cpp(FA3), TRT, bw24 | **HAVE** (flash_attn.cu, FA-2; FA-3/4 scoped not wired) | **HIGH** |
| 3 | **Quantized KV cache (low-bit, in-kernel dequant)** | kv-cache | Store K/V at 8/5/4-bit packed, dequant inside attention. bw24: q8_0-K/q5_1-V. | mem, throughput | vLLM(FP8/NVFP4), SGLang(FP8/FP4), llama.cpp(Q8/Q5_1), ExLlama(Q4), bw24 | **HAVE** (lib.rs:87-101, flash_attn.cu:130-155) | **HIGH** — direct 2x KV capacity on fixed 24 GB |
| 4 | **KIVI asymmetric KV quant (K per-channel, V per-token)** | kv-cache | Exploit that K outliers cluster in channels, V doesn't → different quant axes. Free quality upgrade to #3. | mem | KIVI(research) | **MISSING** | **HIGH** — cheap design rule, shapes bw24's KV-quant scale layout, near-lossless at lower bits |
| 5 | **CUDA-graph capture (decode loop)** | scheduling | Record decode forward once, replay → kill kernel-launch + CPU-sync overhead. | decode, latency | vLLM, SGLang, llama.cpp, bw24(planned) | **MISSING** (no `cudaGraph*`/capture in code; survey lists as planned task #5) | **HIGH** — 5-10%/token at <10 ms steps; mandatory per spec |
| 6 | **MMVQ (quantized matvec for decode)** | quant/GEMM | One warp/row fused dequant+dot, high occupancy at batch≤8. | decode, throughput | llama.cpp, bw24 (dp4a qmatvec) | **HAVE** (qmatvec.cu; resident-quant GEMM) | **HIGH** — the decode GEMM workhorse |
| 7 | **MMQ (quantized matmul for prefill)** | quant/GEMM | Register-blocked tile GEMM, dequant-in-register; wins at N≥128. | prefill, throughput | llama.cpp, bw24 | **HAVE** (qmatvec.cu dispatches per-dtype) | **MED** — prefill is short for single-stream chat; matters for long prompts |
| 8 | **Resident-quant weights (no f32 dequant tax)** | quant | Keep GGUF k/i-quant blocks packed in VRAM, dequant on the fly. 8 dtypes. | mem, throughput | llama.cpp, bw24, (TRT weight-only, Marlin) | **HAVE** (model.rs:28-62; Q8_0/Q4_K/Q6_K/Q5_K/Q3_K/IQ4_XS/IQ3_S/NVFP4) | **HIGH** — this is the whole edge premise |
| 9 | **Op fusion (RMSNorm/RoPE/bias/residual into GEMM+attn)** | quant/other | Fuse the "glue" elementwise ops into surrounding kernels → fewer launches, fewer HBM round-trips. | decode, prefill | TRT-LLM, MLC, vLLM(partial) | **PARTIAL** (FA fuses softmax; norm/RoPE/residual still separate launches) | **HIGH** — pairs with CUDA-graph to hit bandwidth ceiling |
| 10 | **EAGLE / EAGLE3 speculative decode** | spec-decode | Small draft from target hidden states proposes K tokens, verify in 1 pass. | decode, throughput | vLLM, SGLang, llama.cpp | **MISSING** (only MTP head present) | **HIGH** — mandatory per spec; 1.5-2x at batch=1 |
| 11 | **MTP / NextN multi-token-predict spec decode** | spec-decode | Aux head drafts K, verify-all-columns, greedy accept, snapshot/rollback. | decode, throughput | vLLM, llama.cpp, bw24 | **HAVE** (spec.rs:1-420, CacheSnapshot rollback) | **HIGH** |
| 12 | **Tree-attention verification (Medusa / SpecInfer / Sequoia)** | spec-decode | Verify many candidate continuations in ONE pass via tree mask; accept longest valid subtree → more accepted tokens/step. | decode, latency | Medusa, SpecInfer, Sequoia(research) | **MISSING** (bw24 spec verify is linear) | **MED-HIGH** — bolt onto existing MTP/EAGLE verify; Sequoia hardware-aware tree sizing fits single-fixed-chip philosophy |
| 13 | **Lookahead decoding (draft-free Jacobi n-gram)** | spec-decode | Parallel Jacobi trajectories + n-gram pool verify in one fused pass. NO draft model/weights. | decode, latency | research (LMSYS) | **MISSING** | **HIGH** — pure attn-mask + n-gram layer, no extra weights, exploits bw24's spare FLOPs (762 TFLOP FP4 vs 896 GB/s); fallback when no draft head shipped |
| 14 | **N-gram speculative (prompt/corpus lookup)** | spec-decode | Match current suffix to n-gram table, propose tokens, ~zero overhead. | decode, throughput | vLLM, SGLang | **MISSING** | **MED** — cheap, big on code/repetitive text; overlaps #13 |
| 15 | **Staged-expert MoE — selective per-token routed staging (EDGE-1)** | memory/spilling | Host-resident experts; per token, stage ONLY the top-K selected experts H2D, run qmatvec, accumulate. Shared expert stays GPU-resident. | mem, throughput | **bw24** (others do static/CPU-compute placement — see §4) | **HAVE** (hybrid_forward.rs:233-282) | **HIGH** — fits 35B-A3B in 24 GB where llama.cpp OOMs (validated) |
| 16 | **Hot-expert resident cache (SLRU / LRU across tokens)** | memory/spilling | Keep recently-used experts resident, skip re-staging on reuse. | throughput | (LMCache tier interface; bw24 survey *claims* it) | **MISSING** (code re-stages every expert every token — no cross-token cache; see verification) | **HIGH** — biggest EDGE-1 speedup left on the table; routing has temporal locality |
| 17 | **KV/expert tiered spilling (VRAM↔pinned-host↔NVMe)** | memory/spilling | Memory hierarchy keyed by chunk-hash; dedicated copy streams + eviction barrier; layerwise load/compute pipeline. | mem (capacity), latency | vLLM, SGLang(HiCache), LMCache, bw24(planned task #8) | **PARTIAL** (HostExps host-tier exists for experts; no NVMe tier, no double-buffered stream, no SLRU) | **MED** — mandatory per spec; capacity win, but NVMe ~7 GB/s is slow tier |
| 18 | **Async H2D pinned round-robin upload** | memory/spilling | Pinned host buffers + event-synced async copy to overlap H2D with compute. | throughput, latency | vLLM, LMCache(load_stream), bw24(planned) | **PARTIAL** (stage_expert is plain copy; pinned/async pipeline designed in ARCHITECTURE §3.9, not wired) | **HIGH** — required to hide EDGE-1 staging behind compute |
| 19 | **Model weight offload to CPU (mmap + partial GPU layers)** | memory/spilling | mmap weights, offload top-N layers to GPU, rest on CPU; H2D at layer boundary. | mem | vLLM(UVA+prefetch), llama.cpp(n_gpu_layers), bw24 | **PARTIAL** (GGUF mmap loader exists; layer-boundary CPU compute not the model — bw24 keeps compute on GPU via staging) | **MED** — staging (EDGE-1) is bw24's better answer for MoE; dense-layer offload still useful for >24 GB dense |
| 20 | **NVFP4 / MXFP4 block-scale GEMM (4-bit weights, tensor-core)** | quant | Microscaling FP4 weights, dequant-in-register to feed mma. | mem, throughput | vLLM, SGLang, TRT(sm_90+), bw24 | **PARTIAL** (NVFP4 dtype loads + per-tensor scale; dense NVFP4 GEMM path via vLLM/Marlin kernel noted as plan) | **HIGH** — best perplexity-per-byte lever on sm_120 Blackwell |
| 21 | **Marlin W4A16 GPU GEMM (dense/attn linears)** | quant | 4-bit weight-only mixed GEMM, dequant-in-register → mma.sync; bandwidth-friendly decode. | decode, mem | KTransformers, vLLM, bw24(planned default) | **PARTIAL** (named as planned default decode/MoE GEMM; resident dp4a path is current) | **HIGH** — confirmed-best correct sm_120 MoE GEMM |
| 22 | **Trellis/vector quant (EXL3 / QTIP)** | quant | Tail-biting trellis VQ beats scalar-block quant at low bpw; tiny metadata. | mem | ExLlamaV3 | **MISSING** | **LOW-MED** — better quality/bit, but non-GGUF format + bespoke decode GEMM; competes with NVFP4; large standalone build |
| 23 | **W8A8 / INT8 activation quant** | quant | Quantize activations+weights to INT8 dynamically. | throughput, mem | vLLM | **PARTIAL** (q8_1 input quant for gate/up exists, decode.rs:45-56) | **MED** — decode is bandwidth-bound, compute speedup limited |
| 24 | **GQA fast-path in attention** | attention | Special-case warp org when Q/K head ratio is 8/4/2. | throughput | llama.cpp, TRT(XQA), bw24 | **HAVE** (FA supports GQA; qwen35 16/4) | **MED** |
| 25 | **XQA-style fused decode (RoPE+bias+KV-dequant in attn read)** | attention | Fuse preprocessing into the attention KV read; multi-block spreads few-head decode across SMs. | decode, mem | TRT-LLM | **PARTIAL** (split-K = the multi-block idea; RoPE/bias fusion into attn read not done) | **MED-HIGH** — overlaps #1+#9; fuse RoPE+dequant into the decode read |
| 26 | **Hybrid attention: full-attn + GDN linear-attn per layer** | attention | Some layers full-attn (O(T²)), some Gated DeltaNet linear (O(T)); dual cache. | throughput, mem | bw24 (Qwen3.5/3.6); SGLang(Mamba state) | **HAVE** (hybrid_forward.rs:70-120, hybrid.cu) | **HIGH** for hybrid models — structural long-ctx win (bw24 EDGE-3) |
| 27 | **Sliding-window attention (SWA) + ring KV** | attention | Restrict attn to recent window; ring-buffer KV → constant memory. | mem, throughput | vLLM, SGLang, llama.cpp(iSWA) | **MISSING** (no SWA path; depends on model) | **MED** — model-dependent; helps very-long-ctx modes |
| 28 | **Attention sinks + rolling window (StreamingLLM)** | kv-cache | Keep first ~4 sink tokens + recent window → stable infinite streaming in bounded KV. | mem | research; vLLM(sink flag) | **MISSING** | **MED** — cheap "endless chat" mode; the "never over-compress first tokens" fact matters for any KV quant/evict |
| 29 | **H2O heavy-hitter KV eviction** | kv-cache | Track cumulative attn mass, evict low-importance tokens to a fixed budget. | mem | research | **MISSING** | **MED** — per-stream, fits edge; but lossy/heuristic, risks judge/coding quality; optional long-ctx mode only |
| 30 | **SnapKV prompt-time KV selection** | kv-cache | At prefill→decode boundary, pool attn scores, keep clustered important KV. | mem, prefill→decode | research | **MISSING** | **MED** — good for long-prompt-then-generate (RAG/judge); lossy, optional |
| 31 | **LoRC low-rank KV compression** | kv-cache | SVD-factor K/V projections, progressive per-layer rank. | mem | research | **MISSING** | **LOW** — changes KV math, breaks exact-match validation gate; EDGE-3 hybrid already captures much KV-shrink |
| 32 | **CacheGen KV entropy-coding compression** | kv-cache | Layer-adaptive bits + delta + arithmetic coding of stored KV bitstream. | mem, transfer-latency | LMCache | **MISSING** | **LOW-MED** — only if NVMe KV spill becomes measured bottleneck; big standalone codec |
| 33 | **RoPE / YaRN context extension** | other | Frequency-scale RoPE for ctx beyond training; free math. | latency (ctx) | all | **HAVE** (bw24_gguf config rope params, model-driven) | **MED** — inherited from model, ~free |
| 34 | **Transposed-V cache layout** | kv-cache | Store V col-major for coalesced gather. | mem, throughput | llama.cpp | **PARTIAL** (quantized KV layout fixed by ggml block; V-trans not separately done) | **LOW-MED** — minor decode locality win |
| 35 | **Unified KV cache (single pool)** | kv-cache | One cell pool vs per-stream → −20% alloc overhead at single-stream. | mem | llama.cpp | **HAVE** (bw24 is single-stream by design; dual cache is per-layer not per-seq) | **LOW** — already the default posture |
| 36 | **Backend graph reuse (skip rebuild on identical ubatch)** | scheduling | Reuse compute graph when shapes match → save 5-10%/step. | latency | llama.cpp, bw24(forward structure) | **PARTIAL** (decode reuses fixed structure; no formal can_reuse gate) | **MED** — subsumed by CUDA-graph (#5) |
| 37 | **Chunked prefill / variable ubatch** | batching | Split long prompt into ubatches → bounded peak memory, no OOM. | prefill, mem | vLLM, SGLang, llama.cpp | **PARTIAL** (forward processes all T; no explicit ubatch splitter) | **MED** — enables 4K+ prompts on 24 GB without OOM; single-stream relevant |
| 38 | **Guided/structured decoding + jump-forward FSM** | sampling | Constrain logits to grammar; skip deterministic FSM runs. | throughput (fewer retries) | vLLM, SGLang | **MISSING** | **MED** — real win for JSON/tool-use workloads; jump-forward skips forwards |
| 39 | **GPU-resident sampler (top-k/p/temp/penalties, no host roundtrip)** | sampling | Fused logits processing on device. | latency, throughput | vLLM, SGLang | **PARTIAL** (greedy argmax only in current path) | **MED** — needed for real generation quality knobs; low overhead |
| 40 | **Safetensors loader** | other | mmap safetensors reader parallel to GGUF. | — | (HF ecosystem) | **PARTIAL** (reader exists `safetensors.rs`; MoE path NOT wired, hf_mapping.rs:65) | **MED** — mandatory per spec; finish wiring |
| 41 | **CacheBlend non-prefix KV reuse** | prefix-reuse | Stitch independently-computed RAG-chunk KV with positional fix + selective recompute. | prefill (RAG) | LMCache | **MISSING** | **NONE** (single-stream) — see §3 |
| 42 | **Cross-request prefix/radix KV reuse** | prefix-reuse | Hash/trie match prefixes across separate requests, reuse KV. | prefill (multi-req) | vLLM, SGLang, LMCache, TGI | **MISSING** | **LOW** — only same-session multi-turn helps single-stream; see §3 |
| 43 | **PagedAttention / paged KV block manager** | memory | Fixed-size KV pages + block tables, anti-fragmentation across requests. | mem, throughput (multi-req) | vLLM, SGLang, TRT, TGI | **MISSING** (bw24 uses contiguous/grow-in-place) | **LOW** — fragmentation is a multi-request problem; single stream uses simple contiguous/ring |
| 44 | **Continuous / in-flight batching** | batching | Admit new requests mid-batch, interleave prefill/decode. | throughput (multi-req) | vLLM, SGLang, TRT, TGI | n/a | **NONE** — batch=1, nothing to interleave; see §3 |
| 45 | **Disaggregated prefill/decode (PD split)** | parallelism | Separate prefill & decode onto different GPUs/nodes. | throughput (multi-GPU) | vLLM, SGLang | n/a | **NONE** — single GPU; see §3 |
| 46 | **Tensor/Pipeline/Context parallelism** | parallelism | Split model across multiple GPUs. | mem, throughput (multi-GPU) | vLLM, SGLang, TRT | n/a | **NONE** — single GPU; see §3 |
| 47 | **CPU-expert GEMM with AMX/AVX-512 int8 (KTransformers CPUInfer)** | parallelism | Run routed-expert FFN on CPU via AMX tiles. | mem | KTransformers | n/a | **NONE** — host has NO AMX, NO AVX-512 (verified); see §3 |
| 48 | **CPU-expert GEMM with AVX2+VNNI (KTransformers fallback)** | parallelism | int8 dpbusd MoE on AVX2+VNNI for INT4 experts. | mem | KTransformers | **MISSING** (feasible on this host) | **LOW** — feasible but slow; EDGE-1 GPU-staging wins at batch=1; keep only as cold-expert fallback |
| 49 | **YAML static module placement (hot-GPU/cold-CPU)** | scheduling | Regex-match modules → device/impl, static up-front. | mem, throughput | KTransformers | n/a | **NONE** — Python/torch-graph retrofit; EDGE-1 dynamic staging strictly more adaptive; see §4 |
| 50 | **Token merging / pruning** | other | Merge similar tokens mid-inference to shrink seq. | throughput | research/vLLM(latent) | **MISSING** | **LOW** — best on long input; lossy; marginal at chat lengths |
| 51 | **TorchCompile / TVM-Unity / MLC codegen** | other | JIT/AOT fuse+tile+plan kernels, cross-platform. | throughput | vLLM(torch.compile), MLC | n/a | **NONE** as a system — bw24's explicit non-goal is hand-tuned single-target sm_120; only the *lesson* (fuse+tile) applies (= #9) |
| 52 | **KV-offload-to-GPU when weights on CPU (offload_kqv)** | kv-cache | Keep KV in HBM even if weights on CPU → no per-step H2D for KV. | throughput | llama.cpp | **HAVE** (dual cache is GPU-resident by design) | **MED** — already the posture |

---

## 2. PRIORITIZED GAP LIST for bw24 (single-stream wins only)

Ranked by (impact × feasibility on sm_120). Server-only techniques excluded (see §3).

### 2A. MANDATORY per project spec
*(spec items: KV-quant ✓, MTP ✓, EAGLE, safetensors, spilling, CUDA-graph, FA3/4)*

| Rank | Gap | Status | Why / Action | Feasibility |
|------|-----|--------|--------------|-------------|
| **M1** | **CUDA-graph capture of decode loop** (#5) | MISSING | No `cudaGraph*` in code. Biggest cheap decode latency win (5-10%/token); spec-mandated. Capture the fixed single-token decode graph; the EDGE-1 per-token H2D staging is the known tension — capture the dense path, leave staging as a graph-external prologue or use the layerwise-piecewise pattern LMCache documents. | HIGH |
| **M2** | **EAGLE / EAGLE3 draft head** (#10) | MISSING | Have MTP; spec wants EAGLE too. EAGLE3 reuses target hidden states, fits 24 GB (7B target + ~1B draft). Reuse existing snapshot/rollback + verify-all-columns plumbing from spec.rs. | MED-HIGH |
| **M3** | **Finish safetensors MoE wiring** (#40) | PARTIAL | Reader + hf_mapping exist; MoE path explicitly NOT wired (hf_mapping.rs:65 "no MoE safetensors test model"). Wire HostExps to load from safetensors. | MED |
| **M4** | **Tiered spilling: pinned-async + NVMe tier** (#17, #18) | PARTIAL | HostExps is the host tier; missing double-buffered pinned async copy stream + eviction barrier + NVMe tier. Required to hide EDGE-1 staging behind compute. | MED |
| **M5** | **FA-3/FA-4 by-hand decode kernel** (#2, #25) | PARTIAL | FA-2 base + split-K done. FA-3 scoped not wired. Fuse RoPE+bias+KV-dequant into the attention read (XQA pattern) on top of existing split-K. | MED |

### 2B. DISCRETIONARY wins worth porting (genuine single-stream value)

| Rank | Gap | Status | Why / Action | Feasibility |
|------|-----|--------|--------------|-------------|
| **D1** | **Hot-expert SLRU/LRU resident cache** (#16) | MISSING | **Highest-ROI discretionary item.** Code re-stages every selected expert every token (hybrid_forward.rs:258-274) — no cross-token reuse despite the survey claiming an "SLRU runtime hot-expert cache." Expert routing has strong temporal locality; caching hot experts in a VRAM pool eliminates most redundant H2D. Pure CPU-GPU bookkeeping, no new kernel. | HIGH |
| **D2** | **Op fusion: RMSNorm/RoPE/residual into GEMM+attn** (#9) | PARTIAL | Kills launch + HBM round-trip overhead; the prerequisite to hitting the bandwidth ceiling and to a clean CUDA-graph. Pairs with M1. | MED-HIGH |
| **D3** | **Lookahead decoding (draft-free)** (#13) | MISSING | No extra weights to train/ship, pure attn-mask + n-gram pool over existing forward. Exploits bw24's huge spare FLOPs at batch=1. A spec-decode fallback GGUF/llama.cpp lacks by default. | MED |
| **D4** | **KIVI asymmetric K-per-channel / V-per-token quant** (#4) | MISSING | Free quality/bit upgrade to the existing KV-quant kernel; should shape the scale layout of #3 before it's frozen. Near-lossless at lower bits. | MED-HIGH |
| **D5** | **NVFP4 / Marlin W4A16 dense GEMM path finish** (#20, #21) | PARTIAL | Best perplexity-per-byte on Blackwell sm_120; named as planned default but resident dp4a is current. Port vLLM `marlin_moe_wna16`. | MED |
| **D6** | **Tree-attention verification on MTP/EAGLE** (#12) | MISSING | More accepted-tokens/step than linear verify; Sequoia hardware-aware tree sizing matches bw24's single-fixed-chip philosophy. Bolt a tree mask onto the existing verify-all-columns path. | MED |
| **D7** | **GPU-resident sampler + structured decode** (#39, #38) | PARTIAL/MISSING | Current path is greedy argmax. Real generation needs top-k/p/temp/penalties on device, and JSON/tool-use needs grammar + jump-forward. Low overhead, broad utility. | MED |
| **D8** | **Chunked prefill / ubatch splitter** (#37) | PARTIAL | Lets 4K+ prompts run on 24 GB without prefill OOM — a genuine single-stream concern, not batching. | MED |
| **D9** | **StreamingLLM sinks + SWA/ring KV** (#27, #28) | MISSING | Optional "endless chat" / very-long-ctx mode at constant memory. The "never over-compress the first tokens" rule also guards the KV-quant scheme. | MED |
| **D10** | **SnapKV / H2O long-context KV modes** (#29, #30) | MISSING | Per-stream, fits edge; lossy so make them opt-in long-ctx modes after KIVI quant proves insufficient. Quant first, eviction second. | LOW-MED |

---

## 3. EXPLICIT "NOT WORTH IT for single-stream edge"

Do not spend effort here. Each is a multi-request, multi-GPU, or dead-ISA technique.

| Technique | Why it's dead for bw24 | Cite |
|-----------|------------------------|------|
| **Continuous / in-flight batching** (#44) | batch=1 — there are no concurrent requests to interleave. The entire throughput premise is multiplexing. | SGLang scheduler.py:1564-1632 (survey: "Single-stream execution is sequential, no batching advantage"); TRT in-flight batching ("fundamentally a MULTI-REQUEST server throughput win") |
| **PagedAttention / paged-block KV manager** (#43) | Anti-fragmentation across many sequences. One stream uses simple contiguous/ring KV; no fragmentation to fix. | vLLM paged_attn.py / block_pool.py |
| **RadixAttention / hash prefix sharing** (#42) | Peak benefit is multi-request prefix reuse. Single-stream only gains on same-session multi-turn, which is a much smaller, separable feature. | SGLang radix_cache.py:66-216 ("peak benefit is multi-request batch reuse") |
| **Disaggregated prefill/decode (PD)** (#45) | Requires ≥2 GPUs/nodes; adds network latency on one box. | vLLM kv_connector; SGLang disaggregation/prefill.py ("Edge single-GPU has no benefit") |
| **Tensor / Pipeline / Context parallelism** (#46) | Single 24 GB GPU — no second device to split across. | vLLM parallel_state.py; SGLang parallel_state.py |
| **KTransformers CPU-expert with AMX/AVX-512** (#47) | **DEAD on this host.** Core Ultra 9 275HX has NO AMX and NO AVX-512 (verified `/proc/cpuinfo`: avx2 + avx_vnni only). KTransformers' fastest CPU-expert path cannot run here. | kt-kernel AMX path; verified CPU flags |
| **KTransformers AVX2+VNNI CPU-expert fallback** (#48) | *Feasible* but slow; EDGE-1 GPU-staging beats per-token x8-PCIe CPU compute at batch=1. Keep only as an optional cold-expert fallback, never a pillar. | rawint4_avxvnni-moe.hpp; bw24 ARCHITECTURE open-question #4 (defaults to GPU-staging) |
| **YAML static module placement** (#49) | Python/torch-graph retrofit; irrelevant to a Rust single-process engine that places tensors explicitly at load. Also STATIC — EDGE-1's dynamic per-token staging is strictly more adaptive. | KTransformers deepseek-v2-injection.md |
| **CacheBlend non-prefix RAG KV reuse** (#41) | Premise is a library of pre-stored document-chunk KVs reused across queries — the multi-request RAG world bw24 de-scopes. Nothing to blend on one stream. | LMCache CacheBlend (vllm_v1_adapter.py:512-524) |
| **LoRA multiplexing / overlap loading** (vLLM #LoRA, SGLang lora_overlap) | Multi-tenant adapter serving; single task = no throughput gain. | vLLM lora/; SGLang lora_overlap_loader.py |
| **Request routing / priority scheduling / prefill delayer** (SGLang) | Multi-user fairness and small-request batching — no multi-user, no batch. | SGLang schedule_policy.py:348-360 |
| **Two-batch overlap** (SGLang) | Overlaps prefill batch N with decode batch N-1; single-stream has no batch pipeline. | SGLang two_batch_overlap.py |
| **TVM-Unity / MLC cross-platform codegen** (#51) | bw24's explicit non-goal is one-chip hand-tuned sm_120; the compiler-genericity tax is what EDGE-2 rejects. Keep only the *lesson* (fuse+tile = #9). | MLC blog; survey "Do not adopt TVM" |

**Borderline — defer, don't kill:** LoRC low-rank KV (#31, breaks exact-match validation gate), EXL3 trellis quant (#22, non-GGUF, competes with NVFP4), CacheGen codec (#32, only if NVMe spill is measured-bottleneck), Token merging (#50, lossy, marginal at chat lengths).

---

## 4. EDGE-1 cross-reference: is selective per-token routed-only staging novel?

**Claim under test:** bw24 EDGE-1 stages, per token, ONLY the top-K routed experts host→GPU, runs them on GPU, accumulates; shared experts stay resident (verified hybrid_forward.rs:233-282).

Compared against every expert/KV-offload approach in the surveys:

| Engine | Their expert/offload mechanism | Does it do per-token, routed-ONLY, GPU-compute staging? |
|--------|-------------------------------|--------------------------------------------------------|
| **KTransformers (CPUInfer, AMX/VNNI)** | Routed experts live in CPU RAM; **the expert GEMM runs ON THE CPU**, results return to GPU. Avoids per-token H2D of weights entirely. | **NO.** Opposite trade: it keeps the *compute* on CPU to avoid moving weights. bw24 moves the *weights* per token and computes on GPU. Different axis. |
| **KTransformers (YAML placement)** | Regex rules assign each module to a device **statically at load** (hot→GPU, cold-routed→CPU). | **NO — and crucially STATIC.** The hot/cold decision is config-time, not per-token. EDGE-1 re-derives the top-K *every token* (the loop at hybrid_forward.rs:234). Strictly more adaptive. |
| **vLLM** | Has weight offload (UVA + prefetch) for *layers*, and KV offload to CPU. No routed-only per-token expert staging. | **NO.** Weight offload is layer-granular and dense; not expert-selective per token. |
| **SGLang (HiCache)** | Tiers *KV cache* (not experts) GPU↔CPU↔disk by access temperature, async prefetch. | **NO.** It's KV tiering, not expert weight staging. |
| **LMCache** | Tiers/compresses *KV cache* keyed by token-hash across requests; async stream overlap. | **NO.** KV, not experts; cross-request, not per-token-routed. |

**Verdict: EDGE-1's specific combination is genuinely distinctive.** The novel axis is *dynamic, per-token, routed-only weight staging with GPU compute* on a single consumer GPU. KTransformers (the only other host-resident-expert engine) makes the opposite choice (CPU compute, static placement); KV-offload engines tier the cache, not the experts. The pattern itself — "move only the selected experts, compute on GPU, batch=1" — is not something the surveyed engines implement.

**Two honest caveats that temper the novelty:**
1. **The idea is an obvious point in the design space**, not a deep algorithm — it's "MoE weight offload, but stage only the routed subset." vLLM/KTransformers chose other points because at *server* batch sizes the routed subset approaches the full expert set (many tokens → many distinct experts selected), so selective staging loses its advantage. EDGE-1 is specifically a **batch=1 win** — at batch=1 only K≪N_experts experts are ever touched per token, which is exactly why bw24 fits 35B-A3B in 24 GB where llama.cpp OOMs (validated, commit 02af8fc). The novelty is *recognizing this is the right point for single-stream edge*, not the mechanism.
2. **EDGE-1 is currently incomplete in the way that most matters** (gap D1): the code re-stages every selected expert on every token with no cross-token cache (verified — no SLRU/LRU exists despite the survey claiming one). KTransformers/LMCache *do* have the eviction/caching machinery EDGE-1 lacks. So EDGE-1 is novel in *placement strategy* but currently *naive in reuse* — adding the hot-expert SLRU (D1) is what would make it both novel and fast.

---

## Bottom line for "what to build next"

1. **CUDA-graph (M1) + op-fusion (D2)** together get decode to the bandwidth ceiling — the single biggest verified gap, and they unblock each other.
2. **Hot-expert SLRU cache (D1)** is the highest-ROI discretionary item: it completes EDGE-1, which is bw24's most distinctive feature, and it's pure bookkeeping with no new kernel.
3. **EAGLE (M2)** and **finish safetensors MoE wiring (M3)** clear the remaining hard spec requirements.
4. **Ignore everything in §3** — roughly half the famous techniques in these surveys are multi-request/multi-GPU/AMX wins worth exactly zero on this config.

**Files verified during this analysis (absolute paths):**
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu:568-690` — Flash-Decoding split-KV (HAVE)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid_forward.rs:233-282` — EDGE-1 per-token selective staging (HAVE; no cross-token cache)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs:174-217` — HostExps host-resident experts
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs:159-164` — `stage_expert` (plain copy, no pinned/async)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/spec.rs:1-420` — MTP spec decode + snapshot/rollback (HAVE)
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/safetensors.rs` + `hf_mapping.rs:65` — safetensors reader present, MoE path NOT wired (PARTIAL)
- `/proc/cpuinfo` — confirms AVX2 + AVX_VNNI, no AMX, no AVX-512 (kills KTransformers AMX path)

Negative findings (claims in the bw24 survey NOT backed by current code): **no CUDA-graph capture**, **no SLRU/hot-expert cache** (re-stages every token), **no async pinned H2D pipeline** (stage_expert is a plain copy) — all three are described as planned/existing in the survey but are absent in the source.
