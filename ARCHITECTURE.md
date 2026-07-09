# bw24 — Architecture & Tech-Stack Decision

From-scratch LLM inference engine for **one** machine: NVIDIA **RTX 5090 Laptop** (consumer Blackwell **GB203, sm_120**, compute capability **12.0**, **24463 MiB** VRAM, **858 GB/s** measured read wall [95.8% of peak], 82 SMs, 3090 MHz max SM clock, 64 MB L2), Intel **Core Ultra 9 275HX** (24 cores, **no AMX/AVX-512** but **has AVX2 + AVX_VNNI** int8), **elastic free host RAM** (~12–16 GB free, varies with other LLM servers — query at runtime, never hardcode), **2× WD PC SN8000S** NVMe (**PCIe Gen4 x4, ~7 GB/s each**, RAID0 ≈ 14 GB/s), GPU on **PCIe Gen5 x8** (~31 GB/s ceiling; idles at Gen1, re-clocks under load). **Power is not a hard constraint:** the "150 W cap" is an asus-armoury firmware-settings bug (`nv_tgp.max_value` wrongly clamped; spec = 150 TGP + 25 boost = 175 W; user's patch to asus-linux maintainer raises it). Real sustained limit is thermal (87 °C target), not fixed wattage — benchmark at full power (`gpu-full-power on`). Toolchains present (mutable): CUDA **13.1** + **12.8**, driver **595** (CUDA 13.2 runtime); CUTLASS/deps installable as research dictates.

**Goal:** beat vLLM, SGLang, and llama.cpp on *this exact box* for single-stream/low-concurrency decode and MoE-that-doesn't-fit. Must support MoE, weight/expert/KV spilling, GGUF, KV-cache quantization. Copying leading impl 1:1 is allowed and encouraged. (Note: "tokens-per-watt" was planned, but since power cap is a *settings bug being patched*, not silicon limit, watt-normalized targets are secondary — optimize for raw tok/s at thermal-limited full-power state and re-baseline after TGP patch lands.)

This document folds in the adversarial verdicts. Where a verifier **refuted** or marked a claim **uncertain**, the optimistic framing has been downgraded and the load-bearing risk demoted off the critical path.

---

## 0. Hard architectural facts (the constraints that drive every decision)

These are confirmed locally and by multiple primary sources. They are non-negotiable inputs:

1. **sm_120 ≠ sm_100.** Consumer Blackwell has **no tcgen05, no TMEM, no WGMMA, no TMA multicast** (cluster forced 1×1×1), and only **~99 KB shared-mem/block** (vs 228 KB on datacenter). It is **Ampere/Ada-style warp-level `mma.sync` into registers**, *plus* Blackwell FP4/FP6 block-scaled `mma.sync` extensions. **FlashAttention-3 (sm_90a) and FlashAttention-4 (sm_100a) do not run here.** Any kernel gated on tcgen05/TMEM silently fails or refuses.
2. **NVFP4 dense GEMM works on sm_120; NVFP4 *grouped/MoE* GEMM does not (yet).** CUTLASS TMA-warp-specialized grouped tactics fail to initialize even under `compute_120f`; the working fallback path is ~39 tok/s, **below Marlin W4A16 at 46–50 tok/s**. → **Marlin W4A16 is the shipping default for MoE; native FP4 grouped is research, not a pillar.** [CORRECTED — the earlier "MXFP4 block-scale mma.sync is ptxas-rejected on sm_120" claim was a **false negative**: `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X..ue8m0` AND `kind::mxf8f6f4.block_scale` both **assemble and RUN on this GPU** when built with `-gencode arch=compute_120a,code=sm_120a`. The rejection only happens under the bare `-arch=sm_120a` shortcut (which misroutes to `compute_120` PTX) — the same llama.cpp #19662 build-flag bug, NOT a silicon limit. Verified on-device 2026-06-26.]
3. **DECODE here is memory-bandwidth- and power-bound, not FLOP-bound** (single-token AI≈1-2 FLOP/byte, far below any crossover). [CORRECTED re compute: the "FP8/FP4 is not a compute win, FP8≈FP16 ~102 TFLOP/s" claim is true ONLY of the **plain** mma path. **Measured on-device: block-scaled FP8 (mxf8f6f4) = 381 TFLOP/s and block-scaled FP4 (mxf4) = 762 TFLOP/s vs FP16 117 and plain-FP8 219** — block-scale lifts the FP32-accumulate throttle (1.74x over plain FP8), matching FlashInfer #3628.] → For DECODE, low-bit weights + KV quant win via **bytes moved** (compute is irrelevant there). For **PREFILL / batched / block-scaled attention mainloops**, block-scaled FP4/FP8 is a genuine **6.5x / 3.3x compute lever** — a real edge, scoped to compute-bound regimes.
4. **The build toolchain is a correctness landmine.** llama.cpp MMQ **segfaults under CUDA 13.1** on sm_120 (nvcc codegen/-O3 bug, confirmed: zenn write-up + issue #18331) → must build the GGUF/MMQ path with **CUDA 12.8**. Arch flag must be **`sm_120a` (SASS)**; FP4-fast grouped needs **`compute_120f` (CUDA 13.0+)**. There is also a **`sharedMemPerBlockOptin` overflow on Blackwell** that aborts MMQ at runtime regardless of toolkit — must carry the smpbo clamp from llama.cpp PR #22338 **[CORRECTED: PR #22338 was on driver 590.48.01 (not 595), and is an UNMERGED, author-closed PR — it did NOT land upstream, so we carry it as a LOCAL PATCH, not a relied-on upstream fix].**
5. **The Python engines' tax is structural, not kernel quality.** vLLM/SGLang pay per-step Python dispatch + GC + mandatory CUDA graphs (8× penalty without graphs on sm_120 per vLLM #37242) and have **no host expert offload** (they OOM on >24 GB MoE). llama.cpp has offload but **re-copies hot experts every token** and uses per-op dispatch. These are the real seams.
6. **Numbers in the wild are mostly desktop/datacenter.** Desktop 5090 ≈ 1792 GB/s / 575 W / 170 SMs; **this laptop ≈ 896 GB/s (847 measured) / 82 SMs / thermal-bound (~175 W now, more after the TGP patch)** — roughly **half the bandwidth, half the SMs**. Every published decode tok/s must be ~halved (bandwidth) as a first approximation and re-measured at full power before any "beats X" claim (perf claims require **N=5 medians**, benchmarked with `gpu-full-power on`).

---

## 1. TECH STACK DECISION

### Chosen: **Rust host runtime + raw `.cu` kernels (nvcc, `-arch=sm_120a`) embedded as a fatbin, driven via `cudarc`. Build = `cargo` + `build.rs`→`nvcc`→`ninja`. Server = native `axum`/`tokio` OpenAI-compatible HTTP, in-process.**

| Layer | Choice |
|---|---|
| Host runtime / scheduler / KV mgr / spill mgr / loader | **Rust** (single process, single address space, no GC, fearless compute/copy/spill overlap) |
| CUDA driver/runtime access | **`cudarc`** crate (driver + NVRTC + cuBLAS + cuBLASLt + cuDNN; CUDA 13.0–13.3 wrapped, block-scale FP4/FP8 enums exposed) |
| Kernel authoring | **Hand-written `.cu`** compiled AOT by **nvcc** in `build.rs` (ninja-driven) into a **fatbin embedded in the binary**; `cudarc` loads the module + launches. NVRTC used **only** for runtime shape autotuning. |
| GEMM (robust path) | Link **cuBLASLt** through `cudarc` for BF16/FP8/FP4 dense; **Marlin W4A16** (.cu, copied) for weight-only 4-bit |
| API surface | Native Rust OpenAI-compatible server (`axum` + `tokio`); **optional** thin PyO3 binding crossed **once per request**, never per token |
| Two CUDA toolkits | **12.8** for the GGUF/MMQ TUs (13.1 segfaults), **13.1** for cuBLASLt + CUTLASS dense FP4 (`compute_120f`). Dual-toolkit build is deliberate. |

**Rationale.** A solo dev does not beat NVIDIA at matmul; the win is **deleting the two structural taxes** of the Python engines: (a) per-step dispatch + GC jitter that dominates single-stream decode on one fast GPU, and (b) many-GPU scheduler overhead we don't need. A native single-process runtime has **zero per-token FFI crossings** (FFI cost = crossing count, per prior `ffi-crossing-count-dominates` finding). Rust over C++ because: memory-safe concurrency for the copy/compute/spill overlap that *is* the engine on a bandwidth-bound box, one-command reproducible builds, and `cudarc` already wraps exactly the CUDA 13.x surface on this machine (verified). For peak-silicon kernels we **don't reinvent** — we link cuBLASLt and copy already-working sm_120 `.cu` (Marlin, dense NVFP4) behind a thin C ABI, batched at module load, not per call.

**Runner-up: C++/CUDA monolith (the llama.cpp/ggml model).** Rejected because: (1) the only thing C++ buys over Rust here is *not* needing `cudarc` — but `cudarc` is mature and the kernels are still `.cu` either way; (2) the engine's hard part is the **concurrent spill/prefetch/overlap state machine**, exactly where Rust's safety pays a solo dev the most; (3) we still *copy* ggml's kernels regardless of host language. The reference engine `imp` (kekzl/imp, MIT, sm_120a-native) is C++; we lift its CUDA-runtime `.cu` (graphs/conditional-WHILE) verbatim through a C ABI but keep the orchestration in Rust.

**Also rejected:** Python+torch (vLLM/SGLang) — the tax we're explicitly removing; torch nightly cu130 for sm_120 is a dependency nightmare on a box already short on RAM. TVM/MLC — compiler-stack learning cost, weak consumer-Blackwell story. TensorRT-LLM — AOT engine build friction, sm_120 trtllm-gen FMHA cubins **don't exist**.

---

## 2. sm_120 FEASIBILITY LEDGER

Status reflects the *adversarial verdicts*, not the optimistic first pass. **"risky" = do not put on the critical path.**

| Component | Recommended kernel / lib | sm_120 status | Fallback if it fails |
|---|---|---|---|
| **Dense BF16/FP16 GEMM** | cuBLASLt 13.x (`cublasLtMatmul`) | **confirmed** | CUTLASS ex.79 GeForce tiles (fit 99 KB) |
| **Dense FP8 GEMM** | cuBLASLt FP8 (E4M3) | **confirmed** (works; ~99 TFLOP/s, *not* a compute win over BF16) | BF16 |
| **Dense NVFP4 GEMM** | vLLM `nvfp4_scaled_mm_sm120_kernels.cu` (non-TMA) | **confirmed** (correct, prefill/TTFT win) | Marlin W4A16 |
| ~~Dense NVFP4 via CUTLASS ex.79a verbatim~~ | — | **risky** (segfaults: misaligned smem, CUTLASS #2906) | use vLLM's non-TMA kernel or ex.87 (`array_aligned`) |
| **NVFP4 *grouped*/MoE GEMM** | (research only) | **risky** (vLLM's OWN native CUTLASS grouped GEMM = silent garbage ~5 tok/s on SM120 per cutlass #3096 / vLLM forum 2536; the ~39 tok/s figure is the *FlashInfer-CUTLASS* grouped path on compute_120f, 4×RTX-PRO-6000, still < Marlin 46–50) | **Marlin W4A16 grouped (default)** |
| **MXFP4 / block-scale mma** | mxf4 + mxf8f6f4 `mma.sync` block-scale | **confirmed** (assembles+runs w/ `compute_120a,code=sm_120a`; 762/381 TFLOP/s measured) — earlier "rejected" was the bare-`-arch` flag-trap false negative | plain FP8/BF16 |
| **GGUF k-quant matmul (MMQ)** | llama.cpp `ggml-cuda/mmq.cu` + `vecdotq.cuh` | **confirmed** *iff built CUDA 12.8 + 120a-real + smpbo clamp* | cuBLAS dequant→FP16 (5–6× slower; correctness oracle only) |
| **Prefill attention (BF16)** | gau-nernst `learn-cuda/07_attention` v5 (`mma.sync` m16n8k16) | **confirmed** kernel exists & runs; **uncertain** that 94% SOL transfers (desktop/non-causal numbers) | llama.cpp `fattn-mma-f16` |
| **Decode attention (paged, KV-quant)** | llama.cpp `fattn-vec.cuh` (built `GGML_CUDA_FA_ALL_QUANTS=ON`) | **confirmed** on sm_120 today | — (this *is* the safe baseline) |
| **Decode attention (XQA)** | FlashInfer XQA `mla_sm120.cu` (built `12.0f`) | **likely** (XQA decode works per #2555; **page_size 64**, not 16) | llama.cpp `fattn-vec` |
| ~~Prefill via FlashInfer FMHA_V2 sm120~~ | — | **risky** (kernels exist but unwired behind `ENABLE_SM120`) | net-new work; not copy-paste |
| **KV-cache FP8 (E4M3)** | FlashInfer `fmha_v2 e4m3_fp32_*_sm120` (enable + lift) | **uncertain** (source-present, API-disabled; silent corruption on some models per SGLang #19603) | GGUF q8_0-K/q4_0-V dequant-in-VEC (memory-bound, no FP8 scale fragility) |
| **KV-cache MXFP8 block-scaled** | (FlashInfer #3628 RFC) | **risky** (open RFC, no shipping attention) | FP8 per-tensor or int8 KV |
| **Sampling (top-k/p/min-p, rejection)** | FlashInfer `sampling.cuh` (dual-pivot rejection, CUB + curand) | **confirmed** (sampling.cuh + topk.cuh verified present; #3170 = "DGX Spark SM121 support audit" [not a "sampling audit"], sampling is §7 of it) | CPU sampler (llama.cpp), <1 ms on the 275HX |
| **Grammar mask apply** | xgrammar `apply_token_bitmask_inplace_cuda.cu` | **confirmed** (elementwise, arch-agnostic) | CPU mask |
| **CUDA graphs / conditional-WHILE decode loop** | imp `src/runtime/cuda_graph.cu` (relaxed capture) | **confirmed** primitives work on this box; **uncertain** the megaloop win applies to host-routed MoE / offload (it falls back to per-token there) | per-token graph replay; enforce-eager last resort |
| **Spill mechanism (pinned/async/streams/io_uring/mmap)** | CUDA driver + `liburing` | **confirmed** (arch-agnostic; verified) | — |
| **GPUDirect Storage (cuFile)** | — | **risky** (likely unsupported on GeForce) | host-pinned bounce buffer |
| **GGUF parse + async upload** | llama.cpp `gguf.cpp` + `llama-model-loader.cpp` | **confirmed** (pure host) | — |

---

## 3. COMPONENT ARCHITECTURE

### 3.1 GEMM (dense linear)
Three-tier dispatcher keyed on dtype × regime (M). **(A) BF16/FP16 dense:** default **cuBLASLt** (robust, NVIDIA-tuned floor). **(B) Weight-only 4-bit (default decode + MoE):** **Marlin W4A16** — dequant→BF16 `mma.sync`, bandwidth-bound, immune to the broken grouped-GEMM path, and *currently the fastest correct sm_120 MoE*. **(C) NVFP4 dense (opt-in prefill/TTFT accelerator only):** vLLM's non-TMA `nvfp4_scaled_mm_sm120_kernels.cu`. Treat FP4 as a *VRAM saver + prefill win*, never a decode-compute win; gate behind a startup correctness self-check vs a dequant-BF16 oracle.
**Copy from:** `vllm/csrc/quantization/marlin/*` (gptq/awq marlin); `vllm/csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`; cuBLASLt via `cudarc`. *Do not copy CUTLASS ex.79a verbatim — it segfaults (CUTLASS #2906); use ex.87 `array_aligned` if hand-rolling.*

### 3.2 Attention
**Split prefill from decode.** **Prefill (BF16):** start from **gau-nernst v5** (`mma.sync` m16n8k16 + `cp.async` + ldmatrix.x4 + XOR-swizzle + 2-stage pipeline). Add causal masking, GQA (broadcast K/V across a group in the n-dim, llama.cpp `ncols2`-style), paged/ragged indexing — and **re-benchmark on-target with causal+GQA at N≥5**, because the famous 94% SOL is desktop/non-causal. **Decode (memory-bound, the real workload):** **llama.cpp `fattn-vec.cuh`** (per-thread vector dot, separate `type_K`/`type_V` templates) — *the known-good sm_120 KV-quant decode path today*. Build with **`-DGGML_CUDA_FA_ALL_QUANTS=ON`** to avoid the ~18× asymmetric-K/V slow path. XQA (FlashInfer, page_size 64) is a later throughput upgrade once verified on driver 595.
**Copy from:** `github.com/gau-nernst/learn-cuda/tree/e83c256/07_attention` (v5 + `common.h`); `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` + `fattn-common.cuh` + `fattn.cu` dispatch.

### 3.3 KV cache management + paging
**Hybrid: paged physical pool + thin radix index + 2-tier spill.** Physical KV = flat GPU pool, FlashInfer layout `[num_pages, 2, page_size, n_kv_heads, head_dim]`. **Decouple logical block granularity (fine, for fragmentation/radix matching) from the kernel's physical page size — set physical `page_size=64`** (the only shipping sm_120 XQA kernel is tuned for 64, not 16). Logical map = per-sequence block table + a radix/HiRadixTree for token-level prefix reuse; eviction = vLLM `FreeKVCacheBlockQueue` LRU + ref-count + chained block hash. Spill: GPU layer-first, host/disk page-first, write-back + write-through-selective (host RAM is only 12–16 GB free).
**Copy from:** `vllm/v1/core/{block_pool.py, kv_cache_utils.py, kv_cache_manager.py}` (manager logic); SGLang HiRadixTree + `sgl-kernel/csrc/kvcacheio/transfer.cu`; FlashInfer paged layout + `plan()/run()` split for graph stability.

### 3.4 KV quantization
**Two tiers. Tier 1 default = GGUF asymmetric `q8_0`-K / `q4_0`-V** dequant-in-VEC kernel — memory-bound, no FP8 dequant-scale fragility, no silent-corruption class (SGLang #19603 showed FP8 KV silently garbages some models on sm_120). **Tier 1-alt = FP8 E4M3 per-head** *only after a per-model numeric gate*, justified purely by **halved capacity/bandwidth** (not compute — FP8 attention math is BF16-rate here). **Tier 2 (research) = MXFP8 block-scaled** (FlashInfer #3628 RFC) is the only true full-rate FP8 path but doesn't exist as attention on sm_120 yet. Always compile the **full asymmetric K/V matrix**.
**Copy from:** `llama.cpp/ggml-cuda/fattn.cu` `FATTN_VEC_CASES_ALL_D` + `fattn-vec.cuh`; vLLM `reshape_and_cache_flash` (FP8 write + per-head scales) for the alt path.

### 3.5 MoE kernels + expert offload
**Two layers. GPU kernel:** copy vLLM's dispatch (`moe_align_block_size` → `sorted_token_ids/expert_ids/num_tokens_post_padded`, fixed `BLOCK_SIZE_M` = one expert/block) + register-fused gate+up (SonicMoE/Triton trick). **Quant path = Marlin W4A16 grouped, period** — native NVFP4 grouped is off the critical path (broken/slow). **Offload executor (the differentiator):** keep attention + shared expert + dense FFN + router + KV resident; routed experts in pinned host RAM with a **persistent GPU expert-slot cache** (N≪n_expert fixed-address slots, **SLRU + second-miss admission**) so the ~15–20% hot experts stay resident → per-token PCIe ≈ 0 after warmup. **CPU expert *compute* is weak-but-not-dead** — this CPU has no AMX/AVX-512 (KTransformers' headline AMX lever is gone) BUT **has AVX2 + AVX_VNNI** (verified: int8 VPDPBUSD dot-product). So CPU int8 expert GEMM is feasible as a *fetch-vs-recompute* alternative for cold experts (open question #4: VNNI CPU GEMM vs x8-PCIe fetch at batch=1). Default keeps compute on GPU, CPU/RAM/SSD as paging tiers, with VNNI CPU-expert as a measured fallback, not a pillar.
**Copy from:** `vllm/model_executor/layers/fused_moe/{fused_moe.py, moe_align_block_size.py, experts/marlin_moe.py}` (these are PYTHON modules; the Marlin MoE C++ kernel is `csrc/.../moe/marlin_moe_wna16/ops.cu`); `llama.cpp/ggml/src/ggml-backend.cpp` `copy_experts` lambda (~line 1626 on current master; selective expert sub-row copy) + **issue #20757** (SLRU two-tier slot cache design — *CORRECTED: this is a CLOSED/completed feature-REQUEST, not an active RFC; it will NOT land upstream on its own — we implement it ourselves, lifting the design*); imp single-CUDA-graph scheduler + Expert Deferral idea; SGLang `eplb/expert_distribution.py` as the hot-expert signal.

### 3.6 Weight quant / GGUF formats
**Three tiers. Tier 1 ship-first = GGUF k-quants (Q4_K_M / Q5_K / Q6_K) via llama.cpp MMQ** — the compatibility baseline *and* the decode-throughput winner on this box (Q4_K_M beats native NVFP4 on single-user decode by 24–30%). Q6_K for sensitive tensors (`attn.v`, `ffn_down`). **Tier 2 = NVFP4 dense** as a prefill/TTFT accelerator + VRAM saver, not a decode play. **Tier 3 = per-layer mixed bitrate** (Q6_K attn, NVFP4 MLP) for tight fits. Build MMQ on **CUDA 12.8 + 120a-real**, port the **smpbo clamp** and avoid `-O3` on MMQ TUs.
**Copy from:** `llama.cpp/ggml-cuda/{mmq.cu, mmq.cuh, vecdotq.cuh, dequantize.cuh}`; vLLM Marlin; NVIDIA TensorRT-Model-Optimizer (`NVFP4_DEFAULT_CFG`) only to *produce* NVFP4 checkpoints.

### 3.7 Spilling tiers
**3-tier, prefetch-and-cache-first.** **Tier 0 (24 GB VRAM):** residents + per-layer LRU hot-expert slot cache + 4–8 async staging buffers. **Tier 1 (pinned host RAM, budget ~10–12 GB):** hottest experts in one contiguous `cudaHostAlloc` buffer. **Tier 2 (mmap NVMe):** cold-expert bulk + full GGUF, via `mmap` + page cache, with an `io_uring` O_DIRECT reader for predicted-cold experts to dodge page-cache thrash. **Budget honestly:** each SN8000S is **~7 GB/s (Gen4 x4)**, RAID0 ≈ 14 GB/s — *half* the Gen5 x8 PCIe ceiling, not "comparable"; treat disk as the slowest tier. Transfer = dedicated copy stream per direction + event-synced double-buffer; PDL edges for tail/head overlap. The **primary lever is the large resident SLRU cache** (24 GB headroom vs the PoC's 8 GB), *not* the io_uring machinery (steady-state hit rate ~98–100% leaves PCIe mostly idle).
**Copy from:** `dvmazur/mixtral-offloading` (`expert_cache.py` LRU + speculative next-layer-gate prefetch); MoE-Infinity EAM priority; llama.cpp RFC #20757; vLLM `cpu_offload_gb` copy-stream pattern; ZeRO-Inference DeepNVMe `io_uring` pattern.

### 3.8 Scheduler
**Single-process, CPU-light continuous batching, one per-step token budget.** A step = `{req_id: num_tokens}` (vLLM unified abstraction) so chunked prefill + decode piggyback with no phase split. **Running-first** scheduling for low ITL/jitter (beats SGLang prefill-priority for interactive single-stream). **SGLang `event_loop_overlap`** (result_queue, run step N+1 while processing N's tokens) — the single biggest CPU-overhead lever. **Preemption = recompute-first** (cooperating with always-on prefix cache), *not* swap — the verdict refuted swap-as-default: on this PCIe-x8 box recompute beats KV-swap for short/interactive contexts; KV swap is an opt-in, cost-model-gated path for >8–16 K contexts only. Spec decode: ship **ngram/suffix (CPU, zero VRAM) first**; EAGLE3 second, gated on sm_120 draft/verify kernels.
**Copy from:** `vllm/v1/core/sched/scheduler.py` + `request_queue.py`; SGLang `managers/scheduler.py` `event_loop_overlap`; vLLM prefix-cache block pool; TRT-LLM `GUARANTEED_NO_EVICT` policy idea; `vllm docs/features/speculative_decoding` (ngram/suffix).

### 3.9 GGUF loader
**Copy llama.cpp's reader structurally, verbatim.** `mmap` (MAP_SHARED + `posix_fadvise SEQUENTIAL`; **no MAP_POPULATE** for partial-offload so only touched pages fault in). Parse magic/version(==3)/counts/KV/tensor-infos; `data_offset = GGML_PAD(end, align)`. Map `general.architecture` + `[arch].*` → `ModelConfig` via a **data-driven arch registry** (avoid the giant per-arch switch). Per-tensor `{file_idx, offset, ggml_type, ne[], tier}` placement. **Upload via llama.cpp's `load_all_data` async pipeline** (4 round-robin pinned 64 MB buffers, event-synced, `tensor_set_async`) — reuse the same machinery for streaming paged experts. Upload quantized blocks verbatim; dequant in-kernel.
**Copy from:** `llama.cpp/ggml/src/gguf.cpp` (`gguf_init_from_reader`), `src/llama-mmap.cpp` (`read_aligned_chunk`, O_DIRECT), `src/llama-model-loader.cpp` (`load_all_data`), `ggml-common.h` (block structs). safetensors as thin secondary (8 B header + JSON).

### 3.10 CUDA runtime / graphs / allocator
**Lift imp's three-tier runtime.** **Allocator:** slab/arena for weights (plain `cudaMalloc`, never moves) + paged 64-token-block KV free-list (makes attention graph-safe: block tables are static device-pointer args) + `cudaMallocAsync` mem-pool **with a release threshold** for transient prefill scratch (so it coexists with the other LLM servers on this box). **Streams:** 1 compute + 1 H2D + 1 D2H/token-out; event-fork the copy stream for next-expert prefetch during matmul; PDL edges. **Graphs:** capture-once + `cudaGraphExecUpdate`, **`cudaStreamCaptureModeRelaxed`** (Global hangs CUTLASS grouped GEMM). For dense + graph-safe MoE, the `cudaGraphNodeTypeConditional` **WHILE megaloop** (verified working on this box) runs N decode steps on-GPU streaming via mapped-pinned ring buffer — *but* the verdict is clear: **with offload active or host-routed MoE, this falls back to per-token replay**, so don't sell it as the headline for the 30B-MoE-with-spill case. The real MoE graph win requires **on-device routing** (device top-k + fixed-capacity grouped GEMM with static block/expert tables) — that's the open engineering task. **Skip building a from-scratch megakernel** (all published ones are sm_90/sm_100).
**Copy from:** `kekzl/imp/src/runtime/cuda_graph.cu` (Capture/Runner/ConditionalRunner, `apply_pdl_edges`); vLLM `cudagraph_dispatcher.py` (bucket/pad); llama.cpp `ggml-cuda.cu` (`cudaGraphExecUpdate` heuristics).

### 3.11 Sampling + tokenizer
**Fully GPU-resident, sorting-free, lowest-risk component (no tensor cores at all).** Keep logits on GPU; one **fused logits-processor kernel** (penalties + logit-bias + grammar bitmask to −inf, xgrammar `LogitsBitmaskKernel` verbatim); temperature via fused online-softmax; sample via **FlashInfer Dual-Pivot rejection sampler** (`sampling.cuh`, CUB + curand, verified on this box, 0.042 ms @128k batch=1). Greedy/temperature-only bypass the rejection sampler. Structured output: **llguidance** (Rust/C, no Python, stable per-token cost) running on idle CPU cores overlapped with the GPU forward. **Tokenizer:** GGUF-native — port `llama.cpp/src/llama-vocab.cpp` + `unicode.cpp` (the 53 pre-tokenizer regexes are the spec); use the Rust `tokenizers` crate when `tokenizer.huggingface.json` is embedded.
**Copy from:** FlashInfer `include/flashinfer/sampling.cuh` (+ its `topk.cuh`/`math.cuh`/`utils.cuh`/`vec_dtypes.cuh` include subtree); xgrammar `apply_token_bitmask_inplace_cuda.cu`; `guidance-ai/llguidance` (`parser/llguidance.h`); llama.cpp `llama-vocab.cpp`. *Pin one stream or vendor FlashInfer #3625 fix to avoid the multi-stream top-k corruption bug.*

---

## 4. HOW WE BEAT vLLM / SGLang / llama.cpp

Optimized metric (in priority order): **single-stream decode tok/s @ low context → prefill TTFT → MoE-with-offload tok/s**, all at full power. We explicitly **do not** chase high-concurrency batched throughput (the Python engines' mature turf, irrelevant to a single-user laptop), and we de-prioritize tok/s-per-watt (the cap is a patched settings bug, not a silicon envelope — but it stays a secondary reported number since the laptop still cares about heat/battery).

**Edge 1 — Remove the runtime tax (strongest, best-supported).**
- vs vLLM/SGLang: no Python per-step dispatch, no GC jitter, no mandatory-graph warmup (they suffer 8× without graphs on sm_120), zero per-token FFI crossings. Cold-start latency win is immediate.
- vs llama.cpp: replace per-op dispatch with capture-once graphs + (for dense/graph-safe paths) the on-GPU conditional-WHILE decode loop removing the ~1.3 µs/step CPU round-trip.
- **Beat-targets (re-measure at N=5 on-box; halve desktop numbers):** **8B-Q4 single-stream ≥ 100 tok/s** (beat llama.cpp ~75–95 laptop by 15–25%, beat vLLM on cold-start latency outright). **0.6–1.5B ≥ 500 tok/s** (Python engines can't reach this — launch overhead dominates).

**Edge 2 — Block-scaled FP4/FP8 (confirmed feasible AND a real compute win, scoped).**
- Block-scaled FP4 (mxf4) = **762 TFLOP/s measured (6.5× FP16)**, block-scaled FP8 = 381 (3.26×) — genuine compute lever for **prefill/TTFT** (compute-bound) + VRAM savings. Q4_K_M MMQ stays the *decode* winner (decode is bytes-bound, compute irrelevant). So: FP4/FP8 block-scale for prefill+capacity, MMQ for decode.
- **Beat-target:** prefill **~3–5k tok/s laptop** (≈ half desktop's 6–10k), with block-scaled FP4 prefill as the lever competitors mostly don't wire on sm_120 yet. (tok/s-per-watt deferred — power cap is a settings bug being patched, not a fixed silicon envelope; raw full-power tok/s is the primary metric.)

**Edge 3 — MoE-that-doesn't-fit (real but conditional, *demoted from "biggest win"*).**
- vLLM/SGLang OOM (0 tok/s). llama.cpp `--n-cpu-moe` (`-ncmoe`, flag confirmed in `common/arg.cpp`) offloads routed experts to CPU and is the real baseline to beat [the prior "8–31 tok/s desktop 5090" figure was an UNSOURCED private number — DELETED; measure llama.cpp `-ncmoe` on-box as the live baseline instead of citing a figure]. Our SLRU resident cache + correct sm_120 kernels target **a ~1.5–2× beat over llama.cpp in the offload regime** (measured, not a published claim). With only 12–16 GB free host RAM, a 120B's experts spill to ~7 GB/s SSD; the cache delivers **zero** benefit for the 30B-A3B-Q4 *primary* target (which fits 24 GB and runs fully resident at bandwidth limit — Edge 1's megakernel is the lever there).

**The moat is whole-system:** native Rust runtime + the *right copied/correct* sm_120 kernels (cuBLASLt + Marlin + vLLM dense NVFP4 + llama.cpp MMQ/fattn-vec + block-scaled FP4 prefill) + this-laptop tuning (99 KB smem tiles, 82 SMs, 896 GB/s, thermal-bound not 150 W, x8 PCIe, ~7 GB/s NVMe). **Not** a faster matmul, **not** native FP4 MoE. The two-tier SLRU offload cache (issue #20757) is a *closed proposal nobody is shipping upstream* — so building it well IS a real (if implementable-by-others) edge, not a commodity.

---

## 5. BUILD ORDER

Dependency-ordered. Each phase ends in something runnable and benchmarkable. **Benchmark gate every phase: N=5 medians, set `maxmemory`/VRAM headroom, on this box (or the `dev` EC2 for non-GPU parts).**

### Phase 0 — Toolchain + smoke (FIRST MILESTONE, build immediately)
Goal: **a Rust binary that loads a small GGUF, uploads one quantized tensor, runs one cuBLASLt BF16 GEMM and one custom `mma.sync` kernel on the GPU, and reads it back correct.** Proves the whole spine: `cargo` + `build.rs`→nvcc fatbin, `cudarc` module load/launch, dual-toolkit (12.8 for MMQ TUs, 13.1 for cuBLASLt), `-arch=sm_120a`, smpbo-clamp probe, NVMe `mmap`. No model forward yet.
- Deliverable: `bw24 probe` prints GPU caps, runs the 4 smoke kernels, exits 0.
- Copy: `cudarc` README launch pattern; one cuBLASLt call; one gau-nernst `mma.sync` snippet.

### Phase 1 — Dense forward pass, single-stream, BF16/Q4_K
GGUF loader (`gguf.cpp` port) + arch registry + async upload pipeline → a **dense** model (e.g. Qwen3-8B Q4_K_M) forward: MMQ matmul (CUDA 12.8 build), gau-nernst prefill attention + llama.cpp `fattn-vec` decode (`FA_ALL_QUANTS=ON`), RoPE/RMSNorm, FlashInfer sampler + GGUF tokenizer. Per-token replay (no graphs yet). **Greedy, batch=1.**
- Gate: matches llama.cpp logits within tolerance; **records baseline tok/s** vs llama.cpp on the same GGUF.

### Phase 2 — CUDA runtime: graphs + allocator + paged KV
Lift imp's capture-once + `cudaGraphExecUpdate`; slab/arena allocator + paged-64 KV pool; conditional-WHILE decode loop for the dense path. Continuous-batching scheduler (running-first token budget) + SGLang overlap loop.
- Gate: graphed dense decode beats Phase-1 per-token replay and **beats llama.cpp 8B-Q4 single-stream by ≥15%** (Edge 1, beat-target ≥100 tok/s).

### Phase 3 — MoE (resident) + Marlin
`moe_align_block_size` dispatch + Marlin W4A16 grouped GEMM + register-fused gate+up. Run 30B-A3B-class MoE **fully resident** in 24 GB.
- Gate: correctness self-check vs dequant-BF16 oracle; bandwidth-bound decode tok/s recorded. (No FP4 grouped — off critical path.)

### Phase 4 — Spilling tiers (MoE that doesn't fit)
SLRU GPU expert-slot cache + pinned host RAM tier + `io_uring`/mmap NVMe tier + speculative next-layer-gate prefetch + double-buffered copy stream. Recompute-first preemption.
- Gate: 120B-class MoE runs (where vLLM/SGLang OOM); **≥1.5× llama.cpp `--n-cpu-moe`** measured live on-box in the offload regime (no pre-committed tok/s figure — establish it from the llama.cpp baseline run).

### Phase 5 — FP4/FP8 upside + KV quant + spec decode
Opt-in NVFP4 dense prefill (vLLM non-TMA kernel) gated behind correctness + microbench; FP8/q8q4 KV quant with per-model numeric gate; ngram/suffix spec decode; structured output (llguidance). Tune tiles/launch params and spill thresholds to this exact part.
- Gate: prefill TTFT win measured (Edge 2) with block-scaled FP4; tok/s-per-watt reported as secondary.

### Phase 6 — Hardening + "beats competitors" sign-off
On-device MoE routing for graphable MoE (the open task); XQA decode upgrade; EAGLE3 spec (if sm_120 kernels verify); full N=5 benchmark suite vs all three engines on the target models. Server polish (OpenAI-compatible, streaming).

---

## 6. TOP RISKS & OPEN QUESTIONS (ranked)

**R1 — The headline decode win is unmeasured on *this* part.** Every cited tok/s/SOL is desktop @400–575 W / 1792 GB/s or datacenter. Laptop is ~half bandwidth + half SMs; gau-nernst's 94% SOL was *non-causal, desktop, @400 W*. (Power is NOT the binder — it's a settings cap being patched; thermal @87 °C is the real sustained limit.)
→ *Mitigation:* Phase-0/1 microbench mandatory before any beat-claim (already started: 847 GB/s, FP16 117 / block-FP8 381 / block-FP4 762 TFLOP/s measured); recompute SOL for 896 GB/s + 82 SM; add causal+GQA and re-profile at N=5 with `gpu-full-power on`. Keep llama.cpp as the live baseline in every benchmark.

**R2 — Build-toolchain correctness landmines silently degrade or crash.** CUDA 13.1 + MMQ segfaults; `120a` (not `120f`) gives garbage/illegal-instruction for FP4 grouped; driver-595 smpbo overflow aborts MMQ at runtime; `-O3` MMQ codegen bug; FORCE_CUBLAS cache trap = 5–6× slow.
→ *Mitigation:* Pin MMQ TUs to **CUDA 12.8 + 120a-real**, FP4-dense to **13.1 + compute_120f**; **port llama.cpp's smpbo clamp (PR #22338)**; avoid `-O3` on MMQ; build-time smoke test that asserts MMQ output on a probe prompt (not just that it links); pin known-good llama.cpp + CUTLASS commits.

**R3 — CUDA-graph instability on sm_120 (EngineDeadError, silent 120 W spin-hang).** Documented on real 5090 even with `120f` (HF discussion #9).
→ *Mitigation:* `plan()/run()` split (capture only static `run()`), bucketed/padded batch sizes, cap captured graph sizes, keep a **PIECEWISE** and an **enforce-eager** fallback as first-class config; validate on driver 595/CUDA 13.2.

**R4 — Native FP4 MoE never becomes viable; FP8 attention gives no compute win.** Grouped GEMM TMA-WS tactics still fail; FP8/FP4 attention is BF16-rate (FP32-accumulate throttle).
→ *Mitigation:* already off the critical path — **Marlin W4A16 is the MoE default**, FP4 is dense-prefill upside only. Sell the engine on Edge 1 (runtime) + Edge 3 (offload), treat FP4 as gated upside. Re-test grouped GEMM each CUTLASS bump with `cutlass_profiler` on-box.

**R5 — The MoE-offload edge is smaller than hoped.** [CORRECTED: llama.cpp issue #20757 is a CLOSED feature-request, NOT a merged/in-flight implementation — so it is NOT shipping the same SLRU cache; we'd implement it ourselves and the edge is less "time-limited" than first framed.] With 12–16 GB free RAM the 120B working set spills to ~7 GB/s SSD; the 30B *primary* target doesn't spill at all.
→ *Mitigation:* Reframe Edge 3 as ~1.5–2× over llama.cpp in the offload regime, not 6–25×. Lead with Edge 1. Measure hot-expert working-set vs free pinned RAM per target model (open question) before fixing offload targets.

**R6 — KV-quant silent corruption.** FP8 E4M3 KV silently garbages some models on sm_120 (SGLang #19603), and asymmetric K/V hits the 18× slow path without `FA_ALL_QUANTS`.
→ *Mitigation:* Default to GGUF q8_0-K/q4_0-V (no FP8 scale fragility); gate FP8 KV behind a per-model numeric check; always build the full asymmetric K/V matrix.

**Open questions to resolve empirically (each blocks a target):**
1. Sustained H2D bandwidth at full power (PCIe link idles at Gen1, re-clocks to Gen5 x8 ≈31 GB/s under load) — sets prefetch break-even. (Phase 4)
2. Does `compute_120f` actually make grouped NVFP4 correct *and* ≥ Marlin on *this laptop* (vs the 4× RTX PRO 6000 benchmarks)? (Phase 5, gated)
3. Hot-expert working-set vs free pinned RAM for Qwen3-MoE / GLM / DeepSeek — does it fit Tier 1 or fall to SSD? (Phase 4)
4. Fetch-vs-recompute crossover for one expert at batch=1 over x8 PCIe vs AVX2 CPU GEMM. (Phase 4)
5. Can MoE routing be made on-device/graph-safe to get the WHILE-megaloop win for MoE? (Phase 6, the open engineering task)
6. Does gau-nernst v5 hold its SOL ratio after causal+GQA+paging on the laptop? (Phase 1/2)
7. GGUF tokenizer fidelity per target model: embedded `tokenizer.huggingface.json` vs llama.cpp vocab+merges — avoid silent tokenization drift. (Phase 1)
