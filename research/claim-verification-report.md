I'll consolidate these verdicts into the report. Let me check the verdicts that are marked "refuted" but contain nuance, since the report's lead section depends on getting those exactly right.

Looking at the data, I have everything I need. Let me produce the report.

# Adversarial Verification Report: ARCHITECTURE.md Claims

Scope: 5 clusters, 47 distinct claims, each checked against a fetched primary source. Verdict tally: **39 confirmed, 5 refuted, 3 confirmed-but-with-unverifiable-numeric-sub-claims** (the magnitude/figure could not be sourced even though the mechanism held).

---

## 1. FABRICATIONS / REFUTED — fix these in ARCHITECTURE.md

Five claims were marked **refuted**. None is a wholesale fabrication of a feature, but each contains a **false load-bearing detail** (wrong status, wrong number, wrong type, or fabricated figure). Fix each as noted.

### 1.1 llama.cpp PR #22338 "smpbo clamp on driver 595" — TWO factual errors
- **What's wrong:** (a) Driver was **590.48.01**, NOT 595. (b) The PR was **never merged** — closed by the author 2026-04-25 (`merged: false`, `merged_at: null`).
- **What's right:** It is a real PR, it is an smpbo (`sharedMemPerBlockOptin`) overflow clamp (`if (info.devices[id].smpbo > 1024*1024) { ... = prop.sharedMemPerBlock; }`), fixing a SOFT_MAX error on Blackwell/RTX 5090 + CUDA 13.1.
- **Fix in doc:** Change "driver 595" → "driver 590.48.01". Stop treating this as landed behavior — it is an **unmerged, closed PR**. If the architecture relies on this clamp existing upstream, it does not; you'd carry it as a local patch.
- Source: https://patch-diff.githubusercontent.com/raw/ggml-org/llama.cpp/pull/22338.diff

### 1.2 llama.cpp RFC/PR #20757 "active RFC" — status is WRONG
- **What's wrong:** Described as an "active RFC." It is **closed (state_reason: completed)**, closed by the author 2026-04-08, and it is an **Issue/feature-request, not a PR**.
- **What's right:** The content is accurate — a two-tier GPU+RAM expert cache for MoE offload with pluggable SLRU eviction + frequency-gated admission, citing `ggml-backend.cpp:1445-1564` and `common/arg.cpp:2284,2291`.
- **Fix in doc:** Do not call this an active/open RFC. It is a closed proposal. Any architecture decision that assumes this is "in flight upstream and will land" must be revisited — it will not arrive on its own.
- Source: https://github.com/ggml-org/llama.cpp/issues/20757

### 1.3 llama.cpp `--n-cpu-moe` "8-31 tok/s on a desktop 5090" — figure is FABRICATED/unsourced
- **What's wrong:** No primary source publishes "8-31 tok/s for large MoE on a desktop 5090." Targeted searches found nothing; published 5090 MoE numbers vary wildly (e.g. Qwen3.5-35B-A3B tg 201-211 t/s; Kimi K2.5 ~12-14 t/s cross-socket). The figure appears to be a private/local measurement dressed up as a published benchmark.
- **What's right:** The `--n-cpu-moe` (`-ncmoe`) flag exists (`common/arg.cpp:2336`) and offloads routed experts to CPU.
- **Fix in doc:** Keep the flag claim. **Delete or explicitly label the "8-31 tok/s" range as an unsourced local measurement** — do not cite it as a benchmark.
- Source: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

### 1.4 FlashInfer #3170 "sampling audit" — MISLABELED
- **What's wrong:** #3170 is titled **"DGX Spark (SM121) Current Support Audit"** — a 15-section architecture-support audit. Sampling is just section 7. Calling it "the sampling audit" misrepresents both its scope and its target arch (SM121, not a sampling deep-dive).
- **Fix in doc:** Re-label #3170 as "SM121 (DGX Spark) full support audit." If the doc cited it as evidence about the sampler specifically, that citation is too narrow / mis-scoped.
- Source: https://github.com/flashinfer-ai/flashinfer/issues/3170

### 1.5 SageAttention3 "has prebuilt sm120 wheels" — FALSE
- **What's wrong:** There are **no prebuilt sm120 wheels**. README mandates source compilation (`python setup.py install`). `CachedWheelsCommand` wires no prebuilt-wheel download; a comment notes FORCE_BUILD "instead of attempting to find prebuilt wheels."
- **Fix in doc:** Remove any claim/assumption of a `pip install`-able prebuilt SageAttention3 sm120 wheel. **Installation requires from-source compilation** (`compute_120a,code=sm_120a`). This is a build-pipeline cost the architecture must budget for.
- Source: https://raw.githubusercontent.com/thu-ml/SageAttention/main/sageattention3_blackwell/README.md

### 1.6 Hardware: "cuDNN SDPA ~160 TFLOPS on 5070 Ti" — FABRICATED half of the claim
- **What's wrong:** The "~160 TFLOPS on 5070 Ti" figure does not exist in the cited source (gau-nernst blog benchmarks **only** the desktop 5090). No "5070 Ti" or "160 TFLOPS" appears anywhere in it, and no other primary source attributes that number. **Appears fabricated or mis-sourced.**
- **What's right:** The 5090 half is exactly correct — cuDNN `F.sdpa()` = 203.61 TFLOPS / 97.19% of the 209.5 SoL.
- **Fix in doc:** Keep the 5090 number. **Delete the "~160 TFLOPS on 5070 Ti" figure** unless an actual 5070 Ti benchmark source is produced.
- Source: https://gau-nernst.github.io/fa-5090/

---

## 2. UNVERIFIABLE — treat as assumption, verify before relying

These claims have a **confirmed mechanism but an unverifiable magnitude/figure**. The direction is real; the specific number is a private/local measurement with no primary source. Flag each as "treat as assumption, verify before relying on it."

| # | Claim | What's confirmed | What's unverifiable |
|---|---|---|---|
| 2.1 | `GGML_CUDA_FA_ALL_QUANTS` avoids a **~18x-slow** asymmetric K/V path | Flag exists (`ggml/CMakeLists.txt:208`); symmetric-only-by-default behavior is real (4 symmetric instances default; asymmetric pairs only under the `#ifdef`); slowdown direction confirmed by issue #20866 + discussion #22411 | The **specific ~18x magnitude** is in no public source. Treat 18x as a private benchmark. |
| 2.2 | NVFP4 grouped GEMM **~39 tok/s** vs Marlin W4A16 **46-50 tok/s** on sm_120 | Numbers are real and sourced — but to **community reports** (NVIDIA/cutlass issue #3096, vLLM forum post 2536), **not a vLLM PR/issue**. | **Source attribution in the doc is likely wrong** if it credits vLLM. Also a key nuance: the 39 tok/s is the **FlashInfer-CUTLASS** grouped GEMM on `compute_120f`, NOT vLLM's own native CUTLASS grouped GEMM — which the same sources say produces **garbage output (~5 tok/s)** on SM120. If the architecture plans to use vLLM's native CUTLASS grouped path, that path is reported broken. |
| 2.3 | `--n-cpu-moe` "8-31 tok/s" (also listed under Refuted §1.3) | Flag exists | tok/s range unsourced — see §1.3. |

**Action:** For 2.1 and 2.2, keep the qualitative claim, strip or footnote the exact figure. For 2.2 specifically, **re-attribute the benchmark to cutlass #3096 / vLLM forum (community), not vLLM core**, and add the "vLLM native CUTLASS grouped GEMM = garbage on SM120" caveat — this is a real architectural risk, not a footnote.

---

## 3. CONFIRMED — held up against primary source

### Cluster A — vLLM (source files, PRs, dispatch)
- NVFP4 sm_120 dense GEMM kernel exists (`Fp4GemmSm120`, `Sm120` ArchTag, `cutlass_scaled_fp4_mm_sm120a`) — *path moved to `csrc/libtorch_stable/quantization/fp4/...` on current main (torch-stable-ABI migration); old `csrc/quantization/fp4/...` was historically valid.*
- NVFP4 blockwise MoE kernel exists, gated by `ENABLE_NVFP4_SM120` (same path move).
- Marlin C++ kernels exist (`csrc/libtorch_stable/quantization/marlin/`; MoE at `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`). *Correction baked in: `fused_marlin_moe`/`fused_moe` are **Python** module names, not csrc files.*
- PR #21309 = CUTLASS NVFP4 (w4a4) SM120 ✓. *#31085, #36095, #37242 are **ISSUES not PRs**; #31085 is MXFP4-backend-selection, not "backend dispatch."*
- Scheduler/KV files all exist (`scheduler.py`, `block_pool.py`, `kv_cache_utils.py`, `kv_cache_manager.py`, `request_queue.py` under `sched/`).
- `cutlass_scaled_mm_supports_fp4(120)==True` confirmed in `nvfp4_scaled_mm_entry.cu`.

### Cluster B — llama.cpp
- All 16 listed source files resolve (HTTP 200); `load_all_data` at `llama-model-loader.cpp:1408`.
- Issue #19662 = MXFP4 mma.sync block-scale build failure on bare sm_120, resolved by building `120a` — number + content match.
- MMQ segfaults under CUDA 13.1 on sm_120; build with CUDA 12.8 — confirmed (zenn write-up + issue #18331; root cause = nvcc codegen/O3 bug, not llama.cpp logic).
- `ggml-backend.cpp` selective expert sub-row copy for `mul_mat_id` exists (`copy_experts` lambda ~line 1626; line numbers drifted as the claim predicted).

### Cluster C — FlashInfer & SGLang
- `sampling.cuh` (dual-pivot rejection sampler), `topk.cuh`, `xqa.py` (arch check includes 12; `mla_sm120.cu` at `csrc/xqa/`) — all confirmed.
- FlashInfer #2555 (SM120 mis-route to Ampere FA2), #2577 (NVFP4 mm_fp4 returns zeros on SM120), #3628 (RFC MXFP8 block-scaled prefill SM120a), #3625 (multi-stream top-k race — a **PR**, fixes bug #3618), #2460 (fp4 GEMM tiles + streamK — a **PR**), #2166 (MLA runtime failure on SM120) — confirmed.
- #3628 microbenchmark: block-scaled QMMA.SF ~202 TFLOP/s FP32-acc vs throttled ~51/102 plain — confirmed exactly.
- SGLang `scheduler.py` (`event_loop_overlap`), `eplb/expert_distribution.py`, HiRadixTree (HiRadixCache), `kvcacheio/transfer.cu`, issue #19637 (SM120 perf plan), #19603 (FP8 E4M3 KV silent garbage on some sm_120 models) — confirmed.

### Cluster D — gau-nernst, imp, CUTLASS, xgrammar, llguidance
- fa-5090 blog (hand-written mma.sync FA-2 kernel, Ampere features only, no TMA) — confirmed.
- learn-cuda `07_attention` v1..v5 + `common.h`; **v5 = 197.74 TFLOPS / 94.39% SoL > flash-attn 190.58, ~cuDNN 203.61**; bench is **desktop 5090 @ 400W, non-causal** — all confirmed.
- gn-kernels (GQA + INT8/FP8 MMA; CuTe-DSL SM120 matmul via TMA+mma.sync, NOT tcgen05) — confirmed.
- `kekzl/imp` exists, MIT, sm_120a-native; `cuda_graph.cu` with Capture/Runner/ConditionalRunner + `apply_pdl_edges` + `cudaGraphNodeTypeConditional` WHILE — confirmed.
- CUTLASS `mma_sm120.hpp` (`SM120::BLOCKSCALED::SM120_16x8x32_TN_VS`); examples 79/79a/79c/87 + ex.87 `array_aligned`; issue #2906 (ex.79a smem-misalignment crash) — confirmed.
- SageAttention3 = microscaling FP4 attention, sm_120a block-scaled mainloop — confirmed (*but no prebuilt wheels — see §1.5*).
- xgrammar `apply_token_bitmask_inplace_cuda.cu`; llguidance `parser/llguidance.h` — confirmed.
- flash-attn-4 PyPI 4.0.0b19 (Jun 2026), CuTeDSL, Hopper+Blackwell datacenter; #2307 (FA4 no consumer SM120 support); #2361 (`FLASH_ATTN_CUDA_ARCHS=120` segfaults nvcc on backward) — confirmed.

### Cluster E — Hardware specs
- Desktop 5090: 1792 GB/s, 512-bit GDDR7, 575W, 170 SMs, GB202 — confirmed.
- Laptop 5090: GB203, 82 SMs, 256-bit, 896 GB/s, ~150W (TGP configurable 95-150W) — confirmed.
- Desktop 5090 BF16 attention SoL ~209.5 TFLOPS — confirmed.
- WD PC SN8000S = PCIe Gen4 x4 NVMe ~7 GB/s (spec 7.3, real-world >6) — confirmed.
- PCIe: Gen5 x8 ≈31, Gen4 x4 ≈7.88, Gen5 x16 ≈63 GB/s — confirmed.
- Core Ultra 9 275HX = 24 cores (8P+16E), no AMX, no AVX-512 — confirmed.

---

## 4. NET ASSESSMENT

**The architecture rests overwhelmingly on confirmed ground.** ~83% of claims (39/47) confirmed outright, including every load-bearing *existence* claim — every source file, kernel, build flag, scheduler component, hardware spec, and the bulk of the PR/issue references resolve to a real primary source. **No invented file paths and no invented PR/issue numbers** were found; the worst path issues are stale prefixes from the torch-stable-ABI migration (`csrc/` → `csrc/libtorch_stable/`), which are cosmetic and self-correcting.

**What actually needs revisiting (load-bearing refutations):**

1. **SageAttention3 has no prebuilt sm120 wheel (§1.5).** If the build/deploy plan assumed `pip install`, that's wrong — budget for from-source `compute_120a` compilation. This is a real pipeline cost, not a doc typo.

2. **vLLM's native CUTLASS NVFP4 grouped GEMM is reported broken on SM120 (§2.2).** The "39 tok/s" path is **FlashInfer-CUTLASS**, not vLLM's own kernel, which community sources call garbage (~5 tok/s). Any decision to lean on vLLM's native grouped path for MoE on consumer Blackwell needs to be re-examined — Marlin W4A16 (46-50 tok/s) is the actually-working baseline. This is the **single most consequential** correction.

3. **Two upstream proposals are dead, not in-flight.** The smpbo clamp (#22338, §1.1) is an unmerged closed PR, and the two-tier SLRU MoE cache (#20757, §1.2) is a closed/completed feature-request, not an active RFC. If the architecture assumed either lands upstream, it must plan to **carry them as local patches** — they will not arrive on their own.

**What's cosmetic (fix for accuracy, no decision impact):**
- Several "PR #X" labels are actually issues (#31085, #36095, #37242, #3170) or PRs mislabeled as issues (#3625, #2460); #3170 is mis-scoped as a "sampling audit." Re-type/re-label these.
- Two figures (18x asymmetric-KV penalty §2.1; 8-31 tok/s `--n-cpu-moe` §1.3; 160 TFLOPS 5070 Ti §1.6) are unsourced private numbers — strip or footnote. The 5070 Ti figure should be deleted; it's the only number with no corroborating source at all.

**Bottom line:** the doc's factual skeleton (what exists, what the hardware can do) is sound. The risk is concentrated in (a) one **wrong performance/availability assumption** about vLLM native NVFP4 grouped GEMM, (b) **two dead upstream patches treated as live**, and (c) **one missing-wheel build assumption**. Fix those four and the architecture is on verified footing; the remaining edits are labeling hygiene.