# BW24-BUILD-MAP — master build-decision map

The single source of truth for what bw24 builds, takes, or hand-rolls, in what order, and why.
Target: **batch=1–4, single-stream, sm_120 (RTX 5090 / desktop Blackwell), GGUF + safetensors, 24 GB resident.**

Decision legend:
- **TAKE** — port a portable *algorithm* (no python/torch/CUTLASS dependency) ~1:1.
- **HANDROLL-BETTER** — take the *idea* from an existing engine, write a better-fit implementation for B-small / sm_120.
- **MUST-HANDROLL** — no existing kernel runs on sm_120 (wgmma/tcgen05/AMX) or the only impls are multi-request-only; we have to write it.
- **HAVE** — already implemented and validated in bw24.

> Sourcing correction applied (adversarial check #1): the GGUF tokenizer keys
> (`tokenizer.ggml.model/pre/tokens/merges/bos_token_id/eos_token_id`) are **not** read by any engine config
> file. `ModelConfig` (`crates/bw24-gguf/src/config.rs`, imported at `crates/bw24-engine/src/cache.rs:7`) parses
> only arch / nextn / ssm / moe fields. Those tokenizer keys are read **only** in the diagnostic dumper
> `crates/bw24-gguf/src/bin/inspect.rs:86-90`. The new tokenizer crate must parse `tokens[]`/`merges[]` from raw
> GGUF metadata itself — `config.rs` does not surface them.

---

## 1. Master table

| Component | bw24 status | Decision | Source engine (port from) | Priority | sm_120 note |
|---|---|---|---|---|---|
| tokenizer | none (raw `u32` IDs in, raw IDs out — `run_gen.rs:15`) | TAKE algo / MUST-HANDROLL host glue | llama.cpp GPT2-BPE (`llama-vocab.cpp:279-752`, `:1964-1992`, `:695-710`) | **BASE (blocking)** | CPU-only; no GPU involved |
| sampler — greedy | HAVE (host argmax, `forward.rs:119`) | HAVE | bw24 | BASE (done) | host scan; bit-exact reference |
| sampler — temp/top-k/top-p/min-p/penalties | none | HANDROLL-BETTER (host first) | llama.cpp CPU chain (`llama-sampler.cpp:18-109/135-190/265-287/1351-1404/1543-1595`) | **BASE** (host) / PERF (GPU) | host f32 over 152K vocab = single-µs; GPU-fused warp-shuffle deferred |
| generation-loop | HAVE (`decode.rs:77` greedy prime+decode, gate `run_gen.rs:25`) | HAVE (harden EOS/stop) | bw24 `decode.rs` | BASE (done) | n/a (host loop) |
| quant-gemm (prefill) | in progress (int8 `m16n8k32`, `qmatvec_gemm.cu:54`) | HANDROLL-BETTER | bw24 GEMM-PLAN | **PERF (headline)** | int8 MMA shape RUNS on sm_120; NO wgmma/tcgen05 |
| decode GEMV | HAVE dp4a baseline | MUST-HANDROLL (NVFP4 TC GEMV) | DECODE-GAP-PLAN §3.1 | PERF (critical) | warp-per-row MMVQ; no existing impl fits |
| attention — prefill QK/PV | bf16 FA-2 `m16n8k16` scalar-ish tiles (`flash_attn.cu:109`) | HANDROLL-BETTER; TC-prefill **MUST-HANDROLL** | llama FA + bw24 `mma.sync` primitive | **PERF (critical)** | llama tile-prefill uses `nvcuda::wmma` (Volta) → won't lower; explicit `mma.sync` required |
| attention — decode | HAVE scalar (`fa_decode`, quant-KV dequant) | HANDROLL-BETTER | llama FA-vec (`fattn-vec.cuh:86-254`) + GQA broadcast; TRT `mla_sm120` idea-only | PERF | warp-per-token shuffle softmax; TMA paging over-built for resident KV |
| kv-cache | HAVE (q8_0 K / q5_1 V fused, `cache.rs:65`, commit 9ebf958) | HAVE | bw24 | BASE (done) | inline dequant `dq_q8_0_elem`/`dq_q5_1_elem` |
| spec-decode (MTP) | HAVE (greedy, batched partial-accept replay, `spec.rs:186`) | HAVE; tree-verify HANDROLL-BETTER | bw24 `spec.rs` + SGLang tree (`eagle_utils.py:48-159`, `frozen_kv_mtp_worker.py:81-199`) | PERF (tree) / EXTRA (EAGLE3) | mask + extra verify columns; no special kernel |
| moe | HAVE Stage-1 (per-token H2D staging, `hybrid_forward.rs:203`); no residency cache | HANDROLL-BETTER (router fuse + SLRU) | vLLM `topk-moe.cu` (idea) + ktransformers residency mask (idea) | **PERF** | FlashInfer CUTLASS grouped GEMM = DEAD (SM90, sm_120 bodies absent) |
| cuda-graph | none (grep `cudaGraph`/`BeginCapture` → empty) | MUST-HANDROLL (TAKE pattern) | llama.cpp capture (`ggml-cuda.cu:4468-4525`) | PERF (late, gated) | `cudaStreamBeginCapture` RUNS; gated on GPU-argmax (`decode.rs:70` barrier) |
| model-load | HAVE (dual-source GGUF+safetensors, resident-quant, commit 41f0bc6) | HAVE | bw24 `model.rs` + `source.rs` | BASE (done) | `GpuTensor::Quant` block-packed; NVFP4 two-level scale |
| serving-api + scheduler | none (CLI-only, `run_gen.rs`) | MUST-HANDROLL minimal (RR per-agent); HTTP BASE | round-robin handroll + SGLang RadixAttention idea (`radix_cache.py:206-267`) | **BASE (server)** / EXTRA (prefix cache) | vLLM/SGLang continuous batching DEAD at B=2–4 |
| spilling (VRAM↔host↔disk) | none | MUST-HANDROLL-if-needed | lmcache async overlap + ktransformers (idea) | EXTRA | pure cudaMemcpy/stream; no kernels; deferred until >24 GB |

---

## 2. BASE-FIRST ordered build sequence

These are the components required before bw24 is a *usable engine at all*. bw24 is today a logits validator: it
parses pre-tokenized `u32` IDs off the CLI (`crates/bw24-engine/src/bin/run_gen.rs:15`) and prints raw IDs. The
forward/decode/KV/model-load machinery is HAVE and validated; the gap is text-in / text-out / serve.

Build in this order — each gate must be green before the next starts.

### BASE-1 — tokenizer (encode + decode + chat template) — BLOCKING
- **Decision:** TAKE the GPT2-BPE *algorithm* 1:1; MUST-HANDROLL the Rust host glue.
- **Port source:** llama.cpp GPT2-BPE — unicode-regex pre-split + priority-queue bigram merge by rank
  (`llama-vocab.cpp:279-752`), merge-rank lookup load/call (`llama-vocab.cpp:1964-1992`), byte-unicode fallback
  (`llama-vocab.cpp:695-710`). Cited in `research/inference-maps/llamacpp.md:98-100`.
- **Concrete plan:** new host-only `bw24-tokenizer` crate (no CUDA). Parse `tokenizer.ggml.tokens[]` / `merges[]` /
  `pre` name from **raw GGUF metadata** (the loader exposes it; `config.rs` does not — see correction box above;
  the only existing reader is the diagnostic `crates/bw24-gguf/src/bin/inspect.rs:86-90`). Detok = reverse
  token→piece + byte-decode. Chat-template (CHATML/qwen) is a small separate host fn (`llama-chat.cpp`). Wire a
  `--prompt "text"` path into `run_gen.rs` replacing the `u32` parse at line 15.
- **Validation gate (integer-exact):** encode/decode round-trip on the model's own GGUF matches llama.cpp
  `llama-tokenize` token-for-token on a fixed prompt corpus (Qwen3.5 = GPT2-BPE pre-tokenizer). No tolerance.

### BASE-2 — host sampler chain (temp / top-k / top-p / min-p / penalties)
- **Decision:** HANDROLL-BETTER, host-side first. Greedy is HAVE (`crates/bw24-engine/src/forward.rs:119`) — keep
  it as the bit-exact reference.
- **Port source:** llama.cpp CPU sampler semantics — penalties ring-buffer (`llama-sampler.cpp:18-109`), temp
  (`:265-287`), top-k partial-sort (`:135-190`), top-p (`:1351-1404`), min-p (`:1543-1595`).
- **Concrete plan:** plain host arithmetic over the 152K f32 logit vector — single-µs at B=2–4, rides the D2H sync
  `decode.rs:70` already pays. **Do NOT** port vLLM's Triton 3-pass top-k/top-p — only profitable at hundreds of
  requests (MUST-HANDROLL-if-ever).
- **Validation gate:** fixed RNG seed → token stream matches a llama.cpp reference run with identical params.

### BASE-3 — gen-loop stop conditions (EOS / n_predict / stop-strings / context guard)
- **Decision:** HAVE loop; the remaining BASE work is small host glue. `decode.rs:77` is a complete greedy
  prime-then-decode loop; `max_ctx` already computed at `decode.rs:79`; `n_predict` already via `BW24_NGEN`.
- **Concrete plan:** EOS = compare emitted ID to `eos_token_id` (parsed in BASE-1); stop-strings = host substring
  match on the detokenized tail; context guard = turn the implicit `max_ctx` cap into a clean error/truncation.
  Wire the BASE-2 sampler in. **Do NOT** adopt vLLM continuous-batching / SGLang scheduler — DEAD at B=2–4.
- **Validation gate:** existing `am_p == am_d` MATCH gate (`run_gen.rs:25`) stays green; add an EOS-stop test.

### BASE-4 — minimal HTTP server + round-robin per-agent scheduler (= complete usable base)
- **Decision:** MUST-HANDROLL minimal. Reject the big engines' continuous batching + preemption (DEAD at B=2–4).
- **Port source / idea:** none for the scheduler — `decode.rs::decode_step` *is* the per-agent primitive; the
  server is a thin Rust loop over N `{KV cache, pos, prefill_done}` structs, interleaving one decode step per agent.
  ExLlamaV3 validates eager single-stream B=1 as the right architecture.
- **Concrete plan:** a few-hundred-line async Rust server (axum/hyper) with token streaming, one request queue,
  round-robin dispatch. **Scope note (anti-overstating):** this is a thin few-hundred-line task, not a
  multi-week project. Gated on BASE-1 (tokenizer) for text I/O.
- **Validation gate:** N concurrent agents each produce the same token stream as their isolated single-stream run
  (no cross-contamination).

> Model-load (`model.rs` + `crates/bw24-gguf/src/source.rs`, commit 41f0bc6) and kv-cache (`cache.rs`, commit
> 9ebf958) are already BASE-done HAVE — no gaps to close before the engine is usable.

**BASE complete = tokenizer + host sampler + EOS/stop glue + HTTP/round-robin server.** Only then start PERF.

---

## 3. PERF section (deferred until BASE done)

The two headline gaps both route through int8 MMA work: **prefill 43× vs llama.cpp** (commit 8d1c0b7) and
**decode 2.1×**.

### PERF-1 — quant-GEMM prefill tuning (HEADLINE) — HANDROLL-BETTER
- **State:** `crates/bw24-engine/cu/qmatvec_gemm.cu` exists and is wired. `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
  (line 54) — an Ada/Ampere-class int8 MMA shape that **genuinely runs on sm_120** (`build.rs:17`:
  `arch=compute_120a,code=sm_120a`). BM=64×BN=128×BK=32, 4 warps, decode-block→int8-once-into-smem.
- **5 dtypes, 2 kernels (verified):** Q8_0 / Q4_K / Q5_K via the single-`(dw,da)`-scale template; **Q6_K + NVFP4
  via `qmatvec_gemm_kernel2`** 16-wide sub-accumulation (the "can't fold two 16-elem scales into one int8 block"
  problem, `qmatvec_gemm.cu:127-140`). Dispatched at `m >= GEMM_M_THRESHOLD(16)` behind `BW24_GEMM`; decode (m=1)
  stays on validated dp4a.
- **Why int8 over bf16 / why not Marlin:** 219 vs 117 TFLOP/s sm_120, keeps weights quantized (the VRAM win that
  fits 30B-A3B in 24 GB), shares the q8_1 activation format with decode. Marlin (`marlin_template.h` m16n8k16
  LOP3-dequant) correctly rejected — W4A16-centric, needs an offline repack per dtype; the int8 path reuses the
  existing dp4a decoders bit-for-bit.
- **Remaining work (GEMM-PLAN):** (a) bit-equivalence gate vs dp4a in `crates/bw24-engine/src/bin/kernel_check.rs`
  — **actual thresholds in source:** f32-qmatvec path `rel < 1e-4` (`kernel_check.rs:241`), int8-activation fast
  path `rel < 3e-2` (`kernel_check.rs:264`) — the s32 accumulate is exact vs dp4a, only the final f32 scale rounds;
  (b) end-to-end argmax holds at 268 (qwen3) / 271 (qwen35) / 1178 (35B-MoE) with `BW24_GEMM=1`; (c) smem swizzle +
  per-dtype vectorized load tuning. GEMM-PLAN is honest: first tuned lands ~3500–5500 pp512, not llama's 6240; the
  residual is MMQ-style smem-swizzle work, not an MMA-throughput wall.

### PERF-2 — attention prefill (TC QK/PV) — MUST-HANDROLL — CRITICAL
- See §4 below — no existing kernel fits (llama tile-prefill is `nvcuda::wmma`). This is the 43× prefill bottleneck.
  Reuse the proven `mma.sync.aligned.m16n8k16` primitive (`flash_attn.cu:109`) for the FA QK/PV phase.
- **Reuse 1:1 (HAVE):** online-softmax rescale (`exp2`, self-normalizing — no 2.079 bias bug), LSE-merge split-K,
  quant-KV inline dequant — all in `flash_attn.cu`, architecture-neutral.
- **Gate:** prefill argmax holds at 268/271/1178; pp512 rises toward 6240.

### PERF-3 — attention decode (FA-vec) — HANDROLL-BETTER
- **Port source:** llama FA-vec warp-per-token shuffle-reduction softmax (`fattn-vec.cuh:86-254`,
  `research/inference-maps/llamacpp.md:19/69`) onto the existing quant-KV cache, plus **GQA broadcast** (Qwen3.5 =
  32Q/8KV = 4:1, load each KV pair once per warp → ~1.3× KV-BW). TRT-LLM `mla_sm120.cu` = TAKE the
  warp-specialization idea only; its TMA-paged-KV is over-built for B=1 resident KV.
- **Gate:** token stream bit-identical; decode tok/s rises toward the 847 GB/s SOL ceiling.

### PERF-4 — decode GEMV (NVFP4 tensor-core) — MUST-HANDROLL — CRITICAL
- See §4 below. This is the *decode* gap (NOT the prefill GEMM) — a warp-per-row vectorized MMVQ rewrite at ~86%
  of 847 GB/s SOL (DECODE-GAP-PLAN §3.1). Keep dp4a as the baseline/reference.

### PERF-5 — MoE: fused router + SLRU residency cache — HANDROLL-BETTER
- **State:** `hybrid_forward.rs:203` does host router (dtoh logits → softmax-over-256 → stable DESC sort → top-8
  renorm) then per-routed-expert `stage_expert` H2D into one scratch slot → `qmatvec_view`
  (`hybrid_forward.rs:261/265/274`). 29.75 GB of experts stay host-resident — this fits 35B-A3B in 24 GB where
  vLLM/SGLang OOM. **Gap (D1): no cross-token residency cache** — every token re-stages over PCIe (scratch sized
  for ONE expert, restaged in the j-loop).
- **Two improvements (both portable, idea-only):**
  - **Router fuse** (idea from vLLM `topk-moe.cu`): a fused warp `softmax→iterative-argmax` kernel kills the host
    round-trip. Low effort.
  - **SLRU residency cache** (pattern from ktransformers, `research/inference-maps/ktransformers.md:30-45`):
    per-layer GPU expert-residency bitmask + residency-keyed dispatch split
    `if resident: qmatvec_view(slot) else: stage_expert()+qmatvec_view()` — the exact one-change that converts
    always-stage Stage-1 into cache-aware Stage-2, dropping per-token PCIe to ~0 after warmup. Use ktransformers'
    frequency / front-loading static placement as a cheap warm-start admission prior.
- **Reject:** FlashInfer CUTLASS grouped GEMM (SM90 macro, sm_120 kernel bodies absent = DEAD); SGLang/vLLM Triton
  grouped GEMM (correct but low B=1 occupancy — staged `qmatvec` is the better fit).
- **Gate:** MoE argmax stays 1178; cache-hit path bit-identical to stage-every-token; per-token PCIe → ~0 after warmup.

### PERF-6 — spec-decode tree verification — HANDROLL-BETTER (base spec already HAVE)
- **State:** `spec.rs::generate_spec` is a complete greedy MTP — NextN draft (T=1 on own scratch KV) → batched
  verify `decode_step_t` (T=K+1, all-column logits, `spec.rs:177-203`) → greedy accept-prefix → partial-accept
  replays `draft[0..n_acc]+bonus` as ONE T=(n_acc+1) batched forward, **single weight read** (the profitability
  lever, `spec.rs:182-186`). Snapshot/rollback in `cache.rs:39-45` (append-only truncate for KV-len, full copy for
  conv/ssm). Token-identical-to-greedy is the documented invariant.
- **The one real improvement:** bolt a **tree mask** onto the existing `full_attn_verify` (`spec.rs:263`) to verify
  many draft candidates in one masked pass (SGLang `eagle_utils.py:48-159`, `frozen_kv_mtp_worker.py:81-199` —
  conceptual, portable mask + extra verify columns, no special kernel). This is the only path to *exceed* the
  raw-kernel decode bar.
- **Do NOT implement** (DECODE-GAP-PLAN debunks both): per-column snapshot (adds D2D writes on the >70%
  full-accept rounds → net slowdown); linear-attn-only replay (breaks the exact-match invariant).
- **Gate:** spec token stream stays bit-identical to greedy `generate`, AND measured tok/s > no-spec.

### PERF-7 — cuda-graph capture — MUST-HANDROLL (TAKE pattern) — LATE, GATED
- See §4 prerequisites. **Honest scope (DECODE-GAP-PLAN L4):** single-digit % wall-clock, ceiling ~18%, NOT
  inflated estimates — decode is GPU-bound (52.9 tok/s; only ~2.5 ms of ~13.9 ms is non-GPU-busy).
- **TAKE source:** llama.cpp 2-call-warmup + `cudaStreamBeginCapture(Relaxed)` + replay (`ggml-cuda.cu:4468-4525`,
  `research/inference-maps/llamacpp.md:84`). `cudaStreamBeginCapture` is sm_120-compatible; cudarc exposes the graph API.
- **Hard prerequisite (verified):** `decode.rs:70` `e.dtoh(&logits)` is one CPU↔GPU barrier per token that
  **cannot be captured mid-graph**. Must first move argmax + pos/token-counter onto the device (resident device
  buffer, device `t_kv` counter for indexed KV-append) — DECODE-GAP-PLAN Lever 3 / the GPU-fused sampler. That
  gates this entirely.
- **Discipline:** this lever *subsumes* all "fuse N launches" gains — do NOT sum op-fusion savings on top. Use
  **bucketed capture** at t_kv thresholds (512/1024/2048) to avoid a max-ctx-masking regression at short context.
- **Gate:** full 128-token generation bit-identical to non-graph; no perf regression at short t_kv.

---

## 4. MUST-HANDROLL — no sm_120 fit exists

These are the cases where there is **no existing kernel we can port** — either the only impls use instructions that
do not run on sm_120, or the only impls are multi-request-only and DEAD at B=2–4. Confirmed by grep: the tree has
**no** `wgmma`, **no** `tcgen05`, **no** `nvcuda::wmma`, and **no** `cudaGraph`/`BeginCapture`.

### 4a. Attention prefill — tensor-core QK/PV
- **Why no kernel runs:** llama.cpp's tile prefill uses `nvcuda::wmma` (a Volta-era intrinsic) — it does **not**
  lower well to sm_120. There is no portable sm_120 TC-prefill kernel to take.
- **The fit we write:** explicit `mma.sync` tiles. The primitive already exists and is validated in bw24 —
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` at `crates/bw24-engine/cu/flash_attn.cu:109` (the proven
  sm_120 `ldmatrix` discipline, same family as the int8 GEMM `m16n8k32`). Reuse it for the FA QK/PV phase.
- **Stakes:** this is the 43× prefill bottleneck (commit 8d1c0b7). PERF-critical.

### 4b. Decode GEMV — tensor-core NVFP4
- **Why no kernel runs:** the decode path is currently the scalar dp4a baseline; there is no existing
  warp-per-row NVFP4 tensor-core MMVQ that targets sm_120 at B=1. The multi-request CUTLASS/Triton grouped paths
  are DEAD here (SM90 bodies / low B=1 occupancy).
- **The fit we write:** a warp-per-row vectorized MMVQ rewrite at ~86% of the 847 GB/s SOL (DECODE-GAP-PLAN §3.1).
  This is **not** the prefill GEMM (PERF-1) — it is the separate decode root-fix.

### 4c. CUDA-graph capture — no existing capture code
- **Why nothing exists:** grep for `cudaGraph`/`BeginCapture` is empty; decode is eager per-token kernel launches.
  The *pattern* is portable (TAKE llama's capture), but bw24 has zero capture code today, and the capturable shape
  does not exist until the mid-graph `dtoh` barrier (`decode.rs:70`) is removed.
- **What blocks the TAKE:** the GPU-resident argmax + pos/token-counter must land first (4d below). So even though
  the pattern is portable, the *enabling* work is must-handroll.

### 4d. GPU-fused sampler (argmax / softmax→cumsum→categorical)
- **Why no kernel fits:** the host sampler (BASE-2) is correct at B=2–4, but once CUDA-graph lands it reintroduces
  a mid-graph D2H barrier that cannot be captured (DECODE-GAP-PLAN Lever 3). The one piece llama's `ggml_cumsum`
  does poorly on sm_120 is the cumsum (single-warp sequential).
- **The fit we write:** a single fused warp-shuffle GPU kernel — argmax for greedy, or `softmax→cumsum→categorical`
  in one launch writing the token ID to a resident device buffer, with a `__shfl_sync` warp-scan cumsum. Gate:
  integer-exact vs the host version on a fixed seed.

### 4e. Serving scheduler — round-robin per-agent
- **Why no kernel/framework fits:** vLLM/SGLang continuous batching + preemption is explicitly DEAD at B=2–4 (both
  the inventory and the scheduler analysis agree). There is no existing minimal scheduler to take.
- **The fit we write:** a thin Rust round-robin loop over N `{KV, pos, prefill_done}` agents — `decode_step` is the
  per-agent primitive. Not a kernel; a host state machine. (Also BASE-4.)

### 4f. Spilling — VRAM↔host↔disk (deferred)
- **Why must-handroll:** the lmcache choreography (layer-wise async overlap `start_load_kv`/`wait_for_layer_load`,
  prefetch layer i+1 while i computes, `-1`-sentinel paged slot-mapping, in-flight-store-before-evict barrier) is
  an *idea-take of a memcpy choreography*, not a framework dependency — all pure cudaMemcpy/stream ops, no kernels,
  no python/torch. MoE expert staging (PERF-5) is already a special case. **Deferred entirely** until a model +
  context exceeds 24 GB resident (resident-quant + q8_0/q5_1 KV push that out). EXTRA.

---

## 5. EXTRA section (deferred until BASE + the relevant PERF item land)

| Item | Decision | Source idea | Trigger to start |
|---|---|---|---|
| RadixAttention prefix cache | TAKE the idea | SGLang CPU token-ID trie (`radix_cache.py:206-267`, `:560-587`) | once multi-turn agents are the workload; high-ROI, zero kernel cost |
| EAGLE3 speculative decode | TAKE the algorithm | `speculative.cpp:419-894` | only if/when a draft model is wired |
| FP8 E4M3 KV (A/B vs q8_0/q5_1) | experiment only, not a rebuild | SGLang/FlashInfer | optional precision A/B; current cache is the conservative validated choice |
| Quantized K-shift (dequant→Hadamard→RoPE→requant) | defer | llama `llama-kv-cache.cpp:1826-1876` | only if context-window sliding/rotation is added |
| Marlin/TRT `ColumnMajorTileInterleave` offline repack | HANDROLL-BETTER-if-needed | Marlin / TRT-LLM | only if PERF-1 smem bank-conflict tuning proves the layout is the ceiling |
| ExLlamaV3 autotune-cache blob | EXTRA | `coop_autotune_v1.bin` | only if kernel tiles get parametrized per model |
| Tiered spilling | MUST-HANDROLL-if-needed (see 4f) | lmcache + ktransformers | only when a target model + context > 24 GB resident |

> **ExLlamaV3 trellis format = NONE (not portable).**

---

## Cited source files

Engine maps: `research/inference-maps/{llamacpp,sglang,ktransformers,vllm,flashinfer,tensorrt_llm,cutlass_marlin,exllamav3,lmcache}.md`,
`research/INFERENCE-FEATURE-MAP.md`. Perf roadmaps: `research/basics/{GEMM-PLAN,DECODE-GAP-PLAN}.md`.
bw24 source anchors (all under `crates/`): `bw24-engine/src/bin/run_gen.rs:15/25`, `bw24-engine/src/forward.rs:119`,
`bw24-engine/src/decode.rs:70/77/79/134`, `bw24-engine/src/spec.rs:177/182/203/263`,
`bw24-engine/src/hybrid_forward.rs:203/261`, `bw24-engine/src/cache.rs:7/65`,
`bw24-engine/src/bin/kernel_check.rs:241/264`, `bw24-engine/cu/qmatvec_gemm.cu:54/127`,
`bw24-engine/cu/flash_attn.cu:109/133`, `bw24-engine/build.rs:17`, `bw24-gguf/src/config.rs`,
`bw24-gguf/src/bin/inspect.rs:86`.
