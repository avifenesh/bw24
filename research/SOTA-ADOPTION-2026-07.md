# SOTA Sweep Adoption Report — July 2026

## Scope

Source-level sweep of 8 inference engines, adversarially judged against bw24's measured cost centers and wall ledger:

| Engine | Source location / version |
|--------|--------------------------|
| vLLM | v0.23.0-0.24.0 on disk; web-refreshed July 2026 |
| SGLang | wheel `~/.venvs/torch/lib/python3.12/site-packages/sglang/` (post-June-2026 build) |
| FlashInfer | v0.6.12 on disk; web-refreshed through v0.6.14 (2026-07-02) |
| TensorRT-LLM | csrc/ on disk (flashinfer-vendored tinygemm2, xqa); web-checked July 2026 |
| ktransformers | v0.6.3.post1, clone HEAD 2026-06-25, web-checked 2026-07-03 |
| LMCache | map-era + June 2026 dev-branch commits verified |
| CUTLASS/Marlin | sgl-kernel 0.3.21 + Marlin on disk; CUTLASS 4.6.0 (June 11 2026) |
| ExLlamaV3 | master v0.0.43 (2026-06-14) + dev through 2026-07-03, cloned to scratchpad |

**Method:** 49 candidates extracted from source-level maps -> adversarial judge vs bw24 cost centers (C1-C7), wall ledger closed negatives, exactness law (8 FP-order lessons), measured shares in rig5090.jsonl, and HANDOVER/ROADMAP state. 27 survived, 22 rejected.

**Date:** 2026-07-04

---

## Ranked Adoption Table (27 candidates, rank_score descending)

| Rank | Name | Engine | Cost Center | Verdict | Expected Size | Exactness Risk |
|------|------|--------|-------------|---------|---------------|----------------|
| 7.0 | Single rig-native NVFP4 layout (FP4 TC prefill + MMVQ decode) | flashinfer | C3 | ADOPT-NOW | +13% pp512, larger at 6k TTFT | Medium-high (MMVQ addressing rewrite) |
| 7.0 | Expert-grouped layerwise MoE prefill | ktransformers | C6 | ADOPT-NOW | 3-10x MoE TTFT | Real but bounded (sel-rank reduce order) |
| 6.5 | Hybrid-aware prefix cache (block-hash KV + GDN/conv checkpoints) | vllm | C5 | ADOPT-NOW | saves ~8.7s 27B prime per shared-prefix hit | Medium-high (chunked WY FP reorder) |
| 6.0 | Chunked WY blockwise-inverse GDN prefill | flashinfer | C3 | ADOPT-NOW | +9-15% pp512, scales to 6k | Medium (prefill FP-order, batched-prime precedent) |
| 6.0 | cp.async smem-staged weight pipeline for batched MMVQ | vllm | C1 | ADOPT-NOW | +5-9% e2e 27B spec | Low-moderate (load-timing only) |
| 4.0 | Marlin-style offline repack into fragment-ordered weight layout | cutlass_marlin | C1 | ADOPT-NOW | +3-8% e2e 27B spec | Low (index arithmetic, same dot order) |
| 6.0 | Grouped expert matvec — single launch over all selected experts | tensorrt_llm | C6 | INVESTIGATE | +5-15% MoE decode | Low-medium (cross-expert combine order) |
| 6.0 | Single-launch fused expert-FFN with expert-id indirection (Marlin-MoE) | cutlass_marlin | C6 | INVESTIGATE | +10-25% MoE decode | Medium-low (off spec path) |
| 5.0 | DFlash block-diffusion draft + target-latent KV injection | sglang | C4 | ADOPT-LATER | +30-50% conditional on drafter | Low (verify gates all output) |
| 4.5 | Miss-tail overlap via pending/deferred expert tasks | ktransformers | C6 | ADOPT-LATER | +5-12% on 35B decode | Low (ordering-only) |
| 4.0 | Layer-overlapped async tier transfer (LayerDoneCounter) for MoE prefetch | sglang | C6 | INVESTIGATE | +15-25% MoE decode if miss>=15% | Low (byte-copy only) |
| 3.5 | Radix-tree prefix cache with recurrent-state-aware nodes | sglang | C5 | ADOPT-LATER | 0% current; large on multi-session | Medium (split-point restore) |
| 3.0 | Async copy-stream discipline for MoE expert tier | lmcache | C6 | ADOPT-LATER | +10-20% C6 if misses 10-20% | None |
| 3.0 | Packed GDN decode: fused in_proj + packed recurrent kernel | sglang | C7 | ADOPT-LATER | 2-4% on 9B, 1-2% on 27B | Medium-low (mixed-dtype layers) |
| 3.0 | PDL on eager verify chain + tiny GDN projections (vllm/FI) | vllm | C7 | INVESTIGATE | 2-5% e2e | Low (scheduling only; __restrict__ hazard) |
| 3.0 | Async expert prefetch (vllm offloader pattern) | vllm | C6 | ADOPT-LATER | unknown, 0% dense | Low (same bytes/kernel) |
| 3.0 | XQA specDec shared-KV multi-row verify walk | flashinfer | C2 | INVESTIGATE | 4-8% at p3 | Low if fixed-chunk holds |
| 2.5 | Routing-independent MoE launch (device-side index packing) | exllamav3 | C6 | ADOPT-LATER | 10-20% MoE if host sync >=20% | Low (launch plumbing only) |
| 2.2 | PDL for tiny-launch chains (CUTLASS 4.6 DSL) | cutlass_marlin | C7 | INVESTIGATE | 2-3.5% e2e 27B | Low (no FP change; race hazard) |
| 2.0 | cp.async.cg multi-stage smem pipeline for batched MMVQ | cutlass_marlin | C1 | INVESTIGATE | +6-15% 27B if staging helps | Low-moderate (proven precedent) |
| 2.0 | Deep-K single-CTA matvec decomposition (exl3_gemv) | exllamav3 | C1 | INVESTIGATE | 3-8% if exactness survives | HIGH (accumulation reorder) |
| 2.0 | PDL across decode/verify kernel chain (flashinfer) | flashinfer | C1 | INVESTIGATE | 3-7% e2e 27B | Low (scheduling only) |
| 2.0 | TMA bulk async weight staging (tinygemm2 pattern) | flashinfer | C1 | INVESTIGATE | 2-4% e2e 27B | None |
| 2.0 | tinygemm2 warp-specialized TMA pipeline | tensorrt_llm | C1 | ADOPT-LATER | +8-15% 27B if staging headroom real | Low-moderate |
| 1.5 | XQA SPEC_DEC packed-mask multi-token verify | tensorrt_llm | C2 | ADOPT-LATER | 5-8% 9B spec | Highest (FP-order gate ~50%) |
| 1.0 | NVFP4 KV cache (4-bit K/V + FP8 scales) | flashinfer | C2 | ADOPT-LATER | 3-6% at p3, strategic VRAM | Low mechanism / Medium quality |
| 1.5 | Disk-persisted runtime autotuner (exl3 sweep) | exllamav3 | C1 | ADOPT-LATER | <2% attributable | Low |

---

## ADOPT-NOW Transfer Plans (6 items)

### 1. Single rig-native NVFP4 layout (rank 7.0) — C3 prefill

The FORMAT-DECISION (committed `8d04e92`) already reserves rig-native internal layouts. The vendored `cutlass_fp4_sm120.cu` + `cutlass_ffi.rs` proved 4.4x GEMM kernel speedup and argmax-exact output (`824b2af`). The `BW24_FP4` seam (`ff951f8`) showed +27% pp512. Transfer: extend the loader repack (which already handles safetensors/NVFP4 per `model.rs`) to emit the CUTLASS sm120 blockscaled interleaved layout as the single on-device format; route `prime_cache` GEMMs at m>=16 to the existing `cutlass_fp4_sm120.cu` path; write an MMVQ-address-swizzle variant of `qmatvec.cu` (the b4/r2w8 family) that reads the new layout in logical block order so per-(token,row) dot FP order is unchanged. Gate with the full battery: msweep bit-exact, kernel-check, run-gen argmax==82, run-spec K=1..8 on all three configs.

### 2. Expert-grouped layerwise MoE prefill (rank 7.0) — C6 prefill/TTFT

`hybrid_forward.rs:509-551` currently runs MoE prefill as the per-token m=1 GEMV walk even under batched `prime_cache`. Transfer: in `prime_cache`'s MoE branch, sort the chunk's (token, expert) pairs by expert using the host-side `sel` arrays, stage each activated expert once via `moe_cache.rs` SLRU (hits skip H2D), and run the existing int8-MMA path (`mmq_fp4.cu` / `qmatvec_gemm.cu`) at m=tokens-routed-to-expert; scatter/gather rows via a small index kernel. Gate: run-gen argmax + the prime batched-vs-tokenwise A/B protocol already established for dense. Prerequisite: run the pending 35B/gemma4 interleaved baseline bench vs llama first (HANDOVER open item) so the win is measured against a real number.

### 3. Hybrid-aware prefix cache with GDN/conv checkpoints (rank 6.5) — C5 TTFT

bw24 already owns every hard primitive: `VerifyCkpt` (in `spec.rs`), prefix-stable `gdn_scan` re-run from snapshot, and `ssm_conv_ring_rebuild_f32` (in `hybrid.cu`). Transfer: in `cache.rs`, implement SHA-style block hash chain for KV indices; checkpoint GDN/conv state every N blocks during `prime_cache`; restore path = the existing VerifyCkpt rebuild kernels. Avoid the vllm #45238 pitfall (checkpoint on shared-prefix boundaries, not just last block). Gate: decode/verify dispatch identity is untouched; run-gen argmax + first-token A/B. Build order: one-layer prototype first, full battery, then expand.

### 4. Chunked WY blockwise-inverse GDN prefill (rank 6.0) — C3 prefill

`gdn_scan` in `hybrid.cu` is 17.6ms of ~101ms pp512 (17%) at exact llama parity. Transfer: implement the chunk-parallel matmul form (cumsum-log gate kernel + small triangular-inverse kernel + 7 GEMMs per chunk driven through existing f16/bf16 MMA paths in `qmatvec_gemm.cu`). Prefill-only; decode keeps the sequential scan in `hybrid.cu` so decode==verify dispatch identity is untouched. Gate: one-layer prototype + run-gen argmax==82 before building all layers (the stream-K lesson: 1e-6 reorder flipped 82->68). Ship behind `BW24_GDN_CHUNKED=0` fallback seam.

### 5. cp.async smem-staged weight pipeline for batched MMVQ (rank 6.0) — C1 decode

`qmatvec.cu:1099-1300` (the b4/r2w8 family) has measured long_scoreboard 10.9-16.7/issue at only 50-60% DRAM after the register double-buffer fix. Transfer: new `_ca` variants in `qmatvec.cu` behind `BW24_MMVQ_BV` dispatch — reuse the existing `cp_async16`/`cp_async_window` helpers from `qmatvec_gemm.cu:171-185`; 3-4 stage smem ring per warp, zero register growth; identical per-(token,row) dot accumulation order. Apply the same recipe to the untouched k-quant batched variants in `mmq_q45k.cu` (~19% of 9B verify). Gate: msweep bit-exact-bad=0 vs pf/r2w8, kernel-check ALL GREEN, run-spec K=1..8.

### 6. Marlin-style offline repack into fragment-ordered weight layout (rank 4.0) — C1 decode

Repack GGUF NVFP4 blocks at model load into the exact per-warp walk order of the batched MMVQ kernels. Fuse the current 6 scattered LDGs per block into 1-2 LDG.128/cp.async 16B transactions; hoist q4_K interleaved sub-scales into a contiguous stream. Implementation: one-time repack kernel in the loader (layer-streamed to avoid dual-VRAM spike on 24GB); no runtime format change for the rest of the engine. Exactness: same bytes, same dot order -> bit-identical gates achievable. Gate with msweep + kernel-check + run-spec. Composes multiplicatively with item 5 (repack makes clean 16B cp.async possible).

---

## INVESTIGATE Items — One Cheap Measurement Each

| Item | Settling measurement |
|------|---------------------|
| Grouped expert matvec (TRTLLM-Gen) | One nsys trace of 35B decode: if launch/host-gap share >=15% of MoE step, upgrade to ADOPT-NOW |
| Single-launch fused expert-FFN (Marlin-MoE) | Same nsys trace — confirm per-expert launch serialization share |
| Layer-overlapped async tier transfer | nsys miss-stall decomposition on 35B decode (counters already instrumented in `moe_cache.rs`) |
| PDL on verify chain + GDN (vllm) | A/B llama.cpp on-rig with `GGML_CUDA_PDL=0` vs `1` — if llama gains >=3%, build it |
| PDL tiny-launch chains (CUTLASS 4.6) | Same llama PDL toggle; then one-pair spike (ffn_up->ffn_down b4) via cuLaunchKernelEx C shim |
| PDL decode/verify chain (flashinfer) | Same llama PDL A/B, confirms whether overlap on serial 54.9us chains pays on sm_120a |
| cp.async.cg multi-stage (cutlass_marlin) | One A/B of `_ca` variant on the b4 kernel via existing `BW24_MMVQ_BV` seam + MSWEEP DRAM-cold harness |
| XQA specDec shared-KV multi-row verify | Re-read the design-B burial note vs current fixed-chunk code (30 min theory check); if unlocked, one-layer prototype |
| Deep-K single-CTA (exl3_gemv) | Prototype ONE shape (ffn_down b4, 640 blocks) and run argmax gate — ~70% expected gate failure |
| TMA bulk async staging (tinygemm2) | ncu experiment: LDG throughput vs FMA chain stall — determines if load mechanism is the residual bottleneck |

---

## NOT Taking and Why (22 rejections, 5 themes)

**Already built / closed-negative equivalent (9 items).** CUTLASS sm120 NVFP4 prefill GEMM, Marlin NVFP4 W4A16 decode GEMM, CUTLASS TmaWarpSpecialized grouped GEMM (all three already in-repo as `cutlass_fp4_sm120.cu` / `cutlass_ffi.rs`). Prefill-routing-stats SLRU warming (MoE prefill already warms SLRU). FP4 block-scale grouped MoE GEMM (FORMAT-DECISION already adopted the policy). PDL on verify chain closed by graph-spec-stage3 idle-gap reclamation measurement. Cross-request prefix-KV cache stale (exact-extension reuse landed). CUDA-graph node re-parameterization (graph-capture already measured negative). Multi-problem fused matvec launch (post-dual-matvec already tested).

**Zero e2e on headline bench / conditional on unbuilt infrastructure (5 items).** Single-kernel UVA KV gather/scatter (no KV host tier exists). CacheGen GPU arithmetic-coding KV codec (no disk tier). Radix-tree adaptive-K spec controller (acceptance lever already cashed). KV offload with QUEST/block-anchor sparse attention (removes keys from attention = exactness violation). NVFP4 KV cache at short context (~0% where the bench lives).

**Exactness/quality violation with no mitigation (3 items).** Warp-shuffle Hadamard rotation + ballot bitplane (premise about fixed 64-key chunks is false; `flash_attn.cu:1464` computes dynamic splits). QUEST block-anchor key removal (breaks bit-identity law). Fused inverse->forward RoPE re-rotation (CacheBlend graded fit=NONE for bw24's GDN hybrid).

**Mechanism real but transfers at ~0% or <2% in bw24's regime (3 items).** Adaptive-K per-round draft length (measured sweep showed K=4 optimal, ~0% gain from adaptation). Disk-persisted autotuner num_sms axis (bw24 grids are not persistent-cooperative). b12x fused-MoE grouped pipeline (C6-only, 0% on headline 9B/27B).

**Duplicate of an already-planned roadmap item with no new mechanism (2 items).** AVX2+AVX-VNNI RAWINT4 CPU expert GEMM (already triaged in `ktransformers.md:92`). Per-round adaptive draft from vllm (same as sglang version, same verdict).

---

## Queue Recommendation

**Current queue (from HANDOVER next-levers):**

```
H1: acceptance agent running on GPU (spec acceptance 0.58 -> 0.75 target)
H2: GDN host-fusing (tiny projections, ~2-3%)
H3: k-quant batched pf/r2 recipe (~2-4% on 9B verify)
H4: 35B/gemma4 re-bench baseline
H5: MoE async prefetch stage (gated on H4)
```

**Merged queue with ADOPT-NOWs slotted by rank_score x readiness:**

```
GPU timeline (serialized, each occupies the rig):
  H1  acceptance agent (in progress)
  A5  cp.async smem-staged MMVQ _ca variants (rank 6, ~day build+A/B)
       — directly extends the just-landed r2w8; same kernel file, same harness
  A6  Marlin-style offline repack (rank 4, ~day)
       — composes with A5; repack makes 16B cp.async clean
  H2  GDN host-fusing (rank ~2, few hours)
  H3  k-quant batched pf/r2 recipe (same harness as A5)
  A4  Chunked WY GDN prefill (rank 6, one-layer prototype first)
       — GPU needed for gate battery
  A1  Single rig-native NVFP4 layout (rank 7, largest engineering item)
       — loader repack + MMVQ addressing; prefill GEMM already proven
  A3  Hybrid-aware prefix cache (rank 6.5, build when multi-session live)

CPU-preparable in parallel (research packs, layout tools — pipeline law:
research never occupies GPU):
  A2  Expert-grouped MoE prefill: sort/gather logic is pure Rust host code;
       kernel path already exists (int8-MMA MMQ). Also: run H4 baseline first.
  A6  Repack index math: pure-host one-time loader kernel, testable offline
  A3  Radix tree + hash chain: pure Rust host bookkeeping, tests w/o GPU
  INVESTIGATE: all 10 nsys/ncu measurements and the llama PDL A/B toggle

Gated on external (model availability):
  DFlash drafter for Qwen3.6-27B — check HF collection; SpecForge training
       only after MMVQ tranches exhaust (acceptance becomes sole structural lever)
```

**Ordering rationale:** A5+A6 slot immediately after H1 because they (a) attack the #1 cost center (C1, ~60-73% of the 27B round), (b) are cheapest to A/B on existing harness, (c) compose multiplicatively, and (d) are CPU-preparable layout math + a few-hundred-line kernel variant. A4 (chunked GDN) and A1 (single layout) are larger but attack the weakest ratio (C3, 0.34-0.45x llama at 6k) — they slot after because their gate batteries are longer and the TTFT front matters less than decode tok/s for the current headline bench. A2 (MoE prefill grouping) is CPU-preparable now but GPU-gated on H4 (the 35B baseline that doesn't exist yet). A3 (prefix cache) is ordered last because its value is 0% until multi-session serving is live.
