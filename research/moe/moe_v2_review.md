Confirmed: the current `HybridModel::load` asserts `moe.is_none()`, so this MoE model has **never** loaded through this path. The non-MoE dense hybrid (qwen35, the 9B daily-driver mentioned in commits) presumably has `nextn_predict_layers = 0`, so `n_layer == n_layer()` there and the loop was correct for that model. For the qwen35**moe** target, `n_layer = 41` includes the MTP block, so the loop `0..41` over-runs by one. **This is a real blocker the plan does not address.**

I have everything I need. Let me write up the complete review.

## REVIEW: MoE + EDGE-1 implementation plan vs. code reality

I verified every claim against the actual source, the real GGUF file (`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`, 31GB), llama.cpp's `qwen35moe.cpp` + `build_moe_ffn`, and cudarc 0.19.8 APIs. The router math, dtype handling, residency budget, and the 3D row_bytes fix are **correct**. But there is **one BLOCKER** the plan does not address (and which silently breaks the argmax gate), plus several MEDIUM/LOW issues.

---

### Answers to the explicit HARD checks

1. **Residency / 24GB fit — PASS, big margin.** Measured from real tensor sizes: GPU-resident non-expert weights + LM head = **2.31 GB** (token_embd stays host in `EmbedHost`; output head 0.54 GB is resident). Expert scratch = 3 slots × one expert = **2.8 MB**. KV cache for the 4-token gate is negligible (~0.16 MB). Experts stay HOST: `HostExps.bytes: Vec<u8>` is never `htod`'d — `expert_bytes()` returns `&[u8]` and only `stage_expert` does H2D of a single 0.86/1.11 MB block. Host RAM for experts = **29.75 GB** (41 layers × 725.6 MB), fits the 50 GB free. The "no full upload" / BUG-4 avoidance is real.

2. **Router — EXACT match to llama.cpp.** `build_moe_ffn` with `gating=SOFTMAX, norm_w=true`: `probs=ggml_soft_max(logits)` over all 256 (plain, scale=1.0); `selected=argsort_top_k(probs,8)`; `weights=get_rows(probs,...)` (unbiased); renorm `sum → clamp(6.103515625e-5) → div`. Plan reproduces this exactly, including the clamp-before-divide and "no w_scale".

3. **w_scale — correctly omitted.** `hparams.expert_weights_scale` defaults `0.0f` and `qwen35moe::load_arch_hparams` never sets it; `build_moe_ffn` line 1641 skips scaling when `w_scale==0.0`. Plan's "NO w_scale" is right.

4. **Shared gate 1-D (BUG-2) — correct.** `ffn_gate_inp_shexp` is `ne=[2048]` (out_f=1). Plan uses `e.linear(z, …, in=2048, out=1)` + `sigmoid`, not `matmul`/`out_features()` (which would read `ne[1]` OOB). llama.cpp: `build_lora_mm(ffn_gate_inp_shexp, cur)` → `ggml_sigmoid` → `ggml_mul`. Matches.

5. **row_bytes / stride — correct, verified arithmetically.** gate/up Q6_K [2048,512,256]: total 220,200,960 → stride 860,160, row_bytes 1680. down Q8_0 [512,2048,256]: total 285,212,672 → stride 1,114,112, row_bytes 544. The `row_bytes = raw.len()/(out_f*n_expert)` fix and the `expert_stride == out_f*row_bytes` assert both hold. `expert_bytes(e) = bytes[e*stride..(e+1)*stride]`.

6. **Per-token distinct experts — handled.** `moe_ffn` loops `for tok in 0..t`, recomputing `sel`/`w` per token and re-staging each expert. No reuse of token-0's set.

7. **silu(gate)*up — correct.** `silu_mul_f32` computes `(g/(1+e^-g))*up[i]`; plan passes `(gate, up)`. Shared expert `build_ffn(SILU,PAR)` = `down(silu(gate)*up)`. Matches.

8. **Q6_K gate/up + Q8_0 down dtypes — correct.** Verified per-tensor in the file. Down transpose (in=512/out=2048/row=544) carried purely through args to `qmatvec_f32`. The f32 dequant `deq()` path handles all three. Stage-1 stays off `BW24_FAST`, so no q8_1 div-by-32 concern.

9. **Staging H2D — correct.** `scratch.slice_mut(off..off+len)` → `CudaViewMut<u8>` implements `DevicePtrMut`; `memcpy_htod(host:&[u8], &mut dst)` is valid. `qmatvec_view` does `w.slice(range)` → `CudaView<u8>`, passed via `b.arg(&wv)` (PushKernelArg impl exists). Same default stream → H2D is ordered before the dependent kernel. Correct.

---

### BUGS

**BLOCKER — B0: loader/forward run 41 layers, executing the MTP block (block 40) as a trunk layer. Breaks the argmax gate.**
- Reality: `block_count=41`, `nextn_predict_layers=1`. bw24's `ModelConfig.n_layer = block_count = 41` — this equals llama.cpp's `n_layer_all`, **not** `n_layer()`. llama.cpp computes `n_layer() = n_layer_all - n_layer_nextn = 40` and the main graph loops `il < 40` (`qwen35moe.cpp:181`); block 40 is loaded by `load_block_mtp` and runs **only** in the separate `graph_mtp` (DECODER_MTP), never in the prefill that `llama_logits` exercises.
- The plan's own note ("0..cfg.n_layer = 40 trunk layers; MTP block 40 is excluded because n_layer excludes nextn") is **factually wrong for this file**: `cfg.n_layer` is 41 and the loop `for il in 0..cfg.n_layer` (`hybrid.rs:63`, and the matching forward/decode loops) hits blocks 0..40 — i.e. it builds and executes the MTP block as if it were a normal decoder layer. That injects an extra full-attn+MoE layer the reference never runs → argmax will not match.
- Also: block 40 carries `nextn.eh_proj/enorm/hnorm/shared_head_norm` and its `ffn_gate_inp`/`ffn_gate_inp_shexp` are **BF16** (blocks 0-39 are F32) — loadable (BF16 dequantizes to Float) but it should not be in the trunk at all.
- Severity: BLOCKER (the only gate that matters, argmax, fails).
- Fix: introduce a trunk count `n_layer_trunk = cfg.n_layer - cfg.nextn_predict_layers` (= 40) and loop `0..n_layer_trunk` in `HybridModel::load`, `forward`, and `decode_step`. Do NOT build/run block 40. (This is a pre-existing latent bug masked until now by the `moe.is_none()` assert; it surfaces the moment this model loads.)

**MEDIUM — B1: argsort tiebreak claim is unverifiable against the actual reference path and likely irrelevant, but the stated justification is wrong.**
- Plan comments `total_cmp().then(a.cmp(&b))` "matching CUDA cub radix". The CPU reference uses non-stable `std::sort` with strict `>` (`cmp_argsort`), and `llama_logits` runs the **CUDA** backend with `n_gpu_layers=999` — which uses cub `DeviceRadixSort`/`DeviceSegmentedRadixSort` (or the fused `topk-moe.cu`), not the CPU path. Exact-tie ordering between these is not guaranteed identical to `then(ascending-index)`.
- Practical impact: after softmax over 256 f32 logits, exact ties are essentially impossible on real inputs, so top-8 set-equality will hold regardless. Severity MEDIUM only because the *comment asserts a correctness equivalence that isn't established*; the behavior is fine.
- Fix: keep `total_cmp().then(a.cmp(&b))` (deterministic, NaN-safe) but soften the comment to "deterministic tiebreak; ties unreachable in practice." Validate via Stage-B top-8 set-equality rather than asserting bitwise sort identity.

**LOW — B2: `Vec::with_capacity(cfg.n_layer)` and any layer-count-derived sizing will be off by one once B0 is fixed.** Mechanical; fix alongside B0 (use the trunk count).

**LOW — B3: prefill `moe_ffn` per-token serialization is O(t·8·3) H2D + launches.** Fine for the 4-token gate (~96 matvecs/layer), but `hybrid_forward.rs::forward` calls `moe_ffn(e, m, &z, t)` once with the full t and the method loops tokens internally — correct, just slow. Not a correctness issue. No fix needed for the gate.

**LOW — B4: redundant `use bw24_gguf::GgmlType;` in model.rs.** `model.rs:7` already imports it; the appended `use` will trigger an unused/duplicate-import warning (or E0252 if not glob). The plan flags this itself ("drop if duplicate"). Drop it.

**LOW — B5: `run_hybrid.rs:28` sorts top-5 with `partial_cmp(...).unwrap()` (NaN-panics).** Display-only, doesn't affect routing. The forward already asserts `non-finite==0` first, so unreachable in practice. Optional: switch to `total_cmp`.

**NIT — B6: kernels appended to `cu/hybrid.cu` but `silu_mul`/`add` live in `cu/kernels.cu`.** Functionally fine — `Engine::func()` searches all four fatbins. Just keep `axpy_f32`/`add_scaled_rows_f32` in whichever .cu the build compiles (hybrid.cu is built into `BW24_HYBRID_FATBIN`, so it works).

---

### VERDICT: FIX-AGAIN

One BLOCKER (B0) must be fixed before the argmax gate can pass — without it the plan executes 41 layers including the MTP block, which the llama.cpp reference never runs in prefill, so argmax cannot match. Everything else (router math, dtypes, 3D stride/row_bytes, residency, staging, shared-gate 1-D, silu order) is correct and the 24GB budget passes with ~21GB of headroom (2.31GB resident + 2.8MB scratch).

Minimal fix to unblock: compute `let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;` and loop `0..n_trunk` in all three places (`hybrid.rs` load, `hybrid_forward.rs` forward, `decode.rs` decode_step). Note `ModelConfig` already carries `nextn_predict_layers` (config.rs:83) and `n_layer_total` (line 84) — the trunk count is `n_layer - nextn_predict_layers`, NOT `n_layer`. After that fix, the plan should RUN in 24GB and is positioned to argmax-match.

Files verified (all absolute):
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/config.rs` (MoeConfig, n_layer=block_count=41, nextn=1)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs` (GpuTensor, EmbedHost, load row_bytes=raw/out_f)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid.rs` (loop `0..cfg.n_layer` — the B0 site, line 63)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid_forward.rs` / `decode.rs` (FFN dispatch sites + their per-layer loops)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` (qmatvec/qmatvec_f32 grid, matmul, linear, silu_mul; new helpers' signatures all type-check vs cudarc 0.19.8)
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec.cu` (qmatvec_f32 reads `W + o*row_bytes`, deq dispatch Q8_0/Q4_K/Q6_K)
- `/home/avifenesh/projects/llama.cpp/src/models/qwen35moe.cpp` (n_layer trunk loop, load_block_mtp, build_layer_ffn) and `/home/avifenesh/projects/llama.cpp/src/llama-graph.cpp:1496` (build_moe_ffn router)