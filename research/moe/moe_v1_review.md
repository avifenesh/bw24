Confirmed: `n_expert_groups` defaults to 0 (no group routing — the DeepSeek group branch is skipped since `n_expert_groups > 1` is false), `expert_weights_scale` defaults to 0.0, and the call site hardcodes `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with `norm_w=true`. Everything I need is verified. Let me write the review.

I have verified every load-bearing claim against the actual `build_moe_ffn` in `/home/avifenesh/projects/llama.cpp/src/llama-graph.cpp`, the model graph in `qwen35moe.cpp`, the ggml argsort kernels (CPU `std::sort` and CUDA cub radix), the GGUF tensor dims, and the bw24 source. Here is the adversarial review.

---

# bw24 MoE FFN — adversarial review verdict: FIX-FIRST

The router math is **mostly correct** (softmax-before-topk is right, contradicting the "assume the router has a bug" hint — but the *selection-vs-weighting* split has a latent bug, and there are two correctness-fatal bugs outside the router). Plan is ~85% sound; 3 must-fix bugs will break the argmax gate, plus several that break compilation or residency.

## Answers to your 7 hard questions

1. **softmax-then-topk vs topk-then-softmax** — Plan is CORRECT. `build_moe_ffn` (llama-graph.cpp:1538) does `probs = ggml_soft_max(logits)` over all 256 FIRST (gating_op hardcoded `SOFTMAX` at qwen35moe.cpp:508), THEN `ggml_argsort_top_k(selection_probs, 8)` at :1599. The `SOFTMAX_WEIGHT` re-softmax-after-topk branch (:1617) is NOT taken for qwen35moe. Plan matches.

2. **norm_topk_prob / scale** — Plan is CORRECT. `norm_w=true` (qwen35moe.cpp:508) → renorm by `ggml_sum_rows` then `ggml_clamp(sum, 6.103515625e-5, INF)` then `ggml_div` (lines 1631-1643). Clamp constant matches exactly. `w_scale`: `expert_weights_scale` key is **absent in the GGUF** → defaults to `0.0f`, and `:1645` only scales when `w_scale != 0.0 && != 1.0` → skipped. Plan's "skip w_scale" is correct.

3. **3D expert slicing** — Math is CORRECT but with an off-by-design subtlety. Verified GGUF ggml `ne`: `gate_exps`/`up_exps` = `[in=2048, out=512, n_expert=256]` Q6_K; `down_exps` = `[in=512, out=2048, n_expert=256]` Q8_0. `expert_slice` computing `per_expert = out_f * row_bytes` and `start = eidx*per_expert` is correct because ggml stacks experts on the last axis contiguously. The plan correctly flags `down_exps` is transposed (`in=512, out=2048`). OK.

4. **shared-expert gating** — Plan is CORRECT. Always-on (no top-k), `sigmoid(ffn_gate_inp_shexp · h)` scalar gate (qwen35moe.cpp:531-540), `ffn_shexp = ffn_shexp * sigmoid_gate`, then `moe_out + ffn_shexp` (`:543`). Added AFTER routed sum, on the SAME `cur` (= attn_post_norm). Plan matches.

5. **silu on gate not up** — CORRECT. `ggml_swiglu_split(ctx0, cur, up)` = `silu(src0)*src1` = `silu(gate)*up` (confirmed via GLU backward grad at ggml.c:6868). bw24 `silu_mul_f32(gate, up)` = `silu(gate)*up` (kernels.cu:83). Plan matches.

6. **8 experts × right routing weights** — CORRECT in intent: `weights = ggml_get_rows(probs, selected_experts)` gathers the UNBIASED softmax probs at the selected indices, renorms, multiplies the down output (`:1792 experts*weights`), sums. Plan's `w[j] = probs[sel[j]]` then renorm matches. **But see BUG-3** (weight↔expert pairing after host sort).

7. **reuses hybrid attention unchanged** — YES, confirmed. `forward` only swaps the FFN block (lines 35-46); attention mixers (`full_attn`/`linear_attn`) and both residuals are untouched. Correct.

---

## BUGS (severity-ordered)

### BUG-1 — CRITICAL — Host argsort tie-break ≠ CUDA argsort, can flip top-1
The gate's ground truth runs `n_gpu_layers=999`, so llama.cpp uses the **CUDA** argsort (`ggml/src/ggml-cuda/argsort.cu`): cub `DeviceSegmentedRadixSort::SortPairsDescending` (or `DeviceSegmentedSort`). The plan uses Rust `sort_unstable_by` on `probs[b].cmp(probs[a])`. These disagree on **ties** (equal probs) and the radix sort over `n_expert=256` is not a comparison sort. With softmax over 256 experts, exact-equal float probs near the top-8 boundary are plausible; a different tiebreak selects a different 8th expert → different weighted sum → possible argmax flip. The CPU reference (`cmp_argsort`, ops.cpp:8184) uses `std::sort` (also unstable) which disagrees with BOTH. 
**Fix:** match the actual execution backend. Since the gate is GPU llama.cpp, replicate cub descending-radix semantics: for equal keys, segmented radix sort is **stable** (preserves original index order → ascending expert index on ties). Use a STABLE sort by `(-prob, index)`: `idx.sort_by(|&a,&b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)))`. Also use `total_cmp` not `partial_cmp().unwrap()` (NaN-safe; `unwrap` panics on NaN). Add the micro-check in §5 comparing host top-8 indices to llama.cpp `ffn_moe_topk` cb tensor for layer 0 — this is the bisection anchor, do it FIRST.

### BUG-2 — CRITICAL — `gate_inp_shexp` is 1-D `ne=[2048]`; `matmul`/`out_features` is wrong
Verified GGUF: `ffn_gate_inp_shexp.weight` ggml `ne = [2048]` (single dim, F32). The plan calls `e.matmul(&m.gate_inp_shexp, z, t)` expecting `[T,1]`. But `GpuTensor::out_features() = ne[1]` (model.rs:22) → **index out of bounds / wrong value** on a 1-element `ne`. `matmul`→`linear` would use `out_f = ne[1]` (garbage) and `in_f = ne[0] = 2048`. This is a dot-product (`out_f=1`), not a matmul as written. 
**Fix:** in `load_t`, this loads as `GpuTensor::Float` (F32) — fine — but the forward must treat `out_f=1` explicitly: `e.linear(z, m.gate_inp_shexp.float_data(), t, n_embd, 1)`. Do NOT route through `matmul`/`out_features`. (llama.cpp `build_lora_mm(ffn_gate_inp_shexp, cur)` produces `[1, n_tokens]`.)

### BUG-3 — HIGH — weight↔expert index pairing must use the SAME sorted order
The plan builds `sel = &idx[..8]` then `w[j] = probs[sel[j]]`, and in the expert loop uses `(j, &ex)` so `w[j]` pairs with expert `ex=sel[j]`. That pairing is internally consistent ✓. BUT the renorm `wsum` and the per-expert `w[j]/=wsum` must happen on exactly the 8 selected probs in the selected order — the plan does this correctly. The real trap: if you ever change `sel` ordering (e.g. re-sort by index for the slice loop) without re-gathering `w`, the weights detach from experts. **Not a bug as written**, but flag it as an invariant: `w[j]` and `sel[j]` must stay index-locked. No fix needed if BUG-1's stable sort is applied consistently to build both `sel` and `w`.

### BUG-4 — CRITICAL (residency) — 33 GB model cannot be resident on 24 GB; `htod_bytes` will OOM at load
The plan's §1 VRAM note acknowledges this but the loader (`load_exps` → `e.htod_bytes(raw)`) unconditionally uploads ALL 41 layers of Q6_K/Q8_0 exps = 33 GB to a 24 GB 5090. **`HybridModel::load` will OOM before the gate ever runs.** "Spill to host" / "partial-layer check" is hand-waved — there is no host-spill path in the engine (`htod_bytes` always allocates device memory). 
**Fix (decisive):** the correctness gate does NOT need all 41 layers resident. Either (a) gate on a **truncated layer count** (load+run first N layers, compare intermediate `ffn_out`/`z` at layer 0 vs llama.cpp cb tensors — this is the bisection check in §5 anyway), or (b) implement a host-resident exps store with per-expert H2D staging in `qmatvec_view` (only the 8 selected experts' rows per token need to be on device). Option (b) is the real Stage-1 path; do it, because (a) cannot produce a full-model argmax to compare. Without resolving this, the entire gate is unrunnable. This is the biggest gap in the plan — it is presented as a footnote but is a blocker.

### BUG-5 — HIGH — `add_scaled_rows` row indexing assumes shexp is token-major `[T, n_embd]`, but verify `down_shexp` output layout
`shexp = e.matmul(&m.down_shexp, &sg_act, t)` returns `[T, n_embd]` token-major (matmul output is `[m=T, out_f]`). `add_scaled_rows(src=shexp, scale=g_sig, dst=moe_out, ncols=n_embd, nrows=t)` computes `r = i/ncols` = token index, `scale[r]` = per-token sigmoid. ✓ Correct IF `g_sig` length == t. With BUG-2 fixed (`out_f=1`, `gate_scalar` is `[T,1]`=`[T]`), `g_sig` length == t ✓. Consistent. **Fix:** just ensure BUG-2's `out_f=1` produces a length-`t` buffer; then this is correct.

### BUG-6 — MEDIUM — `qmatvec_view` x-pointer offset for per-token slice
The kernel `qmatvec_f32` (qmatvec.cu:303) reads `xrow = x + (long)t*in_f` with grid `blockIdx.y = t`. The plan passes `m=1` and a `CudaView zt = z.slice(tok*n_embd..(tok+1)*n_embd)` of length `in_f=n_embd=2048`. With `m=1`, grid `blockIdx.y` only takes value 0, so `t=0`, `xrow = base+0` → reads the view from its start. ✓ Correct. BUT for `down_e`, the activation `act` is length `n_ff_exp=512` and `in_f=512` — also a full-length view, fine. **Caveat:** cudarc `.slice(range)` on a `CudaSlice<u8>` yields a `CudaView<u8>` whose base pointer IS offset — verify `launch_builder.arg(&wv)` passes the offset pointer, not the parent base. cudarc does honor view offsets in `arg()`, so OK — but this is exactly the kind of thing to assert in the §5 micro-check (dequant expert 0 on host vs `qmatvec_view`). Keep that check mandatory.

### BUG-7 — MEDIUM — `expert_slice` returns `(in_f, out_f)` but plan's call sites must NOT confuse gate/up (in=2048,out=512) with down (in=512,out=2048)
`down_exps.expert_slice` returns `in_f=512, out_f=2048`. The plan's loop passes `(dr, din, dout, ...)` into `qmatvec_view(..., din, dout, ...)` — correct as written (uses the slice's own dims). Just confirm `act` length for down is `n_ff_exp=512=din` ✓ and output `y` length is `dout=2048=n_embd` ✓. No bug, but the transpose is the #1 silent-corruption trap; the §5 micro-check covers it.

### BUG-8 — LOW — `qmatvec_view` signature: `x: &CudaView<f32>` but `down_e` activation `act` is a `CudaSlice<f32>`
Plan calls `e.qmatvec_view(&m.down_exps, dr, &act, ...)` passing `&act` (a `CudaSlice`) where the signature wants `&CudaView`. Type mismatch → won't compile. The plan's own note says "wrap with `.slice(0..n_ff_exp)`". **Fix:** either accept `impl PushKernelArg`/generic, or wrap: `&act.slice(0..n_ff_exp)`. Trivial but must be done or it won't build.

### BUG-9 — LOW — `moe_out.slice(...)` returns a `CudaView` (immutable); `axpy_into` writes through it
`axpy_into(dst: &CudaView<f32>)` then `dst[i] += ...` in the kernel mutates memory behind an immutable view. cudarc `CudaView` is a read view; to write you need `slice_mut` → `CudaViewMut`. The plan passes `moe_out.slice(...)` (immutable). **Fix:** use `moe_out.slice_mut(tok*n_embd..(tok+1)*n_embd)` and type `dst: &mut CudaViewMut<f32>` in `axpy_into`. As written it either won't compile or aliases incorrectly. Same concern for `add_scaled_rows` dst (it takes `&mut CudaSlice` — fine, that one's OK).

### BUG-10 — LOW — `MoeWeights` field comment says `gate_exps: Q6_K [n_embd, n_ff_exp, n_expert]` — consistent, but the `n_ff_exp` here is `expert_ff_length=512`, and `expert_shared_ff_length` is ALSO 512 (verified). Don't assume they differ; both are 512 in this model. Harmless, but the plan's `n_ff_sh = m.gate_shexp.out_features()` correctly reads it from the tensor rather than hardcoding — keep that.

---

## What's correct and should NOT be changed
- Softmax-over-256 BEFORE top-k (the hint's "assume router bug" is a red herring for THIS axis — the order is right).
- Unbiased probs for both selection and weighting (no `exp_probs_b`, no DeepSeek group routing: `n_expert_groups=0`).
- F16-min clamp `6.103515625e-5` and renorm-by-sum.
- Shared expert: always-on, sigmoid scalar gate, added after routed sum, on the same normed input.
- `silu(gate)*up` ordering.
- Residual: `ffn_out` added to `ffn_residual` = the pre-post-norm tensor (`x1`), confirmed at qwen35moe.cpp:204-221. Plan's `e.add(&x1, &ffn_out, &mut x2)` matches exactly.
- Reusing hybrid attention untouched.

---

## VERDICT: FIX-FIRST (do not code the forward until these are resolved)

**Blockers before any gate run, in order:**
1. **BUG-4** (residency/OOM) — the gate is literally unrunnable as planned. Decide host-staged exps now; this reshapes `qmatvec_view`. This is the single biggest under-specified risk.
2. **BUG-2** (1-D `gate_inp_shexp`) — silent wrong shared-gate, breaks argmax. One-line fix but mandatory.
3. **BUG-1** (argsort tie-break) — use stable `total_cmp` sort with ascending-index tiebreak to match cub radix; verify against `ffn_moe_topk` cb tensor first.

**Then compile-blockers:** BUG-8, BUG-9 (view mutability / slice types).

**Then run the §5 micro-checks BEFORE the full gate** — they are correctly specified and will catch BUG-6/BUG-7 (slicing/transpose) cheaply. Keep "no `BW24_FAST`" for the gate (matmul falls back to validated `qmatvec_f32`; `qmatvec_view` always uses it).

Files to touch are correct as listed. The math invariants in the plan's footer are accurate. Once BUG-1/2/4 and the compile-blockers are fixed, the implementation will argmax-match.