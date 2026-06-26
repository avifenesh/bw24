# Decode-Gap Plan (adversarially verified; NOTE: readers used pre-session docs — Lever 0 NVFP4 already fixed)

This is decisive. Two project-grounded facts reshape the entire answer:

1. **9B-NVFP4 is currently BROKEN in bw24** (argmax 1543 vs llama 268) — the global-scale fix is unshipped. bw24's 59.6 is **Q8_0**, not NVFP4. So "beat llama 126 on 9B-NVFP4" requires first *correctly running NVFP4 at all*.
2. The prompt's "78 tok/s" baseline doesn't exist; the real baseline is **59.6 (Q8_0) / 52.9 (fast)**. The "847 GB/s ceiling at 47%/77%" framing maps to the project's own ledger: bw24 ~40% of ceiling, llama ~86%.

I have everything I need. Writing the plan.

---

# bw24 Decode-Gap Closure Plan — Ranked, Honest, Overlap-Aware

## 0. Reality reset (the prompt's premises are partly fictional — correct them before planning)

The prompt asks to go "78 → 77%+ of 847 GB/s to beat llama 126 on 9B-NVFP4." Three of those anchors do not survive contact with the repo, and the plan is built on the corrected ones:

| Prompt premise | Repo reality | Source |
|---|---|---|
| bw24 at "78 tok/s" | bw24 9B is **59.6 tok/s (Q8_0)**, 52.9 fast-path | `COMPETITOR-SETUP.md:295`, `benchmarks.md:45,75` |
| "47% of 847 ceiling" | bw24 Stage-B = **40% of ceiling**, 2.1× slower than llama | `benchmarks.md:20` |
| "beat llama 126 on **9B-NVFP4**" | **bw24's NVFP4 path produces WRONG output today** (argmax 1543 vs llama 268). The 59.6 is Q8_0. | `benchmarks.md:51-60` |
| "715 launches/token" | ROADMAP and GRAPH-PLAN say **320 launches/token** | `ROADMAP.md:48`, `GRAPH-PLAN.md:9` |
| CUDA-graph "15-20% BW" | GRAPH-PLAN's own honest ceiling: **~18% wall-clock, likely single-digit % real**, decode is GPU-bound not launch-bound | `GRAPH-PLAN.md:9,140` |

**Consequence:** the headline "beat 126 on NVFP4" has a hidden **correctness prerequisite** (NVFP4 global-scale fix) that gates everything else. You cannot speed up a kernel that outputs garbage. This is Lever 0 below, and it is non-negotiable.

The adversarial verdicts are correct on the central point: **5 of the 6 "bandwidth" levers and ~9 of the 12 "launch-inventory" levers are either already shipped or ~0% real.** The genuine gap is **GEMV memory throughput + dtype coverage**, with CUDA-graph as a secondary single-digit lever that *subsumes* every "N launches × 1.5µs" claim. Do not sum them.

---

## 1. Ranked levers (post-adversarial honest gain, dependency order)

Ranking is by **honest, non-overlapping contribution to closing the 59.6 → 126 gap**, with dependency order noted. "Real" reflects the adversarial verdict, re-grounded against the repo.

### Lever 0 — NVFP4 global-scale correctness fix `[PREREQUISITE, not a speedup]`
- **Honest gain:** 0% speed; **infinite** in the sense that without it the 9B-NVFP4 target is unmeasurable (output is wrong).
- **Why first:** The 9B "bar" (126.6) is an NVFP4 model. bw24 NVFP4 = argmax 1543 ≠ llama 268. Every downstream speed lever on NVFP4 is meaningless until this lands.
- **Dependency:** blocks all NVFP4 speed work. Q8_0/Q4_K/Q6_K are unaffected (`benchmarks.md:60`), so if you choose to chase the gap on Q8_0 instead, this becomes optional — but then you are not beating "126 on NVFP4," you are racing a different model.

### Lever 1 — Extend int8 dp4a fast path to all decode-hot dtypes (the real ceiling-closer) `[REAL, LARGEST]`
- **Honest gain (post-verdict):** **1.3–1.8× on the affected matmuls.** The adversarial pass flagged this (MMVQ-occupancy verdict) as "potentially the LARGEST real lever." Today **only Q8_0 uses dp4a; everything else falls to Stage-A f32 dequant at 3.6× slower** (`benchmarks.md:23-24`). Q4_K/Q6_K dp4a already landed (`benchmarks.md:46`); the residual is NVFP4 + the linear-attn projections (wqkv/ssm_*) still on Stage-A.
- **Why this is the gap:** llama is 86% of ceiling, bw24 is 40%. That ~46-point gap is overwhelmingly the GEMV-throughput/dtype gap, **not** elementwise launches (verdict topic 3, overall lever).
- **Caveat:** this is per-matmul on dtypes still on Stage-A. On an all-Q8_0 model the dp4a path is already in, so the *incremental* gain there is from occupancy (Lever 2), not dtype coverage.

### Lever 2 — MMVQ small-K / weight-streaming GEMV throughput `[REAL but NARROW]`
- **Honest gain (post-verdict):** **~1–2% of total decode**, *not* the 6–10% the readers claimed. The "81/82 idle SMs" diagnosis is **wrong** (out_f ≫ 82 saturates SMs in waves). The one genuinely real sub-lever is **small-K rows_per_block** for layers with `in_f < 4096` (nblk < 128 leaves threads idle after the first stride iteration) — but all the big matmuls (QKV, gate/up/down, lm_head) have `in_f ≥ 4096`, so gain there is **zero**. Vectorized 16B loads, `__restrict__`, byte_perm, Q8_0 padding: all **~0% and/or already done** (Q8_0 block is 34B, never 16B-aligned; padding would *increase* DRAM bytes → slower on a BW-bound kernel).
- **Worth doing:** the `__restrict__` annotation (`qmatvec.cu:387,416,663`) is a free, zero-risk one-liner — add it, but bank ~0–1%, not 5–8%.

### Lever 3 — GPU-side argmax + resident pos/token (remove the per-token host sync) `[REAL, and it UNBLOCKS Lever 4]`
- **Honest gain:** **a few % standalone**, but its real value is **enabling CUDA-graph capture.** `decode.rs:70` does `dtoh(logits)` → `lib.rs:322-325` `clone_dtoh + stream.synchronize()`: **exactly one hard CPU↔GPU barrier per token.** A mid-graph dtoh+sync **cannot be captured.** So this is a *prerequisite* for Lever 4, not merely additive with it.
- **Dependency:** must land before Lever 4 (CUDA-graph) yields anything.

### Lever 4 — CUDA-graph capture of the decode step `[REAL, but SMALL and SUBSUMES every launch-count lever]`
- **Honest gain (post-verdict + repo):** **single-digit % wall-clock, ceiling ~18%.** `GRAPH-PLAN.md:9` is explicit: decode is GPU-bound (52.9 tok/s), only ~2.5ms of ~13.9ms wall is non-GPU-busy, and only the launch-dispatch slice of *that* is recoverable. After max-ctx masking overhead (Option B) it can even be a **net loss at short context** without bucketed capture.
- **CRITICAL OVERLAP RULE:** this lever **claims essentially all** of the launch-overhead pool. Every "fuse N launches" lever (op-fusion, rms_norm fusion, elementwise fusion, conv/glog/kv fusion) draws from the **same pool**. **Do not add their gains to this one.** Post-graph, those fusions are worth ~0 incrementally (they remain useful as *graph hygiene* — `INFERENCE-FEATURE-MAP.md:99` calls op-fusion "the prerequisite to a clean CUDA-graph").

### Lever 5 — Op-fusion (gate+up+SwiGLU; NVFP4 scale-into-epilogue; add+rmsnorm) `[MOSTLY OVERLAPS Lever 4]`
- **Honest gain (post-verdict):** **~0.5–2% as graph hygiene**, NOT the 4–6% claimed. The "RMSNorm into MMVQ" mechanism **does not exist** in llama.cpp (RMSNorm fuses as RMS_NORM+MUL separately). The defensible piece is folding the **NVFP4 macro-scale into `qmatvec_nvfp4_dp4a` epilogue** (`qmatvec.cu:706`) — pure ALU, no occupancy risk, removes one read+write of `y` per NVFP4 matmul — and **gate+up+SwiGLU** fusion (8 full-attn layers only, since `full_attention_interval=4`). HBM-unique residual is ~1–2 MB/token vs ~8.86 GB/token weight traffic = fractions of a percent.

### Lever 6 — MTP profitability (NO-REPLAY) `[the proposed Options A/B are NOT real; a corrected approach is needed]`
- **Honest gain (post-verdict):** the readers' two options are **dead on arrival**:
  - **Option A** (per-column snapshot) measures against a **strawman** — `spec.rs:419-426` *already* does a single batched replay reading every weight ONCE. Option A *adds* K×24-layer×2.195MB D2D snapshot writes on **every** verify including the >70% full-accept rounds that currently cost zero → **net slowdown**.
  - **Option B** (linear-attn-only replay) is **incorrect** — `cache.rollback(snap, 0)` (`spec.rs:423`) truncates full-attn KV back to `pos`, so skipping full-attn layers leaves committed positions with **no K/V**, breaking the exact-match invariant. It's also explicitly deferred (`MTP-PLAN.md:158`).
  - **Storage math is wrong** too: real config is **24 linear layers** (not 18) → K=4 ⇒ 211MB.
- **What's actually real for MTP profitability:** raising **accepted-tokens-per-round**, not shaving replay. `INFERENCE-FEATURE-MAP.md:35` (D6) flags **tree-attention verification** (verify many candidates in one masked pass) as MED-HIGH and bolt-on to the existing verify-all-columns path. That, plus ensuring full-accept rounds (the common case) carry zero added overhead, is the profitable direction. **Honest gain: model-dependent; only profitable if acceptance rate × draft cost beats the no-spec step.**

---

## 2. Realistic cumulative projection (NOT a sum of best-cases)

Baseline: **bw24 9B = 59.6 tok/s (Q8_0)** / **52.9 fast-path**. Target: **beat llama 126.6 (NVFP4)**.

Compounding multiplicatively from the fast-path baseline (52.9), applying overlap discipline:

| Step | Lever | Multiplier (honest) | Running tok/s |
|---|---|---|---|
| base | fast-path Q8_0 | — | **52.9** |
| L1 | dp4a on all hot dtypes (NVFP4/linear-attn off Stage-A) | ×1.3–1.5 (only on still-Stage-A matmuls; partial on an all-Q8_0 model) | ~63–72 |
| L2 | MMVQ small-K (`in_f<4096` only) | ×1.01–1.02 | ~64–73 |
| L3 | GPU argmax / no host sync | ×1.02–1.04 | ~66–76 |
| L4 | CUDA-graph (subsumes L5 launch savings) | ×1.05–1.12 (capped ~1.18) | ~70–85 |
| L5 | op-fusion (graph-hygiene only — **no additive gain**) | ×1.00–1.01 | ~70–86 |

**Realistic cumulative on 9B: ~70–86 tok/s** (point estimate **~78**, with a thermal-bound 150–175W ceiling clipping the top of the range, per `INFERENCE-FEATURE-MAP.md:14`).

### Does it beat llama 126.6? **No. Plainly: it does not reach 126 on raw decode.**

The honest projection lands at **~70–86 tok/s vs the 126.6 bar — still ~1.5–1.8× short.** This matches `COMPETITOR-SETUP.md:295` which states the gap is ~2.1× and that the *realistic intermediate win is beating vLLM/SGLang no-spec (~70–90) first, not llama.* Anyone claiming 126 from these levers is summing best-cases and ignoring the L4/L5 overlap and the thermal ceiling.

---

## 3. What else is needed to actually beat 126 (since the levers above do not)

The levers above close the *launch + dtype-coverage* gaps. To reach 126 you need to attack the **GEMV bandwidth SOL itself** and/or change the throughput regime:

1. **A real tensor-core / mma NVFP4 decode GEMV** (not dp4a-on-CUDA-cores). llama's 126 comes from a kernel sitting at ~86% of 847 GB/s. bw24's dp4a path tops out around 40–55%. Closing *that* is a Stage-2 kernel rewrite (warp-per-row vectorized MMVQ matching llama's tuned mmvq), **not** in any reader lever. This is the single largest remaining item and is explicitly the diagnosis in `benchmarks.md` and verdict topic 3.
2. **MTP spec-decode that is net-profitable** — the only path to *exceed* the raw-kernel bar (llama itself uses MTP to go 42→66 on 27B). For 9B there's "no 9B draft on disk" (`COMPETITOR-SETUP.md:291`), so MTP needs the draft head wired first, then tree-verification (D6) to push accepted-tokens/round.
3. **Confirm thermal headroom** — at 150–175W the box may be power-limited before it is bandwidth-limited; verify with `nvidia-smi` power draw during decode that you're hitting the 847 GB/s wall and not the watt wall.

**Bottom line: the ranked levers get bw24 to ~78 (beating vLLM/SGLang no-spec, a real win), but reaching/beating llama's 126 requires a tensor-core NVFP4 GEMV rewrite and/or profitable MTP — neither of which is in the reader findings.**

---

## 4. Concrete code changes per lever (file:line)

**Lever 0 — NVFP4 global-scale fix**
- Load `<w>.scale` + `<w>.input_scale` per-tensor F32 globals at GGUF/safetensors load; apply `y = nvfp4_blockdequant(W)@x * w_scale` (`benchmarks.md:56-57`).
- `crates/bw24-gguf/src/dequant.rs` — NVFP4 dequant currently ignores globals (this file is already modified per git status).
- Epilogue scale multiply: `crates/bw24-engine/cu/qmatvec.cu:706` (apply `acc *= w_scale` before `mmvq_block_reduce_write`).

**Lever 1 — dp4a all dtypes**
- Pattern after the landed Q4_K/Q6_K dp4a kernels in `crates/bw24-engine/cu/qmatvec.cu` (`qmatvec_q4_K_dp4a` ~:416, `qmatvec_nvfp4_dp4a` ~:663).
- Route linear-attn projections (wqkv/ssm_*, `decode.rs:180-187`) and NVFP4 tensors through `quantize_q8_1` + dp4a instead of Stage-A `qmatvec_f32` (`lib.rs:142-152`). Gate is `BW24_FAST` (`lib.rs:418-435`).

**Lever 2 — MMVQ small-K + `__restrict__`**
- Add `__restrict__` to `W` in `qmatvec.cu:387, 416, 663` (free).
- small-K rows_per_block: launch-grid change for `in_f<4096` matmuls — mirror llama `mmvq.cu:455-473` `should_use_small_k`; bw24 launcher `lib.rs:142-152` / kernel stride loop `qmatvec.cu:398`.

**Lever 3 — GPU argmax + resident pos/token**
- Replace `dtoh(logits)` (`decode.rs:70`) + `lib.rs:322-325` `clone_dtoh + synchronize()` with a GPU argmax kernel writing the token id to a resident device buffer; keep `pos_d` resident and bump via device counter (`GRAPH-PLAN.md:79` describes the device `t_kv` counter + indexed KV-append).
- KV-append offset must read the device counter, not host `kvl.len` (`GRAPH-PLAN.md:79`, `decode.rs:125-126`).

**Lever 4 — CUDA-graph**
- New capture path: `cudaStreamBeginCapture`/`EndCapture` + `cudaGraphInstantiate`/`cudaGraphLaunch` around the per-token decode (model llama `ggml-cuda.cu:4443, 4512-4519`). bw24 launchers are at `lib.rs:241,255,269,308`.
- Use **bucketed capture** at t_kv thresholds (512/1024/2048/4096) per `GRAPH-PLAN.md:81` to avoid max-ctx masking regression.

**Lever 5 — op-fusion (graph hygiene)**
- NVFP4 scale-into-epilogue: fold `scale_inplace` (`lib.rs:493, 498-506`) into `qmatvec.cu:706`. (Overlaps Lever 0's epilogue edit — do both in one kernel change.)
- gate+up+SwiGLU: fuse `decode.rs:50-55` (8 full-attn layers, `full_attention_interval=4`, `config.rs:596`).
- add+rmsnorm: `decode.rs:39→42` only (post-ffn add at :61 feeds next layer's fresh buffer — not a clean pair).

**Lever 6 — MTP (corrected)**
- Do NOT implement reader Options A/B. Keep the existing single-batched-replay (`spec.rs:419-426`) and `cache.snapshot/rollback` (`cache.rs:98-139`).
- Pursue tree-attention verify (`INFERENCE-FEATURE-MAP.md:35` D6): add a tree mask to the verify-all-columns path (`spec.rs:204-216`).

---

## 5. Validation gate per lever (argmax must hold; MTP self-consistency must hold)

| Lever | Gate |
|---|---|
| **L0 NVFP4 fix** | **9B-NVFP4 argmax == llama.cpp** (must move from 1543 → 268). This is the whole point; no perf measurement until argmax matches. Same pattern as the validated 35B-MoE argmax=1178 and 9B-Q8_0 argmax=268. |
| **L1 dp4a all-dtypes** | Per-dtype rel-error vs Stage-A/ggml oracle ≤ 3e-3 (same bar used for landed Q4_K/Q6_K, `benchmarks.md:46`), AND end-to-end **argmax unchanged** on a fixed 512-token prompt. Use `examples/dequant_oracle_diff.rs` / `tools/ggml_dequant_ref`. |
| **L2 MMVQ small-K + restrict** | `__restrict__` is correctness-neutral; small-K must keep **argmax identical** and dp4a rel-error ≤ 3e-3 on small-`in_f` layers specifically. |
| **L3 GPU argmax** | Emitted token id must **exactly equal** the prior CPU-argmax token on a fixed seed/prompt, every position. No tolerance — it's an integer match. |
| **L4 CUDA-graph** | **argmax of the full 128-token generation identical** to non-graph path (bit-exact token stream). Plus a **perf regression check at short t_kv** (graph must not be slower than non-graph at t_kv<512, per `GRAPH-PLAN.md:81`). |
| **L5 op-fusion** | Fused-kernel output rel-error vs unfused ≤ 1e-3 (NVFP4 late-scale rounding) and **argmax unchanged**. Test against Stage-A oracle. |
| **L6 MTP** | **Self-consistency: spec-decode token stream must be bit-identical to greedy non-spec** (the exact-match invariant `spec.rs:2-4` rests on). Any tree-verify change must preserve this AND show measured tok/s > no-spec (profitability, not just correctness). |

---

## Skeptic's summary
- The real gap is **GEMV bandwidth + dtype coverage**, not elementwise launches. ~9 of 12 launch levers and 5 of 6 BW levers are already-shipped or ~0%.
- **CUDA-graph and op-fusion overlap fully** on launch overhead — counted once (~single-digit %, capped ~18%), never summed.
- **9B-NVFP4 is currently incorrect in bw24** — Lever 0 is a correctness prerequisite, not a speedup.
- Honest cumulative: **52.9/59.6 → ~70–86 tok/s** (point ~78). **This beats vLLM/SGLang no-spec (~70–90) but does NOT reach llama's 126.6.**
- To beat 126 you need what the readers did not propose: a **tensor-core NVFP4 GEMV at ~86% SOL** and/or **net-profitable MTP with tree verification**.