# CUDA-Graph Decode Capture — Build Plan

All confirmed. `CudaStream::cu_stream()` is public at core.rs:732, returning the raw `sys::CUstream` needed by `stream::begin_capture`. The Engine exposes `stream()` returning `&Arc<CudaStream>`. I have everything required to write the plan with verified citations.

# CUDA-Graph Decode Capture for bw24 — Concrete Implementation Plan

## 0. Verdict up front (honest ceiling)

CUDA graphs help bw24 decode **only** by amortizing per-kernel CPU launch dispatch into one `cuGraphLaunch`. Decode at `9B-Q8_0 = 52.9 tok/s` is **GPU-bound** (`research/benchmarks.md:75`), so the realistic win is the slice of wall time that is *not* GPU-busy: roughly **2.5 ms of ~13.9 ms wall ≈ ~18% wall-clock headroom**, and only the launch-dispatch fraction of that is recoverable. Graphs do **not** add fusion, better occupancy, or bandwidth — those are separate levers (Task #5 MMVQ occupancy, Task #2 KV-quant). Treat ~18% as a ceiling, not a target; real gain after KV-append/masking overhead is likely **single-digit percent** unless launch overhead dominates more than current profiling shows.

---

## 1. What changes per step vs. what is static

`decode_step` (`crates/bw24-engine/src/decode.rs:12-64`) takes `(token: u32, cache: &mut Cache)`. Auditing the body:

- **Static across steps:** all weights (`layer.mixer`, `layer.ffn`, `self.output*`), model cfg, kernel grid/block dims, KV/recur cache *base device pointers* (allocated once in `Cache::new`, `crates/bw24-engine/src/cache.rs:36-67`).
- **Per-step inputs that change:**
  1. The embedding row — `e.htod(&self.embd.gather(...))` at `decode.rs:20` (host gather → fresh device buffer).
  2. The RoPE position — `pos_d = e.htod_i32(&[pos as i32])` at `decode.rs:17`, derived from `cache.pos`.
  3. The KV length `t_kv = kvl.len` at `decode.rs:128`, which grows every step and is passed into `fa_decode` (`decode.rs:137`).

Only (1) and (2) are *values*; (3) is a *shape/loop-bound* and is the hard part (see §4).

---

## 2. DecodeScratch pre-allocation (kill per-step allocation)

Every kernel in `decode_step` currently allocates its output via `e.zeros(n)` = `alloc_zeros::<f32>` (`crates/bw24-engine/src/lib.rs:264`). Per-layer that is `h, x1, z, act, x2` (`decode.rs:23,31,34,47,53`) plus, inside `full_attn_decode`, `q, gate, qn, kn, attn, gsig, attn_g` (`decode.rs:107-145`), and the `fa_decode` internal partials `part_o/part_m/part_l` (`lib.rs:521-523`). Each `alloc_zeros` is a `cuMemAllocAsync`-class op. Under capture these become graph **alloc nodes** that re-reserve memory on every replay — unacceptable.

**Fix:** a `DecodeScratch` struct holding one `CudaSlice<f32>` per intermediate, allocated once (sized by `n_embd`, `n_head*head_dim`, `n_ff`, and the `fa_decode` partials at max splits — see §4), reused every step. Hold it in the `Cache` or a sibling `DecodeGraph` struct so its device pointers are stable across captures.

```rust
struct DecodeScratch {
    h: CudaSlice<f32>, x1: CudaSlice<f32>, z: CudaSlice<f32>,
    act: CudaSlice<f32>, x2: CudaSlice<f32>,
    q: CudaSlice<f32>, gate: CudaSlice<f32>, qn: CudaSlice<f32>, kn: CudaSlice<f32>,
    attn: CudaSlice<f32>, gsig: CudaSlice<f32>, attn_g: CudaSlice<f32>,
    part_o: CudaSlice<f32>, part_m: CudaSlice<f32>, part_l: CudaSlice<f32>,
    pos_d: CudaSlice<i32>, embd_row: CudaSlice<f32>, logits: CudaSlice<f32>,
}
```

The matmul-output buffers (`qf,k,v`, `gate,up`, `ffn_down`, `wo`, `self.output`) also currently return freshly-allocated `y` (`matmul` at `lib.rs:342`, `qmatvec` `y = alloc_zeros` at `lib.rs:82`, `fa_decode` partials at `lib.rs:521-523`). These need `*_into` variants too (§3).

---

## 3. `*_into` pointer-stable launcher variants

Every launcher that today *returns* a `CudaSlice` (allocating internally) must gain a sibling that *writes into* a caller-owned buffer, so the captured graph's kernel nodes bind fixed device pointers. Mechanical pattern, per launcher:

- `rms_norm` (`lib.rs:269`) already takes `dst: &mut CudaSlice<f32>` — graph-ready as is.
- `add` (`lib.rs:318`), `mul` (`lib.rs:329`), `silu_mul` (`lib.rs:307`), `sigmoid` (`lib.rs:619`) — already `dst`/`y: &mut` — graph-ready.
- `copy_into` (`lib.rs:66`) — already in-place into the KV cache; graph-ready (its `dst` pointer is the resident `kvl.k`).
- **Needs an `_into` variant:** `qmatvec`/`matmul` (`lib.rs:80-90, 342`) — return `y`; add `matmul_into(&self, w, x, m, out: &mut CudaSlice<f32>)` that skips the `alloc_zeros` at `lib.rs:82`. Same for the fast-paths (`qmatvec_q8_0_fast` etc.) and `matmul_pre`.
- **`fa_decode`** (`lib.rs:516`) allocates `part_o/part_m/part_l` internally (`lib.rs:521-523`). Add `fa_decode_into(..., part_o, part_m, part_l: &mut ...)` reading scratch sized for `n_splits` at **max** `t_kv` (see §4).
- `q_gate_split`, `rope_neox`, the linear-attn repack kernels — already `&mut`-out; confirm none call `e.zeros` internally before reuse.

The grid/block `LaunchConfig` for each (e.g. `fa_decode` cfg at `lib.rs:526`) must be **frozen at capture time** — for `fa_decode` this is tied to `n_splits`, which depends on `t_kv` (§4). All `b.arg(...)` device-pointer args become graph kernel-node params bound at capture; the scalar args (`&hd, &nh, &t_kv, &scale`) are captured by value into `CUDA_KERNEL_NODE_PARAMS`.

---

## 4. The `cache.pos` / variable-`T_kv` problem and its resolution

This is the central obstacle. Two values move every step:

- `pos` → only feeds `rope_neox` via `pos_d` (`decode.rs:17,119`). **Easy:** keep `pos_d` at a fixed device address in `DecodeScratch` and overwrite it host-side (`memcpy_htod` of one i32) before each `cuGraphLaunch`. RoPE reads the new value; no re-capture needed.
- `t_kv = kvl.len` → feeds `fa_decode` both as the **K/V view length** (`decode.rs:131-132` slices `t_kv * kv_dim`) **and** as the kernel scalar `tkvi` plus `n_splits = ceil(t_kv/256)` which sets `grid_dim.y` (`lib.rs:520, 526`). This is a **shape/launch-config dependency**, which static graphs cannot express. This is the real problem.

### Resolution: static-max-ctx + masking (preferred), not re-capture

**Option A — re-capture on every step (rejected):** re-running `begin_capture`→`end_capture`→`instantiate` for 320 nodes per token costs as much as eager dispatch — it defeats the entire win. Use only as the MoE fallback (§5).

**Option B — static max-context graph + causal mask (chosen for dense decode):**

1. The KV cache is *already* allocated at full `max_ctx * kv_dim` (`cache.rs:51-52`), so the K/V base pointers never move — only `kvl.len` advances. Capture the graph with the K/V **view spanning the full `max_ctx`**, not `t_kv` (replace `e.view(&kvl.k, t_kv*kv_dim)` at `decode.rs:131` with the full-capacity view).
2. Fix `n_splits` at capture to `ceil(max_ctx/256)` so `grid_dim.y` (`lib.rs:526`) is constant; `part_o/m/l` scratch is sized for that max.
3. Pass `t_kv` (the live length) into the kernel as a **value read from a small device-resident counter buffer**, not as a host-baked scalar. `fa_decode_f32` must be modified to treat keys `>= t_kv` as masked (`-inf` score, contributing 0 after softmax) — exactly how causal masking works in FlashAttention. The split-K combine (`fa_decode_combine_f32`, `lib.rs:534`) already does log-sum-exp merge, which tolerates empty splits (all-`-inf` → `l=0`, `m=-inf`) provided the combine guards against `0/0`.
4. Per step: host writes the new `pos_d` and bumps the device `t_kv` counter, then `cuGraphLaunch`. `copy_into` (KV append, `decode.rs:125-126`) writes the new token's K/V into the resident cache at offset `kvl.len * kv_dim` — that offset also depends on `kvl.len`, so the **append offset** must likewise be read from the device counter inside a small append kernel, or `copy_into`'s `dst.slice_mut(off..)` must be replaced with a fixed-base + indexed-write kernel.

**Cost of Option B:** `fa_decode` now scans the full `max_ctx` keys (most masked) every step rather than `t_kv` — wasted bandwidth on padding for short contexts. This *eats into the ~18% ceiling* and can make graphs a net loss at small `t_kv`. Mitigation: **bucketed capture** — instantiate a handful of graphs at `t_kv` thresholds (e.g. 512/1024/2048/4096), select the smallest bucket ≥ current `t_kv` per step, and re-capture only when crossing a bucket boundary (amortized ~once per 512 tokens). This is the pragmatic middle ground between A and B.

`ARCHITECTURE.md §3.10` (line 112) endorses the related **capture-once + `cudaGraphExecUpdate`** path: re-capture into a throwaway graph, then `cuGraphExecUpdate(exec, new_graph, ...)` (`cudarc` sys `cuGraphExecUpdate` at `driver/sys/mod.rs:10428`) patches node params in-place *if topology is unchanged*. Topology **is** unchanged across steps under Option B (same nodes, same edges; only scalar `t_kv`/`pos` differ), so `cuGraphExecUpdate` — or the narrower `cuGraphExecKernelNodeSetParams_v2` (`sys/mod.rs:10362`) on just the RoPE and FA nodes — is the lowest-overhead per-step update and avoids even the device-counter complexity.

---

## 5. MoE exclusion (hard, by construction)

MoE decode **cannot** be captured. `hybrid_forward.rs` `moe_ffn` computes router logits on-device then transfers them to host for top-k expert selection (`e.matmul(&m.gate_inp,...)` then `e.dtoh(&logits)` per the router pattern, `crates/bw24-engine/src/hybrid_forward.rs:199-230`), and `stage_expert` issues a data-dependent `memcpy_htod` of only the routed experts (`lib.rs:101-106`). The set of experts — and therefore which device buffers the matmuls read — changes every token based on activation values. A static graph bakes fixed pointers and a fixed node set; data-dependent routing + host-side top-k violates both.

**Gate boundary:** capture is enabled **only** when the model is dense, or hybrid with all `Ffn::Dense` layers (`decode.rs:37`). The moment any layer is `Ffn::Moe` (`decode.rs:51`), fall back to per-token eager dispatch. `ARCHITECTURE.md §3.10` line 112 states the same verdict: "with offload active or host-routed MoE, this falls back to per-token replay." The `cudaGraphNodeTypeConditional` WHILE-megaloop is explicitly *not* pursued for host-routed MoE.

---

## 6. cudarc 0.19.8 graph API (verified) + raw-sys gaps

The project pins `cudarc = "0.19"` with `driver` + `dynamic-loading` (`crates/bw24-engine/Cargo.toml:9`); resolved source is `cudarc-0.19.8`. Safe wrappers exist for the core flow:

| Step | cudarc 0.19.8 API | Location |
|---|---|---|
| Get raw stream | `CudaStream::cu_stream() -> sys::CUstream` | `src/driver/safe/core.rs:732` (Engine exposes `stream()` → `&Arc<CudaStream>` at `lib.rs:52`) |
| Begin capture | `unsafe stream::begin_capture(cu_stream, mode)` | `src/driver/result.rs:788` |
| End capture | `unsafe stream::end_capture(cu_stream) -> CUgraph` | `result.rs:798` |
| Query capture | `unsafe stream::is_capturing(...) -> CUstreamCaptureStatus` | `result.rs` (`is_capturing`) |
| Instantiate | `unsafe graph::instantiate(graph, flags) -> CUgraphExec` | `result.rs:1494` |
| Replay | `unsafe graph::launch(graph_exec, cu_stream)` | `result.rs` (`graph::launch`, wraps `cuGraphLaunch`) |
| Upload (warm) | `unsafe graph::upload(graph_exec, cu_stream)` | `result.rs` (`graph::upload`) |
| Destroy | `unsafe graph::exec_destroy / graph::destroy` | `result.rs:1507, 1514` |

Capture mode enum: `CUstreamCaptureMode_enum { GLOBAL=0, THREAD_LOCAL=1, RELAXED=2 }` (`src/driver/sys/mod.rs:4730-4734`). Use **`CU_STREAM_CAPTURE_MODE_RELAXED`** — `ARCHITECTURE.md §3.10` warns Global mode hangs CUTLASS grouped GEMM. Instantiate flag: `CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2` (`sys/mod.rs:3411-3416`) to pre-upload nodes at instantiate time.

**Raw-sys calls needed (no safe wrapper in 0.19.8):** the per-step *update* path. cudarc exposes `graph::instantiate/launch/destroy` but **no** safe wrapper for exec-update. Call the sys bindings directly (all present, dynamic-loaded):
- `sys::cuGraphExecUpdate(exec, new_graph, &mut err_node, &mut result)` — `sys/mod.rs:10428`; or `cuGraphExecUpdate_v2` — `sys/mod.rs:10445`.
- `sys::cuGraphExecKernelNodeSetParams_v2(exec, node, &params)` — `sys/mod.rs:10362` — to patch only the RoPE/FA scalar args, avoiding a full re-capture.
- Node handles must be retained at capture (the safe `end_capture` returns only the `CUgraph`; walk it via `cuGraphGetNodes`/store handles, or re-capture into a fresh graph and feed it to `cuGraphExecUpdate`).

No high-level `CudaGraph` builder exists in cudarc 0.19.8 — all six lifecycle calls are `unsafe`. Wrap them in a bw24-local `DecodeGraph { exec: CUgraphExec, nodes: Vec<CUgraphNode>, scratch: DecodeScratch }` RAII type that calls `exec_destroy`/`destroy` on `Drop`.

---

## 7. The `BW24_GRAPH` gate

Follow the existing env-flag convention — `BW24_NOFA` (`decode.rs:134`, `hybrid_forward.rs:102`), `BW24_FAST` (`lib.rs:349`), `BW24_IQ_FAST` (`lib.rs:~366`):

- `BW24_GRAPH` **unset** (default) → current eager path; zero behavior change, safe.
- `BW24_GRAPH=1` → in `generate` (`decode.rs:68`), after priming the prompt, check eligibility: dense-only (no `Ffn::Moe`), no offload/spilling active. If eligible, build `DecodeScratch` + capture once (or per `t_kv` bucket per §4) and replay via `cuGraphLaunch` in the generation loop (`decode.rs:78-82`); else log a one-line "graph disabled: MoE/offload" and fall back to `decode_step`.
- Pair with an internal `decode_step_into(&self, e, token, cache, scratch)` that drives the `*_into` launchers and contains zero `e.zeros`/`alloc` calls, so the same function body is used both for the capture pass and (under fallback) eager replay — guaranteeing capture and eager produce identical kernel sequences.

Validation gate: capture must reproduce the eager argmax bit-exactly (the project's standard — cf. MoE+EDGE-1 argmax=1178 == llama.cpp, commits `02af8fc`/`8a66118`). Add a graph-vs-eager argmax diff to the decode validation before trusting `BW24_GRAPH`.

---

## 8. Implementation order (smallest-risk first)

1. **`DecodeScratch` + `*_into` launchers** (`matmul_into`, `fa_decode_into`, plus reuse of existing `&mut`-out kernels) and a `decode_step_into` that allocates nothing. Verify argmax-identical to `decode_step` *without* any graph — this is independently useful (cuts per-step alloc churn) and de-risks capture.
2. **Static-max-ctx + masked `fa_decode_f32`** (or bucketed capture) to fix the `t_kv` shape dependency — the only kernel change required.
3. **`DecodeGraph` capture/replay** wrapping the unsafe cudarc calls, RELAXED mode, behind `BW24_GRAPH`.
4. **Per-step update** via `cuGraphExecKernelNodeSetParams_v2` on RoPE/FA nodes for `pos`/`t_kv`, or `cuGraphExecUpdate` per bucket boundary.

Steps 1-2 are where the engineering lives; step 3 is ~80 lines of unsafe glue. Keep expectations at the **~18% wall-clock ceiling** (GPU-bound), with net gain likely lower once max-ctx masking overhead is paid — measure before/after at multiple `t_kv` to confirm graphs are not a regression at short contexts.

---

### Key citations
- Decode anatomy / per-step inputs: `crates/bw24-engine/src/decode.rs:12-64` (alloc churn `:23,31,34,47,53`; `pos_d :17`; embd `:20`; KV append `:123-128`; `fa_decode :137`)
- KV/recur cache (max_ctx pre-alloc): `crates/bw24-engine/src/cache.rs:36-67` (`:51-52`)
- Launchers: `lib.rs` — `zeros :264`, `matmul :342`, `qmatvec :80`, `fa_decode :516` (internal partials `:521-523`, cfg `:526`), in-place kernels `rms_norm :269 / add :318 / mul :329 / silu_mul :307 / sigmoid :619`, `copy_into :66`, `stream() :52`
- MoE host-routing (capture-incompatible): `crates/bw24-engine/src/hybrid_forward.rs:199-230`; `lib.rs:101-106`
- cudarc 0.19.8 graph API: `result.rs:788/798/1494/1507/1514` (+ `graph::launch`, `is_capturing`); raw sys `cuGraphExecUpdate sys/mod.rs:10428`, `cuGraphExecKernelNodeSetParams_v2 :10362`; capture mode enum `sys/mod.rs:4730`; instantiate flags `:3411`; `CudaStream::cu_stream() core.rs:732`
- Design verdict + RELAXED + MoE fallback: `ARCHITECTURE.md §3.10` (line 112)
- ~18% ceiling / GPU-bound 52.9 tok/s: `research/benchmarks.md:49,75`
- Gate convention: `BW24_NOFA` `decode.rs:134`, `BW24_FAST` `lib.rs:349`
