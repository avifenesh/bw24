# CUDA-GRAPH-PLAN — bw24 decode lever `cuda_graph`

Concrete plan for the **Decode lever** closing the **75 -> 126 tok/s** gap.

## 0. Problem statement (ncu evidence)

ncu on decode shows it is **launch / serialization-bound + power-bound**:

- **~50% duty cycle** — the SM idles half the time between the matvecs while ~715
  small serial kernels per token are dispatched one at a time from the host.
- **Power-bound** — 161 W of a 175 W cap; the card is *also* near its power
  ceiling, so we cannot expect launch-removal alone to fully close the gap.
- Decode runs **batch=1, T=1**: every kernel is tiny, so per-launch host overhead
  (~1-5 us kernarg + queue) dominates over the actual GPU work.

The three structural fixes, in dependency order:

1. **GPU-resident argmax + device pos/token counter** — removes the mid-step
   D2H sync at `decode.rs:91` (`e.dtoh(&logits)?`). This sync is a hard
   capture-blocker: a stream that synchronizes mid-flight cannot be captured into
   a single replayable graph. **This is the prerequisite — nothing else lands
   without it.**
2. **Op-fusion of the serial tail** — collapse the element-wise/norm kernels that
   sit *between* the matvecs (add+rmsnorm, gate+sigmoid+mul, scale-into-epilogue),
   cutting both launch count and the duty-cycle gap even before graphs.
3. **CUDA-graph capture/replay** — collapse the remaining ~715 launches/token to a
   single `cudaGraphLaunch` replay, amortizing all host launch overhead.

### Honest ceiling

- **Graph alone: ~18%.** Decode is *also* power+compute bound (161/175 W), so
  removing launch overhead recovers the idle half of the duty cycle but not the
  compute/power half. ncu duty ~50% -> graph closes the launch-idle portion only.
- **Fusion adds more** on top of graph (fewer kernels = less tail = higher duty
  even pre-graph, and a smaller graph to replay).
- **Combined realistic landing: 75 -> ~95 tok/s, not 126.** The last ~95 -> 126
  almost certainly needs **MTP speculative decode on top** (multiple tokens per
  forward pass amortizes the power-bound compute) — out of scope for this lever,
  tracked in MTP-PLAN.
- This is a deliberately *capped* claim: do not market `cuda_graph` as the full
  51-tok/s win.

---

## Phase 1 — Capture prerequisite: GPU-resident argmax + device counters

**Goal:** remove every host sync from the steady-state decode step so the kernel
sequence is a pure device-side DAG.

### 1.1 The barrier we are removing

`decode.rs:39-93` (`decode_step_h`) ends with:

```
let logits = e.matmul(&self.output, &hn, 1)?;   // decode.rs:90  -> logits [n_vocab] device
let host = e.dtoh(&logits)?;                     // decode.rs:91  <-- D2H SYNC (the barrier)
cache.pos += 1;                                  // decode.rs:92
```

The generation loop (`decode.rs:108-114`) then runs host argmax
(`forward.rs:119-122`, linear scan over `[n_vocab]` f32) and re-enters
`decode_step` with the next token. Each step also does an H2D of `pos`
(`pos_d = e.htod_i32(&[pos as i32])?` at `decode.rs:45`). `sampler.rs:5-7`
explicitly documents the GPU-fused sampler as deferred *"only needed once
CUDA-graph removes the D2H barrier"* — this phase is that work.

### 1.2 Device argmax kernel (greedy path only)

New kernel in `cu/kernels.cu` (links into `BW24_ENGINE_FATBIN` via `build.rs`):

```cuda
extern "C" __global__ void argmax_logits_f32_to_u32(
    const float* __restrict__ logits,   // [n_vocab] device
    uint32_t*    __restrict__ token_out, // [1] device, persists across replays
    int n_vocab);
```

- Launch: **grid=(1,1,1), block=(256,1,1)** (single CTA, 8 warps), shared mem
  `8*(4+4)=64 B`. Each thread scans `ceil(n_vocab/256)` elements (Qwen3
  n_vocab=151936 -> ~594/thread), keeps thread-local `(best_val,best_id)`.
- Reduce: warp butterfly via `__shfl_down_sync(0xffffffff, ...)` reusing the
  pattern at `cu/moe_router.cu:53-65` and `cu/kernels.cu` warp_reduce; per-warp
  partials -> shared -> warp-0 final reduce -> thread 0 writes `token_out[0]`.
- **Tie-break = smallest index wins**, matching the host `>` scan at
  `forward.rs:120-122` (strictly-greater keeps the first max). This is the
  bit-identical contract for the gate (§4).

### 1.3 Device position / token counters (resident GPU state)

Add to a per-decode-session state (or `Cache`):

- `gpu_token_d: CudaSlice<u32>` `[1]` — written by the argmax kernel, read as the
  next step's embed index.
- `gpu_pos_d: CudaSlice<i32>` `[1]` — replaces the per-step
  `htod_i32(&[pos])` at `decode.rs:45`. `rope_neox` already consumes a device
  `pos_d` (`decode.rs` rope call site), so it stays device-resident; a tiny
  `inc_i32` kernel bumps it inside the graph after KV append.

Host `cache.pos` stays mirrored locally (cheap counter) for ring-buffer / context
bookkeeping; `gpu_pos_d` is purely the device copy fed to kernels.

### 1.4 Embed-from-device

The embed at `decode.rs:48` currently gathers on host (`self.embd.gather(..)` then
`htod`). For replay, the embed must read `gpu_token_d[0]` on-device: add an
`embed_gather_u32(embd_table, gpu_token_d, x_out, n_embd)` kernel so the token id
never round-trips to host inside steady state. (Warmup/prime can keep the host
path.)

### 1.5 Engine plumbing

Add an Engine method `argmax_token_device(&self, logits, n_vocab) -> CudaSlice<u32>`
following the launch pattern of `moe_router_topk` and the `func` lookup at
`lib.rs:57-60`. No `dtoh` of logits anywhere in steady state — only a single
`[1]` u32 D2H *after the whole graph completes* (and even that can be deferred to
end-of-sequence in the bulk-extract variant).

**Phase-1 exit:** decode runs greedy with zero mid-step host sync; first N tokens
bit-identical to the host argmax reference.

---

## Phase 2 — Op-fusion of the serial tail

**Goal:** shrink the ~715 launches/token (and the graph that captures them) by
fusing the element-wise/norm kernels sandwiched between matvecs. Land these as
independent PRs *before or alongside* capture — each is independently gated on
bit-identical output and a decode tok/s rise, and each shrinks the eventual graph.

Ordered by ROI / simplicity:

### FUSION #1 — add_residual + rms_norm  (first PR, ~1 kernel)

`decode.rs:60` (`e.add(&x,&mixed,&mut x1)`) then `decode.rs:63`
(`e.rms_norm(&x1, post_attn_norm, &mut z)`), and symmetrically the FFN residual
`decode.rs:82` -> next layer's `decode.rs:52` `rms_norm`. Fuse into
`residual_rms_norm_f32(a, b, w, out, ncols, eps)`: one block per row, accumulate
`sum += (a[i]+b[i])^2`, scale, weight-multiply, store. Same launch geometry as
the existing `rms_norm` (grid=(nrows,1,1), block=(256,1,1)). **Saves ~1
launch/layer x n_layer.** Lowest risk, do first.

### FUSION #2 — q_gate_split + sigmoid + mul  (full-attn path)

`decode.rs:177` (`q_gate_split`), `decode.rs:212` (`sigmoid(gate,gsig)`),
`decode.rs:214` (`mul(attn,gsig,attn_g)`). Fuse into
`q_gate_split_sigmoid_mul_f32`: read fused `qf` once, write `q`, write
`gate=sigmoid(...)`, and `attn_g = attn * sigmoid(gate)` in one pass (kernel
shape per `cu/hybrid.cu:216-229`). **Saves ~2 launches/full-attn-layer.**
Full-attn branch only.

### FUSION #3 — scale-into-epilogue (NVFP4 / dp4a matmul)

NVFP4-quantized matvecs currently call a separate `scale_inplace`
(`lib.rs` scale call sites; `scale_f32` kernel does `y[i]*=s`) after the qmatvec
writes f32. Fold the per-tensor macro-scale into the qmatvec **epilogue** —
multiply by `scale` before the store in `qmatvec_nvfp4_dp4a` (tensor-core GEMM
already does this; extend to the dp4a/MMVQ warp-per-row path). **Saves ~3
launches/layer** when NVFP4 is the active quant. Verify the active quant
distribution first — only pays off if NVFP4 weights dominate.

### Stretch (defer until #1-#3 land + gated)

- **silu_mul + add_residual + next rms_norm** (`decode.rs:76,82,52`) — one fused
  FFN-tail kernel; layout-tricky (act is `[n_ff]`, norm wants row-major).
- **quantize_q8_1 inlined into first sibling matmul** (`decode.rs:70-71,169-170,
  241-242`) — fold the q8_1 quantize into the first qmatvec of each sibling group,
  reusing the quantized row from smem. Highest complexity; dedup already removed
  the 4x redundancy, so this only removes the standalone launch.

Each fusion preserves a non-fused fallback path (gated by the `uses_q8_1_fast` /
quant-type checks already in `decode.rs:69`) so non-matching layers still work.

---

## Phase 3 — CUDA-graph capture / replay

**Goal:** collapse the (now-fused, sync-free) steady-state step into one
`cudaGraphLaunch`.

### 3.1 cudarc 0.19 graph surface

`Cargo.toml:10` pins `cudarc 0.19` (driver + sys). The high-level Rust graph
wrappers are minimal, so call the **raw driver API via `cudarc::driver::sys`**:
`cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED)`,
`cuStreamEndCapture -> CUgraph`, `cuGraphInstantiate -> CUgraphExec`,
`cuGraphLaunch(exec, stream)`. Capture on the Engine's `CudaStream` (`lib.rs:51`
already holds streams). Fallback if sys bindings are insufficient: a thin C shim
in `bw24_runtime` linking the CUDA driver, exposing `begin/end/instantiate/launch`.

### 3.2 llama.cpp warmup pattern (reference)

Port the 2-call warmup + relaxed capture from
`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4468-4525`:

1. **Call 1** (`warmup_counter==0`): execute the step inline (no capture).
2. **Call 2** (`warmup_counter==1`): execute inline; confirm properties stable.
3. **Call 3+** (`warmup_counter>=2`): `cuStreamBeginCapture(Relaxed)`, run the
   step ops (embed-from-device -> per-layer loop -> output_norm -> lm_head ->
   `argmax_logits_f32_to_u32` -> `inc_i32` on `gpu_pos_d`), `cuStreamEndCapture`,
   `cuGraphInstantiate`. Subsequent tokens: `cuGraphLaunch(exec)`.

### 3.3 Bucketing by `t_kv`

The captured graph is valid only while kernel launch geometry is constant. The
one quantity that changes each step is the KV length `t_kv` (flash-decode loop
bound, `rope` pos). Strategy:

- **Bucket graphs by `t_kv` range** (e.g. powers-of-two or fixed strides) and
  keep a small `HashMap<bucket, CUgraphExec>`. On crossing a bucket boundary
  (or ring-buffer wrap), re-capture for the new bucket.
- Properties tracked CPU-side before each launch: `n_vocab, n_embd, n_head,
  head_dim, n_layer, t_kv_bucket`. Mismatch -> reset warmup -> re-capture.
- Single-stream, single-token, batch=1 keeps the capture scope minimal (no
  concurrent-kernel hazards inside the graph).

### 3.4 MoE excluded from capture (data-dependent routing)

MoE FFN layers (`decode.rs:79` -> `moe_ffn_il`) route per-token to a
data-dependent expert set; the kernel arguments (which expert weights, the
SLRU residency cache lookups behind `Engine.moe_cache` at `lib.rs:49`) change
every token and can trigger H2D admits. **MoE layers are NOT captured.** Options:

- Capture only the dense + attention sub-graphs around MoE, and run MoE eagerly
  between graph launches (segmented replay), **or**
- For dense models (no MoE) capture the whole step. The dense path
  (`decode.rs:65-78`) is fully static and the primary capture target.

Document this clearly: `cuda_graph` is a full win for dense/hybrid-dense models;
MoE models get the attention/dense-segment subset only.

### 3.5 Steady-state replay loop

Per token after warmup: `cuGraphLaunch(exec)` (entire step, no host work) ->
optional `[1]` u32 D2H of `gpu_token_d` (or defer to end-of-sequence) -> loop.
H2D of `pos` is gone (device counter). The ~715 launches collapse to 1 replay +
1 tiny copy.

---

## 4. GATE (hard, must pass to merge)

1. **Bit-identical argmax.** Device `argmax_logits_f32_to_u32` output token id ==
   host `argmax` (`forward.rs:119-122`) for the first N (>= 256) greedy tokens on
   a fixed prompt. Tie-break (smallest index) verified explicitly. Any mismatch
   blocks the merge.
2. **Decode tok/s rises.** Measured on the existing decode bench (same harness as
   the prefill bench in commit `8d1c0b7`), each phase shows a non-trivial,
   reproducible tok/s increase vs the pre-change baseline. No regression on
   prefill.
3. **Fusion correctness.** Each fused kernel (#1-#3) produces output matching the
   non-fused path within f32 tolerance (and bit-identical for the integer-pathway
   pieces) before it counts toward the launch-reduction metric.
4. **Per-phase gating.** Phase 1 lands and gates alone (argmax identity + no
   slowdown). Each fusion PR gates alone. Graph capture gates on identical
   generated token stream vs eager decode for >= 256 tokens across at least two
   `t_kv` buckets.

---

## 5. File / line reference index

- D2H capture barrier: `crates/bw24-engine/src/decode.rs:91`
  (`let host = e.dtoh(&logits)?;`)
- Per-step H2D pos (replace with device counter): `decode.rs:45`
- Decode step body + embed: `decode.rs:39-93`, embed at `decode.rs:48`
- add+rms_norm fusion sites: `decode.rs:60`, `decode.rs:63`; FFN residual
  `decode.rs:82`; next-layer norm `decode.rs:52`
- q_gate_split / sigmoid / mul fusion sites: `decode.rs:177`, `decode.rs:212`,
  `decode.rs:214`
- silu_mul + quantize_q8_1 stretch sites: `decode.rs:70-71`, `decode.rs:76`,
  `decode.rs:169-170`, `decode.rs:241-242`
- MoE FFN (excluded from capture): `decode.rs:79`; MoE cache `lib.rs:49`
- Generation loop / host argmax call: `decode.rs:108-114`
- Host argmax reference (bit-exact contract): `forward.rs:119-122`
- Sampler D2H-barrier deferral note: `sampler.rs:5-7`
- Engine struct + stream + module-load: `lib.rs:38-60`
- cudarc 0.19 (driver/sys for graph API): `Cargo.toml:10`
- Warp-reduce / argmax kernel reference: `cu/moe_router.cu:53-65`
- Fusion kernel shape reference: `cu/hybrid.cu:216-229`
- llama.cpp warmup + capture reference:
  `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4468-4525`
