# Lever C - grouped Step35 prefill experts

Branch: `lane/cx-moe-grouped-prefill`
Train base: `13f5ddb8`
Pipeline dependency: local `lane/cx-pipeline-prime` tip `62fac3c0`

## Increment 0 - read-first mechanism contract

### The cost being removed

The Step35 prefill path reaches the sequential MoE fallback because its sigmoid router denies
the pairs and device-router arms. The anatomy receipt in
`research/pp-prefill-20260807/PROGRESS.md` measured that fallback class at 28% of prime time:
about 835K expert-kernel launches per pp4096 prime, with expert work dispatched one token at a
time even though routing is already known for the whole prefill chunk.

This lane will group routed rows by expert inside the existing Step35-legal host-routing family.
It will not relax the sigmoid-router predicates and will not route Step35 through pairs, device
routing, grouped-decode fusion, or any uniform-slab fallback that bypasses expert metadata.

### Chosen grouped mechanism

For each prefill MoE layer:

1. Compute router logits with the same `router_gemv` decision used by the sequential path.
   The cuBLASLt `m=t` router is forbidden because its logits and selected experts are
   chunk-size-dependent.
2. Apply the existing host sigmoid-router oracle, including correction bias, top-k ordering,
   scaling, and renormalization.
3. Bucket routed token rows by expert while retaining each route's original top-k slot.
4. Gather one `[m_e, n_embd]` activation matrix per expert and run the existing row-parallel q8
   expert kernels at `m=m_e`: `quantize_q8_1`, gate/up `qmatvec_expert_q8`,
   clamp-aware `ffn_act_lim`, a second `quantize_q8_1`, then down
   `qmatvec_expert_q8`.
5. Scatter each result back to its original token and top-k slot, then reduce slots in original
   top-k order with the same fused multiply-add chain as sequential `axpy_into`.

The local resident expert slab is legal input because it is only a storage source for the same
per-expert q8 kernels. Mixed-layout banks continue through metadata-aware cache/staging and never
enter uniform-only fused kernels.

### Why this targets exact arithmetic

- Router logits, route selection, weights, and route order come from the same m-invariant helper
  and host sigmoid code as the sequential oracle.
- `quantize_q8_1` and `qmatvec_expert_q8` index token rows independently. Raising `m` changes the
  launch grid, not any row's reduction program or quantized bytes.
- Step35's layer-specific SwiGLU clamp and macro scales stay on `ffn_act_lim`.
- Slot scatter preserves the router's original top-k position. Slot reduction therefore performs
  the same ordered `fmaf(weight, expert_output, accumulator)` sequence as the token loop.
- The shared-expert branch is unchanged.

The older `MEMRA_MOE_GROUPED` prototype is not the default implementation for this lane: its
expert arithmetic uses f32 dequantized `qmatvec_view`, while served Step35 uses q8 expert math.
That documented numeric-class difference is unnecessary here because the existing q8 expert
kernel already supports `m>1`.

### Dispatch seam and chunk uniformity

The default promotion is restricted to the served prime walker, not decode/spec:

- every Step35 prefill chunk with more than one row uses the grouped q8 arm;
- `MEMRA_MOE_GROUPED=0` restores the sequential prefill oracle;
- explicit `MEMRA_MOE_GROUPED=1` retains the existing opt-in meaning outside prefill;
- no chunk-local size threshold chooses between arithmetic classes.

Thus all normal chunks of one request use the same dispatch class. The final single-row tail, if
present, may use the sequential path because grouping cannot change a one-row launch and both
paths are required to be bit-identical.

### Planned gates

| Gate | Required verdict |
|---|---|
| grouped-vs-sequential MoE oracle | bit-identical model-backed output |
| `kernel-check` | ALL GREEN |
| `chunkinv35` / `tickinv35` | PASS, with canaries retaining teeth |
| `ppsplit` | unsplit, serial split, and pipelined split bit-identical and live |
| `run-gen` PP-2 | argmax MATCH |
| `run-spec` K=1..8 | 8/8 PASS at pinned acceptance |
| performance | pp512/2048/4096, N=5 interleaved against the 417.6 tok/s pipeline baseline |

Raw build, gate, and performance logs will be retained under `research/leverC-20260808/raw/`.
Every reported median will state N and thermal regime. The supplied steering path
`~/.lanectl/inbox/cx-leverC.md` was absent at lane start; no alternate Lever C steering file was
present under `~/.lanectl`.

## Increment 1 - grouped q8 prefill implementation

The local pipeline-prime tip `62fac3c0` was merged first, preserving its measured 417.6 tok/s
pp4096 baseline and current `ppsplit`/chunk/tick gate surfaces.

The grouped implementation now has two exact storage cases behind one host-routed dispatch:

- **Resident uniform Step35 bank:** the host sigmoid oracle produces the original pair ids,
  expert ids, and weights. An expert CSR feeds the existing `moe_pairs_matvec_q8_dec` kernel,
  which decodes one expert weight group and applies it to all routed rows in that expert segment.
  Gate/up outputs remain indexed by original pair id; `ffn_act_lim` applies the layer's Step35
  clamp; down uses the same CSR; `moe_pairs_scatter` performs the original slot-ordered FMA chain.
  This reuses only the pairs arm's arithmetic kernel, not its softmax router or plain-SiLU
  dispatch policy.
- **Spill, remote-slab, or mixed-layout bank:** the existing A2 expert groups remain
  metadata-aware. Uniform q8-supported experts run `quantize_q8_1` plus
  `qmatvec_expert_q8` at `m=m_e` from a local slab or live SLRU slot. Direct no-cache/frozen
  staging and mixed layouts retain the sequential oracle's f32 class, with each expert's
  authoritative `qtype`, `row_bytes`, `len`, and source.

`moe_router_logits` is now shared by sequential and grouped dispatch, so both use
`router_gemv` under the exact-prefill policy. All prefill entry points call the prefill wrapper;
decode and speculative verification keep the original wrapper and dispatch class.
`MEMRA_MOE_GROUPED=0` is the live rollback seam, while an explicit nonzero value preserves the
old opt-in behavior for other callers.

Local proof:

| Check | Verdict |
|---|---|
| `cargo check -p memra-engine` | PASS, CUDA 13.1, auto-detected sm_120a |
| worktree scope | source + flag catalog + this increment only |

Target-rig exactness and performance remain pending; no winner claim is made from the local
compile.

## Increment 2 - first Box2 oracle is red

Box2 ran commit `a1e04b43` on the model-backed Step35 artifact. The first grouped layer took the
intended resident arm, but `MEMRA_MOE_GATE=1` rejected its output:

| Check | Verdict |
|---|---|
| grouped arm engagement | `il=3 t=19 dispatch=resident-q8` |
| grouped vs sequential bytes | **FAIL**, 55,032 / 77,824 f32 elements differ, max diff 1.358427e-5 |
| model-backed `kernel-check` | `ALL GREEN` |

The remaining acceptance battery was stopped after preserving the failure and kernel receipts.
No invariance or performance claim is made from this build.

The failed exactness assumption was the whole-layer arithmetic class. The sequential resident
Step35 path at an unclamped layer runs the fused per-token q8 pair
`moe_gate_up_silu8_q8` plus `moe_down8_fma_q8`. The first Lever C arm instead ran
`moe_pairs_matvec_q8_dec` for each projection, a separate activation kernel, a separate down
matvec, and `moe_pairs_scatter`. Matching row-dot and slot-reduction descriptions were not enough
to make those two complete kernel chains byte-identical.

The correction is to batch the existing fused per-token program over the prefill token axis:
host sigmoid routing still supplies the exact `sel` and `w`, while the established
`moe_gate_up_silu8_dev_q8_rows` / `moe_down8_fma_dev_q8_rows_g` family executes all tokens in one
launch pair. Those kernels are rows twins of the sequential fused program, not the denied
softmax router. Step35's clamped layers cannot enter the plain-SiLU rows arm and retain the
clamp-aware grouped q8 fallback.
