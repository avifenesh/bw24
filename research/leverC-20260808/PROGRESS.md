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
