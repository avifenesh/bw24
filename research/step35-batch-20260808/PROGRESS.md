# lane/step35-batched-decode — the REAL batched step35 decode arm (kill the B=1 pin)

Predecessors (binding, not re-derived): `research/step-sku-20260807/PROGRESS.md` (the b2-geometry
garbage receipt + the fail-closed pin, commits `ca6edb8d`/`a0ba3e36`), `research/step37-p2-20260806/`
(the step35 mixer family + why it is dedicated), `research/pp2-batch-20260806/RESULTS.md` (the
batched PP-N stage split + the #87 fences), `research/gemma4-serve-20260807/` (the eager-only
fail-closed floor shape).

Boxes: box1 `18.195.123.14` (2x PRO 6000, PP-2, flock `/tmp/memra-gpu.lock`, shared with
v072battery + leverb — windows only), box2 `13.59.112.147` (2x PRO 6000, us-east-2, memra @
a131e8c7 in `~/memra`, Step shards landing at `/data/models/step37`) — batteries go to box2.

---

## Read conclusions (write-first, before code)

### Where the B=1 pin sits — three layers, all found

1. **Server**: `worker.rs::chunk_cap_for` (`worker.rs:3902-3913`) returns **1** for any
   `cfg.step35.is_some()` model, not overridable upward by `MEMRA_DECODE_BATCH_CAP`. This is why
   decode aggregate is 34-flat across c=1..8: every session still decodes each tick, but as B=1
   chunks through `decode_step_batch_ppn`'s `b1_stage_fast` walk (`decode_layers_eager`) —
   round-robin across the trunk, one full 45-layer weight stream per session per tick.
2. **Engine, ppn body**: `decode_batch.rs:774-779` — `decode_step_batch_ppn` refuses
   `step35.is_some() && !b1_stage_fast` with a named Err (the fix for the b2ab garbage: before it,
   B>1 walked `decode_batch_layers`' generic Full arm — global n_head=96 over-reading wq on the 12
   full-attn layers, 128-dim rope on all 45 layers, no SWA window, no head-wise gate).
3. **Engine, unsplit body**: `decode_batch.rs:569-573` — the same refusal as an assert (B=1
   already routed to the shared eager trunk at :504).

### What the batched arm needs geometrically (per layer il, per session bi)

From `step35_decode_attn` (`hybrid_forward.rs:7060-7137`), the eager T=1 arm being twinned:

| mechanism | value | batched consequence |
|---|---|---|
| n_head | 64 full (il%4==0) / 96 SWA | wq/wo/attn_gate widths are per-LAYER; batch across B at fixed il is fine — the geometry varies per layer, NOT per session |
| n_head_kv | 8 uniform | — |
| head_dim | 128 | q [B,nh*128] |
| rope | n_rot 64 full / 128 SWA; base 5e6 full / 1e4 SWA; `rope_freqs` factors on FULL only | `rope_neox2` already takes per-call (hd, n_dims, base, ff) and a **per-row pos array** (`pos: &CudaSlice<i32>`, one entry per token) — batching B rows through it at fixed il is the exact prefill call shape. NO new rope kernel. |
| SWA window | 512, per-layer 3:1 | per-SESSION view offset: `off_bi = if swa && len_bi > win { len_bi - win }`, `t_kv_bi = min(len_bi, win)` — sessions differ in len, so offsets/extents are PER ROW. The qwen seqs kernel (`fa_decode_batch_seqs_v4`) reads per-cache base pointers from the ptr table + per-row `pos_d` but has **no per-row view-offset parameter** — it cannot express the SWA window. |
| head gate | separate `attn_gate.weight [n_embd, n_head_l]`, sigmoid per (token,head), pre-wo, input = post-attn_norm hidden | `attn_head_gate(a, g, dst, hd, nh, t)` already takes t rows — batches at t=B directly. Gate projection = one more matmul at m=B off the same q8_1 activation. |
| MoE FFN | sigmoid router (scale 3.0, norm, +bias), 288 experts, 3 leading dense, per-layer SwiGLU clamp on 43/44 | `moe_ffn_il_zq8(e, m, z, zq8, t, il)` is the SAME call the batched body already makes for MoE at t=B (`decode_batch.rs:1228`). The sigmoid-router host path handles t>1 (routing at t rows via `moe_route_sigmoid`); dev/pairs arms are correctly denied by predicate (`sigmoid_router().is_none()` gate). At t=B<16 it rides `moe_ffn_sequential_zq8` — host routing + per-token expert dispatch. WORKS TODAY, not fast, correct. |
| dense layers 0-2 + clamp | `ffn_act_lim` handles per-layer clamp | the batched body's Dense arm uses plain `silu_mul` — must route step35 dense through the clamp-aware path (only layers 43/44 have live limits, and they are MoE+shexp; leading dense layers 0-2 have NO clamp on this artifact — verify via `swiglu_clamped_at`) |

### The blocking kernel gap, precisely

The ONLY structural gap between qwen's batched tick and a step35 one is **phase B attention**:
`fa_decode_batch_seqs_v4` takes one shared `(head_dim, n_head, n_head_kv, t_kv_max, sp0)` and
per-cache base pointers — no per-row view offset, no per-layer n_head grouping. Everything else
(batched projections via `matmul_pre` at m=B, rms_norm at B*nh rows, rope_neox2 with per-row pos,
append via the per-seq `append_kv_quantized_view` loop, attn_head_gate at t=B, MoE via
`moe_ffn_il_zq8`) composes from existing, already-gated pieces.

And the per-seq FALLBACK arm already in the batched body (`decode_batch.rs:1087-1107`) shows the
honest shape: per-session `fa_decode_kvmod` over that session's own (offset) view — exactly what
`step35_decode_attn` does at T=1. B calls to `fa_decode_kvmod` per attn layer instead of one
z-batched launch. Decode is weight-stream-bound; the batched projections (wq/wk/wv/gate/wo at m=B,
MoE expert reads amortized across B rows where sessions share experts — they don't share, but the
FFN dense/router weights DO stream once) carry the win. The per-seq fa_decode loop costs launch
overhead, not weight bandwidth (KV is per-session state, read once per session either way).

### Chosen arm shape

**A step35-specific batched layer walk** (`step35_decode_batch_layers`), NOT a generalization of
`decode_batch_layers`:

- per layer il: ONE `rms_norm` at B rows + ONE `quantize_q8_1` -> batched wq/wk/wv/gate
  projections at m=B (`matmul_pre` — B=2..8 rides the b2/b4/b8 verify-tier mmvq: per-row
  bit-identical to m=1, the isolation contract's kernel class; IQ4_XS trunk = q8_1-fast via
  `iq_fast_enabled`, dp4a at m>1 — see "numeric class" below) -> q/k RMSNorm at B*nh rows ->
  ONE `rope_neox2` (B rows, per-row pos, per-layer n_rot/base/ff) -> per-session loop:
  {`append_kv_quantized_view` row bi, SWA/global view arithmetic from THAT session's own
  `kvl.len` (the iso-gap law: each session's own t_kv), `fa_decode_kvmod`} ->
  ONE `attn_head_gate` at t=B -> `matmul(wo, m=B)` -> add_rms_norm -> FFN
  (`moe_ffn_il_zq8` at t=B for MoE; clamp-aware dense arm).
- The walk is `(lo, hi)`-scoped from birth so `decode_step_batch_ppn` can call it per stage —
  PP-2 wiring is a call-site change (the pp2-batch seam lesson), and the #87
  `fence_stages_behind` + per-stage engine/pos_d/ptr-less structure carries over.
- No pointer table needed (the per-seq loop indexes `caches[bi]` host-side like the fallback arm);
  no BatchLayerCtx dependency — simpler, and avoids uploading state addresses that the per-seq
  loop doesn't consume.

**Numeric class**: per-session per-(token,row) EXACTNESS TARGET is bit-identity to the same
session's B=1 serial run **in the batched-body numeric class**. Note the b1_stage_fast walk today
is `decode_layers_eager` (the m=1 FUSION chain) — the batched arm at B=1 will sit on the batched
side of the accepted decode-config FP gap, same as qwen (`b1_fast` exists for exactly this
reason). So the serve-level geometry gate compares batched c>1 text vs c=1 text (which rides
b1_fast) — these must agree at the TEXT level (greedy argmax), and the engine-level gate compares
B>1 rows vs the same-session B=1 batched-body run bit-for-bit. Both gates below.

**IQ4_XS at m>1**: `mmvq_supports(IQ4_XS)` is false, so `matmul_pre` at m=2..8 falls to the
grid.y=m dp4a tail — each column is the exact m=1 dp4a program. But wait: m=1 decode on IQ4_XS
rides `qmatvec_iq4_XS_fast`/dp4a too (no mmvq kernel), so per-row parity holds by the same
grid.y=m argument the decode-parity law documents. To verify on-box, not assumed.

### The gates (in build order)

1. **RED FIRST — `b2geo35` standing gate**: extend `b2-geometry-ab.sh` into
   `tools/step35-b2-geometry-gate.sh` — c=2 and c=4 batched greedy text must equal the c=1
   serial reference byte-for-byte, PLUS the server log must show decode chunks >1 formed
   (else the gate is vacuously green under the pin). Register in `tools/fast-gate/models.tsv`
   like tickinv35. Today it must be RED-by-construction: with the pin, chunks stay at B=1, the
   "batched evidence" assertion fails -> red.
2. Engine arm + unit shape; `decode-batch-gate`-style bit-identity B∈{2,4,8} vs per-session
   serial batched-body runs, on box2.
3. Lift `chunk_cap_for` step35 pin -> exact-tier cap (8; step35 is IQ4_XS+MoE so exact16 is
   refused by the MoE predicate — cap 8).
4. PP-2: route the ppn body's step35 case through the new walk per stage.
5. Batteries: b2geo35 GREEN, kernel-check, run-gen (batched-prime+tokenwise MATCH), run-spec
   K=1..8 with drafter, chunkinv35/tickinv35 no-regress, serve c=1..8 byte-vs-serial.
6. Perf: c=1/2/4/8 N>=3 vs the 34-flat baseline, one flock hold.

### Session-composition hazard named upfront

`fa_split_keys`/rung logic doesn't apply (per-seq fa_decode fallback shape has no shared rung).
The per-session view arithmetic reads ONLY `caches[bi].kv[il].len` — no cross-session term — so
isolation holds by construction. `pos_d` is per-row (each session's own pos), matching what
`rope_neox2` already consumes at t=B.

## Ledger

| item | state |
|---|---|
| read conclusions + arm shape | DONE (this file) |
| red b2geo35 standing gate | open — next commit |
| engine arm (step35_decode_batch_layers) | open |
| unsplit + ppn routing | open |
| chunk_cap_for pin lift | open |
| bit-identity gate B∈{2,4,8} | open |
| serve gates (b2geo35 GREEN, c=1..8) | open |
| kernel-check / run-gen / run-spec / chunkinv / tickinv | open |
| perf c-scaling vs 34-flat | open |
