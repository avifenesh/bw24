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

## Built (commits d8dad0c8..a50abe62)

- **b2geo35/b2geo35c** (`tools/step35-b2-geometry-gate.sh`, registered in fast-gate): c=2/c=4
  byte-vs-serial PLUS batched evidence (spawn-log `decode chunk cap >= 2` + the engine's
  one-shot `[step35-batch] first B>1` line) — the evidence half is what makes the gate
  non-vacuous under a B=1 pin. Canary = `MEMRA_STEP35_BATCH=0` re-pin must break it.
  c=1 reference pinned to the batched class via `MEMRA_SERVE_B1FAST=0` (the gate2 pin —
  within-config byte identity, not the cross-config FP gap).
- **The arm** (`step35_decode_batch_layers`, decode_batch.rs): batched at m=B for every
  weight-streaming op (attn_norm+q8, wq/wk/wv/attn_gate, q/k norms, ONE rope_neox2 with
  per-row pos + per-layer n_rot/base/ff, attn_head_gate, wo, add_rms_norm, dense FFN via
  clamp-aware ffn_act_lim, MoE via moe_ffn_il_zq8 at t=B); per-session for KV state (append
  + SWA/global view from THAT session's own kvl.len + fa_decode_kvmod — the per-seq fallback
  shape). Stage-scoped `[lo,hi)` from birth; ppn calls it per stage, #87 fences unchanged.
  IQ4_XS numeric class: no mmvq/batched kernel exists, so m=1 AND m=B both ride the
  grid-(out_f,m) dp4a family — per-column the m=1 program (decode-parity by construction).
- **Routing**: unsplit body + ppn body both route step35 to the walk at any B; the generic
  `decode_batch_layers` is unreachable for step35 at every B. B=1 under the door keeps
  `b1_stage_fast` (the m=1 fusion chain).
- **Pin lift**: `chunk_cap_for` step35 -> min(MEMRA_DECODE_BATCH_CAP, 8) (exact16
  structurally refused: MoE). `MEMRA_STEP35_BATCH=0` restores cap 1 + engine Err backstops.
- **Graph promotion fix (found on the walk)**: a solo greedy step35 session with budget >=
  gs_min (384) graph-promoted into `decode_step_dc_cap_masked`, whose full-attn arm REFUSES
  step35 — and the refusal lands on the cache-consumed degrade path = the request DIES.
  step35 is now a named exclusion next to eager_only.
- **decode-batch-gate --plen**: the pp battery's synthetic prompts (20-35 tok) sat INSIDE
  step35's 512 window, so the per-session view-offset arm never fired — the chunkinv35
  vacuous-coverage class again. The battery runs --plen 520.

## Results — ALL GREEN on BOTH PRO 6000 pairs (raw/ box1 + box2)

### The c-scaling table (the number that matters — was 34-FLAT at every c)

Decode aggregate tok/s, DEFAULT batched serve (naked config + spec OFF per #87, drafter
attached, PP-2 dev01, 128 tok/req, N=3 medians per cell, warm steady-state, one flock hold
per box; points JSONL committed):

| c | box1 median (min-max) | box2 median (min-max) | vs 34-flat |
|---|---|---|---|
| 1 | 83.7 (72.8-83.9) | 81.0 (70.8-81.1) | 2.4x |
| 2 | 99.4 (94.7-99.4) | 96.3 (91.8-96.4) | 2.9x |
| 4 | 118.2 (117.5-118.2) | 116.2 (116.2-116.9) | 3.4x |
| 8 | **130.3** (130.1-130.5) | **129.4** (129.4-129.4) | **3.8x** |

(The c=1 gain over the old 34 is NOT this lane's arm — it is the b1_stage_fast eager walk
measured under this battery's server build; the arm's own contribution is the SCALING,
c=8/c=1 = 1.56x where it was 1.00x, and the absolute 129-130 agg at c=8 = 3.8x the pinned
baseline's 34.15. Cross-run vs the 2d2eb676 receipt is clock-drift-caveated per the H100
law; the 34-flat SHAPE vs rising-in-c shape is the claim, and both are same-box receipts.)

### Gate verdicts

| gate | verdict | receipt |
|---|---|---|
| G1 kernel-check | **ALL GREEN, 0 FAIL** (fresh fatbin) | box1/kernel-check-20260808T045135Z.log |
| G2 decode-batch-gate --mode pp, B=1,2,4,8 x2 reps, --plen 520 | **ALL GREEN, 0 failing arms** — split + unsplit@ppncache + b1-stagefast + epilogue, 24 steps x 128896 f32 per row, **0 differing bits at every width** (B=8: 24,748,032 f32 x3 arms) | box1/dbg-pp-step35-20260808T045135Z.log |
| G3 run-gen argmax | **MATCH x2** (prefill==decode 6776; batched-prime==tokenwise 6776) | box1/rungen-20260808T034540Z.log |
| G4 run-spec K=1..8 + drafter | **PASS 8/8**, acceptance **digit-identical** to the pinned baseline (14/18, 15/34..15/136) | box1/runspec-20260808T034540Z.log |
| G5 chunkinv35 | **CHUNK-INVARIANT** (no regress) | box1/chunkinv35-20260808T034540Z.log |
| G5 tickinv35 | FAIL — **PRE-EXISTING**: the fix is lane/tick-seg f01710ca, NOT an ancestor of base a131e8c7 (verified by git ancestry; 1.813e0@6 = the exact pre-fix signature). Resolves at merge time when both lanes land on train | box1/tickinv35-20260808T034540Z.log |
| G6 b2geo35 naked | **PASS on both boxes** — c=2/c=4 byte-identical to c=1, chunk cap 8, batched-walk line present | box1/b2geo35-naked-20260808T051110Z.log, box2/b2geo35-naked-20260808T050301Z.log |
| G6c b2geo35 canary | **CANARY OK on both boxes** — MEMRA_STEP35_BATCH=0 re-pin breaks the evidence assertion (chunk cap 1, no walk line) | box1/b2geo35-canary-20260808T051201Z.log, box2/b2geo35-canary-20260808T050358Z.log |
| S2 serve c=8 byte-vs-serial | **PASS** — 8/8 byte-identical + batched evidence at B=8 | box2/battery-box2-20260808T050301Z.log |

Cards verified 0 MiB on both boxes at battery exit.

### Round-1 traps (receipted, box1/)

1. **Stale fatbin from a seeded target dir**: round 1 seeded `target/` from another lane's
   build cache; the .cu mtimes predated it so nvcc never reran — `kernel
   fa_prefill_qw_db_w_hd128 not in any fatbin` panicked G1/G2 while every OTHER gate
   passed. touch *.cu + rebuild; the binary now greps the kernel string (113 hits).
2. **Bare `wait` in the gate script** also waited on the never-exiting server job — the
   gate hung AFTER producing fully-correct c=2 responses. Fixed: wait on the curl PIDs.
3. **`[prime-batch] failed ... single primes serve`** in the serve logs: expected — step35
   still has no batched prime core (`prime_cache_batch` refuses); prefill concurrency rides
   single primes. Decode batching is untouched by it. Named follow-up, not this lane.

## Ledger

| item | state |
|---|---|
| read conclusions + arm shape | DONE |
| red b2geo35 standing gate | DONE — registered red 9a12b53a, GREEN post-arm on both boxes |
| engine arm (step35_decode_batch_layers) | DONE (c5cd6a35) |
| unsplit + ppn routing | DONE (same commit) |
| chunk_cap_for pin lift (cap 8) | DONE (same commit) |
| graph-promotion step35 exclusion (found on the walk) | DONE (same commit) |
| bit-identity battery B∈{1,2,4,8} over PP-2, plen 520 | **ALL GREEN, 0 differing bits** |
| serve gates (b2geo35 + canary both boxes, c=8 byte-vs-serial) | **ALL GREEN** |
| kernel-check / run-gen / run-spec / chunkinv35 | **ALL GREEN**; tickinv35 pre-existing red (lane/tick-seg fixes it, not an ancestor here) |
| perf c-scaling | **DONE: 34-flat -> 130 agg at c=8 (3.8x), both boxes agree within 1%** |
| FLAGS.md (MEMRA_STEP35_BATCH) | DONE |

Boxes: box1 lane workspace `~/stepbatch-memra` (rsync tree), raw mirrored to raw/box1/.
Box2 lane clone `~/step35-batch` @ 57fc87d3 (own clone — the shared `~/memra` untouched),
raw mirrored to raw/box2/. Both released 0 MiB.
