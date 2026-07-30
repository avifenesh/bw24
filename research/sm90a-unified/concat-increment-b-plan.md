# Concat scheduler increment (b): continuation primes (task #21 plan, 2026-07-30)

Increment (a) shipped v0.48.0: phase (d) batches FRESH same-lane dark prefills via
prime_cache_batch (1.22x rig5090 / 1.086x H100 harvest wall, no interactive regression).
Increment (b): batch CONTINUATION primes (sessions with pos>0 — chat follow-ups on
parked caches; the dominant judge-lane shape).

## Why it's blocked today (measured facts, hybrid_forward.rs:867+)

- `prime_cache_batch` asserts `c.pos == 0` and `kvl.len == 0` per layer.
- Per-seq positions are built as `0..t` (`pos_ds`); a continuation needs `pos0..pos0+t`.
- `attn_pre_vl8` (the varlen split/QK-norm/RoPE/append kernel, struct `crate::AttnPreVl`
  with fields qf/kf/vf/q/gate/qn/kn/kc/vc/t/pad) derives RoPE position from the row
  index — needs a per-seq `pos0` field (use the `pad` slot or widen the struct; kernel
  in kernels.cu, grep attn_pre_vl8).
- The varlen FA twin assumes T_kv == T (fresh causal). Continuation needs T_kv = len+T
  with the causal mask offset by len (attend to the full cached prefix). This is the
  "carried-pos varlen twin": same kernel, extra per-seq `len0` param feeding both the
  KV range and the mask offset.
- Append goes to kv offset len (kc/vc write cursor) — attn_pre_vl8 append already
  writes at kvl.len? (it sets kvl.len += t after) — VERIFY: the kernel writes rows at
  position derived from row index; carried case must write at len0 + row.
- GDN/SSM (qwen hybrids): chunked scan needs per-seq INITIAL STATE (carried from cache)
  — batch scan API takes zero state today. v1 scope: gate increment (b) to
  attention-only models (gemma-4 family: Full/Windowed mixers), leave hybrids on the
  single-chunk path (`model.has_ssm()`-style check at the worker candidate filter).
- Worker side (worker.rs phase (d), the increment-(a) block): drop the
  `c.pos == 0`/`fed.is_empty()` filter for eligible models; candidates carry
  (pos0 = cache.pos); prime_cache_batch grows `pos0s: &[usize]` (or reads pos from
  each cache — cleaner: caches already carry pos; keep signature, read `c.pos`).

## v1 plan (REVISED after reading attn_pre_vl8 — 2026-07-30 session 3)

attn_pre_vl8 = 4 sub-kernels launched from lib.rs:7546 (q_gate_split_vl, attn_rms_vl,
attn_rope_vl, append_kv_vl; struct AttnPreVl at lib.rs:616 has a spare `pad: i32`).
Three simplifications kill most of the planned kernel work:

1. **append needs NO kernel change**: pass kc/vc PRE-OFFSET by len0*tok_bytes from the
   host (fresh case len0=0 is today's behavior, bit-identical).
2. **RoPE pos0 rides the `pad` field**: attn_rope_vl reads seq.pad as the position
   offset (pos = pad + row). One-line kernel change, fresh passes pad=0 (identical).
3. **FA stays PER-SEQ in v1**: after the batched pre+append, run the existing
   single-seq continuation attention (fa_prefill_view: q rows T against t_kv=cache
   len — already shipped for the T=K verify path) per sequence, instead of a new
   T_kv>T batched vl twin. The batching win is the PROJECTIONS at m=sum_T (GEMM
   arms dominate prime cost); per-seq FA at judge shapes is small. The batched
   carried-mask FA twin becomes increment (b2) if measurements demand it.

Kernel facts (flash_attn.cu, verified 2026-07-30):
- `attn_rope_vl` (line ~4058): `float theta = (float)tok * powf(...)` — change to
  `(float)(tok + sq.pad)`; fresh passes pad=0 -> bit-identical. C-side struct is
  `attnpre_t` inside `attnprevl_t v.s[blockIdx.z]` — confirm its `pad` field name
  matches the Rust AttnPreVl layout (lib.rs:616: ...t: i32, pad: i32 — last field).
- `append_kv_vl` (line ~4083): writes `sq.kc + t*k_tok_bytes` with `t = tt` ("fresh:
  t0 == 0" comment) — host passes kc/vc pre-offset by len0*tok_bytes, NO kernel change.
- `q_gate_split_vl` / `attn_rms_vl`: position-free (row-local) — no change.
- The batched-FA call after the pre (fa3_on branch / mma favl) is the fresh-only piece:
  v1 branches to per-seq `fa_prefill_view` (lib.rs ~7600, T=K verify path: q rows T
  vs t_kv=cache len quantized views) when any pos0 > 0.

Steps:
1. hybrid_forward prime_cache_batch: accept pos0 = cache.pos per seq (relax fresh
   asserts; capacity len0+t <= max_ctx; pos_ds = pos0..pos0+t), kc/vc offset, pad=pos0.
   Reference the single-seq prime_cache continuation branch per layer (windowed-view
   handling comes from fa_prefill_view's caller there).
2. attn_rope_vl: pos = seq.pad + row.
3. Per-seq FA branch when any pos0 > 0 (or always, if bit-equal on fresh — it is not:
   fresh uses the batched vl FA; keep the fresh path byte-identical, branch on carried).
4. Worker phase (d): second candidate pass for CONTINUATION sessions (pos>0, fed
   non-empty prefix already in cache, prefill_queue = the new suffix) on attention-only
   models; same lane/model/budget rules as (a).
5. GDN/SSM hybrids excluded v1 (batch scan lacks initial-state API).
6. Gates: greedy text identity per slot vs sequential single-seq continuation runs
   (the (a) protocol), serve-smoke, battery; judge-profile A/B both rigs.

## Status
- [x] hybrid_forward pos0 plumbing (carried asserts relaxed, pos_ds from pos0, fresh fast paths gated !carried)
- [x] attn_rope_vl pos0 via pad_ (DONE 2026-07-30, builds clean; fresh bit-identical)
- [x] FA vl carried: NOT NEEDED v1 (per-seq continuation cores; b2 if measured)
- [x] worker filter: pos==fed.len() invariant, gemma4 carried excluded
- [x] parity + A/B: prime-batch-gate --carried ALL GREEN; serving 9/9 exact cross-binary, 1.20x r2 wall x3
