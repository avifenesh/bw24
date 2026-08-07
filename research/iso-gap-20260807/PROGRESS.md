# lane/iso-gap — Task #91: serve isolation at STAGGERED depths (LADDER-RUNG STRADDLE class)

## 1. The defect, restated from its receipts (do-not-code-before-this bar: MET)

### 1.1 The measured divergence (the receipt)

`research/spec-gate-20260806/RESULTS.md` §2.2 + `logs/exact/exactness.json` (arm `REF_LOAD`),
2026-08-06, local 5090, q9 (Qwen3.5-9B NVFP4+MTP + owntrim drafter), greedy, 768 tokens:

- **Spec OFF, no gate-lane code involved.** A 768-token greedy request run SOLO (`REF`) vs the
  same request sharing batched decode with 4 background sessions **staggered to different
  depths** (`REF_LOAD`: target fires first, 0.5 s head start, fillers arrive with a much larger
  prompt) diverged at byte **1347** (~token 331); a second run moved the divergence to byte
  **2379**. `exactness.json`:
  `"baseline_batchshape_solo_vs_loaded": "DIVERGES at byte 1347 — batch-vs-solo decode is NOT
  bit-identical (pre-existing)"`. The moving byte is itself the proof the loaded config is
  nondeterministic (arrival timing + batch composition).
- The equal-depth serve gate (16 prompts, 96 max_tokens, all sessions arriving TOGETHER) passes —
  the gate's blind spot is exactly the staggered-depth shape.

### 1.2 The docs' own account (commit 446c5203, docs/SERVING.md §The isolation contract)

The contract was re-scoped to *"byte-identical at equal depth, gated to 96 tokens; long
staggered-depth batches are an open gap"*, attributing the gap to two documented laws:

1. `fa_decode_batch_seqs_v4` "carries a single `split_keys` for sessions at different depths
   (the LADDER-RUNG STRADDLE law `fa_decode_rows` documents for the row axis)";
2. "the batched-linear tier selection changes with B".

### 1.3 The prior fix in the same class (the row axis — issue #10)

Commit `4eda65d6` (2026-07-13, g7e-proven): batched spec VERIFY picked ONE split size from the
batch-max t_kv while eager decode picked `fa_split_keys(t_kv)` per token — a ladder rung inside
the batch changed a row's combine FP order vs its eager twin; greedy ties flipped at depth
(g7e rung at 2048; `MEMRA_FA_SPLIT=64` pin → PASS proved the mechanism). Fix:
`fa_decode_rows` (lib.rs ~9861) groups consecutive rows by their OWN ladder value, one launch
per group. Second instance: commit `a3211c7d` (2026-08-02) — graph segment fingerprints missing
the ladder rung replayed a captured `split_keys` partition against eager's other side; ladder
value joined the segment tuple. Third instance (sibling class): `c809181d` (chunkfix,
2026-08-07) — SWA prefill arm keyed on the chunk's t_kv instead of the request's seq_end.

**The class**: any kernel/split selection keyed on a batch-AGGREGATE quantity (batch max,
row 0, whole-batch predicate) instead of the session's OWN state makes one session's FP
program a function of its batchmates.

### 1.4 What the code says today (the selector map, HEAD = 006aca75)

The serve tick (`worker.rs`) → `decode_step_batch` (`decode_batch.rs:369`) →
`batch_layer_ctx` (`decode_batch.rs:813`) computes per step:

- `sp0 = fa_split_keys(t_kvs[0], n_head_kv)` (`decode_batch.rs:891`) — **keyed on row 0**;
- `seqs_fa = ON && all rows fa_seqs_eligible && all rows' fa_split_keys == sp0`
  (`decode_batch.rs:892-896`) — a whole-batch predicate. When true, ONE
  `fa_decode_batch_seqs_v4` launch (z = session) with the shared `sp0`; the kernel derives
  each z's `ns_eff` from its OWN `T_kv = pos_seq[z]+1` (`flash_attn.cu:7871`, ONE-PARTITION
  law). When false (a rung crossing INSIDE the batch), ALL rows fall to the per-seq loop
  (`decode_batch.rs:1052+`), each row running `fa_decode_kvmod` at its own t_kv.
- The per-seq fallback is DOCUMENTED as executing "the exact program its isolated run would"
  and kernel-check pins seqs-vs-loop bit identity (`kernel_check.rs:3305+`) — but only at
  depths `[96,128,257,511]` / `[200;8]`, i.e. **all within one rung** (sp8 on this rig's
  ladder). The pin never crosses a rung, and gate2 of decode-batch-gate uses prompts of
  length 20..55 + 32 steps — **also never crosses a rung**. Equal-rung is the shared blind
  spot of every existing gate.
- The 5090 ladder (82 SMs, q9 n_head_kv=4 → the `n_head_kv <= 4` branch, lib.rs:487):
  `t_kv <= 512 → sp8`, `<= 16384 → sp64`, else sp128. **The live rung boundary is 512.**
- Solo sessions additionally ride `decode_step_b1_fast` (m=1 fused trunk, H3 2026-08-05) —
  a DELIBERATE cross-config FP gap documented at `decode_batch.rs:487+`; the token contract
  covers it. `MEMRA_SERVE_B1FAST=0` is its seam. Attribution of any serve-level divergence
  must pin this OFF to isolate the depth-coresidence axis from the solo-fused axis.

### 1.5 The open question the repro must answer

The guard at `decode_batch.rs:896` LOOKS per-session-correct (falls back to per-seq eager on a
straddle). If every bit-identity pin it leans on were honest across the rung boundary, staggered
depths could not move a session's bytes at fixed B. The receipt says they do (at serve level,
where B also fluctuates). So the repro must separate:

- **H-A (rung straddle at fixed B)**: at B=2, X's logits differ between {X alone at B=1,
  batched body} and {X with Y at a straddling depth} — an engine-tick-level break, the
  mission's named class. Mechanism candidates: a pin hole in seqs-vs-loop across the rung, an
  aggregate key not yet mapped, or the fallback loop not matching the seqs program at the
  crossing step.
- **H-B (B-composition / b1fast axis)**: the serve divergence is carried by B fluctuating
  (1↔2..5) with arrival timing, i.e. the documented cross-config gap, and fixed-B staggered
  depth is actually clean. Then the fix scope is the serve-level selection keying (what program
  a session gets must not flip mid-stream with co-residency), and the honest report may be a
  measured tradeoff instead of a free fix.

## 2. Repro plan (in flight)

1. **Engine-level probe** (`iso-gap-probe`, new bin, gate2's shape with STAGGERED depths):
   prime X and Y to chosen depths, decode X for N steps at B=2 {X,Y} vs B=1 {X} (b1fast pinned
   OFF both sides → same batched body, isolates co-residence), bit-compare X's logits per step.
   Arms:
   - control-same-rung: X=300, Y=310 (both sp8 all steps) → expect bit-identical;
   - straddle: X=480, Y=800 (X sp8, Y sp64 → batch straddles rung 512 for ~32 steps, then
     merges) → the class under test;
   - straddle-reverse: X=800, Y=480 (X's own rung sp64 both ways);
   - deep-control: X=800, Y=810 (both sp64) → expect bit-identical.
   Model: q9 GGUF (`/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`),
   local 5090 (geometry: 33 layers, interval-4 full attn, n_head=16, n_head_kv=4, hd=256).
2. **Seam bisect on any FAIL**: `MEMRA_FA_SPLIT=64` (one global rung — the issue-#10 proof
   move; if FAIL→PASS the rung is the mechanism), `MEMRA_BATCH_FA=0` (force per-seq loop
   always), `MEMRA_BATCH_APPEND=0`.
3. **Serve-level A/B** (the mission's O1 vs O2): target request solo vs target + one
   background session HELD at a straddling depth, greedy, fixed prompts, `MEMRA_SERVE_B1FAST=0`
   arm + default arm — attributes the serve receipt between H-A and H-B.
4. Fix per the chunkfix family: per-session selection keying (group-by-own-rung inside the
   batched FA, the `fa_decode_rows` precedent) if H-A; measured tradeoff report if >1% on the
   shipped default.
5. Gates: new isolation gate sweeping the straddle boundary (register as fast-gate arm per
   chunkinv precedent), kernel-check ALL GREEN, run-gen argmax MATCH, run-spec K=1..8 PASS,
   serve-smoke; N=5 interleaved perf per research/benchmarks.md.

## 3. Rig state at start

Local 5090 (24.5 GB): owner's llama-server (332 MiB) + hermes python (394 MiB) resident, GPU
idle (0% util). q9 artifact + drafter present under /data. Build green at HEAD 006aca75.
