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

## 4. VERDICT (2026-08-07, local 5090, q9)

### 4.1 H-A is DEAD at the engine tick — the named class does not exist in the shipped code

`iso-gap-probe` (new bin): X solo (B=1, batched body, `b1fast` pinned OFF via the gate2 seam)
vs X co-resident (B=2..8), per-step full-vocab logits bit-compare, greedy. Eight shapes, zero
bit diffs (raw/ probe-*.log):

| arm | dx / dys | steps | verdict |
|---|---|---|---|
| control-same-rung | 300 / 310 | 96 | PASS bit-identical |
| straddle (X crosses 512 rung mid-run) | 480 / 800 | 96 | PASS |
| straddle-reverse | 800 / 480 | 96 | PASS |
| deep-control | 800 / 810 | 96 | PASS |
| straddle long-horizon | 400 / 800 | 300 | PASS |
| B=4 mixed rungs | 480 / 800,2100,300 | 96 | PASS |
| B=8 three-rung herd | 480 / 300..3000 | 64 | PASS |
| auto (rig-scanned boundary 513) | 481 / 801 | 96 | PASS |

Canary (wrong token into X's co-resident feed at step 1): caught, ndiff=248320 (full vocab).
Why it holds: `batch_layer_ctx` (decode_batch.rs:891-896) is per-session-correct — every row
either shares ONE `fa_split_keys` rung (the seqs kernel then derives each z's split partition
from its OWN `pos_seq[z]+1`, flash_attn.cu:7871 ONE-PARTITION law) or ALL rows take the
per-seq eager loop at their own t_kv. The issue-#10 row-axis fix pattern was already applied
to the seqs axis at its birth (a98f51b1). No fix needed; the property was UNGATED, not broken.

### 4.2 The serve receipt reproduced and attributed — the carrier is the solo<->batched
### program flip at the CO-RESIDENCE boundary, not depth

serve-ab.py / serve-ab2.py (one boot per arm, spec OFF, greedy 768 tok, q9 thinking-stream
comparator per the spec-gate vacuous-pass trap; raw/serveab-*):

| arm | env | shape | vs O1 (solo default) |
|---|---|---|---|
| O1R | default | solo repeat | byte-identical (deterministic) |
| O2 | default | Y first, X joins | DIVERGES at byte 659 |
| O3S | B1FAST=0 GS=0 | solo | diverges at 659 (= the config gap itself) |
| O3L | B1FAST=0 GS=0 | Y first, X joins | == O3S byte-identical; also == O2 |
| O4a/O4b | default | X first, Y at 2.0s | diverges at 1248 / 1361 (jitter) |
| O5 | default | X first, Y at 6.0s (X done) | byte-identical |

Kill shots: O2 == O3S == O3L. With the program family pinned, a staggered-depth co-resident
moves ZERO bytes (the engine probe's serve echo), and the loaded default stream equals the
pinned stream byte-for-byte — the flip accounts for the ENTIRE solo-vs-loaded divergence.
The receipt's moving byte (1347/2379) is arrival-tick jitter of the flip boundary
(reproduced: 1248 vs 1361 at fixed 2.0s delay). The docs' second suspect (batched-linear
tier changes with B) is also innocent at serving widths (O2==O3L across B 1<->2 under load).

### 4.3 Fix scope (the honest one)

The mission's fix principle — selection keyed on the session's OWN state — is ALREADY the
shipped batched-path design (4.1). The remaining program-flip is not a per-session-keyable
selection: the m=1 fused trunk and GraphSession replay are structurally solo-only (their
fusion/capture premise is b_n==1), so "one program regardless of co-residents" = run the
batched body always. That is the existing deployment pin `MEMRA_SERVE_B1FAST=0
MEMRA_SERVE_GS=0`, and its cost crosses the 1% stop-and-report bar by an order of magnitude:

**Perf (N=5 rep-adjacent interleave, one boot per run, warm, solo c=1 greedy 512 tok,
spec OFF, raw/perfab-*):** default median 136.56 tok/s (135.1/137.0/136.7/136.6/125.9),
pinned median 124.73 (125.1/125.0/124.7/123.9/124.6) -> pinned/default = **0.913x, -8.7%**
(consistent with the H3 lane's +8.33% q9 receipt). Per the brief: REPORT the tradeoff, do
not ship it. The default stays; the contract prose now states the real scope and the pin.
c>1 is unaffected (the fast path only fires as the batch drains to 1 — FLAGS.md receipt).

### 4.4 What ships instead

1. `isogap`/`isogapc` fast-gate arms (tools/iso-gap-gate.sh): pin the staggered-depth
   within-config isolation ACROSS the rung boundary — the shape every prior gate missed
   (kernel-check seqs pin: one rung; serve gate: equal depth; decode-batch-gate: 20-55-token
   prompts). Straddle placed per-rig (`--auto` scans the SM-keyed ladder: 5090 boundary 513,
   188-SM pod 2049). Registered on the decode/decode_batch map row + flash_attn.cu row.
   Verified live through fast-gate --probes (both PASS; canary catches).
2. docs/SERVING.md isolation contract re-scoped: the "open gap" paragraph's two-mechanism
   attribution replaced with the measured one; deployment pin named with its measured cost.
   docs/TESTING.md documents the new arms.

### 4.5 Gate battery (this tree, local 5090)

- kernel-check: ALL GREEN (raw/kernel-check.log)
- run-gen q9: argmax MATCH prefill==decode 268 (raw/run-gen-q9.log)
- run-spec q9 K=1,2,4,8: SELF-CONSISTENCY PASS x4 (raw/run-spec-q9-k*.log)
- serve-smoke: 0 failed (raw/serve-smoke.log)
- fast-gate --probes isogap,isogapc: PASS + canary-caught (raw/fastgate-isogap.log)
- Shipped-binary invariance: lane adds a probe bin + gate script + docs only; memra-server
  sha256 identical before/after lane commits (32b716a23c00abbb) — no perf surface moved,
  no board update owed.

### 4.6 Known residuals (stated, not hidden)

- The engine probe and gate cover the default flash module (q9 class, hd256 v4). The
  gemma hd512/g-module and fp8-KV batched arms route through the same rung guard but are
  not exercised by the isogap arm (its model is q9); their seqs arm is fp8-excluded by
  `fa_seqs_eligible` so they always take the per-seq loop — lower risk, zero straddle
  surface, but unpinned.
- Two mid-battery OOMs from peer-lane co-residency were recorded with GPU state
  (raw/probe-straddle-oom-gpustate.txt) and re-run clean per the evidence discipline —
  no conclusion rests on a dead run.
