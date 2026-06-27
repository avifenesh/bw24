> **2026-06-27 ncu UPDATE — this doc's bandwidth premise is WRONG at the kernel level. Read this first.**
> ncu now runs (sudo -n works; `RmProfilingAdminOnly=1` but sudo lifts it). Profiled `qmatvec_nvfp4_mmvq` directly:
> - `dram__throughput.avg.pct_of_peak` = **37%** (NOT 52%, NOT bandwidth-bound — the 52% was a whole-decode tok/s-implied average, an upper bound on a kernel that ISN'T the DRAM wall).
> - binding stall = `long_scoreboard` **9.46** + `lg_throttle` 3.82; `issue_active` 44%; `inst_issued` 0.44/cyc; `warps_active` 88%.
> - L2 sector hit-rate 4.6% (weights stream once — expected).
> **Verdict: LATENCY-bound (low memory-level parallelism), not bandwidth/occupancy.** 48 resident warps can't hide the ~470cy weight-load latency because each warp's inner block-loop is a tight load→table→dp4a dependent chain. **P1 (aligned load) SHIPPED: 80.6→82.9 (+2.3), as predicted.** P2 (cooperative layout) **DISPROVEN by A/B**: bw24 already has BOTH layouts (dp4a=4-warp-coop, mmvq=warp-per-row); both land 83.1±0.1 — decomposition is NOT the lever. **Real lever = MLP: unroll block-loop + hoist weight loads (multiple LDG in flight/warp). No 1.58x cap — that was a bandwidth-wall artifact.** Levers below are superseded by this.

> **GRIND LOG (2026-06-27, ncu-driven, connected-pieces view). Decode 9B-NVFP4 best config = BW24_FAST+MMVQ+FA_VEC.**
> 69.7 (start) -> 80.6 (memset-kill, prior) -> **82.9 (P1 aligned-load) -> 86.4 (int4 activation, ALL 10 matvec kernels).** All argmax-bit-stable (142/40492), kernel_check rel unchanged.
> **The bound MOVES as you hit it (this is the connected-pieces dynamic — a rising stall after a win is the NEXT step, not a failure):**
> - start: long_scoreboard 9.46 + lg_throttle 3.82 (LSU queue full from 8 scalar int activation loads/block).
> - after int4 activation (2x int4=128-bit load): lg_throttle 3.82->**0.28** (queue freed), long_scoreboard 9.46->**15.2** (now the SOLE bound = pure weight-load latency), issue_active 44->50%, DRAM 37->43%. tok/s 82.9->86.
> **TRIED & DISPROVEN (don't repeat — measured):**
> - 4-way batched g-block unroll (load 4 blocks up front, then compute): REGRESSED 82.9->78.0. Batching loads while lg_throttle still 3.82 made long_scoreboard WORSE (20.9) — one long serial wait at batch edge.
> - 1-ahead software pipeline (issue next-block loads, compute current): cut long_scoreboard 15.2->8.78, issue_active 50->55% (mechanism WORKED) but full Blk struct pushed regs 40->43, crossing the 42-reg cap -> occ 88->75%, NET -1% (85.0). Reg-trimmed (pipeline only qs words): regs back to 40, occ 89%, long_scoreboard 12.68, but still 84.9 < 86.4 — compiler already schedules the simple unrolled loop's loads well; manual pipeline only adds overhead. **Lesson: the simple int4 loop is at the compiler's scheduling optimum for THIS reg budget.**
> **NEXT (connected, not yet tried): long_scoreboard 12-15 = weight-load latency that 48 warps can't fully hide because every token re-reads ALL weights from HBM. The structural lever is cutting per-token weight TRAFFIC (the thing latency is proportional to), which connects prefill+decode: e.g. the q6_K lm_head (1.4ms, ~53% SOL, worst single kernel) + low-bit weight residency. Also CUDA-graph (different bound: ~18.7% launch gap, stacks multiplicatively). MTP/spec changes the regime entirely (amortizes weight reads across accepted tokens) — the path PAST raw-kernel SOL.**

# DECODE-GEMV SOL PLAN — closing the NVFP4 decode bandwidth gap

**Author:** lead CUDA architect (synthesis of 4 adversarially-verified levers)
**Date:** 2026-06-27
**Target HW:** RTX 5090 Laptop, sm_120 (GB203), 82 SMs, 847 GB/s achieved mem BW (94.6% of 896 peak). Warp-level `mma.sync` + `dp4a` + Blackwell FP4/FP8 block-scale only. No wgmma, no tcgen05.

---

## 0. The number we are chasing (do not re-derive — measured this session)

| Engine | decode tok/s | implied weight BW | % of 847 GB/s SOL |
|---|---|---|---|
| bw24 9B-NVFP4 (BW24_MMVQ + BW24_FA_VEC) | **80.6** | 442 GB/s | **52%** |
| llama.cpp 9B-NVFP4 | 126 | 691 GB/s | 82% |
| **bw24 target** | **~120** | ~824 GB/s | **82%** |

Total headroom = 82/52 = **1.58×** (442 → 691 GB/s). Decode is **memory-bandwidth-bound** (weights read ~once/token, arithmetic intensity ~2). The 52→82 gap **is** the weight-read-bandwidth gap.

**HARD CONSTRAINT FOR THIS DOC: no lever or stack of levers may exceed 1.58×.** We do not sum best-cases. Where two levers raise the *same* bound, they do not add (see §5).

---

## 1. Ranked levers (by VERIFIED / corrected SOL gain, in dependency order)

The four analyzed dimensions, after adversarial verification, collapse to **ONE real code lever plus the structural lever that the verification repeatedly named as "the rest of the gap."** Three of the four "dimensions" are the **same edit** described from different angles; they DO NOT STACK.

| Rank | Lever | Verdict | Corrected SOL gain | Status |
|---|---|---|---|---|
| **P1** | NVFP4 weight-qs byte-load → aligned 32-bit (`get_int_b4`) on BOTH kernels | INFLATED→real | **0–5% SOL, expect ~1–3%** (≈ +1 to +4 tok/s) | SHIP — cheap, bit-safe, but do NOT bank double digits |
| **P2** | Restructure mmvq → llama-style 4-warp-per-row cooperative block-striding (activation/L2 reuse) | not yet analyzed as its own lever; named by 3 of 4 refutations as "the larger real lever toward 82%" | **UNQUANTIFIED — this is where the bulk of the 30pp must come from, if anywhere.** Requires a measurement spike before commit | INVESTIGATE FIRST (spike), then decide |
| ~~—~~ | ~~Occupancy / nwarps / rows-per-block / `__launch_bounds__` retune~~ | INFLATED→**~0%** | **0%** (kernel already at hardware-max 12 CTA/SM = 48 warps = 100% occ; regs=40 ≤ 42 cap) | DROP (do as a no-cost safety pin only) |
| ~~—~~ | ~~Quant-fusion (fuse `quantize_q8_1` into matvec prologue)~~ | **FANTASY** | **~0%, net-negative risk** | DROP |

### Why P1 is the only thing to *land first*

All three of {load-vectorization, occupancy-launch, llama-recipe-diff} converged on the **identical one-line edit**: replace the manual byte-OR weight load with an aligned word load. They are **not three additive 5–12% wins**; they are three readings of the same SASS fact. The honest combined gain of that single edit is **0–5% SOL** (best estimate ~1–3%), because:

- **SASS confirms the edit is real** (verified against the shipped fatbin): `(int)qss[0]|((int)qss[1]<<8)|...` compiles to **18× `LDG.E.U8.CONSTANT`** per loop iter; nvcc does NOT fold it. `get_int_b4` → one `LDG.E.CONSTANT` 32-bit. So it is **not** a compiler no-op.
- **BUT it does not move HBM bytes.** The 16 qs bytes are contiguous → same 32B L2 sectors either way. Implied weight read is already 5.48 GB/tok vs 5.06 GB theoretical = ratio **1.08** → the kernel is already moving near-minimum bytes. The waste is LSU **issue throughput / L1 sector-request replays**, not DRAM bytes.
- **The inner loop is ALU-bound, not LSU-bound** (the code's own comment qmatvec.cu:976 records the codebook was moved to `__byte_perm` *"because this loop was ALU-bound (19% of BW ceiling)."*). Cutting LSU instructions frees the MIO pipe but the ALU/FMA pipe stays critical and re-consumes the freed slots. This is the textbook diminishing-return case.

So P1 is **worth shipping** (free, bit-identical, fixes a genuine inefficiency, makes the kernel match llama's load path), but it **does not close the gap** on its own.

### Dropped levers — explicit reasons

- **Occupancy / launch-geometry retune → DROP.** Verified via `cuobjdump -res-usage`: REG 40, SHARED 0 → 128 thr × 40 reg = 5120 reg/CTA → 65536/5120 = **12 CTA/SM**, thread cap 1536/128 = **12 CTA/SM** = 48 warps = **100% theoretical occupancy**. 48 resident warps far exceeds the ~16–24 needed to hide ~470-cycle DRAM latency. `__launch_bounds__(128,12)` is a no-op (40 ≤ the 42-reg cap). Changing `ROWS_PER_BLOCK` or `nwarps` will NOT help a fully latency-hidden kernel and risks the prior tile-redesign regression pattern (Task #20). **Do `__launch_bounds__(128,12)` ONLY as a zero-cost safety pin against future reg bloat; expect 0%.**
- **Quant-fusion → DROP (FANTASY).** Activation bytes are **0.0013–0.049%** of weight bytes per matvec and stay **L2-resident** (16 KB activation in a tens-of-MB L2). The 9% GPU-time of `quantize_q8_1` is launch/occupancy latency on a microscopic under-occupied kernel, **not bandwidth** — that is CUDA-graph territory (≈1.2× cap, explicitly out of scope per brief). Worse, fusing into the matvec prologue would FORCE every sibling matvec (q/k/v share `h` 3-way decode.rs:228-229; wqkv/gate/beta/alpha 4-way decode.rs:300-302; gate+up decode.rs:69-70/129-130, spec.rs:94) to **re-quantize the identical row** — exactly the 13.5%→9% regression that shared-quant already fixed (lib.rs:304-307). Strict pessimization. **Do not pursue.**

---

## 2. PASS-BY-PASS implementation

### PASS P1 — Aligned 32-bit NVFP4 weight load (BOTH kernels)

**Goal:** replace `8× LDG.E.U8` per sub-block with `2× LDG.E.32` (`get_int_b4`), matching llama's `vec_dot_nvfp4_q8_1`. Bit-identical on little-endian.

**Alignment proof (verified, governs the safe form):**
`row_bytes = (in_f/64)*36`. NVFP4 asserts `in_f % 64 == 0` (lib.rs:438). Block stride 36 = 4×9 → 4-aligned; `qs = b+4` → 4-aligned; `qss = qs + s*8` → 4-aligned. So `*(const int*)qss` is **always legal**.
**`int2`/`LDG.E.64` is NOT safe in general:** `row_bytes` is a multiple of 36, which is 4-aligned but only 8-aligned when `in_f/64` is even (i.e. `in_f % 128 == 0`). Odd `(in_f/64)` rows start at a 4-but-not-8 boundary → an `int2` load faults on alternate rows. **Use two 32-bit loads (`get_int_b4`), NOT `int2`.** (This is the safety note all four refutations flagged; the original "exact_changes" that proposed `int2` were wrong on this point.)

**Step 1 — add the helper** (mirror the existing `get_int_b2` at qmatvec.cu:467-471):

```cuda
// Mirrors llama.cpp get_int_b4 (vecdotq.cuh:27-29). Safe for any 4-byte-aligned source.
__device__ __forceinline__ int get_int_b4(const void* p) {
    return *(const int*)p;   // single LDG.E.32; NVFP4 qss is provably 4-aligned (row_bytes%4==0, qs=b+4, qss=+s*8)
}
```

**Step 2 — edit `qmatvec_nvfp4_mmvq`** (BW24_MMVQ fast path, the measured-80.6 config), qmatvec.cu **:650-651**:

```cuda
// FROM:
int q4a = (int)qss[0] | ((int)qss[1] << 8) | ((int)qss[2] << 16) | ((int)qss[3] << 24);
int q4b = (int)qss[4] | ((int)qss[5] << 8) | ((int)qss[6] << 16) | ((int)qss[7] << 24);
// TO:
int q4a = get_int_b4(qss);
int q4b = get_int_b4(qss + 4);
```

**Step 3 — edit `qmatvec_nvfp4_dp4a`** (DEFAULT path when BW24_MMVQ unset; the build gate can select either, so BOTH must change), qmatvec.cu **:979-980** — identical replacement.

**Step 4 — fix the false comment** at qmatvec.cu **:977** (`"36-byte blocks => no 4-align guarantee"`). It is FALSE: `row_bytes=(in_f/64)*36` is a multiple of 4, `qs=b+4`, `qss=+s*8` — all 4-aligned. Replace with a note that `get_int_b4` (4-byte) is used and that `int2`/64-bit is NOT safe (rows are only 4-aligned unless `in_f%128==0`).

**Leave the scale bytes alone:** `ue4m3_to_f32_d(d_bytes[s])` (qmatvec.cu:660 / :990) are 1-byte E4M3 scales (4 B / 36 B block = 11% of payload, 2 byte-loads/lane). Correctly byte-typed; vectorizing them is not worth the unpack complexity. Win is entirely in the 32-byte qs payload.

**GATE P1 (must pass before P2):**
1. **Correctness — bit/argmax gate (mandatory, change is mathematically identity):**
   - `cargo run --release --bin kernel_check -- <9B-NVFP4 GGUF>` — the NVFP4 cases must still pass:
     - `qmatvec_nvfp4_fast` vs GEMM raw (kernel_check.rs:437-438)
     - mmvq raw vs Stage-A (kernel_check.rs:483-536, the `nvfp4` mmvq case)
   - Because the edit is bit-identical, expect **rel unchanged** (no loosening). If `rel` moves at all, the edit is wrong — STOP. (Reference cautionary case: the Q6_K ql-offset bug ran with rel 0.34 while still "working"; weight-load rewrites silently corrupt — re-pass the full argmax 268/220 end-to-end gate.)
2. **SASS gate:** rebuild fatbin (build.rs targets compute_120a), then `cuobjdump -sass <out>/qmatvec.fatbin` and confirm `qmatvec_nvfp4_mmvq` and `qmatvec_nvfp4_dp4a` now show `LDG.E.CONSTANT` (32-bit) on the qs stream instead of runs of `LDG.E.U8.CONSTANT`. Count should drop from 18 byte-loads → ~9 word-loads per iter.
3. **Perf gate:** run the existing decode tok/s bench (the `run-gen` decode arm, current 80.6) AND capture `ncu --metrics sm__inst_executed_pipe_lsu_per_cycle,dram__throughput.avg.pct_of_peak_sustained_elapsed` before/after. **Expect: LSU instructions ↓ ~4× on the qs stream; achieved DRAM throughput ↑ a little or flat; tok/s +1 to +4.**

**Honest projected cumulative after P1:** **80.6 → ~82–84 tok/s** (~53–54% SOL). Bank **~+2 tok/s**, not more.

> **DECISION RULE:** If P1's `ncu` shows DRAM throughput essentially unchanged (likely) and tok/s ≤ +3, that **confirms the kernel is not LSU-bound** and the remaining 28pp is structural/L2 — proceed to P2's spike. If DRAM throughput jumps materially (unlikely given the 1.08 byte-ratio), the LSU was the bottleneck and the gap may be closer than expected — re-measure ceiling.

---

### PASS P2 — (INVESTIGATE) llama-style 4-warp-per-row cooperative decomposition

**This is the only candidate big enough to move the needle toward 82%, and it is the LEAST verified.** Three of four refutations independently named it as "the larger real lever toward 82% … which this [P1] fix does not touch at all," and all three declined to quantify it. So P2 is **a measurement spike first, an implementation second** — do not commit a tile rewrite on faith (Task #20 GEMM tile redesign already regressed once on a wrong-occupancy premise).

**The structural difference (verified from source):**
- **bw24 mmvq** (qmatvec.cu:624): `block=(32, ROWS=4)` — **4 INDEPENDENT rows/CTA, one warp per row.** Each warp strides the *entire* `in_f/32` block list alone (`for g=lane; g<nsb; g+=32`). Reduction = pure `warp_reduce_sum` (5 `shfl`, no smem, no `__syncthreads`) — a **cheaper tail** than llama.
- **llama GENERIC** (mmvq.cu, sm_120 takes the GENERIC table not Turing): `block=(32,4)` = 4 warps but `rows_per_cuda_block=1` — **all 4 warps cooperate on ONE row** via `blocks_per_iter = vdr*nwarps*warp_size/qi` strided loop, then a smem cross-warp reduce. The 4 warps share the **same activation `x`** → higher L1/L2 activation-hit-rate and a different (possibly better) weight-stream spatial-locality pattern across the row.

**The hypothesis to test:** bw24's 4-independent-rows layout has worse activation/weight spatial locality than llama's 1-row/4-warp cooperative striding, and *that* — not the byte-load — is the bulk of 52→82. **This is unproven.** The activation is tiny and L2-resident, so the mechanism (if real) is weight-stream L2-sector hit-rate / DRAM-row-buffer locality, not activation bytes.

**SPIKE (do this BEFORE writing the kernel):**
1. `ncu` the current `qmatvec_nvfp4_mmvq` on a representative decode matvec (e.g. ffn_down 14336→4096): capture `dram__throughput.avg.pct_of_peak`, `lts__t_sector_hit_rate.pct` (L2 hit rate), `l1tex__throughput`, and `sm__warps_active`. 
2. `ncu` llama's `mul_mat_vec_q` on the *same* tensor shape. Diff the L2 sector hit-rate and DRAM throughput. **If llama's L2 hit-rate / DRAM-%-of-peak is materially higher at the same byte volume, the cooperative layout is the lever and P2 is worth building.** If they are equal, P2 is NOT the answer and we are short of 82% (see §3).

**IF the spike confirms it — implementation sketch (only then):**
- Add a `qmatvec_nvfp4_coop` variant: `block=(32, NWARPS=4)`, `grid=(out_f, m)`, all 4 warps stride one row via `blocks_per_iter`, then smem cross-warp reduce (port llama mmvq.cu:594-end reduction). Keep the `get_int_b4` load from P1.
- This **replaces** bw24's warp-reduce tail with a smem+`__syncthreads` tail (slightly more expensive per-row), betting the L2-locality win exceeds the reduction cost.
- Gate behind a new env (e.g. `BW24_MMVQ_COOP`) so it can be A/B'd against the current mmvq without flipping the default.

**GATE P2:**
1. Correctness: same kernel_check NVFP4 mmvq + argmax 268/220 gate, rel unchanged (the math is identical, only decomposition/reduction order changes — float reduction order may shift rel by float-noise; accept ≤ the existing int8-act tolerance, not a regression beyond it).
2. Perf: tok/s + `ncu` DRAM-%-of-peak. **Commit only if it raises achieved DRAM throughput and tok/s by more than the spike-predicted margin AND beats the P1-only config.**

**Honest projected cumulative after P2:** **UNKNOWN — refuse to project a number until the spike runs.** Plausible band IF the L2-locality hypothesis holds: **~84 → ~100–115 tok/s** (~54% → ~65–75% SOL). This is the band that contains essentially all of the remaining headroom; everything else is rounding. **If the spike shows equal L2 behavior, P2 yields ~0 and the realistic ceiling is ~84 tok/s (see §3).**

---

## 3. REALISTIC cumulative ceiling — does the stack reach 120 tok/s / 82%?

**Honest answer: P1 alone does NOT. P1 lands ~82–84 tok/s (~53–54% SOL). The full 52→82 (1.58×) target depends ENTIRELY on P2, which is unverified.**

Two scenarios, stated plainly:

**Scenario A — P2 spike confirms the cooperative-layout L2-locality win (optimistic, plausible):**
`80.6 → ~83 (P1) → ~100–115 (P2)` = **~65–75% SOL.** This **approaches but likely still falls short of 82% / 120 tok/s by ~5–17 tok/s.** The residual would be the structural L2/DRAM-row-buffer efficiency that even llama's layout only partially captures, plus the out-of-scope ~1.2× CUDA-graph launch cap (decode is 81.3% GPU-active → ~18.7% is launch gap, NOT this workflow's topic).

**Scenario B — P2 spike shows equal L2 behavior (pessimistic):**
`80.6 → ~83 (P1)` and STOP. **~54% SOL. We are ~37 tok/s / ~28pp short of target.** In this case the byte-load was a red herring at scale and the gap is something neither P1 nor P2 touches.

**What ELSE would be needed to actually reach 82% (if P1+P2 fall short) — honestly out of this workflow's stated scope:**
1. **CUDA-graph capture of the whole decode step** (~1.2× cap, ~18.7% launch gap). Explicitly out of scope per brief, but it is the single largest *named, measured* remaining slice and it would stack multiplicatively with weight-BW gains (it raises a *different* bound — launch latency, not DRAM throughput).
2. **`quantize_q8_1` (9%) + `rms_norm` (7.5%) under-occupancy** — microscopic kernels under-filling 82 SMs. Fixing their grid sizing (NOT fusing them) is launch/occupancy, recoverable only inside the CUDA-graph lever.
3. **q6_K `lm_head`** single kernel at ~53% SOL (worst-tuned, 1.4 ms) — a separate per-kernel tuning task, orthogonal to the NVFP4 matvecs.

**Do NOT promise 120 tok/s from P1+P2.** The defensible claim is: **P1 ships ~+2 tok/s for free; P2 is a measurement-gated bet that *could* recover most of the remaining gap but plausibly lands at ~65–75% SOL, not 82%.** Reaching 82% almost certainly also requires the CUDA-graph lever (out of scope) to multiply on top.

---

## 4. Overlap / non-additivity (critical — do not double-count)

- **load-vectorization == occupancy-launch's weight-load half == llama-recipe-diff's weight-load half == ALL P1.** These are **the same one-line edit**. Their raw projections (3–8%, 8–18%, 5–15%) are **three readings of one fact** and MUST NOT be summed. Combined verified value = **0–5% SOL, once.**
- **occupancy/nwarps retune adds 0** on top of P1 (kernel already at 100% occupancy; verified).
- **P1 and P2 raise DIFFERENT bounds and can partially stack:** P1 = LSU/L1-instruction-issue efficiency; P2 = L2-sector hit-rate / weight-stream DRAM locality. But both are on the **same matvec critical path**, so P2's win is measured *relative to the P1-improved kernel*, not the 80.6 baseline — gate P2 against the P1 config, not the original.
- **CUDA-graph (out of scope) raises a THIRD bound** (launch latency) and is the only lever that stacks *multiplicatively* and cleanly with the weight-BW levers — which is why §3 names it as the realistic path to 82% if P1+P2 alone fall short.
- **Quant-fusion overlaps NOTHING positively** — it would *regress* the shared-quant bound. Dropped.

---

## 5. Execution order summary (for the implementation workflow)

1. **P1 (SHIP):** add `get_int_b4`; edit qmatvec.cu:650-651 and :979-980; fix comment at :977; rebuild; pass kernel_check NVFP4 (rel unchanged) + SASS gate (`LDG.E.U8`→`LDG.E.32`) + tok/s. **Bank ~+2 tok/s (→ ~82–84).** Use `get_int_b4` (two 32-bit), NOT `int2`/`LDG.E.64` — rows are 4-aligned, not 8-aligned unless `in_f%128==0`.
2. **Safety pin (free, 0%):** add `__launch_bounds__(128,12)` to the `_mmvq` kernels; verify `nvcc -Xptxas -v` keeps regs ≤ 42. Insurance against future reg bloat dropping occupancy; expect no measurable change.
3. **P2 (SPIKE → maybe BUILD):** `ncu`-diff bw24-mmvq vs llama mul_mat_vec_q L2 hit-rate + DRAM-%-of-peak on the SAME tensor. **Only if llama's L2/DRAM efficiency is materially higher**, build `qmatvec_nvfp4_coop` (4-warp-per-row, smem reduce, behind `BW24_MMVQ_COOP`), gate against the P1 config. Project a number ONLY after the spike.
4. **Re-baseline the ceiling after P2's spike.** If short of 82%, the next levers are CUDA-graph (~1.2×, different bound, out of current scope) and the under-occupied `quantize_q8_1`/`rms_norm`/`lm_head` kernels — NOT more weight-load tweaks.

**Bottom line:** one cheap, certain, bit-safe edit (P1, ~+2 tok/s) plus one measurement-gated structural bet (P2, the only candidate large enough to matter and the least proven). The honest expected landing is **~83 tok/s guaranteed, ~100–115 if P2's L2 hypothesis holds, and 82%/120 only with the out-of-scope CUDA-graph lever stacked on top.** Do not bank 120 from this stack alone.
