# Prefill re-aim lane (#72 phase 2) — the two named probes, both refuted, and the honest headroom

**Date:** 2026-08-06 · **Branch:** `lane/prefill-ilp` · **Base:** `50a59039` (the phase-1 merge)
**Rig:** local RTX 5090 Laptop (sm_120a, GB203, 82 SM), **clocks locked 1860/1860**, persistence ON,
53–70 C, nvcc 13.1.115. One idle co-resident `llama-server` (pid 144655, 332 MiB, 0% util) — recorded,
not contending.
**Denominator:** `research/e2e/prompts/pp512.txt` (512 tokens, the prod anchor), q27 NVFP4.
**Receipts:** `RESULTS.jsonl` (7 slices), `logs/`, `sass/`, `spike/`, `tools/`.
**Engine code changed: NONE.** `git diff --name-only 50a59039 HEAD` is entirely under `research/`.
Battery not run — correctly, and for the same reason as phase 1: there is nothing to gate.

---

## Headline

Both probes are refuted, **and both were refuted before a kernel was built** — one by SASS, one by
ptxas plus a microbench. But the reusable finding is neither verdict: it is that **phase 1's
mechanism story was wrong**, and the correction closes a whole family of levers that two lanes have
now circled.

> **Phase 1 said:** `tensor_cycles/tensor_insts = 16.00` is the m16n8k16.s8 issue interval, and with
> 8 warps/CTA at 1 CTA/SM each scheduler owns 2 warps against a 16-cycle pipe, so the idle cycles
> are **MMA latency exposure from thin warp parallelism**.
>
> **Measured here:** MMA latency is **27–29 cycles** and is **fully hidden at 2 independent
> accumulators**. The kernel already has 4. At the shipped 8 warps/CTA the pipe runs at 31.8–32.0
> cyc/MMA = 2 warps × 16 = **100% issue-saturated**. There is no latency exposure. The pipe is
> issue-bound, and 16.06 cyc/MMA is **the pipe's interval, not the instruction's** — all three
> candidate MMA forms measure the same 16.06.

That single correction is what kills probe 1 and what makes probe 2's 4x real.

---

## PROBE 1 — intra-warp ILP: **REFUTED AT ITS PREMISE**

The probe asked: "how many independent accumulator tiles does one warp own, and are the MMAs
serially dependent on the same accumulator? If the chain is dependent, restructure to 2–4
independent C tiles per warp."

`cuobjdump -sass` on the shipped instantiation `mul_mat_q_nvfp4_w4a8<128,128,1,0,1>` — the
77.97%-of-pp512 kernel:

```
grep -c 'IMMA'                     = 256
grep 'IMMA' | grep -v ', RZ ;'     = 0 matches
```

**Every MMA is `D = A*B + RZ`.** There is no serial accumulator chain. The source already declares
`tile_C C[2]` *fresh inside* the `n`-loop with `ntx=2`, so a warp already owns **4 mutually
independent C tiles per k01 step** and drains each product straight into the persistent f32 `sum`.
The restructure the probe proposed is what the compiler already emits.

**The accounting closes exactly**, so this is a complete audit and not a sampled window:

| quantity | source predicts | SASS measures |
|---|---|---|
| MMAs | 8 j0 × 4 k01 × 2 (C[0],C[1]) × ntx 2 = 128/call × 2 calls = **256** | **256** (fully unrolled, no loop-carried MMA branch) |
| A ldmatrix | ntx 2 × (32/8) = 8/call × 2 = **16** | **16** |
| fold | 512 C-elements → 1024 I2FP + 1024 FFMA + ~512 FMUL | **1024 / 1024 / 584** |

**Register budget forbids it independently.** `ptxas -v`: this instantiation is **252 regs, 0 spill**
— and **six of the eight** 128×128 siblings in the same TU are **already at 255 regs and already
spilling** (8–28 bytes). One added independent C tile costs 4 regs; doubling 4→8 tiles costs **+16**
against **+3** headroom. The stated fallback — double-buffering A/B fragments — costs **+32**. Both
force spills, and spilling the accumulator on a kernel whose stalls are `math_pipe_throttle` 1.25 /
`lg_throttle` 0.01 trades a filled pipe for an LSU-bound one.

**The fallback also has no mechanism**, exactly as the brief anticipated: `short_scoreboard` is 0.16
against `math_pipe_throttle` 1.25, and the SASS shows the `LDS`/`LDS.128`/`LDSM` operand loads for
MMA *i+1* already interleaved between MMA *i*'s issues — ptxas software-pipelined them unasked. The
loads were never the stall.

**No A/B run was warranted**: there is no code change to measure. The denominator was reproduced
first anyway (rep0 1394.6 tok/s vs phase-1's locked 1395.2, 0.04%) so the refutation does not rest
on an unverified rig.

---

## PROBE 2 — `mxf4nvf4.block_scale` feasibility: **the door is real, 4x is real, and it is
## already built**

### (a) Does sm_120a expose it? **YES — and it lowers to a real SASS op.**

| PTX form | ptxas | SASS |
|---|---|---|
| `m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | **ACCEPTED** | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` |
| `m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0` | **ACCEPTED** | `QMMA.SF.16832.F32.E2M1.E4M3.E8` |
| `m16n8k64` … with an `e4m3` B operand | REJECTED | *"Incorrect instruction type … shape '.m16n8k64'"* |
| `m16n8k64` … `scale_vec::2X` with `.ue4m3` | REJECTED | *"Illegal modifier '.scale_vec::2X' … type '.ue4m3'"* |
| `m16n8k32.mxf8f6f4` … any `scale_vec` with `.ue4m3` | REJECTED | *"Incorrect instruction type … shape '.m16n8k32'"* |

The rejections are the load-bearing half: **k64 is `e2m1 × e2m1` ONLY**, and **k32 mxf8f6f4 is
`ue8m0`-only**.

### (b) Operand layout: **MATCHES EXACTLY — zero repack.** Not an estimate; the repo ships it.

NVFP4 stores one UE4M3 per 16 elements. `m16n8k64` with `scale_vec::4X` takes 4 scale bytes per
operand, each covering 64/4 = **16** elements. 16 == 16.

`cu/mmq_fp4.cu:183` (`mma_block_scaled_fp4_nvfp4`) already issues this exact instruction against
memra's stored bytes — *"load_tiles_nvfp4_nvfp4 reads the raw weight bytes directly (pure u32 copy,
no repack)"* — and `cu/qmatvec_gemm.cu:1212` records the probe-verified fragment/scale lane layout
(maxrel = 0 vs f32 oracle) plus the exact cancellation: GGUF e2m1 = 2× HW e2m1, GGUF UE4M3 = 0.5× HW
UE4M3, so feeding both **raw** reproduces the GGUF dequant exactly.

**The cost is not a repack — it is a precision change.** k64 takes FP4 activations only, so this
door *is* W4A4. The W4A8-preserving alternative (k32 mxf8f6f4, e4m3 activations) is equally closed:
`ue8m0`-only means dropping the UE4M3 mantissa to a power-of-two scale, and 1X over k32 means one
scale per 32 elements where NVFP4 has one per 16. Both are lossy on the stored weight — phase 1's
"no shared scale across a wider K span" obstruction reappearing inside the block-scale family.

### (c) The measured issue rate — **the number this lane exists to produce**

Two independent instruments, both clock-locked 1860 and `flock`'d. (i) `clock64()` around a tight
loop of mutually independent MMAs, sweeping NACC and warps/CTA. (ii) full-GPU `cudaEvent` at
82 CTAs × 256 thr, best of 5.

| form | clock64 cyc/MMA | full-GPU cyc/MMA | MAC-rate | vs s8 |
|---|---|---|---|---|
| `m16n8k16.s8.s8.s32` (today's kernel) | 16.06 | 16.12 | **155.0 TOP/s** int8 | 1.000x |
| `m16n8k64.mxf4nvf4.4X` | 16.06 | 16.16 | **618.5 TFLOP/s** | **3.989x** |
| `m16n8k32.mxf8f6f4` e2m1×e4m3 | 16.06 | 16.18 | **309.0 TFLOP/s** | **1.993x** |

**All three forms share the same 16-cycle interval**, and the two instruments agree to 0.4%. So the
paper bound is exact: same interval, 4x the work per issue → **measured 3.989x**. `scale_vec::4X`
costs nothing — the block scales really are free hardware operands.

**The NACC control is what makes this an issue interval rather than latency/NACC** (1 CTA, 4 warps
= 1 warp/scheduler):

| NACC | s8.k16 | mxf4.k64 | mxf8.k32 |
|---|---|---|---|
| 1 | 27.13 | 29.08 | 29.08 |
| 2 | 16.13 | 17.13 | 17.13 |
| 4 | 16.06 | 16.07 | 16.07 |
| 8 | 16.06 | 16.06 | 16.06 |
| 16 | 16.03 | 16.03 | 16.03 |

Cycles/MMA does **not** halve as NACC doubles — it floors at 16.06 from NACC=2 and holds to NACC=16.
NACC=1 exposes true latency: **27.1 cyc (s8) / 29.1 (block-scale)**, hidden by **two** independent
accumulators. And the warp sweep: 8 warps/CTA → 31.8–32.0 cyc/MMA = 2 × 16 = **the pipe is 100%
issue-saturated at the shipped shape**.

Two corrections fall out, both retiring lever families:
- **"More MMAs in flight" is closed.** Not by a flat config sweep this time but by mechanism: latency
  is already hidden 2x over, so more warps or more accumulators can only queue on a saturated pipe.
- **Phase 1's peak inference is superseded.** Measured dense s8 peak is **155.0 TOP/s**, not the
  109.5 inferred by halving the 219 nameplate. The live kernel's 88.6 TOP/s is **57.2%** of measured
  dense peak, not 80.9%.

### (d) Verdict: **build-worthy — NO, because there is nothing left to build**

`cu/mmq_fp4.cu` *is* the m16n8k64 mxf4nvf4 block-scale MMQ prefill GEMM: complete, dispatched behind
`MEMRA_MMQ=1`, with its own exactness ledger. So the door gets **priced by measurement**.

4-arm interleaved pp512, one binary (the door is a runtime dispatch flag), **N=15/arm**, clocks
locked, q27 NVFP4, 12 clean `rc=0` runs, 53–70 C. (`MEMRA_RP=0` on the W4A8/W4A4 arms because an
`rp` weight always forces W4A8 — `mmq_ffi.rs:564`.)

| arm | median tok/s | min–max | vs naked |
|---|---|---|---|
| **NAKED** (shipped default, rp split-plane W4A8) | **1316.3** | 1217.4–1398.0 | 1.0000x |
| W4A8 (`MEMRA_RP=0`, GGUF layout) | 1231.2 | 1148.6–1249.7 | 0.9353x |
| **W4A4RAW** (`MEMRA_MMQ=1`, `RESIDUAL_K=0`) | **2591.2** | 2108.0–2603.1 | **1.9685x** |
| W4A4K32 (`MEMRA_MMQ=1`, `RESIDUAL_K=32`) | 1697.1 | 1693.4–1698.9 | 1.2893x |

Against the 77.97% GEMM share, 1.9685x e2e implies a **2.710x kernel** speedup — **68% of the
3.989x instruction bound actually realized**. Compare the entire fold/ILP/feed/occupancy family:
**+3.17%** (phase 1). The instruction form is the only lever with an order of magnitude on it.

**It is not the default for exactness reasons already on record** (`docs/FLAGS.md:333`,
`research/w4a4-rescue-20260803/`): the k=32 residual correction makes the original 3-prompt corpus
5/5 token-identical 48/48 across four kernel revisions with the full battery green, but widening to
5 untested prompts gives **4/10**, and q27/board-2048 forks at token 0 deterministically (3/3) at
**8 of 9 k depths including k=0**.

**What this lane adds is the root cause.** The ptxas sweep proves k64 accepts `e2m1 × e2m1` only —
no e4m3 B operand at any `scale_vec`. Taking this door *requires* FP4 activations. The divergences
are **activation precision loss, structural to the instruction's operand grammar** — which is why
four kernel revisions could not fix it and why k=0 diverges too. The only build that moves this
verdict is an **FP4-activation accuracy-recovery scheme**, which is a quality-research question, not
a kernel question.

---

## The honest prefill headroom statement

The prefill GEMM is **issue-saturated on a 16-cycle tensor pipe at 57.2% of measured dense s8 peak**,
and every remaining lever splits cleanly into two classes:

**Class 1 — same instruction: ~3%, closed.** Fold removal measured +3.17% (phase 1). ILP has no
dependent chain to break and no register headroom (this lane). Feed/`long_scoreboard` is 0.04, bank
conflicts weigh 0.22 against math-throttle 1.25, occupancy is pinned at 1 CTA/SM by both smem and
regs, and the config axes (X 32/128/256, Y 64/128/192) swept flat 2026-07-06. More warps and more
accumulators are now closed **by mechanism**, not just by flat sweeps: latency is hidden 2x over and
the pipe is 100% issue-saturated. **Realistic remaining headroom on the current instruction: a few
percent, and the cheap ones are spent.**

**Class 2 — change the instruction: 3.989x available at the MMA, 2.4052x e2e if fully realized
(pp512 1316 → 3166), 1.9685x e2e measured today.** This is the only class with real headroom, it is
**already implemented**, and it is gated on **activation precision, not engineering**. It buys 2x
prefill the day an FP4-activation accuracy story exists, and not before.

There is no third class. The Marlin x-stage restructure named by the July 6 row belongs to Class 1
and the profile is hostile to it (DRAM 8–10%, `long_scoreboard` 0.04) — and it is now doubly closed,
because a restructure that changes *when bytes arrive* cannot help a pipe that is issue-saturated.

**So: prefill is done as a kernel-engineering target.** The next real prefill gain is a **quality**
deliverable (FP4-activation accuracy recovery), and the honest statement to carry forward is that
memra's prefill GEMM is within a few percent of what the m16n8k16.s8 instruction can deliver on this
silicon.

**Owner call requested** on whether FP4-activation accuracy recovery gets a lane, given it is the
sole remaining path to 2x prefill and it is a quality problem rather than a kernel one.

---

## Method notes worth keeping

1. **Two probes, zero kernel builds, both answered.** Probe 1 died to `cuobjdump` + `ptxas -v`;
   probe 2 died to a 20-line compile spike plus a 120-line microbench. Phase 1's lesson (a derived
   ceiling was 16x wrong) generalizes: **read the SASS and measure the instruction before building
   the kernel.**
2. **A cycles-per-instruction number needs an ILP control.** 16.06 cyc/MMA only means "issue
   interval" because the NACC sweep shows it flat from 2 to 16. Without that control it is
   indistinguishable from latency/NACC — and phase 1's latency-exposure story is exactly the error
   that control catches.
3. **Two instruments or it did not happen.** `clock64()` on one CTA and full-GPU `cudaEvent` agreeing
   to 0.4% is what makes 16.06 trustworthy; either alone could be a measurement artifact.
4. **Check whether the thing is already built.** Probe 2 was scoped as a feasibility spike for a
   large future build. The build shipped months ago, behind a flag, with an exactness ledger. Reading
   `docs/FLAGS.md` and `git log --all --oneline | grep` first would have reframed the probe from
   "is this possible" to "why is the existing one off" — which is the question that actually mattered.
5. **Incidental find, kept:** the `rp` split-plane layout is **+6.9%** over GGUF-layout W4A8
   (1316.3 vs 1231.2, N=15 each, same battery) — an independent confirmation that `rp` is the right
   default.
