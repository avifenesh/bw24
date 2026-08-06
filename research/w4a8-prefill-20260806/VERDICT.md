# W4A8 prefill — VERDICT

**GO — but not for the reason the queue item asked, and not for 20-30 eng-days of work.**

The FP4-activation scoping brief (`research/fp4-act-scoping-20260806/BRIEF.md` §5.4) listed
"keep the k64 door closed, pursue W4A8" as the pragmatic fallback, at 20-30 engineering days,
citing QServe's W4A8KV4. Priced against the closed prefill campaign's own denominator, that
recommendation was **already implemented twice** in this repo and was **losing 49% of its own
instruction budget to a single wrong PTX mnemonic**.

| | pp512 q27 NVFP4, locked 1860, N=15/arm |
|---|---|
| NAKED (default int8 W4A8) | **1395.5 tok/s** |
| f8f4 arm, as shipped (plain `kind::f8f6f4`) | 1402.7 tok/s = **1.0052x** |
| f8f4 arm, one-line form swap (`kind::mxf8f6f4.block_scale`) | **1696.0 tok/s = 1.2153x** |

Cost of the find + fix: **one day, one line of PTX** (plus a rollback seam and a gate fix).

---

## 1. What the brief proposed, and why it was already answered

The brief's W4A8 row proposes 4-bit weights against 8-bit activations, "leveraging existing FP8
paths." Three facts, established in SCOPE.md before any measurement:

- **A.** memra's *default* prefill GEMM already **is** W4A8: `mul_mat_q_nvfp4_w4a8` takes NVFP4
  weights (e2m1 + UE4M3 per-16 scales), LUT-dequants them to int8 tiles, and multiplies against
  q8_1 int8 activations. It is 78% of q27 pp512.
- **B.** The FP8-flavoured W4A8 route was **also** already built — `MEMRA_MMQ_F8F4=1`, the R-B
  route of `research/prefill-mxf8f6f4-design.md`, folding the per-16 scales into e4m3 weight
  containers at tile load.
- **C.** `research/prefill-ilp-20260806` had measured a **1.993x** MMA-issue advantage for the
  k32 f8f4 route over the k16 int8 default, and the campaign was nonetheless declared closed.

So the honest one-day answer looked like a NO-GO: *"nothing to build, W4A8 is shipped twice, and
its 1.993x didn't convert."* SCOPE.md wrote that prediction down, along with the decision rule
that would refute it: **re-price the shipped route on the closed denominator; if F8F4 ≤ 1.07x
NAKED, NO-GO.**

## 2. Slice 1 — the shipped route delivers +0.52%

Interleaved NAKED/F8F4, one binary, locked clocks, `flock`'d: **1395.4 → 1402.7 = 1.0052x.**
A 1.993x instruction advantage converting to half a percent. Not the +3.9-6.3% the July flip
battery had recorded either.

(One round's NAKED arm died on a quoted `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
memory")` — a co-resident `gemma-gate` from another lane holding 21.6 of 24 GiB. Round dropped,
not interpreted; surviving NAKED reported as N=10.)

The NO-GO was one sentence away. What stopped it was running the **positive control** anyway.

## 3. Slice 2 — the anomaly the seam-miss hypothesis could not explain

`nsys` confirmed the seam works: `mul_mat_q_nvfp4_f8f4` replaces `mul_mat_q_nvfp4_w4a8` at the
same 1600 instances and the same ~78% of pp512, and the activation quantizer swaps too
(`quantize_mmq_q8_1_d4` → `quantize_mmq_e4m3_d4`). Not a dispatch miss. Then `ncu`:

| metric | NAKED | F8F4 | ratio |
|---|---|---|---|
| `smsp__inst_executed_pipe_tensor.sum` | 22,282,240 | 11,141,120 | **0.500** |
| `sm__pipe_tensor_cycles_active.sum` | 356,515,840 | 402,947,768 | **1.130** |
| `sm__cycles_elapsed.sum` | 157,166,591 | 156,148,485 | 0.994 |
| `sm__pipe_alu_cycles_active.sum` | 243,624,960 | 101,793,280 | 0.418 |
| `l1tex__t_bytes.sum` | 651,165,696 | 651,165,696 | 1.000 |

Tensor instructions halve **exactly** — the k32 route issues half as many MMAs for the same
work, as designed. And the kernel gets *no faster*, because tensor-pipe **cycles rose 13%**.
Derived: **16.00 tensor cycles per warp-MMA for NAKED, 36.17 for F8F4.** NAKED's 16.00 matches
`prefill-ilp`'s measured 16.06 issue interval to 0.4%. F8F4 was paying 2.26x the interval for
2x the depth.

The dequant-ALU saving the route was designed for **was** delivered (ALU cycles −58%, total
instructions −35%) and bought nothing, because the kernel is tensor-pipe-cycle-bound. Identical
`l1tex` bytes (1.000) also kills the expected smem-byte win: PTX packs f4 in 8-bit containers.

This pointed at the **instruction**, not the tile.

## 4. Slice 3 — THE FIND: the tile issues the slow form

`tools/f8f4_issue_rate.cu`, five PTX forms, `clock64` over mutually-independent MMAs with an
NACC=1..16 ILP control, plus full-GPU `cudaEvent` at the shipped 82×256 shape. Two full reruns
agreeing within 0.5%:

| form | cyc/warp-MMA | full-GPU rate | vs s8 |
|---|---|---|---|
| `m16n8k16 .s8.s8.s32` (today's default) | 16.06 | 155.2 TOP/s | 1.000 |
| **`kind::f8f6f4` PLAIN, e4m3×e4m3** ← **the shipped f8f4 form** | **32.02** | 155.3 TF | **1.000** |
| `kind::f8f6f4` PLAIN, e2m1×e4m3 | 32.13 | 155.5 TF | 1.002 |
| `kind::mxf8f6f4.block_scale.1X` e2m1×e4m3 ue8m0 | 16.06 | 309.6 TF | 1.995 |
| **`kind::mxf8f6f4.block_scale.1X` e4m3×e4m3 ue8m0** ← **same operands, 2x rate** | **16.06** | 309.4 TF | **1.994** |

NACC control: every form floors from NACC=2 and holds flat to NACC=16, so these are pipe
**issue intervals**, not latency artifacts of thin ILP. The plain form's 32.02 is real.

**The plain form has 2x the issue interval for 2x the K depth — exactly 1.000x the MAC rate of
the int8 default it was supposed to beat.** It is a rate-neutral instruction.

This overturns a documented assumption. `research/prefill-mxf8f6f4-design.md` chose the plain
kind *deliberately*: "CUTLASS also ships a plain `kind::f8f6f4` (no block_scale, no scale regs)
— R-A applies scales in the epilogue anyway, so the plain form is the cleaner instruction for
the tile." Cleaner, and half speed. And `prefill-ilp` slice 2b's 1.993x was measured on
`kind::mxf8f6f4.block_scale.scale_vec::1X` — **so the 1.993x never belonged to the shipped
tile.** The two documents were each internally correct and were never cross-checked against
which mnemonic the `.cu` file actually contained.

It also closes slice 2 with no residual: 32.02 instruction interval + tile overhead = the 36.17
measured on the live kernel; s8's 16.06 = the 16.00 measured on NAKED.

## 5. Slice 4 — the fast form is a bit-exact drop-in

`tools/blksc_identity.cu`: one warp, 128 accumulator elements, real random NaN-free e4m3
operands, the **same A/B fragments** fed to both MMA forms in the same kernel launch. `ue8m0` is
bias-127, so scale byte `0x7F` = 2^0 = identity.

| case | bitwise different | mean(blksc/plain) |
|---|---|---|
| identity `0x7F7F7F7F` both operands | **0 / 128** | 1.000000 |
| scale_a `0x80808080` (2^1) — control | 128 / 128 | **2.000000** |
| scale_b `0x7E7E7E7E` (2^-1) — control | 128 / 128 | **0.500000** |

The two controls are what make case 0 meaningful: the scale operand is genuinely read in every
selected lane and moves the result by exactly the power of two it encodes. So case 0's zero
difference is real identity, not a silently-ignored operand. Same SM80 m16n8k32 8-bit TN
fragment layout, same f32 accumulator ⇒ **the swap is a pure rate change.** The tile's numeric
config, its `f8f4-check` tolerances, and its argmax lineage are untouched *by construction*.

## 6. Slice 5 — the swap, measured and gated

`crates/memra-engine/cu/mmq_nvfp4_w4a8.cu` — `memra_mma_f8f4` now issues
`kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0` with both scale operands at
`0x7F7F7F7F`, documented with the measured rate table so the next reader cannot repeat the
mistake. Build-time rollback seam: **`MEMRA_MMQ_F8F4_PLAIN=1`** restores the plain form.

**Clean interleaved A/B** (`tools/ab_clean.sh`, idle guard: util < 15% and ≤ 1 compute app
required before *every* run, util/clocks/temp recorded per run), pp512, one binary, 3 rounds ×
`MEMRA_PP_REPS=5` = N=15/arm, locked 1860/1860, 56-74 C, 6/6 `rc=0`:

```
NAKED  median 1395.5  [1394.2 .. 1397.1]  spread 0.21%
F8F4   median 1696.0  [1694.8 .. 1697.1]  spread 0.14%
ratio  1.2153x        the two arms' ranges DO NOT OVERLAP
```

NAKED reproduces `prefill-ilp`'s locked reference (1395.2) to 0.02% — this is the closed
campaign's own denominator, not a re-baselined one.

A first attempt was **discarded and labeled**, not published: another lane's
`graph-warmup-stress` ran the GPU at 100% while the `flock` was held (that lane doesn't take the
lock), clocks fell to 1635-1755 MHz, and both arms landed near half the clean denominator.
Interleaved ratio still read 1.20x. Lesson now encoded in the harness: **a locked-clock A/B
needs a per-arm contention guard, not just a lock file.**

### Correctness battery (live path → full bar, both arms)

| gate | NAKED (regression control) | F8F4 (in-config) |
|---|---|---|
| `kernel-check` | `ALL GREEN` rc=0 | `ALL GREEN` rc=0 |
| `run-gen` argmax | prefill 271 == decode 271 **MATCH**; batched-prime 271 == tokenwise 271 **MATCH** | same, **MATCH** / **MATCH** |
| `run-spec` K=1..8 | rc=0 | rc=0 |
| `f8f4-check` | `ALL GREEN` | `ALL GREEN` |

The NAKED column is the control that the swap caused no collateral damage: it only touches
`memra_mma_f8f4`, so any NAKED movement would be a red flag. There is none.

**One gate had to be fixed to get there, and it was broken before this lane.** `kernel-check`'s
`MMQ-W4A8` rung failed 4 tests at rel 3.37e-2 … 4.34e-2 against a hard 2e-2. The **plain-form
control** (`MEMRA_MMQ_F8F4_PLAIN=1`, `logs/kc-plainform-control.log`) reproduces **all four
values to printed precision** — pre-existing, not a regression. Cause: `kernel_check.rs:1649`
states its own premises ("weight FP4 is LUT-dequantized to int8 … the activation stays q8_1
int8 → rel MUST sit in the int8-activation band"), and **both are false** under
`MEMRA_MMQ_F8F4=1`, because `mmq_ffi.rs:1129`'s seam redirects that very entry point to the
f8f4 tile (e4m3 weight containers, e4m3 activations). It was judging a 3-mantissa-bit class
against a 7-bit class's band. This is precisely the **stale-verdict class the H100 lane named as
LAW 2**. Fixed arm-aware (2e-2 int8 / 5e-2 e4m3-act — the bound and the reasoning `f8f4_check.rs`
already carried), printing as `MMQ-W4A8-F8F4` so the classes can never again be read off one
line. Re-gated, both arms `ALL GREEN`. NAKED lands at 4.19e-3 … 7.36e-3 where F8F4 lands at
3.37e-2 … 4.34e-2 on identical inputs — an ~8x split that is itself the evidence one band could
never gate both. This also brings the f8f4 config **inside** the battery (H100 LAW 3) instead of
leaving it to a standalone bin.

## 7. Arithmetic: how much is left

GEMM share of q27 pp512 = 0.7797. Measured 1.2153x e2e ⇒ **1.312x kernel**, against a **1.994x**
instruction bound = **66% realized** — the same realization band as the mxf4 door's 68%. Full
realization would be **1.616x e2e (pp512 → ~2255)**. So roughly **0.4x of e2e headroom remains
inside the tile**, now on the correct instruction, and the campaign's "closed at the issue
interval" conclusion holds for the *int8* path only.

## 8. What this says about the closed campaign

The campaign's central receipt was that the tensor pipe is 100% issue-saturated at the 16.06
interval, and that receipt is **correct** — for `m16n8k16.s8`. What was never checked is whether
the *alternative* route's instruction reached that same interval. It did not, by 2x, and the
tile had been wired to the slower of two forms that compute the identical product. The
saturation argument was sound; the instruction under it was the wrong one.

Generalizable lesson, and the reason this took a day rather than 20-30: **a positive control that
"passes" is not the end of an investigation.** The seam worked, the kernel swapped, the
instruction count halved exactly as designed — every check passed, and the result was still
wrong. The anomaly was only visible in a metric nobody had a hypothesis for (tensor *cycles*
rising while tensor *instructions* halved).

---

## 9. Follow-up brief

**1. `mmq_fp8_blk.cu:214` has the same wrong mnemonic, on a live path. HIGHEST PRIORITY.**
`memra_fp8_mma_f8f4` issues the identical plain `kind::f8f6f4` e4m3×e4m3 form, and it is live via
`fp8_ffi.rs:401` under `MEMRA_PP_FP8` — the **FP8-ST** direction. Same operands, same fragment
ABI, so slices 3 and 4 transfer verbatim: same 2x issue interval, same bit-exact identity-scale
swap. Its comment even asserts "381-TF class, the same op the W4A8-FP8 arm uses" — the same
belief that measured false here. **Not edited by this lane on purpose:** an FP8-ST agent is
active on that file and a concurrent edit would collide. Route it to that lane with slices 3-4
attached; the gate is `fp8_mmq_check`. Expected size: one line, plus a seam.

**2. `mmq_q8_0_f32acc.cu:163` — same form, no action needed.** `accprobe` instrument only, not a
serving path. Worth a comment noting the rate, because it is an *accumulator* instrument and its
absolute numbers are form-dependent.

**3. The default flip is NOT this lane's call, and needs multi-model evidence.** `MEMRA_MMQ_F8F4`
stays default-OFF. The swap makes the seam worth flipping, but the f8f4 arm is a genuinely
different numeric class (3.4e-2 vs 4.2e-3 rel), not a bit-exact win — the bit-exactness proven
here is *between the two MMA forms*, not between f8f4 and int8. Argmax matched and K=1..8 passed
on q27, but CLAUDE.md's prefill-KV acceptance law says a prefill numeric change moves spec
acceptance **model-dependently**. Flip gate: acceptance + quality across the served model set,
not one model's argmax. (The strong prior for "acceptance cannot move *from this swap*" is slice
4 — the swap changes no bits relative to the plain f8f4 arm. The open question is the f8f4 arm
vs int8, which pre-dates this lane.)

**4. The remaining ~0.4x.** 66% of the instruction bound is realized; the tile is now
tensor-cycle-bound on the *right* instruction. `scale_vec::1X` at identity is leaving the
block-scale hardware idle — folding the NVFP4 per-16 scales into the **scale operand** instead
of into the e4m3 values (the R-A route the design doc rejected) would remove the fold work from
the tile-load path and might recover part of it. Cheap next slice, and it now has a working
instruction to build on.

**5. Audit every PTX mnemonic in the repo against a measured rate table.** This bug class is
invisible to correctness gates by construction — both forms compute the same product, so every
numeric gate passes. `f8f4_issue_rate.cu` is the tool; the deliverable is a committed table of
measured issue intervals per mnemonic on sm_120a, so "which form is fast" stops being folklore
carried in prose comments. Two of three plain-form sites found here were wrong.

---

## 10. Evidence index

All raw logs committed beside this file. No `.nsys-rep` committed anywhere — profiles ran into
`mktemp -d` and only `cuda_gpu_kern_sum` CSVs were exported, per the standing security rule.

| file | what |
|---|---|
| `SCOPE.md` | pre-registered question, ceiling arithmetic, decision rule (≤1.07x → NO-GO) |
| `RESULTS.jsonl` | all five slices, per-rep values, ncu tables, NACC controls |
| `logs/ab-f8f4-q27.log` | slice 1 A/B (incl. the quoted OOM round) |
| `logs/f8f4-issue-rate-5form.log` | slice 3, the five-form table + NACC sweep |
| `logs/blksc-identity.log` | slice 4, bit-exactness + the 2x / 0.5x live-operand controls |
| `logs/ab-f8f4-q27-blksc.log` | contaminated A/B — **not published**, kept as the contention receipt |
| `logs/ab-clean-q27-blksc.log` | the published 1.2153x, N=15/arm, per-run GPU state |
| `logs/battery-q27.log` | two-arm battery; found the pre-existing gate mis-scope |
| `logs/kc-plainform-control.log` | the control proving those 4 FAILs pre-date the swap |
| `logs/kc-armaware.log` | both arms `ALL GREEN` after the gate fix |
| `nsys/`, `ncu/` | kern-sum + metric CSVs |
| `tools/` | every harness, incl. `ab_clean.sh`'s idle guard and `nsys_arm.sh`'s scrub |

GPU left unlocked (`nvidia-smi -rgc`) — every harness resets clocks via `trap cleanup EXIT`.
