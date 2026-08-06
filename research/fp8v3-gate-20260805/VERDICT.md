# FP8-ST v3 GATE — verdict

> ## ⛔ CORRECTION 2026-08-06 — **Q1 IS REFUTED. THE Q1 "GO" BELOW IS WRONG. DO NOT FUND A v3 ON IT.**
>
> Q1's instrument (`cu/mmq_q8_0_f32acc.cu`) was **not** single-variable. Its F32 arm issued
> `mma.sync.aligned.kind::f8f6f4.m16n8k32` — which costs **32.02 cyc/warp-MMA** on sm_120a — against
> an S32 arm on `m16n8k32.s8` at **16.06 cyc**. The accumulator class and the MMA *issue interval*
> moved together, 2×. The claim below that "the swap is legal as a single-variable experiment because
> both forms share the fragment ABI" is true at the source level and **false at the cost level**, and
> the SASS census validated the former while never checking the latter.
>
> Re-measured with the form equalized (F32 arm on
> `kind::mxf8f6f4.block_scale.scale_vec::1X … ue8m0` at the identity scale `0x7F7F7F7F` — the
> **bit-identical** e4m3×e4m3 product at 16.06 cyc), 3 adjacent alternating binary pairs per cell,
> both `ACCPROBE_DIST` settings, S32-arm time as a drift control (−0.83 … +0.33% over 24 cells):
>
> | cell | published arm (plain form), re-measured | **equalized** |
> |---|---|---|
> | weighted delta, m=512 | +17.3pp | **−6.7pp** |
> | weighted delta, m=6257 | +17.6pp | **−8.8pp** |
> | geomean ratio f32/s32, m=512 | 1.2033x | **0.9495x** |
> | geomean ratio f32/s32, m=6257 | 1.2163x | **0.9373x** |
>
> **All of the Q1 delta was the MMA form; none of it was the accumulator.** With the interval
> equalized the sign flips — f32 accumulate is 5–9% *faster* than s32 at fixed geometry, because the
> S32 arm additionally pays the 512 `I2FP` converts this document already recorded (and scored as
> "generous to F32"). Q1 does not clear its ≥10pp bar; it does not clear zero. A v3 would pay
> per-128-block mantissa extraction to buy a **negative** accumulator delta.
>
> Q2 (native-e4m3 decode) is untouched by this and is **not** re-priced here; the "decode-v1 first"
> recommendation stands on Q2's own evidence. `ACCPROBE_F32_PLAIN=1` rebuilds the published arm so
> everything below stays reproducible.
>
> Receipts: `research/rp-on-st-20260806/accprobe-form-ab.{sh,log}`, `accprobe-form-ab-summary.txt`,
> `accprobe-sass-census.txt`, and the `accprobe_mma_form_ab` / `accprobe_mma_form_verdict` rows in
> that lane's `RESULTS.jsonl`. Found while auditing a coordinator-routed find on the same defect in
> `cu/mmq_fp8_blk.cu` (third site; `research/w4a8-prefill-20260806/` was the first).
>
> **Law this teaches:** "one free variable" must be verified at the *cost* level (issue rate), not
> only at the source level (fragment ABI / MMA count). A verdict is only as calibrated as its
> control arm.

Lane: `lane/fp8-v3-gate` (off `restructure/public-split`) · 2026-08-05 · RTX 5090 Laptop (sm_120a)
Charter: `research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl` §6 — *a v3 "should not start without a
receipted estimate that s32-vs-f32 accumulate is worth the >= 10pp it would have to buy."*

This lane delivers two receipted estimates and **builds no campaign kernel**. Raw runs, per-cell
rows and negatives are in `RESULTS.jsonl` + the four `*.log` files next to it.

## Shape sheet: 27B only

Owner correction, binding, applied to everything below:

> "why we test 1.7b? was never part of supported models. we tune the fp8 for 27b."

Every measured cell is a Qwen3-27B projection shape. **Zero 1.7B cells were run** — there are no
instrument-only rows to discount. (Both bins still carry an unused 1.7B shape array behind an
optional argv; harmless, never invoked.)

Shape sheet, verbatim from v2 §3 `final_sheet`: `q_proj 5120->12288`, `k/v_proj 5120->1024`,
`o_proj 6144->5120`, `gate/up_proj 5120->17408`, `down_proj 17408->5120`, `square-ref 5120->5120`.
Prefill at m=512 and m=6257; decode at m=1.

## The weighting (owner directive 2)

Thresholds are applied to the 27B shapes **weighted by their real share of projection time**, not to
an unweighted geomean. The weight is each projection's per-decoder-layer call multiplicity:

| cell | projections it stands for | multiplicity |
|---|---|---|
| `q_proj 5120->12288` | q_proj | ×1 |
| `k/v_proj 5120->1024` | k_proj + v_proj | ×2 |
| `o_proj 6144->5120` | o_proj | ×1 |
| `gate/up_proj 5120->17408` | gate_proj + up_proj | ×2 |
| `down_proj 17408->5120` | down_proj | ×1 |

Weighted total = Σ(multiplicity × per-call median); the reported ratio is total/total, so each cell
contributes exactly its share of real time. `square-ref` is a geometry reference, not a model
projection, and is excluded from the weighted numbers (it stays in the unweighted geomeans).

Measured time shares confirm the owner's read — **gate/up + down carry ~73% of prefill projection
time and ~69% of decode**:

| cell | prefill share (m=6257) | decode share (m=1) |
|---|---|---|
| gate/up_proj | 50.1% | 45.4% |
| down_proj | 23.5% | 23.1% |
| q_proj | 16.4% | 16.5% |
| o_proj | 7.5% | 9.2% |
| k/v_proj | 2.4% | 5.9% |

## Q1 — ~~what s32 accumulation is worth: **+16.7pp (m=512) / +23.1pp (m=6257) weighted → GO**~~ **REFUTED 2026-08-06 → NO-GO (−6.7 / −8.8pp once the MMA form is equalized; see the correction at the top)**

**Bar: ≥ 10pp. Cleared by 1.7–2.3x.**

### The instrument

`crates/memra-engine/cu/mmq_q8_0_f32acc.cu` — the **Q8_0 MMQ floor kernel with the accumulator as
its one free variable**. Both arms live in one translation unit templated on `bool F32ACC` and share
every loader, every smem expression, every launch parameter, the same device buffers and the same
bytes:

- arm **S32**: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` — the floor's own MMA.
- arm **F32**: `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` — the exact op v2
  accumulates in.

**[FALSE — see the 2026-08-06 correction. The two forms share the fragment ABI but not the issue
interval: 32.02 vs 16.06 cyc/warp-MMA. This paragraph is the exact reasoning error.]**
The swap is legal as a single-variable experiment because both forms share the fragment ABI at this
shape (4×b32 A, 2×b32 B, 4-reg D), so nothing but the instruction changes.

**Why the floor and not v2's kernel.** v2 chains four k32 MMAs into one accumulator and folds
`(s_blk*dB)` **once per 128-k scale block**, where the floor folds every k32. v2's epilogue f32 work
is therefore already *strictly cheaper* than the floor's — the fold count cannot be the residual gap,
which leaves the accumulator/MMA class as the only remaining named variable. Swapping inside the
floor holds tiles, traffic, occupancy and fold count fixed. Building an s32 v2 instead would change
the arithmetic contract, need a new host reference, and *be* the v3 this gate is deciding whether to
fund.

The bench (`src/bin/accprobe_bench.rs`) is GEMM-only: the `block_q8_1_mmq` activation is synthesized
host-side, so no quantizer sits in the timed region diluting the measured quantity, and all
magnitudes are clamped to ≤126 so no byte is the e4m3 NaN code `0x7F` (an int8 quantizer emits ±127
at every block amax, which would hand the F32 arm NaNs and make it a different experiment).

Neither arm's output is a numeric claim — the two compute different arithmetic on the same bytes by
construction. This measures **time**. Exactness is owned by `fp8-mmq-check` / `kernel-check`.

### The instrument is single-variable — SASS-verified

`q1-sass-census.txt` (cuobjdump `--dump-sass` + `--dump-resource-usage` on the built object):

| instantiation | instr | IMMA | QMMA | I2FP | LDS | LDG | REG | SHARED | STACK |
|---|---|---|---|---|---|---|---|---|---|
| `<128,false,true>` F32 | 2680 | 0 | **128** | 0 | 232 | 104 | 255 | 1024 | 8 |
| `<128,true,true>` F32 | 2824 | 0 | **128** | 0 | 232 | 104 | 255 | 1024 | 56 |
| `<128,false,false>` S32 | 3176 | **128** | 0 | 512 | 232 | 104 | 255 | 1024 | 0 |
| `<128,true,false>` S32 | 3312 | **128** | 0 | 512 | 232 | 104 | 255 | 1024 | 0 |

- Identical **MMA count (128)** and identical **LDS/LDG** — same tiling, same traffic.
- Identical **REG:255 / SHARED:1024** — the delta is not an occupancy artifact.
- S32 emits `IMMA.16832.S8.S8`; F32 emits `QMMA.16832.F32.E4M3.E4M3`. ~~Exactly one thing moved.~~
  **[FALSE. Two things moved: the accumulator class AND the issue interval — `IMMA.16832.S8.S8` is
  16.06 cyc/warp-MMA and `QMMA.16832.F32.E4M3.E4M3` is 32.02. A census of MMA *count* cannot see
  this; it needs the per-form *rate*. Re-census with both F32 spellings:
  `research/rp-on-st-20260806/accprobe-sass-census.txt`.]**
- **The bias runs against the result**: the S32 arm executes *more* instructions (3176 vs 2680) and
  pays **512 extra `I2FP` int→float converts** (inherent to accumulating in s32 and folding into an
  f32 sum), and still wins ~20%. ~~The instrument is generous to F32.~~
  **[The fact is right, the scoring was backwards. Those 512 converts are exactly what S32 loses by
  once the issue interval is equalized: f32 then wins 5–9%.]**

### Numbers

Unweighted geomean `t_f32/t_s32` across four independent runs (see `RESULTS.jsonl` for all 48 cells):

| run | thermal regime | m=512 (N=9) | m=6257 (N=5) |
|---|---|---|---|
| run1 | 70→74C, cold-start clock ramp | +18.9pp | +19.8pp |
| run2 warm confirm | 20s load settle, 84→80C, clocks pinned 1650–1740 MHz | +19.8pp | +20.2pp |
| run3 control `dist=wide` | 64→74C | +19.1pp | +20.0pp |
| run3 control `dist=mid` | same hold | +20.6pp | +20.7pp |

**Weighted** (the number the gate is judged on), from the warm-confirm run:

| m | total f32 | total s32 | weighted ratio | **weighted delta** |
|---|---|---|---|---|
| 512 | 3.9063 ms | 3.3476 ms | 1.1669x | **+16.7pp** |
| 6257 | 50.9258 ms | 41.3577 ms | 1.2313x | **+23.1pp** |

Weighting does not rescue f32 — the dominant cells are where the delta is *largest* at m=6257
(gate/up +23.2, down +24.4).

### Controls and honesty notes

- **Clock-ramp risk, caught and closed.** Run 1's lock hold opened at 180 MHz and closed at
  1732 MHz. Rather than publish that alone, run 2 added a 20 s load settle and re-measured fully
  warm; the numbers reproduced (+19.8/+20.2 vs +18.9/+19.8). The ramp did not manufacture the result.
- **e4m3 denormal control, PASS.** e4m3 codes `0x01–0x07` are denormals; a QMMA denormal slow path
  would have made the headline a byte-distribution artifact. `ACCPROBE_DIST=wide` (magnitudes
  1..=126, ~5.5% denormals, ~15 binades) vs `mid` (`0x30..=0x4F`, e4m3-normal, mid exponents, zero
  denormals): mid is if anything **larger** (+20.6/+20.7 vs +19.1/+20.0). The delta is a property of
  the instruction, not the data.
- ~~**Neither arm is MMA-issue-bound.** Both run 87–141 TFLOP against the f8f6f4 class's ~381 TF
  peak, consistent with v2's own ceiling analysis — yet the accumulator class still moves ~20%. So
  the cost is not "we ran out of MMA slots"; it is the per-instruction throughput of the QMMA
  f32-accumulate form at this geometry.~~
  **[The ~381 TF denominator is wrong: plain `kind::f8f6f4` on sm_120a is a 155-TF class, not 381 —
  381 belongs to the block_scale form. The last sentence is, ironically, the true one: the cost WAS
  "the per-instruction throughput of the QMMA f32-accumulate form", i.e. the form, not the
  accumulator. Naming that and still attributing the delta to the accumulator is the whole error.]**
- **This is an UPPER BOUND, not a forecast.** A real v3 additionally pays per-128-block e4m3 mantissa
  extraction, which this instrument does not charge. The gate asked "is there a receipted case for
  v3", and there is. It did not ask, and this does not answer, "v3 will deliver 20pp."

### Verdict

**GO.** s32 accumulation is worth **+16.7pp to +23.1pp** on the 27B shapes weighted by real prefill
time share, against a 10pp bar. Reproduced across four runs, two thermal regimes and two data
distributions, with the instrument SASS-validated as single-variable and biased against the finding.
The v3 brief is below; **it is not built in this lane.**

## Q2 — native e4m3 decode: **+6.00pp weighted vs a +6.25pp ceiling → GO**

**Bar: ~6% (the byte-stream arithmetic). Met at 96% of theoretical.**

### No new kernel was needed

The tree check answered the "is this a bounded write" question by making it moot: the native-e4m3
m=1 GEMV **already exists and already ships**. `qmatvec_e4m3_mmvq` (`cu/qmatvec.cu:3258-3300`, body
`e4m3_row_dot`) reads the raw checkpoint e4m3 bytes as its weight stream — `row_bytes == in_f`, no
dequant — against the same q8_1 activation every fast decode path produces, behind
`MEMRA_ST_E4M3=1`. Its m=1 correctness is already gated by `kernel-check` (`kernel_check.rs:2181-2270`:
f64 CPU e4m3 reference plus `grid.y=m` and `_b2/_b4/_b8` bit-parity arms, `for mm in [1,2,5,9]`).

What was missing — and all this lane added — is the **A/B perf measurement**. The kernel shipped with
end-to-end evidence only and had never been benched at GEMV level.
(`fp8_mmq_bench.rs` could not answer Q2: both of its arms are prefill tile kernels.)

### The comparison

`src/bin/gemv_e4m3_bench.rs`. Both arms ride `qmatvec_mmvq_raw`, which quantizes the **same** f32 x
to q8_1 and launches the warp-per-row MMVQ for the given qtype:

- arm **E4M3**: `QT_F8_E4M3`, `row_bytes = in_f` → `out_f * in_f` bytes (1.0 B/weight)
- arm **Q8_0**: `QT_Q8_0`, `row_bytes = (in_f/32)*34` → `out_f*(in_f/32)*34` bytes (1.0625 B/weight)

**DRAM-cold discipline**: decode re-reads the whole weight from HBM every tick, so an L2-resident
measurement would be fiction. Each shape allocates `copies = clamp(768 MB / weight_bytes, 1, 64)`
independent weight buffers and both arms rotate through them identically, so consecutive launches
never re-read the same bytes.

The two arms are *not* the same arithmetic (Q8_0: 8 dp4a into s32 per 32-block; e4m3: 8 cvt + 16 fmaf
in f32) — which is why the gap between the measured delta and the byte ceiling is itself the finding.

### Numbers

`ratio = t_q8_0 / t_e4m3`; >1 means e4m3 is faster. Two independent passes inside one lock hold,
N=200 each, medians, 71→73C:

| cell | pass 1 | pass 2 | e4m3 GB/s | q8_0 GB/s |
|---|---|---|---|---|
| down_proj | +7.08 | **+7.15** | 782.5 | 775.9 |
| o_proj | +6.95 | **+6.78** | 693.6 | 690.1 |
| gate/up_proj | +6.12 | **+6.08** | 788.5 | 789.8 |
| square-ref | +5.98 | +6.03 | 665.3 | 666.7 |
| q_proj | +4.81 | **+4.71** | 756.8 | 767.9 |
| k/v_proj | +3.02 | **+3.42** | 345.4 | 354.9 |
| unweighted geomean | +5.65 | +5.69 | | |

**Weighted** (pass 2): total per-layer e4m3 498.81 µs vs Q8_0 528.75 µs → **1.0600x, +6.00pp**.

Per-layer weight bytes: e4m3 372.2 MB vs Q8_0 395.5 MB = 1.0625x → **+6.25pp ceiling**.
**+6.00 / +6.25 = 96% of the theoretical stream advantage realized.**

The time-dominant cells are at or above the ceiling (down +7.15, o_proj +6.78, gate/up +6.08) and
both arms there run 690–790 GB/s, i.e. genuinely bandwidth-saturated — which is exactly why the byte
advantage converts. Only the small `k/v` cell underperforms (+3.42), and it is **not**
bandwidth-bound (345–355 GB/s vs 780+ on the big shapes) — it is launch/occupancy-limited. It carries
5.9% of decode weight, so it cannot pull the weighted result below the bar.

### Verdict

**GO**, and cheaper than the brief assumed: **no kernel authoring** is required for per-tensor-scale
FP8 checkpoints. The remaining work is dispatch/residency — keep e4m3 resident instead of dequanting
to a Q8_0 slab at load — not new CUDA.

**Known gap that scopes this GO**: block-128-scale FP8 checkpoints are **excluded** from the
`MEMRA_ST_E4M3` arm (it requires `blk.is_none()`). They need the per-block-dequant mmvq twin
(`DECISION.md` item B1, scales indexed `[(o>>7)*cols + (e>>7)]`), which is not yet written. So this GO
covers per-tensor-scale checkpoints today; block-128 needs that twin first.

## Known constraint: 27B FP8 e2e does not fit the 24 GB card

Noted per owner directive 3, **not solved here**.

The `PP_FP8` stash keeps the e4m3 bytes resident **on top of** the dequanted Q8_0 slab, so v2 could
only cover 8.7% of tensors under `MEMRA_PP_FP8_BUDGET_MB` before OOM. The honest 27B **e2e** paths
are therefore:

1. **the 96 GB PRO 6000 pod**, where the duplicate fits; or
2. **a native-dispatch build with no resident Q8_0 slab** — which is precisely what the Q2 GO
   enables, since the e4m3 GEMV needs no dequant.

GEMM-only measurement on the local 5090 with 27B shapes — this lane's primary instrument — remains
fully valid, because it allocates only the shapes under test and never the model. Both verdicts above
came out GO, so **both briefs below name the pod as the e2e proving ground.**

## Brief: FP8 MMQ v3 (s32-accumulate prefill) — DO NOT BUILD IN THIS LANE

Funded by Q1's receipt. Written here so the next owner of the work inherits the contract, not a hunch.

**Kernel contract.**
1. Per 128-element scale block, convert the e4m3 weight mantissas into an **int8-compatible product**
   — a representation whose pairwise products are exactly summable in s32.
2. Chain the resulting products through an **s32 accumulator** across the whole 128-block
   (`mma.sync...s32.s8.s8.s32`), never touching f32 mid-chain.
3. **One fold** per 128-block: apply `(s_blk × dB)` once at the block boundary, matching v2's existing
   fold cadence (v2 already folds once per 128; the floor folds every k32 — do not regress to the
   floor's cadence while chasing its accumulator).
4. **Its own host reference is required.** The arithmetic contract is new: it is neither Q8_0's nor
   v2's, so neither existing reference is valid for it. Bit-exactness is against that new reference.
   `fp8-mmq-check` gains an arm; the `run-gen` argmax gate and `run-spec` K=1..8 apply as usual.

**What the receipt does and does not promise.** +16.7pp (m=512) / +23.1pp (m=6257) is the *upper
bound* — the value of the accumulator swap at fixed geometry with mantissa extraction charged at
zero. v3's net is that minus its extraction cost, and the extraction is the entire engineering risk:
if per-128-block mantissa handling costs more than ~13pp at m=6257, v3 lands under the 10pp bar even
though the accumulator itself is worth 23. **The first thing v3 should build is the extraction path
measured in isolation**, against this lane's `accprobe` numbers as the denominator, before any full
kernel.

**Cost to build.** Multi-day, not day-class: a new arithmetic contract, a new host reference, a new
kernel-check arm, plus the extraction micro-benchmark above. Reuse `mmq_fp8_blk.cu`'s tiling and
loaders — the 128-k-block/one-fold structure is already correct and already measured.

**e2e proving ground: the 96 GB PRO 6000 pod.** GEMM-level A/B stays on the 5090 (valid, and the
lane's denominators live there). Any 27B end-to-end prefill claim must be made on the pod, per the
constraint above.

## Brief: decode-v1 (native e4m3 residency) — the cheaper move

Funded by Q2's receipt. This is a **dispatch/residency change, not a kernel campaign.**

1. Keep the FP8 checkpoint's e4m3 bytes resident and dispatch `qmatvec_e4m3_mmvq` directly, instead of
   device-dequanting to a Q8_0 slab at load (ARM B') and paying 1.0625 B/weight forever after.
   Worth **+6.00pp weighted** at m=1, i.e. 96% of the byte ceiling. Kernel and m=1 correctness gate
   already exist.
2. Doing so removes the resident Q8_0 slab — which is option 2 of the 24 GB constraint above, so this
   change *also* opens a local 27B FP8 e2e path that does not exist today.
3. **Prerequisite for block-128 checkpoints only**: write the per-block-dequant mmvq twin
   (`DECISION.md` B1). Per-tensor-scale checkpoints need nothing new.
4. Model-level equivalence between the two *containers* is not a bit-identity question — it is v2's
   teacher-forced disagreement-count + NLL-window protocol
   (`research/fp8st-20260804/mmq-v2/`, scripts reusable as-is).
5. **e2e proving ground: the 96 GB PRO 6000 pod** for the 27B end-to-end decode claim, until item 2
   lands and a no-slab local run becomes possible.

## Recommended next move

**Do decode-v1 first, v3 second.** Q2 buys +6pp of decode for a dispatch change against an already-
gated kernel, and its side effect (no resident Q8_0 slab) unblocks the 27B FP8 e2e story on the 24 GB
card. Q1 buys more headline (+17 to +23pp of prefill) but costs a new arithmetic contract, a new host
reference and an unmeasured extraction path that could eat the entire margin — and its first
deliverable should be the extraction micro-benchmark, not a kernel.

## Artifacts

| file | what |
|---|---|
| `RESULTS.jsonl` | 81 rows — every measured cell (Q1: 48 = 6 shapes × 8 run/m/dist combinations, Q2: 12 = 6 shapes × 2 passes), the instrument descriptions, the SASS validation, the distribution control, both weighted summaries, both verdicts, the OOM constraint |
| `q1-accprobe-27b.log` | Q1 run 1 raw (cold-start ramp) |
| `q1-accprobe-27b-confirm.log` | Q1 run 2 raw, warm/clock-settled — the source of the weighted numbers |
| `q1-dist-control.log` | Q1 e4m3-denormal distribution control, wide vs mid, both m |
| `q1-sass-census.txt` | instrument validation: MMA class/count, instruction census, resource usage — **incomplete: censuses MMA count, never MMA issue rate; see the correction at the top** |
| `q2-gemv-e4m3-27b.log` | Q2 raw, two independent passes |

Code added by this lane (research instruments; no dispatch seam, no default changed):
`crates/memra-engine/cu/mmq_q8_0_f32acc.cu`, `src/bin/accprobe_bench.rs`,
`src/bin/gemv_e4m3_bench.rs`, plus their `build.rs` / `mmq_ffi.rs` / `Cargo.toml` wiring.
