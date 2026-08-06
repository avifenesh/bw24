# Prefill-GEMM rebuild lane (#72) — profile truth and plan re-aim

**Date:** 2026-08-06 · **Branch:** `lane/prefill-gemm` · **Base:** `4a257c8c` (restructure/public-split)
**Rig:** local RTX 5090 Laptop (sm_120a, GB203, 82 SM), clocks locked 1860/1860 MHz, persistence ON,
55 C, one idle co-resident `llama-server` (pid 144655, 332 MiB, 0% util) — recorded, not contending.
**Denominator prompt:** `research/e2e/prompts/pp512.txt` (512 tokens, the prod anchor).
**Receipts:** `RESULTS.jsonl` (4 slices), `logs/`, `nsys/`, `ncu/`, `tools/` (the harness scripts, verbatim).

**Two findings, in order of importance:**
1. The plan's premise is dead — its target kernel does not run, and vendoring already met its
   pass criteria (§"Headline").
2. The replacement ceiling I derived from ncu was **itself wrong by 16x**, and I caught it with a
   probe instead of a multi-day build (§"The fold ceiling was measured"). The lesson there —
   pipe-utilization % is not a speedup ceiling — is the most reusable thing in this document.

---

## Headline: the plan is refuted, and it was refuted by its own Step 1 succeeding

`research/basics/PREFILL-GEMM-REBUILD.md` is the named plan doc. Its Phases 1–4 all edit
`crates/memra-engine/cu/qmatvec_gemm.cu` → `qmatvec_gemm_kernel<QT>`, on the diagnosis that
prefill is **smem-feed-bound by shared-memory bank conflicts** (measured then at 5.3% tensor
pipe / 41.8M conflicts).

**That kernel is no longer the prefill path.** The vendored llama MMQ suite landed and is
default-on (`mmq_w4a8_enabled()`, `mmq_q8_enabled()`, `mmq_q4_enabled()`, `mmq_iq4xs_enabled()`
all default `true` in `crates/memra-engine/src/mmq_ffi.rs`), and it carries prefill on both
deployment-class models. The plan's own Phase-1 **pass criteria were met by vendoring**, not by
the prescribed pad edit:

| Plan Phase-1 gate | Plan's target | Measured today | |
|---|---|---|---|
| bank conflicts | 41.8M → <5M | **6.79M** | 84% of the way, and now irrelevant (see below) |
| tensor pipe | 5.3% → >20% | **60.38%** | **3.0x past the plan's success bar** |

So Phases 1–2 are already banked. Phases 3–4 (FP4 smem tile rebuild, TMA feed probe) attack a
bound this kernel **does not have**.

---

## Where pp512 time actually goes (nsys, `cuda_gpu_kern_sum`, 1 warmup + 3 timed forwards)

### q27 NVFP4 — `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`

| kernel | % GPU time |
|---|---|
| **`mul_mat_q_nvfp4_w4a8<128,128,1,0,1>`** | **77.97** |
| `gdn_chunk_state_f32` | 4.73 |
| `gdn_chunk_output_f32` | 2.13 |
| `qmatvec_nvfp4_dp4a_rp` (the `out_f < 128` tail) | 2.03 |
| `silu_mul_f32` | 1.79 |
| `ssm_conv1d_gdn_f32` | 1.37 |
| `l2_norm_f32` | 1.34 |
| `quantize_mmq_q8_1_d4_kernel` | 1.30 |
| `rms_norm_f32` | 1.12 |
| `fa_prefill_bf16kv_pp` | 1.06 |
| `gdn_chunk_attn_f32` | 1.03 |
| `quantize_q8_1` | 0.96 |
| `gdn_chunk_solve32_f32` | 0.85 |
| all others (11 kernels) | 2.32 |

24 distinct kernels, 1471.26 ms over 4 forwards.

### q9 NVFP4 — `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`

| kernel | % GPU time |
|---|---|
| `mul_mat_q_nvfp4_w4a8<128,128,1,0,1>` | 57.03 |
| `mul_mat_q_q45k<128,0,4>` (Q4_K) | 11.91 |
| `mul_mat_q_q45k<128,0,5>` (Q5_K) | 8.20 |
| **MMQ GEMM total** | **77.14** |
| `gdn_chunk_state_f32` | 3.76 |
| `gdn_chunk_output_f32` | 2.10 |
| `qmatvec_q8_0_dp4a` | 1.93 |
| `ssm_conv1d_gdn_f32` | 1.71 |

### Two things this breakdown settles

1. **GEMM is ~78% on both.** The absolute Amdahl ceiling for *any* GEMM-only work is
   **4.54x on q27** (1536 → 6975 tok/s) and **4.37x on q9** (5185 → 22672) — and that is the
   physically-impossible "GEMM costs zero" bound, not a target.
2. **The non-GEMM 22% is fragmented and offers no lever.** Largest single non-GEMM kernel is
   `gdn_chunk_state_f32` at 4.73%. The whole `gdn_chunk_*` family is 9.4%. Nothing outside the
   GEMM has a slice worth a campaign — which is why this lane correctly points at the GEMM.

---

## The bound, named (ncu, `-s 40 -c 3`, three metric passes)

Target: `mul_mat_q_nvfp4_w4a8<128,128,1,0,1>` — grid (136,4,1), block (32,8,1) = 8 warps.

| | value | reading |
|---|---|---|
| Compute (SM) throughput | **57.06 / 56.76 / 58.12 %** | |
| **tensor pipe** | **60.38 %** | top pipe by 2x |
| fma pipe | 25.45 % | |
| alu pipe | 20.63 % | |
| lsu pipe | 31.73 % | |
| **DRAM throughput** | **9.38 / 9.86 / 8.11 %** | **not bandwidth-bound** |
| L1/TEX | 32.11 % | |
| L2 | 17.35 % | |
| achieved occupancy | **16.66 %** (theoretical 16.67) | **no gap to reclaim** |
| regs/thread | 252 (of 255) | Block Limit Registers = **1** |
| dyn smem/block | 98,816 B | Block Limit Shared Mem = **1** |
| bank conflicts (shared, LSU) | 6.79M | |
| issue active | 52.16 % | IPC 2.10, warp cycles/issued inst 3.83 |
| waves/SM | 6.63 (1.95 on the small launch) | |

**Stall ranking** (warps stalled per issue-active):

```
math_pipe_throttle  1.25   <-- dominant, 2.1x the runner-up
wait                0.59
mio_throttle        0.22
not_selected        0.22
dispatch_stall      0.21
short_scoreboard    0.16
barrier             0.11
long_scoreboard     0.04   <-- global-memory dependency: ~ZERO
lg_throttle         0.01
no_instruction      0.01
```

**Instruction mix** (launch ID 0): 307,934,464 total instructions =
tensor 22,282,240 · FMA 136,574,464 · ALU 120,807,168 · LSU 23,422,464.
Per warp-MMA: **6.13 FMA + 5.42 ALU + 1.05 LSU = 13.8 non-tensor instructions per MMA.**

**Achieved rate:** 22.28M warp-MMAs × (16·8·16 MACs) × 2 = 9.13e10 int8 ops in 1.0297 ms
(1,915,309 cycles/SM at 1860 MHz) = **88.6 TOP/s = 40.5% of the 219 TOP/s s8 peak.**

### What it is

**Issue-slot competition against a busy tensor pipe, caused by the f32 scale-fold.** From
`vec_dot_nvfp4_w4a8_mma` in `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu`:

```c
for (int n = 0; n < ntx; ++n) {
    tile_C C[2];
    mma(C[0], A[n][k01 / 4 + 0], B[0]);
    mma(C[1], A[n][k01 / 4 + 1], B[1]);
    for (int l = 0; l < tile_C::ne; ++l) {
        sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
            dB[l % 2] * (C[0].x[l] * dA[n][l / 2][k01 / 4 + 0] + C[1].x[l] * dA[n][l / 2][k01 / 4 + 1]);
    }
}
```

Two `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` per k01 step, then 4 accumulator elements ×
(2 s32→f32 converts + 2 scale muls + 1 `dB` mul + 1 add). The measured 6.13 FMA / 5.42 ALU per MMA
is exactly that shape. The tensor pipe sits at 60.38% busy and **every hole in it is a fold**.

### What it is NOT — four levers refuted by this one profile

- **Not memory / not feed.** DRAM 8–10%, `long_scoreboard` 0.04. A TMA or cp.async feed rebuild
  (plan Phases 3–4) has no slice to attack. Independently consistent with
  `research/tma-oddm-20260805/VERDICT.md` ("no staging tax"). A cp.async Marlin-style arm
  *already exists* behind `MEMRA_PP_PIPE=1`.
- **Not bank conflicts.** 6.79M, and `mio_throttle` is 0.22 against `math_pipe_throttle` 1.25.
  The plan's smem-pad edit would chase a 0.22-weight stall while the 1.25 one goes untouched.
- **Not occupancy.** 16.66 achieved vs 16.67 theoretical — nothing to reclaim — and both smem
  (98,816 B) and registers (252) independently pin it at 1 CTA/SM, so raising it means *shrinking
  the tile*. And a math-pipe-throttled kernel gains nothing from more warps: they queue on the
  same busy pipe. `research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl` already refuted the
  occupancy lever in both its forms, on a sibling kernel, with receipts.
- **Not "widen K" / "kill the repack"** (memory-doc Steps 2–3). `MMQ_ITER_K` is *already* 256,
  and the vendored `load_tiles_nvfp4_w4a8` is already a pure LUT dequant-to-smem with no per-K
  repack ring. Both steps describe work that vendoring already did.

---

## The ceiling, stated before building anything (the Amdahl discipline)

> **⚠ MEASURED AND REFUTED — see §"The fold ceiling was measured" below. The 1.656x / 1.447x
> figures in this section are WRONG by 16x. A probe that deleted the entire fold moved pp512
> by 3.17%, not 44.7%. The arithmetic is kept here, with its refutation, because the *error*
> is the transferable finding.**

If the f32 fold cost **went to zero** the tensor pipe would go 60.38% → 100%, so the kernel
speedup ceiling is `1/0.6038 = **1.656x**`:

| | kernel | e2e | pp512 |
|---|---|---|---|
| q27 (78.0% GEMM), fold-free | 1.656x | **1.447x** | 1536 → **2224** |
| q9 (77.1% GEMM), fold-free | 1.656x | **1.440x** | 5185 → **7469** |

**This 1.656x is the hard cap on every fold, scheduling, retile, feed, and occupancy lever
combined.** Anything claiming more must change the MMA instruction form.

Partial-credit projections on q27 (base 1536.5):

| kernel speedup | e2e | pp512 |
|---|---|---|
| 1.10x | 1.076x | 1654 |
| 1.20x | 1.149x | 1766 |
| 1.30x | 1.219x | 1874 |
| 1.656x (fold-free, the cap) | 1.447x | 2224 |

---

## The structural finding — why the 2x is not a scheduling problem

The obvious "go wider" move is `m16n8k32` instead of `m16n8k16` (2x the s8 MMA rate on sm_120).
**It is architecturally unavailable to NVFP4 in this kernel, and the code already shows why:**

NVFP4 carries a **UE4M3 block scale per 16 elements**. That caps the K span of a single
accumulate at 16. The two adjacent k16 halves live in `C[0]` and `C[1]` as *separate*
accumulators precisely because `dA[n][l/2][k01/4+0] != dA[n][l/2][k01/4+1]`. Merging them into
one k32 MMA requires one shared scale across 32 K, which the format does not have.

So the 2x is **not** reachable by retiling, padding, pipelining, or warp specialization. It sits
behind exactly two doors:

1. **Integerize the fold** so the chain accumulates in s32 and the block scale folds once —
   this is verbatim the untried lever that `research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl`
   left on the table (`next_lever_if_anyone_resumes`). Two independent lanes converged on it
   from different kernels. Bounded above by the 1.656x cap.
2. **`mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale`** — the 762 TFLOP/s FP4 form where
   the block scales are *hardware MMA operands*, not epilogue math. This is the only path that
   exceeds 1.656x, because it removes the fold *and* the instruction rate limit at once. It is
   also a much larger build (new tile layout, `scale_vec` operand staging, exactness re-proof).

---

## The fold ceiling was measured — and it is 3.17%, not 44.7%

Rather than spend a multi-day s32-chain build against the 1.656x number above, I bounded the
**entire** fold-removal lever family with one compile-time probe (`MEMRA_MMQ_FOLD_CEILING=1`)
that *deletes* the fold: the per-`k01` `dB*(C0*dA0 + C1*dA1)` becomes `s32acc += C0 + C1`,
drained once outside the k-loop with one `dA` and one `y_df` multiply so both scale loads stay
live and nvcc cannot hoist them. MMAs, ldmatrix feed, smem tile, and geometry are identical.
Numerically wrong by construction; default OFF; **naked build provably untouched — `cuobjdump
-sass` before vs after the source edit is 0 diff lines over 174,084.**

Interleaved BASE/CEIL, 3 rounds × 5 reps = **N=15 per arm**, two separately built binaries,
clocks locked 1860, q27 NVFP4, 53 → 70 C:

| arm | median | all 15 reps |
|---|---|---|
| BASE | **1395.2** tok/s | 1394.2 – 1395.7 |
| CEIL (no fold) | **1439.4** tok/s | 1438.1 – 1440.2 |

**+3.17% e2e = 1.041x kernel**, against 1.447x / 1.656x predicted → **ncu overpredicted 16x.**

So the fold is **not** the bound. It co-issues in the MMA's shadow and is already almost
entirely hidden. **Do not build the s32-chain lever.** This also retires the
`next_lever_if_anyone_resumes` note in `research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl`,
which proposed exactly this on the sibling FP8 kernel — two lanes converged independently on a
lever worth 3%.

### Why the arithmetic lied (the transferable lesson)

`1/utilization` assumes the idle 39.6% of tensor-pipe cycles are *blocked by* the competing
pipes. They are not:

```
sm__pipe_tensor_cycles_active.sum / smsp__inst_executed_pipe_tensor.sum
  = 356,515,840 / 22,282,240 = EXACTLY 16.00 cycles per warp-MMA
```

16.00 is the **hardware issue interval** of `m16n8k16.s8`, not a queueing artifact. With 8
warps/CTA at 1 CTA/SM, each scheduler owns **2 warps** against a 16-cycle pipe — so the idle
cycles are **MMA latency exposure from thin warp parallelism**, and the FMA (25.45%) and ALU
(20.63%) pipes run *concurrently inside that exposure, for free*. Removing concurrent work
from a latency-exposed pipe recovers only the sliver that was genuinely serialized.

> **⚠ CORRECTED BY PHASE 2 (2026-08-06) — the "latency exposure" half of that paragraph is
> WRONG; the conclusion it supports is right for a different reason.** Receipts:
> `research/prefill-ilp-20260806/VERDICT.md`.
> Phase 2 measured the interval directly (two instruments, `clock64` per-CTA + full-GPU
> `cudaEvent`, agreeing to 0.4%) with an **NACC=1..16 ILP control**:
> - cycles/warp-MMA **floors flat at 16.06 from NACC=2 through NACC=16** — so it is an issue
>   interval, confirmed.
> - **NACC=1 exposes the real latency: 27.1 cyc (s8) / 29.1 (block-scale)** — and **two**
>   independent accumulators already hide it completely. This kernel has **four**.
> - At the shipped 8 warps/CTA the pipe measures **31.8-32.0 cyc/MMA = 2 warps x 16** =
>   **100% issue-saturated**.
>
> So the idle tensor-pipe cycles are **not** latency exposure and **not** thin warp parallelism —
> the pipe is *issue-bound*, and the FMA/ALU work co-issues in the gaps between issue slots. The
> practical consequence is the same "removing concurrent work buys almost nothing" (+3.17% stands),
> but the mechanism matters: it means **more warps, more CTAs, and more per-warp accumulators are
> all closed by mechanism**, not merely flat in a config sweep. And 16.00 is **the GB203 tensor
> pipe's interval, not `m16n8k16.s8`'s**: `m16n8k64.mxf4nvf4` and `m16n8k32.mxf8f6f4` measure the
> same 16.06 — which is exactly why the wider-K door is worth a real 4x (measured 3.989x).

**LESSON: `pct_of_peak_sustained` pipe utilization is not a speedup ceiling.** Convert to
cycles-per-instruction and check the issue interval before quoting any `1/utilization` figure.
This lane quoted 1.656x in good faith and it was worth 1.041x.

**LESSON 2 (added by phase 2): a cycles-per-instruction number is not an issue interval until an
ILP control says so.** 16.00 alone cannot distinguish "the pipe's issue interval" from
"latency / accumulators-in-flight" — sweep the independent-accumulator count and see whether it
floors or halves. Skipping that control is precisely how this section arrived at a latency story
for an issue-saturated pipe.

~~Corollary worth noting: at **88.6 TOP/s**, if the 219 TOP/s figure is the sparsity-enabled
number then the dense s8 peak is 109.5 and this kernel is already at **80.9% of dense peak**~~
— **SUPERSEDED by phase 2's direct measurement: dense s8 peak is 155.0 TOP/s** (full-GPU
`cudaEvent`, 82 CTAs x 256 thr, best of 5, clocks locked), so the live kernel sits at **57.2% of
measured dense peak**. Do not infer a dense peak by halving a nameplate figure — measure it. The
"3% fold ceiling" conclusion is unaffected (it was measured independently).

---

## Verdict

- **Plan steps 2–5 as written: REFUTED**, with receipts. Wrong file (`cu/qmatvec_gemm.cu` does not
  execute on either deployment model), wrong bound (feed/conflicts/occupancy vs the measured
  math-pipe throttle), and steps 2–3 describe work vendoring already completed.
  `research/basics/PREFILL-GEMM-REBUILD.md` and the memory doc both need this correction before
  anyone spends a day on a pad edit.
- **Profile truth: DELIVERED.** GEMM is 78.0% (q27) / 77.1% (q9) of pp512. The dominant kernel
  runs at 88.6 TOP/s = 40.5% of s8 peak, 60.38% tensor pipe, `math_pipe_throttle` 1.25 dominant,
  DRAM 9%, occupancy pinned at 1 CTA/SM by both smem and regs.
- **Gap state, honest.** q27 pp512 = **1536.5 tok/s** naked (N=3, clock-locked 1860 MHz);
  q9 = **5184.7** (N=3). No code changed this dispatch, so **before == after**. The frozen llama
  reference (pp512 5451) is orientation only; the live metric is memra-vs-memra.
- **Battery: not run** — correctly, because no code changed. There is nothing to gate.
- **Re-aim, corrected by measurement:** the fold lever is **dead** (3.17%, receipted above).
  The bound is the **m16n8k16.s8 issue interval (16.00 cycles/MMA) at 2 warps/scheduler** —
  i.e. MMA *latency exposure*, not fold cost, not feed, not bandwidth. That leaves exactly two
  candidate directions, and **both need their ceiling measured by probe before any build**,
  because this lane just demonstrated that derived ceilings can be 16x wrong:
  1. **More MMAs in flight per scheduler.** The config axes are already swept and closed
     (`research/tune-data/rig5090.jsonl` 2026-07-06: X 32/128/256, Y 64/128/192 all
     flat-to-negative; Y=64 gives 2 CTA/SM but halves warps/CTA and cancels). What is *not*
     swept is ILP *within* a warp — more independent accumulators per warp so the 16-cycle
     interval is filled without more warps. Cheap to probe.
  2. **A wider/faster MMA form.** `m16n8k32` is architecturally unavailable (NVFP4's UE4M3
     scale per 16 elements caps one accumulate's K span at 16 — that is *why* `C[0]`/`C[1]`
     are separate accumulators). The only real door is
     `mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale`, where block scales are hardware
     MMA operands. That changes the issue interval itself, which is the thing actually binding.
     Large build (new tile layout, `scale_vec` staging, exactness re-proof) — worth a
     feasibility probe on the instruction in isolation before committing to it.
  - Also still un-probed and explicitly named by the July 6 row as "the one remaining prefill
    card": the Marlin-style x-stage restructure (cp.async raw-weight ring + dequant-from-smem).
    Note the profile is *hostile* to it — DRAM 8-10%, `long_scoreboard` 0.04 — so it should be
    ceiling-probed, not built on faith.
  - **Owner call requested** on which of these gets the next dispatch.

> **PHASE 2 ANSWERED BOTH, AND BOTH ARE CLOSED (2026-08-06).** Dispatched as `lane/prefill-ilp`;
> receipts `research/prefill-ilp-20260806/VERDICT.md` + `RESULTS.jsonl`. Neither needed a kernel
> build — one died to SASS, one to ptxas plus a microbench.
>
> 1. **Intra-warp ILP — REFUTED AT ITS PREMISE.** All **256** IMMAs in the shipped
>    `mul_mat_q_nvfp4_w4a8<128,128,1,0,1>` accumulate into **RZ** (`grep 'IMMA' | grep -v ', RZ ;'`
>    = 0 matches). There is no dependent chain to break: the source already declares `tile_C C[2]`
>    fresh inside the `n`-loop with `ntx=2`, so a warp owns **4 independent C tiles** per k01 step —
>    and the NACC control shows **2 already saturate**. Register budget forbids it independently
>    (252/255 regs, **six of eight** siblings already spilling; wider C = +16, fragment
>    double-buffer = +32). The instruction accounting closes exactly (256 IMMA / 16 ldmatrix /
>    1024 I2FP + 1024 FFMA), so the audit is complete rather than sampled.
> 2. **mxf4nvf4 — real, 4x confirmed, ALREADY BUILT, blocked on precision.** Exposed on sm_120a
>    (`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`); NVFP4's UE4M3-per-16 matches `scale_vec::4X` with
>    **zero weight repack**; the paper 4x is exact (**measured 3.989x**, 618.5 TFLOP/s). But
>    `cu/mmq_fp4.cu` **already implements it** behind `MEMRA_MMQ=1` and measures **1.9685x e2e** on
>    q27 pp512 (N=15 interleaved, 2591.2 vs 1316.3 tok/s). ptxas proves k64 accepts `e2m1 x e2m1`
>    **only**, so the door *requires* FP4 activations and its exactness block
>    (`research/w4a4-rescue-20260803/`) is **structural to the operand grammar** — which is why
>    four kernel revisions could not fix it and why k=0 diverges too.
> 3. **The Marlin x-stage card is closed too**, without probing it: a restructure that changes
>    *when bytes arrive* cannot help a pipe measured at **100% issue-saturated**.
>
> **Revised re-aim: prefill is DONE as a kernel-engineering target.** Same-instruction headroom is
> a few percent and the cheap levers are spent. The only remaining 2x is FP4-activation accuracy
> recovery — a **quality** deliverable, not a kernel one. That is the open owner call now.

### Measurement note that is itself a finding

Before the clock lock, q27 pp512 drifted **9% inside a single process** (1601.0 / 1468.4 / 1536.5).
That is larger than most levers this lane would claim. `sudo nvidia-smi -pm 1` +
`-lgc 1860,1860` made reps deterministic (0.3690 / 0.3691 / 0.3690 s). **1860 MHz is below the
3090 MHz boost ceiling, so locked absolute numbers read lower than free-clock ones — the locked
value is the only valid A/B denominator, and free-clock numbers must never be mixed with it.**
