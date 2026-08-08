# PTX MMA mnemonic RATE audit — every site in `crates/memra-engine/cu/`

**Task #81. Rig: RTX 5090 Laptop, sm_120a, 82 SMs. Clocks locked 1860/1860, `flock`'d, released after.**

The bug class: **two PTX forms that compute the identical math at different issue rates.** No
correctness gate can see it — the outputs are bit-identical by construction, so `kernel-check`,
the argmax gate and `run-spec` all pass while the kernel runs at half speed. It has already cost
this repo three live kernels and one published-and-wrong GO verdict, and the rate that would have
caught it **was already measured a month earlier** in `research/sm120-empirical-capabilities.md`
and ignored, because a rate that lives only in prose gets re-picked wrong.

So this audit has two products: the site table below, and — the durable half —
`sm120-empirical-capabilities.md` promoted to **THE canonical rate table**, with a pointer comment
at all 22 asm sites.

## Verdict summary

| | count |
|---|---|
| inline-PTX MMA asm sites inventoried | **22** (in 14 files) |
| distinct mnemonic forms | **12** measured + 8 ISA-absent (verified) |
| **wrong form on a live path** | **0** |
| **wrong form total (incl. dead code)** | **1** (`mmq_nvfp4_f8f4.cu:51`, uncalled) |
| SWAP-AVAILABLE (faster equal-math sibling exists) | **2** (both int8 k16) |
| OPTIMAL | 10 |
| NOT-APPLICABLE (no equal-math sibling exists) | 9 |
| DEAD-DOOR | 1 |

**Headline: the live paths are clean.** The two prior lanes' fixes closed the FP8 family, and
nothing else on a default path issues a slow form. But the audit found a **second instance of the
same bug class in a different family** — the int8 pipe is K-free, so both `m16n8k16.s8` sites are
running at half the available int8 rate for the identical product. That is a real, measured,
un-taken 1.42x on those two tiles, and it is *not* a defect the prior lanes' finding predicted.

---

## 1. The rate table (method + results)

Full table, mechanisms, and the ISA-absence list now live in
**`research/sm120-empirical-capabilities.md` § CANONICAL MMA RATE TABLE** — cite that, not this
file, for rates. Condensed here for the site table's sake:

| form | cyc/warp-MMA | MACs | delivered | vs A1 |
|---|---|---|---|---|
| A1 `m16n8k16.s8.s8.s32` | 16.06 | 2048 | 155.2 TOP/s | 1.000x |
| **A2 `m16n8k32.s8.s8.s32`** | **16.06** | 4096 | **309.7 TOP/s** | **1.997x** |
| B1 `m16n8k16.f32.bf16.bf16.f32` | 32.03 | 2048 | 77.7 TFLOP/s | 0.500x |
| B2 `m16n8k16.f32.f16.f16.f32` | 32.03 | 2048 | 77.8 TFLOP/s | 0.501x |
| **B3 `m16n8k16.f16.f16.f16.f16`** | **16.10** | 2048 | **155.2 TFLOP/s** | **1.001x** |
| B4 `m16n8k8.f32.tf32.tf32.f32` | 32.03 | 1024 | 38.9 TFLOP/s | 0.250x |
| C1 `kind::f8f6f4` plain e4m3×e4m3 | 32.03 | 4096 | 155.5 TFLOP/s | 1.002x |
| **C2 `mxf8f6f4.block_scale.1X` ue8m0 e4m3×e4m3** | **16.06** | 4096 | **309.3 TFLOP/s** | **1.99x** |
| C3 `kind::f8f6f4` plain e2m1×e4m3 | 32.03 | 4096 | 155.4 TFLOP/s | — |
| C4 `mxf8f6f4.block_scale.1X` ue8m0 e2m1×e4m3 | 16.06 | 4096 | 309.6 TFLOP/s | — |
| **D1 `m16n8k64.mxf4nvf4.block_scale.4X` ue4m3** | **16.06** | 8192 | **619.2 TFLOP/s** | **3.99x** |
| D2 `m16n8k64.mxf4.block_scale.2X` ue8m0 | 16.06 | 8192 | 619.1 TFLOP/s | 3.99x |

Method: `clock64()` over mutually-independent MMAs, **NACC 1..16** (flat ⇒ pipe *issue interval*,
not latency), converted by full-GPU `cudaEvent` at the shipped 82×256 shape. **3 reruns within
0.5%.** Every arm's **SASS MMA count verified** by `cuobjdump -sass` — non-negotiable, see §5.
Probe `tools/rate_audit.cu`, raw `logs/rate-audit-12form.log`.

**Two findings beyond the inherited f8f4 one:**

- **NEW A — the int8 tensor pipe is K-FREE.** A1 and A2 cost the *same* 16.06 cyc for *twice* the
  depth ⇒ k32 delivers **1.997x** the MACs for the identical product. Every k16 int8 site runs at
  half rate. This is the f8f4 bug class, in the int8 family.
- **NEW B — 16-bit float with f32 accumulate is the slowest tensor path on this silicon.** B1 and
  B2 are both 32.03 cyc / ~77.7 TFLOP/s vs B3's 16.10 / 155.2 = exactly **2.0x**. This *measures*
  the f32-accumulate throttle the capabilities doc previously only *inferred* from plain-vs-block
  FP8, and shows it taxes bf16 and f16 identically — the operand format is free, the **accumulator**
  costs 2x.

Corollaries: the KIND carries the FP8 cost, not the operand format (C3/C4 track C1/C2 exactly);
FP4 scale granularity is free (D1 ≡ D2); tf32 (B4) is the slowest form in the repo.

**ISA sibling oracle — all 7 deeper-K candidates REJECTED by ptxas** (`tools/isa_sibling_check.cu`,
`logs/isa-sibling-check.log`): bf16 k32, f16 k32, bf16 `.block_scale`, s8 k64, f8f6f4-blocksc k64,
f16-accum k32, mxf4nvf4 k128. Plus **tf32 has no m16n8k16 shape** (*"Illegal instruction types
specified for '_mma' with shape '.m16n8k16'"*; ISA offers only `.m16n8k4`/`.m16n8k8`).
⇒ **the k16→k32 int8 lift is the only depth lever the ISA offers**, and B1/B2 have **no deeper form
to escape to** — any bf16/f16 remedy must be an *accumulator* change, never a depth change.

---

## 2. The site table

Live? column: **HOT** = reached on a default config; **model-gated** = default-on but only for that
quant class; **fallback** = only when no specialized tile exists; **cold** = compiled, default-off
door; **never** = emits no work in the shipped build.

All line numbers are **post-annotation** (i.e. as of commit `9fd00b3f`+, which added the pointer
comments and therefore shifted every site down — the numbers in the earlier lane commits' messages
are pre-annotation).

| # | site (file:line) | form | rate | live? | faster equal-math sibling? | verdict |
|---|---|---|---|---|---|---|
| 1 | `mmq_nvfp4_w4a8.cu:219` | A1 int8 k16 | 16.06 | **HOT** — default NVFP4 prefill tile (`mmq_w4a8_enabled()`: `MEMRA_MMQ_W4A8` != "0", **default true**) | **YES, A2** (1.997x instr / 1.42x tile) — but *not a drop-in*: distinct per-16 scale | **SWAP-AVAILABLE (blocked on scale-fold restructure)** |
| 2 | `mmq_iq_experts.cu:157` | A1 int8 k16 | 16.06 | **HOT (model-gated)** — IQ4_XS/IQ3_S/Q4_0 expert MMQ, no env gate (dispatched whenever the expert qtype matches) | **YES, A2** — and the scales are provably equal across each 16-pair ⇒ candidate bit-identical | **SWAP-AVAILABLE (worth ≤1.42x of this kernel's MMA-bound share; needs e2e measurement)** |
| 3 | `mmq_q8_0.cu:152` | A2 int8 k32 | 16.06 | HOT — Q8_0 prefill | no (k64 rejected) | **OPTIMAL** |
| 4 | `mmq_q45k.cu:157` | A2 int8 k32 | 16.06 | HOT — Q4_K/Q5_K prefill | no | **OPTIMAL** |
| 5 | `mmq_q4_0.cu:164` | A2 int8 k32 | 16.06 | HOT (model-gated) — gemma QAT Q4_0 | no | **OPTIMAL** |
| 6 | `qmatvec_gemm.cu:168` | A2 int8 k32 | 16.06 | **fallback** — Q6_K only (mmq has no Q6_K tile) | no | **OPTIMAL** |
| 7 | `mmq_fp8_blk.cu:256` | C2 blocksc | 16.06 | HOT (model-gated) — FP8-blk models via `fp8_blk_mmq_native_enabled()` (`MEMRA_FP8_MMQ` != "0", **default ON**) | already the fast form | **OPTIMAL** (fixed by lane/rp-on-st: +6.54% e2e) |
| 8 | `mmq_fp8_blk.cu:251` | C1 plain | 32.03 | **never** — `#ifdef MEMRA_FP8BLK_PLAIN_MMA` | n/a | **DEAD-DOOR** (rollback seam, deliberate) |
| 9 | `mmq_nvfp4_w4a8.cu:1099` | C2 blocksc | 16.06 | cold — `MEMRA_MMQ_F8F4=1` | already the fast form | **OPTIMAL** (fixed by lane/w4a8-prefill: 1.2153x e2e) |
| 10 | `mmq_nvfp4_w4a8.cu:1094` | C1 plain | 32.03 | **never** — `#ifdef MEMRA_F8F4_PLAIN_MMA` | n/a | **DEAD-DOOR** (rollback seam) |
| 11 | `mmq_nvfp4_f8f4.cu:51` | **C1 plain** | **32.03** | **never** — function is **UNCALLED** | **YES, C2** (1.994x, bit-identical) | **DEAD-DOOR** — the 4th instance of the fixed defect; see §3 |
| 12 | `mmq_q8_0_f32acc.cu:157` | A2 int8 k32 | 16.06 | **never** on a serving path — probe-only (`bin accprobe_bench`) | no | **OPTIMAL** |
| 13 | `mmq_q8_0_f32acc.cu:200` | C2 blocksc | 16.06 | probe-only | already fast | **OPTIMAL** (fixed by lane/rp-on-st; the fix *inverted* the published Q1 verdict, +17.6pp → −8.8pp) |
| 14 | `mmq_q8_0_f32acc.cu:195` | C1 plain | 32.03 | **never** — `#ifdef MEMRA_ACCPROBE_PLAIN_MMA` | n/a | **DEAD-DOOR** (receipt-reproduction door — must stay) |
| 15 | `mmq_fp4.cu:194` | D1 fp4 | 16.06 | cold — W4A4 `MEMRA_MMQ=1` | no (k128 rejected) | **OPTIMAL** |
| 16 | `qmatvec_gemm.cu:1243` | D1 fp4 | 16.06 | cold — double-gated | no | **OPTIMAL** |
| 17 | `flash_attn.cu:988` | **B3 f16-acc** | **16.10** | **HOT** — `MEMRA_FA_F16PV` default ON (lib.rs:352, "DEFAULT since 2026-07-23 stamp v4"), P@V accumulation | none faster exists | **OPTIMAL** (the rate half of why f16pv is default) |
| 18 | `flash_attn.cu:160` | B1 bf16-f32acc | 32.03 | **HOT** — the hottest attention MMA (KQ) | **no equal-math sibling** (bf16 k32 + blocksc both rejected) | **NOT-APPLICABLE** — see §4 |
| 19 | `moe_f16_grouped.cu:365` | B2 f16-f32acc | 32.03 | **HOT** — default MoE grouped prefill GEMM | **no equal-math sibling** | **NOT-APPLICABLE** — see §4 |
| 20 | `hybrid.cu:1508` | B1 bf16-f32acc | 32.03 | HOT (model-gated) — GDN/K4 path | no | **NOT-APPLICABLE** |
| 21 | `hybrid.cu:1518` | B2 f16-f32acc | 32.03 | HOT (model-gated) — K4/K5 coupled channel | no; and the site documents needing 11 mantissa bits | **NOT-APPLICABLE** |
| 22 | `mma_tile.cuh:132` | B1 bf16-f32acc | 32.03 | **never** — **DEAD FILE** (zero `#include` from any `.cu`/`.cuh`) | no | **NOT-APPLICABLE** |
| 23 | `wgmma_common.cuh:35` (bf16), `:67` (tf32) | wgmma | — | **never** — sm_90a-only ISA; gated by `MEMRA_K45_REAL` (an internal `__CUDA_ARCH__ == 900` macro, **not** a user flag) | n/a | **NOT-APPLICABLE** (absent silicon) |
| 24 | `fa3_prefill.cu:53`, `:67` | wgmma bf16 | — | **never** — `build.rs` compiles with `-DMEMRA_FA3_STUB` on sm_120a | n/a | **NOT-APPLICABLE** (absent silicon) |
| 25 | `qmatvec_gemm.cu:1583` | wgmma s8 | — | **never** — double-gated off | n/a | **NOT-APPLICABLE** (absent silicon) |

(22 asm *sites* in 14 files — rows 1-22 are one asm statement each, and rows 23-25 group the 5
wgmma asm statements that share one verdict. `dp4a` is out of scope per charter — it is not
form-ambiguous. Re-derivable with:
`grep -n '"mma\.sync\.aligned\|("mma\.sync\|"wgmma\.mma_async' crates/memra-engine/cu/*.cu crates/memra-engine/cu/*.cuh`.)

**A gate-name trap worth recording** (it nearly mislabeled row 7): `MEMRA_FP8_MMQ` reads **two
different gates** in `fp8_ffi.rs`. `fp8_mmq_enabled()` (:239) is `== "1"`, **default OFF** — it
admits the duplicate e4m3 *stash*. `fp8_blk_mmq_native_enabled()` (:275) is `!= "0"`, **default
ON** — it is what actually dispatches this tile on a native `QT_F8_E4M3_BLK` tensor. Same env name,
opposite defaults, and only the second one makes site 7 live. (`MEMRA_PP_FP8`, which an earlier
trace named here, is a different seam entirely.)

---

## 3. The one wrong form: `mmq_nvfp4_f8f4.cu:51`, and why it is NOT fixed here

`mma_f8f4_16x8x32` issues **plain `kind::f8f6f4`** — 32.03 cyc where the ue8m0-identity
`block_scale` form gives the bit-identical product at 16.06. It is textbook, the same defect fixed
at sites 7 and 9.

**It is left alone deliberately, and this is the honest reason:** the function is **uncalled**. The
file's only live exports are `memra_mmq_nvfp4_f8f4_act_bytes` and
`memra_mmq_nvfp4_f8f4_quantize_act` — the e4m3 *activation quantizer*. The actual f8f4 GEMM tile
lives in `mmq_nvfp4_w4a8.cu`, which already runs the fast form. Swapping it would change **zero
emitted instructions** and produce a fake "fix" commit. Instead the site now carries an explicit
in-code warning: *"If this function is ever wired to a kernel, SWAP IT FIRST."*

This is also the fourth time this exact mnemonic pair has been found wrong in this repo. The
countermeasure is not a fourth fix, it is §6.

---

## 4. Why the bf16/f16 f32-accumulate sites are NOT-APPLICABLE, not "swap available"

Sites 18-22 run at 32.03 cyc — half the rate of site 17's f16-accumulate form. That is the largest
single rate gap on any HOT path in the repo, so it deserves a precise verdict rather than an
attractive one:

- **There is no equal-math sibling.** ptxas rejects bf16 k32, f16 k32, and bf16 `.block_scale`.
  There is no deeper or scaled form to move to. The charter's question — *"does a faster equal-math
  sibling exist?"* — answers **no**.
- **The only lever is the accumulator, and that is a NUMERIC change, not a swap.** f16 accumulate
  doubles the rate but changes the results. It is not in this audit's scope, and it must never be
  sold as "free 2x".
- **It is already taken where it is safe.** `MEMRA_FA_F16PV` (default ON) is exactly this lever,
  applied to the *P@V* accumulation — bounded, post-softmax operands (0 ≤ p ≤ 1). KQ, softmax and
  the final normalize deliberately stay f32.
- **Where it is not taken, that is a defensible reason, not an oversight.** Site 19 accumulates a
  full FFN reduction over `in_f`, where f16 accumulate would lose mantissa; site 21 documents
  needing 11 mantissa bits (bf16's 8 compounded K4→K5 error past the config pin).

So the correct reading of NEW FINDING B is: **32.03 cyc is the price of f32 accumulation on this
silicon, not a wrong mnemonic choice.** Recorded as a mechanism, not booked as an opportunity.

---

## 5. What the k16→k32 swap is worth, and its legality per site

**Instruction bound 1.997x. Tile-level reality 1.42x.** A real MMQ tile also pays the scale fold,
and the k16 form folds **twice** as often (2 C tiles, 2 dA loads, 2 FMAs per element). Both inner
loops replicated verbatim and measured: **0.713 → 0.502 ms full-GPU** (82 CTA × 256 thr, NACC=8,
3 reruns bit-identical) = **1.42x, ~71% of the bound**; the missing 29% is the fold arity.
(`tools/k16_vs_k32_tileloop.cu`, `logs/k16-vs-k32-tileloop.log`.)

Legality differs per site, and it turns entirely on **whether the per-16 scales differ inside a
32-k window** — because one k32 MMA sums both halves inside the s32 accumulator *before* any scale
can be applied:

**Site 1 — `mmq_nvfp4_w4a8.cu:219`: NOT a drop-in.** NVFP4 carries a genuinely distinct scale every
16 k-values: `x_df[i*MMQ_MMA_TILE_X_K_NVFP4 + ksc + sub] = ggml_cuda_ue4m3_to_fp32(src_d[sub])`
written per-`sub` (:284, :432), read as `dA[n][l][k01/4] = x_df[... + k0/4]` (:482), and the fold
(:528) applies them separately:
`dB[l%2] * (C[0].x[l]*dA[n][l/2][k01/4+0] + C[1].x[l]*dA[n][l/2][k01/4+1])`. Merging the MMAs would
sum two differently-scaled halves. A k32 lift needs the scale fold restructured (or the scales
folded into the operands) — **not a one-line swap**, so out of this lane's scope.

**Site 2 — `mmq_iq_experts.cu:157`: candidate bit-identical.** All three loaders index the 16 x_df
slots through the **32-block id** `s>>1`, so slots 2m and 2m+1 provably hold the **same** value:
- IQ4_XS (`load_tiles_iq4xs`, :206-209): `int g = s>>1` ⇒ `x_df[...+s] = d_sb * (float)(ls-32)`
- Q4_0 (`load_tiles_q4_0`, :241-243): `int g = s>>1` ⇒ `x_df[...+s] = half_to_float(*(const uint16_t*)(sb + g*18))`
- IQ3_S (`load_tiles_iq3s`, :282-285): `int ib32 = s>>1` ⇒ `x_df[...+s] = d * (1.0f + 2.0f*(float)sc_nib)`

The header comment states the invariant directly: *"Per-32 scale (group c>>3) replicated into the 2
per-16 x_df slots of that group"* (:182-183).

And the **fold reads exactly such a pair**: the k-loop at :324 steps `k01` by 8, so `k01/4` is
always even and `dA[..][k01/4+0]`/`[+1]` (:337) are slots 2m, 2m+1 of one 32-block — the two slots
just shown to be equal. Since s32 MMA accumulation is **exact**, merging the two k16 MMAs (:333-334)
into one k32 MMA under that shared scale changes no bit. This is the one genuine drop-in candidate
in the repo.

(One correction to my own first pass, worth recording because it is the kind of thing that turns a
"bit-identical" claim into a wrong one: I initially attributed the `g*18` write to IQ3_S and the
`1+2*sc_nib` write to Q4_0. They are **swapped** — `g*18` is Q4_0's 18-byte block stride, and
`d*(1+2*sc_nib)` is IQ3_S's. The conclusion is unchanged because all three index by `s>>1`, but the
per-loader evidence had to be re-read against the function bodies rather than the line order.)

**It is still not applied here, and that is the charter working as intended.** The exception allows
a swap only if it is *trivially the same pattern as the two already-fixed* **and** measures **>3%
e2e**. It fails both tests:
1. Not the same pattern. The prior fixes were **one-line mnemonic substitutions** at identical
   fragment ABI. This changes the **fragment shapes** (`tile<16,4>`→`tile<16,8>`,
   `tile<8,4>`→`tile<8,8>`), the `load_ldmatrix` calls, the `A[ntx][8]` array shape, the `dA` array
   shape, and the fold expression — a tile rewrite, not a substitution.
2. The >3% e2e gate **cannot be measured honestly right now**: the swap does not exist yet, so there
   is nothing to time. Its ceiling is 1.42x *of this kernel's MMA-bound share only* — and this lane
   has **not** measured that kernel's e2e share on a real IQ-bank model, so any percent I quoted
   here would be arithmetic on an unmeasured denominator. Stating "plausibly worth several percent"
   and stopping is the honest position; the charter's >3% door stays shut until someone measures it.

**Recommendation (a separate lane, not this one):** implement the k32 merge in
`mmq_iq_experts.cu`, prove bit-identity against the k16 form on a real IQ4_XS/IQ3_S/Q4_0 bank, then
gate on e2e. It is the single largest *available* un-taken MMA win the audit found.

---

## 6. The countermeasure (the durable half of this lane)

Four wrong-form finds in one repo says the process, not the authors, was at fault: the rate lived
in a research doc nobody read at authoring time. Fixed three ways:

1. **`research/sm120-empirical-capabilities.md` is now THE canonical rate table** — all 12 measured
   forms with cyc/warp-MMA, plus the ptxas-verified list of forms that *do not exist*, plus a
   header telling the reader to check it before writing an MMA. Its "Measured compute peaks" table
   was also **structurally broken** (a 13-line blockquote had been inserted *between* two table
   rows, splitting the table) — repaired, and a six-week-old missing table cell in the ISA table
   fixed too.
2. **All 22 asm sites carry `rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md`**
   plus their measured rate and verdict, *next to the asm*, where the next author will see it.
3. **The SASS-census law**, learned the hard way here (§below) and now written into the canonical
   doc: **census the SASS opcode count and require it to equal the count you intended, before
   reading any cycle number.**

### The SASS census earned its place: it caught two of my own probe bugs

The tile-loop probe's **first result was fiction**, and only the census exposed it:
- **v1:** every accumulator got the same A/B operands ⇒ ptxas CSE'd all NACC copies *and* hoisted
  them out of the loop. `IMMA=2` (k16) / `IMMA=1` (k32) for the **whole kernel at every NACC**. The
  "1.2148x" it printed was 2-vs-1 hoisted MMAs plus N folds. Discarded.
- **v2:** per-accumulator operand index rotated by `4*i` **aliases mod 32** (i and i+8 read the same
  slot), CSE-ing NACC=16 back to 16 IMMA where 32 were required. Fixed with a per-i golden-ratio mix
  computed *before* the timed region.
- **Method note:** ptxas also unrolls the outer loop ~8x, so per-instantiation
  `IMMA = NACC × mma_per_step × it_unroll`, floored at 8. **Only NACC ≥ 8 gives an exact per-step
  count** — the verdict is read off NACC=8/16 and the low-NACC columns are ILP context only.
- The NACC sweep is non-monotone (2.32 → 1.19 → 2.52) yet **bit-identical across all 3 reruns**, and
  `ptxas -v` reports **0 bytes spill at every NACC** ⇒ scheduling codegen, not noise and not
  register pressure. Stated rather than smoothed over.

The main 12-form table was censused too and is **clean** — exactly 512 MMAs at NACC=8 in every arm,
correct opcode family per form (`IMMA` int / `HMMA` 16-bit float+tf32 / `QMMA` FP8-FP6-FP4-in-8b k32
/ `OMMA` FP4 k64).

### One error caught in this lane's own inputs

A dispatch-trace subagent reported `mmq_nvfp4_w4a8.cu:1099` block_scale as **32.02** and `:1081`
plain as **16.06** — i.e. claiming the *default* arms are the slow ones. **That is inverted.** My
own 3-rerun measurement and both prior VERDICTs agree: **plain 32.03, block_scale 16.06**, and the
defaults at sites 7, 9 and 13 are the **fast** form. Recorded because an inverted rate in a
downstream doc would have "justified" reverting two correct fixes.

Two premise corrections from the same trace are **accepted** (both verified): `mmq_fp8_blk`'s tile
gate is **`MEMRA_FP8_MMQ`** (default ON), *not* `MEMRA_PP_FP8` (a different, default-OFF seam);
and **`MEMRA_K45_REAL` is an internal arch-derived macro** (`__CUDA_ARCH__ == 900`), not a
user-settable flag.

---

## 7. Receipts

| artifact | what |
|---|---|
| `tools/rate_audit.cu` | the 12-form probe, each form annotated with the repo sites that issue it |
| `tools/isa_sibling_check.cu` | the ptxas oracle: does a deeper equal-math sibling exist? (7 candidates, all rejected) |
| `tools/k16_vs_k32_tileloop.cu` | tile-level price of the int8 depth lift (both inner loops verbatim from the repo) |
| `logs/rate-audit-12form.log` | 3 reruns + GPU state before/after |
| `logs/isa-sibling-check.log` | the rejection texts, quoted |
| `logs/k16-vs-k32-tileloop.log` | 3 reruns + `ptxas -v` register/spill stats |
| `logs/sass-census-probe.txt` | 60-row SASS opcode census |

Commits: `9c00eb22` (rate table), `258b111a` (tile-level price + the two probe bugs), `9fd00b3f`
(22 site annotations), plus this file and the canonical-doc rewrite.

**No engine instruction was changed by this lane.** All edits are comments; all 14 touched `.cu`/
`.cuh` files were nvcc-verified to compile for sm_120a (`fa3_prefill.cu` with `-DMEMRA_FA3_STUB`,
as `build.rs` does).
