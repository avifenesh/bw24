# W4A8-prefill lane — SCOPE

**Date:** 2026-08-06 · **Branch:** `lane/w4a8-prefill` · **Base:** `98da33bd` (the v0.71 tag candidate)
**Rig:** local RTX 5090 Laptop (sm_120a, GB203, 82 SM), clocks to be locked 1860/1860, nvcc 13.1.115
**Mandate:** evaluate the "pragmatic alternative" named in `research/fp4-act-scoping-20260806/BRIEF.md`
§4 last row and §5 point 4 — *"Fallback: Keep k64 door closed, pursue W4A8"* — the one remaining door
on the closed prefill campaign.

---

## 1. What the brief actually proposes

The brief's W4A8 row (§4) and its §5.4 rationale are:

> **Fallback: Keep k64 door closed, pursue W4A8** — 0x (no mxf4 k64 gain), accuracy risk **LOW**
> (QServe W4A8 shows +0.25 PPL near-lossless, proven speedup 1.2-3.5x), **20-30 days** eng:
> *"Implement W4A8 quantization for memra (different instruction, not mxf4 k64), leverage existing
> FP8 paths."*

Its evidence is §2.7 (QServe, MIT Han Lab): W4A8KV4 at +0.25 PPL on LLaMA-2 7B, 1.2-1.4x throughput
over TensorRT-LLM on LLaMA-3-8B, 2.4-3.5x on Qwen1.5-72B, and the key claim *"existing INT4 methods
suffer 20-90% overhead from dequantization; QServe optimizes this."*

So the proposal, stated precisely, is: **4-bit weights × 8-bit activations, on an instruction other
than mxf4 k64, reusing memra's FP8 activation infrastructure.**

## 2. The arithmetic that this lane must settle first

The binding context says the shipped prefill GEMM kernel is literally named
`mul_mat_q_nvfp4_w4a8`. That is not a coincidence of naming. Three facts from the repo, before any
new measurement:

**Fact A — memra's shipping prefill GEMM is already W4A8.** `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu`
takes NVFP4 (e2m1 + UE4M3-per-16) weights and int8 activations, dequants the 4-bit weights into int8
tiles inside the mainloop, and rides `mma.sync.m16n8k16.s8.s8.s32`. Weights 4-bit in VRAM,
activations 8-bit. **That is W4A8.** It is the default, it is the 77.97%-of-pp512 kernel, and it is
the 88.6 TOP/s the ILP lane measured at the issue-interval wall.

**Fact B — the *second* W4A8 route, on a 2x instruction, is also already built and measured.**
`crates/memra-engine/cu/mmq_nvfp4_f8f4.cu` + the `..._f8f4` tile inside `mmq_nvfp4_w4a8.cu` is the
R-B route of `research/prefill-mxf8f6f4-design.md`: NVFP4 weights folded to e4m3 containers × e4m3
activations on `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Seam
`MEMRA_MMQ_F8F4=1` (`docs/FLAGS.md:148`, dispatch `mmq_ffi.rs:1126`). Gate bin `f8f4_check`.
Shipped 2026-07-09/10, and **already adopted** into the NV-27B ST standing serve config
(`20229b85`).

**Fact C — the instruction-level arithmetic is already measured on this rig, three forms, two
instruments** (`research/prefill-ilp-20260806/` slice 2b, clock-locked 1860, `flock`'d):

| MMA form | cyc/warp-MMA | rate | vs the shipped s8 path |
|---|---|---|---|
| `m16n8k16.s8.s8.s32` — **today's default W4A8** | 16.06 | 155.0 TOP/s int8 | 1.000x |
| `m16n8k32.kind::f8f6f4` e4m3×e4m3 — **the f8f4 W4A8 route** | 16.06 | 309.0 TFLOP/s | **1.993x** |
| `m16n8k64.kind::mxf4nvf4.4X` e2m1×e2m1 — the shut FP4-act door | 16.06 | 618.5 TFLOP/s | 3.989x |

All three share the same 16-cycle pipe interval, so rate scales exactly with K depth.

### The ceiling arithmetic the brief asked for

**What does W4A8-via-the-proposed-path buy over the current 88.6 TOP/s?**

The proposal's own framing ("different instruction, not mxf4 k64, leverage existing FP8 paths")
resolves, on sm_120a, to exactly one instruction: `kind::f8f6f4 m16n8k32` with e4m3 activations.
`ptxas` has already closed the alternatives (`research/prefill-ilp-20260806` slice 2a): `m16n8k64` is
`e2m1 × e2m1` **only** (no e4m3 B operand at any `scale_vec`), and `m16n8k32 mxf8f6f4` is `ue8m0`-only
with `scale_vec::1X`-only. There is no third 8-bit-activation × 4-bit-weight tensor-core form on this
silicon.

So the ceiling is fixed and already known:

```
instruction bound       = 309.0 / 155.0                      = 1.993x   at the MMA
GEMM share of pp512     = 0.7797                             (q27 NVFP4, prefill-gemm phase 1)
Amdahl e2e ceiling      = 1 / (0.7797/1.993 + 0.2203)        = 1.616x   IF fully realized
mxf4-door realization   = 68% of its instruction bound       (measured: 2.710x of 3.989x)
e2e at 68% realization  = 1 / (0.7797/(1.993*0.68 + 0.32) + 0.2203)  ≈ 1.36x   (see note)
```

**But the realization factor does not need estimating — it was measured 2026-07-10.** The f8f4 flip
battery (4 NVFP4 models, interleaved) recorded, on the *same* pp anchor class:

- **pp1845 +3.9% to +6.3% on ALL models**, prime/TTFT **-4.0% to -5.6%**
- gates PASS both arms; e2e spec is **model-signed** (27B ST +7.2%, 27B GGUF -0.3%, 9B GGUF -3.5%,
  9B ST -6.1%) via the prefill-KV acceptance law

So the measured answer is **+3.9-6.3% pp**, not 1.36x, and not the brief's 1.2-3.5x. The 1.993x
instruction headroom converts at roughly **3-6% realization**, because the f8f4 tile pays for its
2x-per-issue with an in-loop e4m3 fold (values × per-16 scale → e4m3 containers, and PTX requires
f4 operands in **8-bit containers** so there is no smem-byte win either) and the tile is not
MMA-issue-bound once the fold lands. `4c3f4b05` records the follow-up: *"wall = tile algorithm, not
occupancy/DRAM/MMA-class"* (y64 occupancy arm -8%), and the expert-tile twin was measured **-6.4%**
and deleted (`ec996bfe`).

### The one-day NO-GO candidate, stated up front

The brief's W4A8 proposal, read against the repo, appears to be **already shipped twice**:

1. as the **default** prefill GEMM (int8 activations, `m16n8k16.s8`) — the very 88.6 TOP/s the
   campaign closed at, and
2. as the **only other** 8-bit-activation instruction on this silicon (`kind::f8f6f4`, e4m3
   activations), measured at **+3.9-6.3% pp** with model-signed e2e, per-model-adopted, flag
   `MEMRA_MMQ_F8F4`.

If that holds under measurement, the verdict is **NO-GO with 20-30 days saved**: there is no
unbuilt W4A8 route, the brief's "0x → pursue this instead" recommendation is pointing at code that
has been in-tree for four weeks, and its "1.2-3.5x proven speedup" citation is a *cross-framework*
number (QServe vs TensorRT-LLM baseline) that does not transfer to memra's denominator, whose W4A8
baseline is already the same MMQ-class kernel QServe is beating.

## 3. First measurement

**Slice 1 — price the f8f4 W4A8 route on the closed-campaign denominator, today, at locked clocks.**

The 2026-07-10 f8f4 battery used pp1845 on a July code base at interleaved N=2. The campaign that
just closed uses **pp512 (`research/e2e/prompts/pp512.txt`), q27 NVFP4, locked 1860/1860, N≥15
interleaved** — and the tile under it has moved (fp8st lanes, rp split-plane default, the whole
phase-1 fold work). Per the H100 lane's LAW 2, *thresholds and verdicts calibrated on old kernels
must be re-swept when the code under them moves*. So:

- **Arms (one binary, runtime flag — same protocol as `tools/ab_w4a4.sh`):**
  - `NAKED` — shipped default (rp split-plane W4A8, int8 acts, `m16n8k16.s8`)
  - `F8F4` — `MEMRA_MMQ_F8F4=1` (the e4m3-act W4A8 route, 1.993x instruction bound)
- **Protocol:** interleaved, 3 rounds × `MEMRA_PP_REPS=5` = **N=15/arm**, `MEMRA_PP_ONLY=1`,
  `flock /tmp/gpu5090.lock`, clocks locked 1860/1860, `nvidia-smi -rgc` on exit, GPU state and
  co-resident compute-apps recorded per run.
- **Reproduce the denominator first** (the ILP lane's locked reference: pp512 q27 NAKED = 1316.3
  median, N=15). If NAKED does not land within a few percent of 1316.3, the rig is not comparable and
  nothing else is published.

**Decision rule:**
- If F8F4 ≤ ~1.07x NAKED → the 1.993x instruction bound converts at single digits on the current
  tile, the brief's W4A8 door is **already open and already priced**, and the lane closes NO-GO with
  the arithmetic + this measurement as the receipt (plus a follow-up note on what *would* move it:
  the tile-algorithm wall named by `4c3f4b05`, not the MMA class).
- If F8F4 > ~1.07x NAKED → the route has re-gained headroom since July; then, and only then, does a
  build slice get scoped (kernel-check + `f8f4_check` + run-gen argmax + run-spec K=1..8 in-config,
  because f8f4 is its own numeric config and the prefill-KV acceptance law applies).

**Correctness bar:** slice 1 is a runtime-flag A/B on an already-gated seam and changes no engine
code — no battery needed for the *measurement*. Any recommendation to change a default would need the
full in-config battery, and this lane will say so explicitly rather than implying gate coverage it
does not have.
