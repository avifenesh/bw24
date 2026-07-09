# Prefill arc: W4A8-FP8 — native-FP4 weights × e4m3 activations via mxf8f6f4 block-scale MMA

Status: DESIGN (2026-07-09). Owner of the exactness/numeric decisions: main thread.

## The gap and why the current path is capped

Prefill trails llama at 0.59–0.78x (README known-gaps; ppmmq lane decomposition). The residual is
structural: our W4A8 MMQ tile dequants NVFP4 → int8 tiles and rides the plain-int8 MMA class
(~219 TF measured for plain FP8; int8 same class), while the silicon's block-scaled paths run
381 TF (mxf8f6f4) and 762 TF (mxf4). The w4a8v2 box arc proved the wall is not occupancy —
mode-2 halved the throttle with zero throughput change at 16.4% warps-active; the tensor-pipe
issue class itself is the ceiling. W4A4 (mxf4, 762 TF) inverts the llama gap in-tree
(1.03–1.06x) but is EXACTNESS-BLOCKED: the e2m1 activation grid (1+1+... ~1.6 effective mantissa
bits) forks argmax on long prompts (p3 reject ×2, agent-loop 1/8 self-consistency FAIL).

## The middle rung nobody is standing on

`mma.sync.m16n8k32.kind::mxf8f6f4.block_scale..ue8m0` executes on sm_120 (verified,
sm120-empirical-capabilities.md) and the mxf8f6f4 kind accepts MIXED A/B formats: A = e4m3
activations, B = e2m1 weights. That buys, over the current int8 path:

1. **Compute class**: 381 vs ~219 TF ceiling (+74% issue-rate headroom where we are pipe-bound).
2. **No weight-dequant hop**: e2m1 planes feed the MMA natively — the NVFP4→int8 tile decode
   (register + smem traffic and ALU work inside the mainloop) disappears.
3. **Half the weight bytes through smem** (4-bit vs 8-bit tiles) — doubles effective k-depth per
   smem byte, helps the cp.async pipeline (PP_PIPE) run deeper.

Exactness position: e4m3 activations (3 mantissa bits + per-block scale) sit far above the
W4A4-blocking e2m1 act grid, and the in-tree precedent says this grid class PASSES our gates —
ST_E4M3 decode + PP_FP8 prefill (e4m3 activations end-to-end) went through the full battery
green as their own numeric config. This is a NEW NUMERIC CONFIG like FA_V3/ST_E4M3: own argmax
baseline, full battery in-config, spec self-consistency is the kill-gate.

## The hard part: scale semantics

NVFP4 = e2m1 values + **e4m3 scale per 16 elements**. The mxf8f6f4 block-scale form applies
**ue8m0 scales (power-of-2) per 32 columns** from the scale operand. Three routes, in preference
order:

- **R1 — epilogue-applied weight scales, MMA runs unscaled (scale operand = 1.0 encoding).**
  Split the k-loop so each MMA covers one 16-element NVFP4 scale group per operand row
  (m16n8k32 spans two groups → accumulate per-instruction into a temp and scale-add into the
  tile accumulator every k=16 half? — NO: k=32 mixes both halves inside the instruction).
  Viable only if we pre-fold: see R2.
- **R2 — fold weight scales into the ACTIVATION quantization per k-block-pair.** Wrong: weight
  scales vary per (row-block, k-block); activations are shared across all weight rows. Dead.
- **R3 — requant scale plane e4m3 → ue8m0 (power-of-2)**: loses up to 2^(1/16)…—- real quant
  error on the scale plane. Same tax family as KQ_NVFP4's asym→sym (measured acceptance tax).
  Only acceptable if the error lands below the argmax-fork threshold — measurable, probably not
  free. Fallback, not the plan.
- **R4 — crack `kind::mxf4nvf4.scale_vec::4X` with e4m3 scale operand** (the NVFP4-native MMA
  kind). Our first PTX form was rejected ("incorrect instruction type") but CUTLASS SM120 NVFP4
  kernels prove the silicon path exists (vLLM runs them). mxf4nvf4 is FP4×FP4 (act side would be
  e2m1 again → W4A4 exactness class) — useful for the blocked speed-mode door, NOT for this arc.
- **R1' — the actual plan: k16-native MMA.** Use `m16n8k16` (or two-step k16 issue) for the
  mixed kind if the PTX form allows k=16 for mxf8f6f4 — then each MMA covers exactly ONE NVFP4
  scale group and the per-group e4m3 weight scale × per-block activation scale folds into a
  per-fragment FMA on the accumulator between MMAs (registers only, no extra smem). Issue-rate
  cost of k16 vs k32 must be microbenched — if k16 halves the rate, the win evaporates; if the
  pipe is issue-slot-bound not k-bound, it holds.

**Probe 0 (before any kernel work): PTX microbench matrix** — {mxf8f6f4 k32 ue8m0, mxf8f6f4 k16
form if it assembles, epilogue-FMA-per-k16 variant} × {measured TF, correct scale math on a
synthetic tile vs f64 reference}. The capabilities doc's microbench harness
(`bw24-probe`) is the vehicle. This decides R1' vs R3 vs abandon in <1 day of work.

## Ceiling math (what winning looks like)

pp1855 27B ST today: 1341 (NV_W4 decode config) / 1480 (ST_E4M3). llama 27B GGUF: ~1900-2350
regime-dependent. int8→mxf8f6f4 ceiling factor 1.74x on the MMA mainloop; real GEMM captures
70-85% → expected +30-50% pp on the pipe-bound models → 27B into the 1750-2200 band = llama
parity-to-above WITHOUT touching the exactness contract. 9B (0.74x, 4631 vs 6287) → ~1.0x band.
35B expert MMQ inherits the same tile → compounding with MOE_MMA.

## Order of work

1. Probe 0 (PTX matrix + scale-math correctness vs f64) — bw24-probe, no engine risk.
2. If R1' holds: mxf8f6f4 MMQ tile as a twin of the existing W4A8 tile (same dispatch shape,
   `BW24_MMQ_F8F4=1` seam), e4m3 activation-quant kernel (mirror of the q8_1 fold).
3. Gate battery in-config (argmax + K=1..8 + agent-loop text audit per the new protocol).
4. A/B vs current W4A8 same-hour; flip default only on clean margin + green gates.

Risks: PTX operand-form fight (time sink — capped by Probe 0), k16 issue-rate cliff, e4m3 act
grid argmax forks at depth (precedent says no, contract says verify), scale-plane register
pressure in the mainloop.
