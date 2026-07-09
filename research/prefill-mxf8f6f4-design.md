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

### Probe 0 form-matrix results (2026-07-09, compile/assemble level)

| form | verdict |
|---|---|
| `m16n8k32 ... .e4m3.e2m1 ... ue8m0` (the mixed form) | **EXECUTES at full 381 TFLOP/s** — zero mixed-form rate penalty — and the full-tile correctness vs f64 is **BIT-EXACT** (maxdiff 0.0, `probe/mixed_f8f4_probe.cu`) |
| `m16n8k16` mxf8f6f4 (R1' — one NVFP4 scale group per MMA) | **REJECTED** — "Incorrect instruction type for shape m16n8k16". k32-only. R1' dead. |
| `scale_vec::2X` on mxf8f6f4 (hardware per-16 scales) | **REJECTED** — "Illegal modifier". 1X-only. |

Note (PTX spec, matters for the byte math): mxf8f6f4 requires f4 operands in **8-bit
containers** — the smem/register byte win over int8 tiles does NOT exist. The wins that remain:
+74% MMA issue ceiling (381 vs ~219 TF) and no in-mainloop dequant ALU. The tile is pipe-bound
(w4a8v2: 16.4% warps-active, occupancy-invariant), so the issue ceiling is the real lever.

### Probe 0 CLOSED (2026-07-09) — R-A is GO

- Fragment layout = the standard SM80 m16n8k32 8-bit layout (CUTLASS `mma_traits_sm120.hpp`
  inherits `SM80_16x8x32_S32S8S8S32_TN`; verified bit-exact empirically).
- e2m1 element placement: **shifted left 2, bits [5:2]** ("middle of the eight-bit container" —
  CUTLASS `fp4_shift_A/B`; f6/f8 need no shift). The container decodes as a 6-bit bias-1 field;
  e2m1 codes embed EXACTLY — no value loss.
- Scale operand: ue8m0 bias 127, per-thread-quad byte selectors behave as documented.
- CUTLASS also ships a **plain `kind::f8f6f4`** (no block_scale, no scale regs) — R-A applies
  scales in the epilogue anyway, so the plain form is the cleaner instruction for the tile.
- Rate: mixed e4m3×e2m1 = 381 TFLOP/s, identical to e4m3×e4m3. The +74% ceiling is real.

### Revised route ladder (replaces R1–R4 above)

- **R-A — mixed e4m3×e2m1, per-32 scale requant, epilogue FMA.** MMA runs with scale=2^0;
  NVFP4's per-16 e4m3 scales requant to per-32 (shared across the k32 the MMA spans) and apply
  in the per-block epilogue exactly like the existing q8_1 scale FMA — same tile skeleton as
  today's W4A8. Weight VALUES stay exact e2m1; the tax is scale GRANULARITY (16→32).
- **R-B — fold per-16 scales into values → pure e4m3×e4m3** (form already measured 381 TF).
  round(e2m1 × scale16 → e4m3): values re-round (~2^-4 rel), granularity kept. ST_E4M3
  precedent says pure-e4m3 weights pass gates on F8-origin lineage; NVFP4-origin fold is a new
  lineage — gate decides.
- Order: probe correctness → build R-A (values-exact is the better first bet under the exactness
  contract) → battery → if scale-granularity forks argmax, try R-B → if both fork, the arc
  closes NEGATIVE with the JSONL row as the record.

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

## R-A implementation spec (2026-07-09, post-Probe-0 deep-dive on the W4A8 tile)

Twin of `mmq_nvfp4_w4a8.cu` (977 lines; reuse its skeleton verbatim — tiling, cp.async PP_PIPE
ring, write-back). File: `mmq_nvfp4_f8f4.cu`, seam `BW24_MMQ_F8F4=1` (default OFF until battery).

1. **VRAM law binds**: weights stay packed-nibble resident (4-bit planes). The 8-bit containers
   exist only in smem tiles, built in-loop by the loader (the CUTLASS-door resident-8bit repack
   OOMs the 27B — measured, docs/FLAGS.md §5).
2. **Loader** (`load_tiles_nvfp4_f8f4`, twin of `load_tiles_nvfp4_w4a8`): per 16-group with
   scales s1,s2 per 32-pair: s32 = max(s1,s2); ratio r_i = s_i/s32 ∈ (0,1].
   - r == 1 (s_i == s32): pure bit-op — nibble<<2 into byte (the CUTLASS middle placement). No
     value change, EXACT.
   - r < 1: recode v' = round_e2m3(kvalue[nibble] × r) via f32 mul + `cvt.rn.satfinite.e2m3x2.f32`
     (2 vals/cvt, Blackwell FP6 convert). Error ≤ ~2^-4 rel (2 extra mantissa bits vs e2m1).
   - smem x_tile: 64 bytes/row/blk64 (containers) + 2 f32 per-32 scales; adapt
     MMQ_MMA_TILE_X_K accordingly. ALU cost rides the 83%-idle warp slots (tile is tensor-pipe
     bound — w4a8v2).
3. **Activations**: e4m3 quantize twin of `quantize_q8_1_mmq` — f32 → e4m3 byte
   (`cvt.rn.satfinite.e4m3x2.f32`) + f32 amax-scale per 32 (D4 layout kept so the epilogue shape
   is unchanged).
4. **MMA**: `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32` — A=weights
   (e2m1-in-container), B=acts (e4m3), PLAIN kind (no scale regs; CUTLASS SM120_16x8x32_TN
   form). f32 accumulator replaces the int32 of the int8 path — epilogue becomes
   sum_f32 × (s32_w × d_act) FMA, same structure, one fewer convert.
   k-iter covers 32 (vs 16) — halve the inner-loop trip count, keep MMQ_ITER_K=256.
5. **Gates**: new kernel-check section (f8f4 tile vs Stage-A f32 oracle, int8-act-class rel
   tolerance ~3e-2); then run-gen argmax in-config, run-spec K=1..8, agent-loop text audit
   (PRINT_TEXT), A/B vs W4A8 same-hour, depth axis 3 points. Acceptance parity on the spec
   configs is the flip-blocker to watch (KQ_NVFP4 precedent).
6. **Measure r==1 frequency** on real checkpoints first (host-side scan of the scale planes,
   trivial script) — if adjacent-scale equality is rare AND the recode tax shows up in gates,
   R-B (whole-plane fold to e4m3, granularity kept) is the fallback; if it's common, R-A rides
   the fast path most of the time.

Expected: +30-50% pp on 9B/27B dense prefill; 35B expert MMQ inherits after (MOE_MMA twin).
