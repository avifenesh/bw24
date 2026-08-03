# CLC work-stealing MMQ prefill (perf-frontier lever #1) — VERDICT: FLAT-NEGATIVE

Lane: `lane/clc-mmq` off `restructure/public-split` (216419d6). Rig: RTX 5090 Laptop
(sm_120a, 82 SMs, CUDA 13.1.115). Date: 2026-08-03.

## What was built

`clusterlaunchcontrol.try_cancel` hardware block-stealing wrapped around the Q4_0 int8-MMA
MMQ prefill GEMM (`cu/mmq_q4_0.cu`, `mul_mat_q_q4_0_clc`): same (it, jt) tile grid as the
static kernel, each block runs its home tile then cancels one unlaunched block and takes
over that block's coordinates (CUTLASS SM120 pingpong pattern, CUDA 13.3 PG §4.12;
try_cancel issued before the mainloop so the cancellation DMA overlaps compute). No k
split, no fixup — every tile computes its full k range in one block with the exact
per-tile math and accumulation order of the static kernel. Rollback seam `MEMRA_MMQ_CLC`
(default static; CLC behind `=1`), deterministic force via `memra_mmq_q4_0_set_clc(0|1|-1)`
(the #23 lesson: every arm must be forceable for gates). `__CUDA_ARCH_LIST__ >= 1000`
compile gate — sm_89/90a builds byte-skip the CLC kernel and the setter reports
"unavailable" (verified: all three arches compile warning-free).

## Bit-identity gate (ran BEFORE perf — pass required to measure at all)

`kernel-check` full battery on the 5090: **ALL GREEN**, zero FAILs
(`kernel-check-clc.log`). New arms, all vs the forced static xy-tiling kernel:

- `MMQ-Q4_0-CLC` + `MMQ-Q4_0-CLC-RP` (g12 real q4_0 tensors, T=16/64/128/512): 0 bit
  mismatches in 16/16 cells (raw and rp split-plane layouts).
- `MMQ-Q4_0-RAGK CLC` (ragged-k in=2112, the shape class that killed SK on Hopper): 0/0.
- `MMQ-Q4_0-NC26 CLC` (need_check=true clamped last row-tile, 6 g26 tensors x
  T=103..2151): 0 mismatches in all 36 cells.

Total: 54 CLC bit-identity arms, all IDENTICAL. The bench re-checks per shape
(belt-and-braces): 19/19 shapes IDENTICAL. Exactness contract holds by construction —
stealing changed which SM ran a tile, never the tile-internal order.

Steal machinery receipts (`clc_steal_count.cu`, `clc_mmq_pattern.cu`): 1536-tile 2D grid
-> 1044 steals, every tile exactly once; MMQ-form (32x8 block, no cooperative_groups,
ctaid.x+y recovery) 1344 tiles -> 852 steals, correct.

## A/B/C table (N=5 sessions x 30 reps, arms interleaved within-session, same process;
medians; kernel-only via the GEMM-only FFI entry, activation quantized once outside the
timed region; `ab-run3-with-sk.log`, JSONL in `clc-mmq.jsonl`)

ratio = static/arm (>1 = arm faster). sk = incumbent stream-k arm forced via
`MEMRA_MMQ_SK_FORM=sk` (band-class numerics, informational).

| shape | in_f | out_f | T | tiles | waves | wave eff | static us | clc us | clc ratio | sk ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| q9-qkv | 4096 | 8192 | 512 | 256 | 3.12 | 78.0% | 319.1 | 327.1 | 0.976x | 1.131x |
| q9-qkv | 4096 | 8192 | 1736 | 896 | 10.93 | 99.3% | 897.3 | 922.5 | 0.973x | 0.954x |
| q9-gateup | 4096 | 12288 | 512 | 384 | 4.68 | 93.7% | 408.6 | 414.6 | 0.985x | 0.985x |
| q9-gateup | 4096 | 12288 | 1736 | 1344 | 16.39 | 96.4% | 1392.4 | 1434.4 | 0.971x | 0.999x |
| q9-down | 12288 | 4096 | 512 | 128 | 1.56 | 78.0% | 444.9 | 478.3 | 0.930x | 1.111x |
| q9-down | 12288 | 4096 | 1736 | 448 | 5.46 | 91.1% | 1398.5 | 1430.4 | 0.978x | 1.038x |
| q9-attngate | 4096 | 4096 | 512 | 128 | 1.56 | 78.0% | 153.3 | 159.2 | 0.963x | 1.042x |
| q9-attngate | 4096 | 4096 | 1736 | 448 | 5.46 | 91.1% | 505.4 | 514.3 | 0.983x | 1.038x |
| **wavequant-84t** | 4096 | 10752 | 128 | **84** | **1.02** | **51.2%** | 150.9 | 155.4 | **0.971x** | **1.479x** |
| wavequant-336t | 4096 | 10752 | 512 | 336 | 4.10 | 82.0% | 403.5 | 419.7 | 0.961x | 1.103x |
| wavequant-168t | 4096 | 10752 | 256 | 168 | 2.05 | 68.3% | 234.5 | 240.4 | 0.976x | 1.261x |
| subwave-32t (control) | 4096 | 4096 | 128 | 32 | 0.39 | 39.0% | 68.8 | 70.8 | 0.971x | 1.271x |
| subwave-64t (control) | 4096 | 8192 | 128 | 64 | 0.78 | 78.0% | 69.4 | 70.3 | 0.986x | 0.872x |
| g12-attn_q | 3840 | 4096 | 512 | 128 | 1.56 | 78.0% | 142.0 | 146.6 | 0.969x | 1.028x |
| g12-attn_q | 3840 | 4096 | 1736 | 448 | 5.46 | 91.1% | 467.0 | 475.4 | 0.982x | 1.020x |
| g12-ffn_gate | 3840 | 15360 | 512 | 480 | 5.85 | 97.6% | 463.2 | 475.8 | 0.973x | 0.937x |
| g12-ffn_gate | 3840 | 15360 | 1736 | 1680 | 20.49 | 97.6% | 1652.7 | 1657.6 | 0.997x | 0.996x |
| g12-ffn_down | 15360 | 3840 | 512 | 120 | 1.46 | 73.2% | 611.8 | 633.0 | 0.967x | 1.292x |
| g12-ffn_down | 15360 | 3840 | 1736 | 420 | 5.12 | 85.4% | 1744.1 | 1756.7 | 0.993x | 1.078x |

Cross-run stability: run1 (N=3) and run2 (N=5, `ab-run1.log`/`ab-run2.log`) show the same
0.93–0.997x band on every shape. Thermal regime: laptop 5090, sustained back-to-back
sessions, arms interleaved within-session so drift cancels.

## Wave-quantization correlation — the priced hypothesis is REFUTED for whole-tile CLC

The lever was priced at +8–15% on tail-wave shapes (wave eff far from 100%). Measured:

- CLC ratio vs wave eff has **no correlation**: 0.971x at 51.2% eff, 0.976x at 68.3%,
  0.93–0.99x across 73–99.3%. The worst-wave shape (84 tiles = 1.02 waves, half the
  machine idle in wave 2) gains nothing.
- The **sub-wave controls prove the mechanism**: at 32 and 64 tiles (< 82 SMs) stealing
  is impossible (every block launches immediately, try_cancel can only fail), yet CLC
  still loses 1.4–2.9% — the deficit is pure machinery overhead (per-tile-round acquire
  fence + try_cancel issue + mbarrier poll + the persistent-loop code shape; ptxas:
  same 255 regs, but the nc arm picks up an 8B stack spill the static kernel lacks).
- Root cause the pricing missed: for a grid of **uniform-duration whole tiles**, the
  GigaThread engine already backfills blocks dynamically as SMs drain — the static grid
  IS a work-stealing schedule at block granularity. CLC can only relocate the *same*
  tail wave onto different SMs; it cannot shrink it. The tail-wave money is reachable
  only by splitting a tile's k range across blocks — which is exactly the incumbent
  stream-k arm (sk ratio tracks wave eff beautifully: 1.48x at 51% eff, 1.26x at 68%,
  ~1.0x at 96%+), and stream-k pays with band-class fold-order numerics (the known
  trade, default-on for plain serving since 2026-07-23).
- CLC's real value per Colfax/CUTLASS is amortizing kernel prologue/epilogue in
  persistent pingpong kernels with heavyweight per-block setup. The MMQ prologue is an
  ids_dst identity fill — there is nothing to amortize.

## Verdict

**Flat-negative: -0.3% to -7% (median ~-2.5%) across all 19 shapes, no winning cell,
no wave-eff regime where CLC recovers.** The exactness half succeeded completely
(bit-identity everywhere, including the ragged-k and need_check classes that broke SK on
Hopper) — CLC work-stealing IS "the legal stream-K" in the sense of preserving FP order,
but on sm_120a it buys no schedule win for uniform-tile MMQ grids: the hardware block
scheduler already does this job, and the k-split (not the steal) is where the
wave-quantization money lives.

Per the flags doctrine (negative/flat experiment -> kill the flag and dispatch arm; the
JSONL row is the record): recommend **DO NOT MERGE the CLC arm**. The lane branch carries
the full implementation + receipts for owner review; if the verdict is accepted, the
kernel/seam should be dropped and only this evidence directory kept. Where CLC could still
pay later: a future persistent-scheduler prefill GEMM with heavyweight per-block setup
(lever #8's setmaxnreg/TMA rewrite) — there the prologue amortization is real, and this
lane's verified try_cancel pattern (`clc_mmq_pattern.cu`) is the drop-in reference.

## Files

- `kernel-check-clc.log` — full battery, ALL GREEN, 54 CLC bit-identity arms.
- `ab-run1.log` / `ab-run2.log` / `ab-run3-with-sk.log` — raw bench output (run3 adds the
  forced stream-k arm; JSONL rows embedded per shape carry all N session medians).
- `clc-mmq.jsonl` — one row per shape from run3 (static/clc/sk medians + per-session Ns).
- `clc_steal_count.cu` / `clc_mmq_pattern.cu` — on-device steal receipts (this rig).
- Code: `crates/memra-engine/cu/mmq_q4_0.cu` (`mul_mat_q_q4_0_clc`, `mmq_clc_on`,
  `memra_mmq_q4_0_set_clc`), `crates/memra-engine/src/mmq_ffi.rs`,
  `crates/memra-engine/src/bin/kernel_check.rs` (CLC arms),
  `crates/memra-engine/src/bin/clc_mmq_bench.rs` (the A/B/C harness).
