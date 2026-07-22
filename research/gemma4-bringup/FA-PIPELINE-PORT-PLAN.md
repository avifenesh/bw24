# FA pipeline port plan — the last ~3.5% of 12B prefill (2026-07-22)

State: 12B pp1736 0.965x llama interleaved (laptop 175W), decode 0.984x. GEMM at parity
(kernel diff 2026-07-22); the残 excess is FA-structural. llama's `fattn-mma-f16.cuh` config
table (commit c818263f2) is the blueprint — measured configs, not guesses:

| shape | ncols (q-rows/CTA) | threads | occupancy | nbatch_fa (BK) | nstages | Q_in_reg |
|---|---|---|---|---|---|---|
| hd256, ncols 32-64 (Blackwell rows 69-72) | 32-64 | 128 | 2 | 32 | **2** | **true** |
| hd512 (rows 77-80) | 8-64 | 64-256 | 4→1 | 32 | 1 | false |

Our current kernels vs that:

- `fa_prefill_w_bf16_pp`/`_g4` (hd256 SWA): 64 rows, 4 warps, occ 2 — geometry matches, but
  **synchronous stage** (ld.global→cvt→st.shared between __syncthreads) and **sQ staged through
  smem** each Q load. llama overlaps the next K/V chunk load (cp.async, nstages=2) with the
  current chunk's mma, and holds Q in registers (no sQ traffic, no Q re-ldmatrix per step).
  Their chunking: nbatch_K2=128 — K/V staged in 128-element sub-tiles, so the double buffer
  costs 2 sub-tiles, not 2 full tiles (smem stays under the occ-2 budget). THE PORT:
  1. Q in registers (we already hold Qf fragments — drop the per-step sQ re-ldmatrix... done
     for pp body; the g4 variant re-uses load_q_frags_bf16 once — OK).
  2. cp.async 16B double-buffered K/V sub-tiles (128 elems = 16 int4 per row-chunk), ring of 2:
     preload chunk 0; loop { cp.async chunk i+1 → buf alt; cp.async.wait_group; compute chunk i }.
     bf16 source (pre-converted) keeps chunk bytes half of f32.
  3. Budget: sK/sV ring = 2 chunks × (32 rows × 128 elem × 2B) = 16KB + sP/sL ≈ 21KB/CTA →
     occ 2 holds. (Full-tile double-buffer = 64KB → occ 1 → loses; the CHUNKING is the trick.)
  Sized from kernel diff: fa_w ~40ms/prime vs llama hd256 class ~17ms → +2-3% pp1736.

- `fa_prefill_bf16_hd512_sp` (globals): 16 rows/CTA vs llama's 64 → they amortize each staged
  K/V tile over 4x more q-rows. Port: widen to 32 rows (2 warps × 16 rows each for GEMM0
  split-K stays; O per warp doubles to 64 CTiles = spill risk — so instead 4 warps × 16 rows,
  each warp owns a 128-dim V/O quarter, GEMM0 split-K 4 ways). smem: sQ 32KB + K/V chunked
  (nbatch style, not full 512) ≈ fits. Sized: fa512 ~26ms → ~15ms → +1-1.5% pp.

Order: hd256 first (bigger share: 40 SWA layers vs 8 globals). Both bit-identity-gated vs the
current kernels where op order is preserved (chunked cp.async staging does NOT change FP order —
the mma consumption order is unchanged; only the copy mechanism differs → bit-identical claim
holds for the hd256 port; the hd512 4-way split-K DOES reorder → own numeric config + battery).

Iteration rig: vast box (final numbers on the laptop only). Correctness: kernel-check windowed
gates + argmax/VERIFY battery per landing.
