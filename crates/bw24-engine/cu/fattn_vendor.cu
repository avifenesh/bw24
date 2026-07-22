// fattn_vendor.cu — llama flash_attn_ext MMA-f16 prefill, vendored (F16ACC arm — WIP SKELETON).
//
// WHY (FA-PIPELINE-PORT-PLAN.md §2 VERDICT, 2026-07-22): the remaining ~3% 12B prefill gap vs
// llama is a NUMERIC-CLASS premium — llama's fa=1 default accumulates S/P/O in f16, halving the
// register+smem footprint of the same tile on the same silicon; our f32-exact FA cannot fit the
// wide-tile geometries that close the gap (every scheduling axis measured/closed: async ring
// flat, occupancy-2 +0.7%, tile width/rpb closed, wide-tile blocked by 99KB/regs at f32).
//
// APPROACH: the mmq vendor playbook (mmq_q8_0.cu precedent) —
//   - source: ggml-cuda @ c818263f2: fattn-mma-f16.cuh (kernel) + fattn-common.cuh (combine,
//     mask helpers) + mma.cuh (tile machinery; the bf16 subset is already proven in-tree in
//     the mmq TUs — the f16 tile ops get the same treatment).
//   - ggml-decoupled, self-contained TU, everything static/internal, C-ABI launchers only.
//   - instantiations: (DKQ, DV) = (256,256) ncols 32/64 [gemma SWA + globals-hd256 models] and
//     (512,512) ncols 32 [gemma4 hd512 globals]. type_K = type_V = F16.
//   - inputs: bw24 prefill Q/K/V are f32 — pre-convert with f32_to_f16_flat (twin of the
//     existing f32_to_bf16_flat). llama expresses causal+SWA through the KQ MASK tensor: build
//     the [T, T_kv] f16 mask device-side once per prime (kernel fattn_build_mask below).
//   - C-ABI: bw24_fattn_f16(Q,K,V,mask,O, dkq, dv, n_head, n_head_kv, T, T_kv, scale) +
//     bw24_fattn_f16_mask_build(mask, T, T_kv, pos0, window, causal, stream).
//
// GATING: BW24_FA_F16ACC opt-in door (the BW24_MMQ W4A4 precedent — an explicit speed/accuracy
// tradeoff class). NEW NUMERIC CONFIG: full battery in-config (kernel-check oracle band at the
// f16 tolerance, run-gen argmax, VERIFY-GATE, spec self-consistency) before any default talk.
// NOTE the A/B-fairness argument for an eventual default: llama's fa=1 IS this numeric class —
// the current exact-class bw24 vs f16 llama comparison undercounts bw24.
//
// STATUS: SKELETON — not yet in build.rs. Port order:
//   [ ] 1. mma.cuh f16 tile subset (tile<I,J,half2>, ldmatrix, mma m16n8k16 f16-accum)
//   [ ] 2. fattn-mma-f16 config table (the CONFIG_CASE rows for 256/512, sm_120 branch)
//   [ ] 3. flash_attn_ext_f16 kernel body (KQ f16 mma, online softmax f16, VKQ f16 accum)
//   [ ] 4. fattn-common combine (parallel-blocks fixup) — only if nbatch_combine needs it
//   [ ] 5. mask builder + host launcher + FFI + dispatch behind BW24_FA_F16ACC
//   [ ] 6. kernel-check gates (f16-band oracle vs CPU sdpa; NOT bit gates) + battery
//
// The port continues from this file; every piece lands compilable or not at all.

// (intentionally no code yet — see port order above; this TU joins build.rs at step 5)
