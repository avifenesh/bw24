# Alloc-free capture — the 12B decode last-cell build (2026-07-23)

Root cause (osrt): cuGraphLaunch 226µs/launch from cuMemAllocAsync/Free NODES baked into the
captured dc step (~400 transients/token; drops during capture add free nodes). Pool-threshold
pin: no effect. Retained capture: stream 96/96 IDENTICAL, tax unchanged. llama pays zero
(static buffers). Recoverable ≈ the whole 2.1% 12B tg cell.

## Route: slot-fed captured step (bit-identical kernels, zero capture-time allocs)

Stage 1 — mechanical `_into` extractions (old fn = alloc + delegate; no behavior change):
- [x] matmul_q4_fused3_into / matmul_q4_fused2_into
- [x] quantize_q8_1_into
- [x] qmatvec_mmvq_into (the m=1 single: wo, lm_head q6_K)
- [x] rms_norm_q8_1_into
- [x] embed_gather_device_into
- (clone_dtod -> existing copy_into)

Stage 2 — gemma4_layer_tail_add_nq_slotted (ffn gate/up fused2 + gelu + down + add_nq chain).

Stage 3 — G4DcSlots struct (persistent, allocated pre-capture; sized for the model's max
shapes incl. per-class hd/nkv) + gemma4_decode_step_dc_slotted mirroring _dc_into 1:1 in
kernel order + the graph door captures the slotted step.

Gates: BW24_GRAPH_GATE stream identity (96/96), argmax battery, tg128 A/B (door must beat
89.6; upside bound ~+2.3ms/128tok = ~91.6+).
