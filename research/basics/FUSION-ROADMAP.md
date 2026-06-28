# bw24 FUSED-KERNEL Build Roadmap

Lead-architect synthesis. Verified against source on the actual silicon (RTX 5090
Laptop, CC 12.0, sm_120a, 82 SM, 1536 thr/SM, 65536 regs, ~100KB smem/SM, 64MB L2,
~847 GB/s sustained, ~1860 MHz clock-locked). nvcc 13.1, `-gencode arch=compute_120a,code=sm_120a`.

## Thesis (binding)

Per component: take what is BETTER in each leading engine and FUSE into ONE kernel
better than either, then TUNE to THIS silicon. The edge over llama/vLLM/SGLang is
that they ship ONE general kernel averaged across all GPUs/models; bw24 fuses both
their best ideas into one kernel pinned to 82 SM / 99KB smem / 1536 thr-SM / sm_120a
warp-mma+cp.async+TMA+clusters (NO wgmma/tcgen05). NOT two engine-mirror paths. NOT a
feature they lack. NOT model-specific hardcoding.

Current clock-locked state vs llama 9B-NVFP4: prefill 2158 vs 5451 (0.40x, LOSING),
decode ~83 vs 117 (0.71x). GEMM dominates the prefill gap.

---

## VERDICT-GATED RANKING (gain x feasibility)

Drop everything that the adversarial pass scored PARITY-ONLY / TWO-MIRROR / INFEASIBLE.
Build only what is a genuine single-kernel fusion no engine ships AND has verified
above-parity headroom on THIS silicon.

| Rank | Component | Verdict | Build? | Honest headroom |
|------|-----------|---------|--------|-----------------|
| 1 | FA prefill+decode fp8-KV-into-MMA + cp.async double-buffer | REAL-FUSION-WIN | BUILD | FA share 1.4-1.8x; decode 0.80-0.90x |
| 2 | GEMM TMA-feed + persistent stream-K wrapping native decode | PARITY-ONLY* | BUILD (de-mirror only) | +25-55% prefill, saturation not ceiling |
| — | decode mmvq_fused (VDR-interleave + epilogue fusion) | PARITY-ONLY | DROP matvec relayout; KEEP only epilogue+graph-wiring | matvec relayout = 0% (probe-tied) |
| — | gemm-fusion-feasibility (parameterize CUTLASS for GGUF decode) | INFEASIBLE | DROP | TransformA_/B_ never called in mainloop |
| — | best-of-each-map "FP4 ceiling above llama" | PARITY-ONLY | DROP the premise | llama already runs the same FP4 mma on sm_120 |

\*GEMM is PARITY-ONLY against llama on the format ceiling, but bw24 is currently
LOSING and genuinely two-mirror (hand cp.async FP4 kernel + separate CUTLASS wrapper).
Collapsing to one TMA-fed persistent kernel is the single biggest absolute prefill
lever even though it only closes toward parity, not past it. It earns rank 2 by
ABSOLUTE GAP CLOSED, not by being above-parity.

### Why the dropped items are dropped (stated explicitly)

- **Parameterize CUTLASS for GGUF decode (INFEASIBLE).** Verified in
  `sm120_blockscaled_mma_tma.hpp`: `TransformA_`/`TransformB_` are declared (71,75) and
  aliased (156,157) but NEVER called in the mainloop — only the hardcoded
  `fp4_shift_A/B` (808-809) runs in `copy_kblock`. `CollectiveMma` cannot host a GGUF
  decode hook. Cannot parameterize; can only hand-copy the STRUCTURE (see GEMM build).
- **decode VDR-interleave weight relayout (0% on this silicon).** Probe-measured:
  bw24 warp-per-row (`blk+=32`) and llama VDR-interleave (`kbx=tid/(qi/vdr)`, nwarps=2)
  are TIED at 472 GB/s, both 52.5% DRAM, both 94.7% warps-active. The claimed
  92-93% vs 89% edge does NOT reproduce. vec4 split-q/d load was SLOWER (437 GB/s).
  Relaying the matvec is pure churn for 0 gain and risks regressing the clean
  `__shfl` reduce. The matvec already runs at 472/548 = 86% of achievable SOL.
- **"FP4 762-TFLOP ceiling above llama" (PARITY-ONLY).** REFUTED: `llama.cpp`
  `mma.cuh:1145` already emits
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.f32.e2m1.e2m1.f32.ue4m3`
  gated by `BLACKWELL_MMA_AVAILABLE` (__CUDA_ARCH__ >= 1200) with `load_tiles_nvfp4`
  + stream-K. llama's 6139 NVFP4 prefill touches the SAME FP4 ceiling on the SAME
  silicon. Same format + same mma + same silicon = same compute ceiling. The only
  bw24 lever llama lacks is the FEED (cp.async.bulk.tensor TMA vs llama's plain
  cp.async) — that improves SATURATION of a shared ceiling, it does not lift it.
- **decode "kill 196 requant launches" / "CUDA graph" (ALREADY SHIPPED).**
  `lib.rs:349-355` quantize_q8_1 already shares one quant per input group; the
  generate_graph / decode_step_dc_cap CUDA-graph path already exists. The marquee
  "wins" describe code already in the tree. Only the in-kernel gate+SwiGLU+output-requant
  epilogue fusion is new, and even that is modest (a few launches/layer).

---

## RANK 1 — FA fused kernel (the only REAL-FUSION-WIN)

### The ONE fused kernel (best-of-each)

ONE prefill kernel `fa_prefill_q` replacement + decode twin `fa_decode_vec_q`, fusing
three engine bests into a kernel none of the three ships on sm_120:

- **bw24's byte edge:** KV stored quantized end-to-end. K e4m3 / V e5m2 (or q8_0-K),
  per-32 scale, 1-byte cache = 1/4 of f16. SAME bytes flow HBM->L2->smem->MMA, zero
  repack.
- **flashinfer's idea (fp8-into-tensor-core):** the smem tile format IS the cache
  format; the PV GEMM runs the native sm_120 op
  `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e5m2.f32` (P->e4m3, V->e5m2)
  at k=32 (2x the k=16 bf16 step), HD_KTILES 16->8. flashinfer ships this ONLY on
  sm90; its sole sm120 attention kernel is `mla_sm120` (MLA-specific). No fp8-KV-into-MMA
  sm120 prefill exists in any engine.
- **llama's double-buffer:** the smem freed by fp8 tiles funds a `cp.async.cg`
  nstages=2 ping-pong (issue next tile's K+V while this tile's QK/softmax/PV run).
  llama's `fattn-mma-f16.cuh` has this double-buffer but ONLY on half2 f16 KV (it
  inflates quant-KV to f16 BEFORE the mma).
- **bw24's proven softmax stays byte-identical:** register-resident FA3 online softmax
  (4 CTiles, row_max4/row_sum4, exp2f+LOG2E, register-O alpha rescale). Already shipped
  (`fa_prefill_f32_pp`), do not touch.

VERIFIED current state (the headroom is real): `flash_attn.cu` has ZERO cp.async (the
one grep hit at line 49 is a comment). `fa_prefill_q` (line 752) does SYNCHRONOUS
per-tile inline dequant -> bf16 smem -> `__syncthreads` -> mma, with the KV load 100%
exposed. Clear headroom; worth building.

### sm_120-exact tune

- smem (verified EXACT): fp8 KV double-buffer = `BK32 * HD256 * 1B * 2(KV) * 2(stages)`
  = 32KB = today's bf16 single-buffer 32KB. Both give 44.5KB total kernel smem, both
  fit occupancy=2 under the 49.5KB-for-2-CTA budget. The double-buffer is FREE in smem.
- `__launch_bounds__(N_WARPS*WARP_SZ, 2)` — keep the shipped 2-CTA/SM target.
- OPCODE CORRECTION (load-bearing): use the cute `SM120_16x8x32_TN` atom form with the
  `.kind::f8f6f4` qualifier, NOT the bare sm89 opcode `mma_sm89.h:151`. Probe on this
  GPU shows the bare unqualified form is the wrong instruction for sm_120; the
  `.kind::f8f6f4`-qualified form runs clean on compute_120a.
- Decode twin: keep GQA-broadcast, adopt dp4a-on-native-q8_0 for QK (skip the per-key
  bf16 inflate: smem write + read + convert), PV stays fp8/affine-dequant, same
  split-K combine via `fa_decode_combine_f32`.

### ACCURACY FALLBACK (must implement, fires by default)

Probe: e4m3 QK dot rel-err is ~5x worse than q8_0 (mean 0.205 vs 0.040, p95 0.448 vs
0.098). e4m3 (3-bit mantissa) QK is too lossy. **Default to the fallback:** keep
q8_0-K dequant-to-bf16 for the QK GEMM, and fp8 ONLY the P->e4m3 / V->e5m2 PV side.
This forfeits the K-side fp8 win but the cp.async double-buffer (the bigger, free win)
and the PV-side k=32 fp8 both survive. Gate the QK-fp8 path behind a build flag; ship
it only if a later kernel_check rel-err pass clears it.

### argmax + ncu gate

1. kernel_check rel-err: PV-fp8 + double-buffer path must hold attention rel-err at the
   shipped q8_0/q5_1 level. QK-fp8 path: only enable if rel-err clears the q8_0 bar.
2. End-to-end argmax UNCHANGED: run_dense 268, run_hybrid 271, MoE 35B-A3B 1178. Any
   mismatch = revert.
3. ncu: confirm `cp.async` issued + `barrier.cluster`/scoreboard stall on the KV load
   drops (load latency that is 100% exposed today must become hidden); PV mma issue
   count halves (k=32). Clock-locked `nvidia-smi -lgc 1860,1860`.

### Honest projection vs llama

- **Prefill FA share: 1.4-1.8x** the current `fa_prefill_q` (NOT 1.6-2.0x — the QK
  fp8 win is forfeited to the accuracy fallback, so only PV runs k=32 fp8; the
  cp.async double-buffer is the real free win). plain fp8 = 219 TFLOP vs bf16 117
  (1.88x, on-box) on the PV side only.
- **End-to-end prefill: single-digit % until the GEMM lands.** Amdahl-capped: GEMM is
  the dominant prefill gap (the 43x is GEMM-driven — `GEMM-PLAN.md:9`, the 512x weight
  re-read), FA is the smaller share. A 1.5x FA kernel moves end-to-end prefill only a
  few percent on its own.
- **Decode: ~0.80-0.90x (90-105 tok/s)** vs llama 117, from removing the per-key bf16
  inflate. Modest — decode is bandwidth-bound and the inflate is ALU not bytes. The
  quant-KV 1/4-byte read margin GROWS at ctx>4k where llama's f16 KV read dominates.
- **vs llama/flashinfer:** they read 4x the KV HBM bytes (f16) AND have no fused
  fp8-KV-into-mma sm120 path. This is genuinely a kernel none of the three ships.

NET: a real per-kernel win on the FA share. NOT a path to closing the prefill gap on
its own — that needs the GEMM.

---

## RANK 2 — GEMM TMA-feed fused kernel (#1 BUILD by absolute gap, PARITY-targeted)

This is the #1 build despite being PARITY-ONLY against llama, because it is the
biggest ABSOLUTE prefill lever (0.40x -> approaching parity closes ~2-3x more tok/s
than FA) AND it kills the genuine two-mirror split. Tracks task #37 (#36 in-progress is
the MMQ port that feeds it).

### The ONE fused kernel (de-mirror + TMA feed)

ONE persistent kernel replacing BOTH the hand cp.async FP4 kernel AND the separate
`cutlass_fp4_sm120.cu` collective wrapper (the two-mirror):

- **CUTLASS's TMA+mbarrier FEED (structure, hand-copied):** TMA `cp.async.bulk.tensor`
  / `cp.async.bulk` (1D non-tensor for raw GGUF superblocks — see correction below) of
  raw quant superblocks into a multi-stage smem ring arbitrated by an mbarrier
  `PipelineTmaAsync`. This takes the weight read OFF the MIO/scoreboard pipe that bounds
  the current FP4 kernel (ncu: Mem 77% / Compute 44%).
- **llama's native-quant DECODE:** consumer ALU-decode of the resident raw superblock
  into the `%8==4`-padded `x_qs` layout (already in `qmatvec_gemm.cu` kernel1, NWARP 8,
  StageMeta PREDEC) — weights stay quantized in VRAM, decode-once.
- **bw24's exact-scale epilogue:** s32->f32 per-32-block dw*da + bias (bit-equivalent to
  dp4a), FP4 block-scale inside the mma, NVFP4 macro-scale folded in.
- **StaticPersistentScheduler over (out_f/BM x T/BN) + stream-K fixup** so 82 SMs stay
  saturated on any matmul shape.

### TWO source-verified corrections to the blueprint (do NOT build the wrong thing)

1. **No producer/consumer WARP split on sm_120.** CUTLASS sm120 does NOT do
   producer/consumer warpgroup specialization (that is sm90/sm100 wgmma). Verified:
   `sm120_blockscaled_mma_tma.hpp` `load()` issues TMA under
   `if (cute::elect_one_sync())` — a SINGLE elected lane, NumProducerThreadEvents=1 —
   and the SAME warps run `mma()` after. Build TMA-issue + mma in the same warps
   (elected-lane TMA), NOT a dedicated producer warpgroup. Keep NWARP=8.
2. **TMA cannot carry raw GGUF k-quant via a tensor descriptor.** CUTLASS's TMA tensor
   path only accepts e2m1/e2m3/e3m2/e4m3/e5m2 packed-float types
   (`is_sm120_f8f6f4`); ZERO GGUF Q4_K/Q5_K/Q6_K support exists in the CUTLASS tree
   (grep: 0 hits). A TMA tensor descriptor cannot model GGUF's interleaved 6-bit
   scales/mins. So for k-quant, use plain `cp.async.bulk` (1D non-tensor byte copy of
   the raw superblock) + mbarrier — a real but WEAKER primitive than the 2D descriptor
   TMA the 63% CUTLASS FP4 number came from. NVFP4 (already packed e2m1) CAN use the
   descriptor TMA.

### sm_120-exact tune

- PERSISTENT grid ~164 CTAs (2/SM), NWARP=8 (256 thr), matching the shipped MMQ port.
- smem budget ~21KB/CTA (8KB decoded-int8 wtile x2 double-buffer + 9KB raw ring x4
  stages + 4KB act tile) supports 2-4 CTAs/SM under the 99KB cap. NSTAGE=3-4.
- Double-buffer the DECODE-output tile so decode(k+1) overlaps mma(k) — the CUTLASS
  `copy_kblock`/`gemm_kblock` register-pipeline (848-849).
- mma per format: `m16n8k32.s8` for Q8_0/Q4_K/Q5_K, `m16n8k64.kind::mxf4nvf4.block_scale`
  for NVFP4. Occupancy/barrier levers are EXHAUSTED (NSTAGE=4 and tile redesign both
  reverted in tasks #20/#21) — gain MUST come from the FEED ceiling.

### argmax + ncu gate

1. End-to-end argmax (the real gate): run_dense 268, run_hybrid 271, MoE 1178 with
   `BW24_GEMM=1`. Any mismatch = revert. (`GEMM-PLAN.md:111`)
2. Perf gate: pp512 must rise from 2158. First fused cut should clear the current
   value materially; the honest target band is ~2600-3200 (NOT >6000).
3. ncu: the MIO/smem-throughput bound (Mem 77%/Compute 44% on the FP4 kernel) must
   drop — the weight read should move off the MIO pipe onto the TMA/bulk path. This is
   the single measurable lever.

### Honest projection vs llama

- **+25-55% first cut: 2158 -> ~2600-3200 pp512.** The single real lever is the
  TMA/cp.async.bulk feed taking the weight read off the MIO pipe (the FP4 kernel is
  self-measured MIO/smem-bound). That is worth ~+20-40% on the FP4 kernel.
- **vs llama 6139: closes to ~0.45-0.55x at best on first fusion.** Reaching parity
  needs TMA-feed + stream-K both mature. **PASSING llama is NOT supported** — llama
  already runs the same FP4 mma on the same silicon, so the format ceiling is shared.
  This buys SATURATION of a shared ceiling, not a higher ceiling.
- The biggest actual prefill lever for the NVFP4 benchmark may be ELSEWHERE: the bench
  shows sub-1% rows for attn_qkv + ffn_up, suggesting a shape-alignment / dispatch
  fallback bug worth more than the fusion. **Investigate that dispatch path FIRST**
  before churning the full hand-written CUTLASS-structure rewrite.

---

## RANK 3 (partial) — decode epilogue fusion + wire the existing CUDA graph

NOT a new fused kernel. Drop the VDR-interleave matvec relayout entirely (0% on this
silicon, probe-tied). Keep only the two real, small levers:

1. **In-kernel gate+up+SwiGLU+output-requant epilogue** (llama `has_fusion`,
   `mmvq.cu:526-668`). bw24 currently does 2x matmul_pre + 1 silu_mul as separate
   launches; fold to one kernel. Worth a few launches/layer.
2. **Wire the already-built CUDA graph into the benchmark.** `run_gen.rs` calls eager
   `decode_step`, NOT `generate_graph` — the shipped graph is currently UNMEASURED.

Honest: ~83 -> ~90-95 tok/s, reaching parity with llama 117 only via the already-shipped
quant-KV margin at ctx>4k. The matvec is already at 86% SOL (bandwidth wall ~118 tok/s);
do not relayout it.

---

## HONEST BOTTOM LINE — does the fused roadmap BEAT llama/vLLM on every single-model scenario?

**No.** It does not deliver a clean win on every scenario. Breakdown:

- **Prefill (NVFP4, the losing path): PARITY-HARD.** llama already runs the identical
  FP4 block-scale mma on sm_120 (`mma.cuh:1145`). bw24 shares the format ceiling; the
  TMA feed only improves saturation. Realistic outcome: 0.40x -> ~0.5x first cut,
  approaching parity after stream-K + feed mature. A win ABOVE llama on prefill is NOT
  supported by the format ceiling. The FA fp8 prefill win is Amdahl-capped to
  single-digit % until the GEMM lands.

- **Decode: PARITY-AT-BEST short-ctx, WIN long-ctx.** Decode is genuinely DRAM-bound
  and bw24's matvec is already 86% SOL (probe-tied with llama). The honest ceiling is
  ~118 tok/s. bw24 reaches parity via launch-overhead removal (epilogue fusion + graph
  wiring), and EDGES llama only at ctx>4k where bw24's 1/4-byte quant-KV reads beat
  llama's f16 KV read. That long-ctx margin is the one durable single-model decode win,
  and it is already shipped.

- **The genuine, defensible wins (a kernel no engine ships):**
  1. fp8-KV-into-tensor-core PV + cp.async double-buffer FA on sm_120 (rank 1) — real
     per-kernel win, accuracy-and-Amdahl-bounded.
  2. quant-KV decode margin at long context (shipped) — grows with ctx.
  3. TMA/bulk-fed persistent GEMM that also serves GGUF k-quant bit-exact, which
     CUTLASS cannot — parity on FP4 throughput, but a CAPABILITY (k-quant native +
     persistent + TMA feed) neither ships fused.

- **PARITY-HARD scenarios and WHY:** (a) NVFP4 compute-bound prefill — shared FP4
  format ceiling with llama. (b) Short-ctx single-stream decode — DRAM bandwidth wall,
  both engines at ~86-90% SOL, no relayout escapes it. (c) Raw FP4 GEMM throughput vs
  CUTLASS (63% peak) — bw24 will not beat it on FP4, but serves GGUF k-quant which
  CUTLASS cannot.

Build order: (1) investigate the sub-1% attn_qkv/ffn_up dispatch rows (cheap, possibly
the largest lever), (2) GEMM TMA-feed de-mirror (biggest absolute gap), (3) FA fp8-PV +
double-buffer (only above-parity per-kernel win), (4) decode epilogue + graph wiring
(parity polish). Do NOT claim a prefill win above llama; the honest target is
parity-approaching prefill + a real long-ctx decode edge + one genuinely-novel FA kernel.
