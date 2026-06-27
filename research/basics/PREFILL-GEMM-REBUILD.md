# PREFILL-GEMM-REBUILD — the path to beat llama pp512

Status: PLAN (read-only research complete; no code changed this workflow).
Target config: RTX 5090 Laptop, sm_120a / GB203, 82 SMs, 1536 thr/SM, 65536 regs/SM, ~99KB opt-in smem/SM, 847 GB/s achieved BW. TC peaks: FP16 117, int8 219, block-FP8 (mxf8f6f4) 381, block-FP4 (mxf4nvf4) 762 TFLOP/s.
Measured starting point: prefill pp512 = 2090 tok/s. GEMMs = 58% of prefill. SSM/attn cluster = ~27%.

Every projected pp512 number below is Amdahl arithmetic over ONE measured kernel ratio. Each phase is gated on an on-device re-measurement. Flags marked [MEASURE] are NOT yet proven on this silicon.

---

## 0. The decision in one paragraph

bw24's two prefill GEMMs (int8 cluster + FP4) are **smem-feed-bound by shared-memory bank conflicts**, not occupancy-bound and not compute-bound. On-device ncu (this laptop, today) shows bw24 `qmatvec_gemm_q8_0` at 5.3% tensor-pipe / 41.8M bank conflicts / 28.4% warps active, vs llama `mul_mat_q<8,128>` on the **same Q8_0 model** at 38.2% tensor-pipe / 0.13M conflicts / **16.7% warps active** — i.e. llama hits 7.2x the tensor throughput at *half* bw24's occupancy. That single fact kills the "the pad costs occupancy and nets flat" objection that reverted the earlier swizzle/pad experiments (qmatvec_gemm.cu:56-60, Task #33): the bound is the conflict-replay on the smem feed, which a K-stride pad removes for free. **Chosen path: extend the hand-rolled warp-mma + cp.async kernel with the llama-MMQ smem tile (pad K-stride to `%8==4`, separate bulk-decode behind a barrier, 8 warps with a re-tiled accumulator) — NOT adopt CUTLASS wholesale, NOT TMA+warp-spec, NOT marlin.** The de-risking testbed is the int8 kernel (bit-exact gate available); the **beat-6139 lever is the FP4 kernel** (762 TFLOP block-scale, which llama/vLLM cannot run on their consumer-Blackwell fast paths), which inherits the identical fixed tile.

---

## 1. CHOSEN REBUILD PATH (and why it beats the alternatives ON THIS LAPTOP)

### Chosen: extend the hand-rolled MMQ-style kernel (pad + bulk-decode-behind-barrier + 8-warp re-tile)

The transferable structure is **llama MMQ's**, ported (not copied 1:1 — bw24's accumulator/token-span differs). Three structural levers, in dependency order:

1. **Pad the smem K-stride so the per-row 32-bit-element count is `%8==4`** (kills bank conflicts). llama: mmq.cuh:177-178 ("K % 8 == 4 for mma"), the constants `MMQ_MMA_TILE_X_K_*` (mmq.cuh:219-225) and their `static_assert(... % 8 == 4)` (mmq.cuh:227-234). bw24 today: `__shared__ int8_t sW[NSTAGE][BM][BK=32]` / `sA[NSTAGE][BN][BK=32]` (qmatvec_gemm.cu:324-325) — 32 int8 = 8 u32/row, `8 % 8 == 0` → the 8 rows of every 8×8 ldmatrix subtile alias the same bank groups → measured 6.6-way conflict.
2. **Separate weight decode into a bulk producer phase** writing int8 weights + scales to distinct smem arrays, gated by a barrier, so the mma loop is pure ldmatrix+mma (zero decode/global-read on the chain). bw24 ALREADY does this for Q4_K/Q5_K (FIX-A, `USE_PREDECODE`, qmatvec_gemm.cu:336, :388) but NOT for Q8_0/Q6_K/NVFP4 (they keep inline-global decode). llama reference: `load_tiles_q8_0` bulk decode-to-smem (mmq.cuh:317-340 / ~:774-834 in upstream).
3. **Double nwarps 4→8 (256 thr/CTA)** — but ONLY after step 1, and ONLY with the accumulator re-tiled. bw24's current accumulator is `facc[BN/8=16][4]` per warp (each warp spans ALL 128 tokens), 64 f32/lane → the prior naive 8-warp + BN-shrink redesign (PERF-1c) collapsed to register/occupancy bound and was reverted (1284→729). llama avoids this with a **minitile accumulator**: `granularity`/`ntx` so each warp owns FEWER tokens (mmq.cuh:1222-1224, `rows_per_warp = 2*granularity`, `ntx = rows_per_warp/tile_C::I`). Adopt llama's minitile scheme together with the warp-count bump, never separately.

llama runs 8 warps (256 thr) by default: `mmq_get_nwarps_device() = 256/warp_size = 8` (mmq.cuh:307-313). It reaches 38% tensor-pipe at 16.7% occupancy — proving the feed, not the warp count, is the lever.

### Why NOT CUTLASS-SM120 collective (flashinfer fp4_gemm_cutlass_sm120.cu / deep_gemm sm120 collective)

- **It WOULD compile and run on sm_120a** (verified by the runs-on-laptop verdict): deep_gemm's `sm120_gemm_tma_warpspecialized...hpp` uses `ArchTag >= 90`, `cute::cluster_*` (clusters: sm_120 HAS), TMA `cp.async.bulk` (sm_120 HAS), and `setmaxnreg` (available on sm_120). It does NOT need wgmma (SM90) or tcgen05 (SM100). flashinfer uses `CutlassTileConfigSM120::CtaShape128x128x128B`, warp-mma based. **Do NOT let the plan drift into claiming CUTLASS "won't run" — it would.** The rejection is on perf-lever and cost grounds:
- **Wrong lever.** bw24 is bank-conflict-bound on the smem feed. TMA replaces cp.async (an address-gen / load-issue optimization). The deep_gemm survey's own decisive finding: TMA+warp-spec is incremental (single-to-low-double-digit %, possibly net-neutral) on a kernel that is *not* load-issue-bound. It does not touch bank conflicts.
- **Wrong occupancy fit.** The collective's 128×128×128 tile at ~232 regs/thread gates the laptop to ~1 CTA/SM. The laptop's 65536 regs/SM and 99KB smem are the binding constraints; the datacenter-tuned collective tile does not fit them well.
- **High integration cost** (link CUTLASS, instantiate `CollectiveBuilder<Sm120,...>`, weight repack to the swizzled layout) for the wrong lever.

The MMA *atom* ports 1:1 (bw24 already emits the identical `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`), so there is nothing to gain there. TMA is parked as a **Phase 4 optional probe**, not the rebuild.

### Why NOT marlin-style (vLLM marlin / W4A16 fp16-mma-after-dequant)

marlin's ceiling on sm_120 is the **FP16 TC peak = 117 TFLOP/s** — LOWER than int8's 219 and far below FP4's 762. The lop3 dequant-overlap trick (dequant.h) is a real fix for an inline-decode stall, but bw24 ALREADY moves decode off the mma chain (FIX-A); converting to FP16 mma would cap throughput below the int8 path it replaces. marlin's `machete` variant needs wgmma (SM90 silicon bw24 lacks) → not feasible. Reject.

### Why NOT "make int8 reach llama and call it a win"

The int8 cluster's hard ceiling is 219 TFLOP. Even a perfectly-tuned int8 grind lands ~4.5x short of 6139 (see §3). int8 is the **de-risking testbed for the swizzle** (bit-exact dp4a gate, rel<1e-3) and a tie-to-llama-Q8_0 result — it is NOT the beat. The beat is FP4.

---

## 2. EXACT FIRST KERNEL TO BUILD

**File:** `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec_gemm.cu`, the `qmatvec_gemm_kernel<QT>` template, instantiated for **Q8_0 first** (GQT_Q8_0, the inline-decode path, simplest decode, bit-exact gate).

**This first kernel is a pure smem-stride/address change — NOT a tile redesign.** It changes only the smem layout + the three address sites; it does NOT touch the accumulator, the token span, the warp count, NSTAGE, or the mma atom. This is what structurally distinguishes it from the reverted PERF-1c (which shrank BN and collapsed occupancy).

### 2.1 The change (Phase 1)

- **mma primitive:** unchanged — `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`mma_s8_m16n8k32`, qmatvec_gemm.cu:70-76).
- **Tile:** unchanged — BM=64 rows, BN=128 tokens, BK=32 contraction, NWARP=4, NSTAGE=3.
- **Pipeline:** unchanged — cp.async.cg ring (NSTAGE=3) + ldmatrix A(x4.b16)/B(x2.b16), single `__syncthreads`/K-step, FIX-A bulk-decode for Q4_K/Q5_K.
- **THE EDIT — pad the K-stride to break the `%8==0` aliasing:**
  - `__shared__ int8_t sW[NSTAGE][BM][BK]` → `[NSTAGE][BM][BK + PAD]` with **PAD = 16 int8** (so per-row 32-bit count = (32+16)/4 = **12 u32, `12 % 8 == 4`** — matches llama's rule, and BK+PAD=48 keeps 16B alignment so `ldmatrix.x4.b16` + `cp.async.cg`-16 stay legal). Same for `sA[NSTAGE][BN][BK+PAD]`.
  - Apply the IDENTICAL stride at the **store side** (`decode_stage_inline` for Q8_0: qmatvec_gemm.cu:410-419; and `fetch_activation`: :365-384 — the `cp_async16(&sA[s][n][0/16],...)` and the int8 stores) AND the **load address** (`ld_A_s8` / `ld_B_s8`, :100-114 — the `stride_bytes` arg becomes `(BK+PAD)`, the address arithmetic `(threadIdx.x%16)*(stride/4)` already parameterizes on stride, so only the caller's stride changes).
  - This is mechanically simpler than the FP4 path's `SWZ_CHUNK` XOR (:907): a flat pad, no XOR remap, so the bit-exact gate is trivial to keep green.
  - NOTE: prefer the **pad over the XOR-swizzle** here because the int8 sW/sA are stored row-major `[row][k]` and read with a row-stride `ldmatrix`; the pad scatters the 8 subtile rows across distinct bank groups with zero address-remap risk. (The 1-bit XOR on the FP4 path was insufficient — still 6.6-way; a true conflict-free transform needs the pad OR a full 3-bit swizzle. Use the pad.)

### 2.2 smem budget (verified ≤99KB)

Per-CTA at NSTAGE=3, BM=64, BN=128, BK+PAD=48:
- sW: 3 × 64 × 48 = 9,216 B
- sA: 3 × 128 × 48 = 18,432 B
- sWd/sWb/sAd/sAsum: 3 × (64+64+128+128) × 4 = 4,608 B
- sWraw (Q8_0: USE_PREDECODE=false → RAW_W=1, NSTAGE_RAW=2): 2 × 64 × 1 = 128 B
- **Total ≈ 32.4 KB/CTA.** vs current (BK=32) ≈ 27.6 KB/CTA. At ≤99KB/SM this still allows **3 CTAs/SM** (32.4×3=97.2KB). Current is 4 CTAs/SM (27.6×3.5). [MEASURE] whether the drop 4→3 CTAs/SM matters — the diagnosis says it will NOT (llama wins at lower occupancy), but the pass-criterion ncu (below) measures it directly. If 3 CTAs/SM regresses, fall to **PAD = 4 int8** (BK+PAD=36, 9 u32, `9%8==1` — that is the dp4a rule, weaker for mma; only use if the 16B-align variant must shrink) or accept the larger pad and rely on the conflict win dominating.
- block-scale feed (int8): scales are plain f32 in sWd/sAd, applied per-K-step in f32 (already bit-exact vs dp4a). No block-scale fragment machinery for int8.

### 2.3 The gate (argmax + per-kernel rel)

- **kernel_check** must stay green: int8 GEMM is bit-exact vs the dp4a path (s32 accumulate is exact; only final f32 scale rounding differs) → `rel < 1e-3`. The pad must be applied identically at store+load or operands corrupt — kernel_check catches it immediately.
- **argmax gate:** the run-gen argmax token IDs must be unchanged (the known 268/271/1178 argmax at the documented prompt). Any argmax drift = the pad address mismatch; revert.

### 2.4 First on-device ncu measurement (confirms the bound BEFORE committing 8-warp work)

```
sudo -n /usr/local/cuda-13.1/bin/ncu -k qmatvec_gemm_q8_0 -c 1 \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  env BW24_GEMM=1 BW24_NGEN=1 BW24_PROMPT="$(python3 -c 'print("word "*500)')" \
  ./target/release/run-gen /data/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf
```

**Pass criterion (all three):**
1. `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` drops from 41.8M toward llama's ~0.13M (target: <5M).
2. `sm__pipe_tensor_cycles_active` rises from 5.3% toward 30%+ (target: >20% with the pad alone, before 8 warps).
3. Measured pp512 rises from 2090 AND argmax unchanged.

**If conflicts drop but tensor-pipe stays flat:** the next lever is the 8-warp re-tile with llama's `granularity=16` minitile accumulator (mmq.cuh:1222-1224) — but only after the conflict fix lands and is measured. If conflicts do NOT drop, the pad address math is wrong (re-derive against `ld_A_s8`).

---

## 3. HONEST REACHABLE pp512 — does the stack beat 6139?

**The GEMM rebuild alone does NOT reach 6139. The FP4 path + the SSM/attn cluster together CAN, but it is not guaranteed and must be measured.**

### Critical baseline correction (measured on-device, this laptop)

The 6139 bar is **NOT the Q8_0 model**. llama Q8_0 pp512 here = **~2329** (3 reps) / 1931 (profiled). bw24's 2090 already roughly ties llama on Q8_0. The **6139 is llama's NVFP4 model** (smaller, FP4 MMQ). So:
- The int8/Q4_K/Q5_K cluster cannot beat 6139 at ANY tensor-pipe % — its ceiling is 219 TFLOP; the repo's own FP8-GEMM-PLAN arithmetic puts a perfect int8 grind ~4.5x short of 6240.
- **FP4 (762 TFLOP, mxf4nvf4) is the ONLY route past 6139.** llama/vLLM cannot run mxf4 on consumer Blackwell fast paths — that is bw24's structural edge.

### Amdahl math

Two independent limits on the linear-scaling fantasy:
- **Tensor-pipe % ≠ runtime.** The realistic per-kernel speedup is bounded by the **SM-throughput ratio 50.6/15.9 = 3.18x**, not the 7.2x tensor-pipe ratio.
- **Non-GEMM is untouched.** GEMM = 58% of prefill, non-GEMM (SSM/attn ~27% + other ~15% = ~37.6% of *time*) is unchanged by any GEMM work. Hard ceiling with GEMM time → 0: `2090 / (1 - 0.58×... )`. Using the measured 58% GEMM time share: even infinitely fast GEMMs cap the stack at **~5559 tok/s**. **6139 is literally unreachable by GEMM work alone.**

Realistic projection ladder (each step [MEASURE]):
| step | GEMM change | projected pp512 |
|---|---|---|
| baseline | — | 2090 |
| int8 cluster smem-pad (3.2–7.2x on the int8 GEMMs, 40% of prefill) | pad only | ~2880–3180 |
| + FP4 smem-pad transfer (13.3%→~38% tensor, ~2.8–5x on the FP4 GEMM) | pad on FP4 | ~3300–4400 |
| + SSM/attn cluster fixed (see below) | non-GEMM | the only path to >5559 |

### The remaining levers to actually beat 6139

GEMM-pad-ALONE lands **~3300–4400 pp512**. To clear 6139:

> **FRAMING (no-magic principle): competitor numbers are a FLOOR, not a ceiling.** Same silicon, no magic — every bw24 kernel below a competitor's throughput is below ONLY because we haven't yet copied the structural detail that makes theirs fast (proven here: the %8==4 pad took int8 from 5.3%→llama's 38% tensor-pipe). So int8/Q4_K/Q5_K MATCHES llama (copy MMQ), the SSM/attn cluster MATCHES llama (those are unoptimized correctness-first ports — copy llama's FA/conv kernels, NOT a fixed ~5559 cap), and FP4/FP8 block-scale (762/381 TFLOP, consumer-Blackwell-only) is the EDGE that pushes strictly past. True target = max(competitor-best-on-this-config) + bw-edge ≥ 6139. The "~5559 GEMM-only ceiling" only holds if SSM/attn stays unoptimized — it won't.

1. **Push the FP4 kernel toward a high fraction of 762 TFLOP** (the pad is the shared prerequisite; the *means* must be the FP4 kernel, not int8). Even FP4 at a modest **20–30% of 762** clears 6139 with margin. **[CORRECTION: an NVFP4 GGUF IS on disk — /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf — benched all session. FP4 pp512 IS measurable now; the earlier "no model on disk" claim was wrong.]**
2. **The 27% SSM/attn cluster needs its own attention** or it caps the stack at ~5559 regardless: `ssm_conv1d` ~10.9%, `fa_prefill` ~10.4% (1 warp/CTA — large lever), `gdn_scan` ~6%. This is separate from the GEMM rebuild (Task #9 FA3/FA4). `fa_prefill` at 1 warp/CTA is the next-largest lever after GEMM.
3. **MTP/EAGLE is decode, NOT prefill** (Task #7, already shipped) — not relevant to pp512.

**Honest verdict:** GEMM rebuild → tie/modest-beat on int8, and the FP4 path is the structural route past 6139, but reaching 6139 also requires the SSM/attn cluster work. Claiming "GEMM rebuild beats 6139" would be dishonest; the measured ceiling of GEMM-only work is ~5559.

---

## 4. BUILD ORDER (phased, each argmax-gated + ncu-measured, with revert criterion)

### Phase 1 — int8 smem-pad (de-risk the swizzle on the bit-exact path)
- **Do:** §2.1 pad on `qmatvec_gemm_kernel<Q8_0>` (sW/sA stride 32→48, identical at store+load).
- **Gate:** kernel_check rel<1e-3; argmax unchanged; §2.4 ncu pass-criterion (conflicts <5M, tensor-pipe >20%).
- **Revert criterion:** if pp512 does not rise above 2090 OR argmax drifts OR conflicts do not drop → revert the pad, re-derive the address math. If conflicts drop but pp512 flat at 3 CTAs/SM (occupancy regression dominates), try PAD variants (§2.2) before abandoning.

### Phase 2 — extend pad to the rest of the int8 cluster (Q4_K/Q5_K/Q6_K) + 8-warp re-tile
- **Do:** apply the identical pad to the FIX-A pre-decode store paths (`decode_stage`, :388) for Q4_K/Q5_K/Q6_K. THEN bump NWARP 4→8 WITH llama's minitile accumulator (`granularity=16`, `ntx` loop, mmq.cuh:1222-1224) so each warp owns fewer tokens and the f32 accumulator shrinks.
- **Gate:** kernel_check per-dtype rel; argmax; ncu tensor-pipe per dtype.
- **Revert criterion:** the 8-warp re-tile is the PERF-1c failure mode — if pp512 regresses (the 1284→729 signature), revert the warp bump ONLY (keep the pad, which is independently green from Phase 1). The pad and the warp-count are independently revertable.

### Phase 3 — FP4 smem tile rebuild (the beat-6139 lever)
- **Do:** replace the FP4 path's insufficient 1-bit `SWZ_CHUNK` (qmatvec_gemm.cu:907, still 6.6-way) with the proven Phase-1 pad transform applied to `sWq`/`sAq` (qmatvec_gemm.cu:994-997). Same mma atom (`mma_mxf4_m16n8k64`, :889), same block-scale feed (ue4m3 in sWsc/sAsc).
- **Gate:** kernel_check FP4 rel (vs the dp4a NVFP4 reference, maxrel was 0 on the probe); argmax. ncu on `qmatvec_gemm_nvfp4_fp4`: conflicts drop from 53.2M, tensor-pipe rises from 13.3% toward 30%+, target a high fraction of 762 TFLOP.
- **[MEASURE BLOCKER]:** pp512 for FP4 requires an NVFP4 GGUF — NONE on disk on this laptop today. Obtain/convert an NVFP4 model before claiming any FP4 pp512. The ncu conflict/tensor-pipe metrics on the current FP4 path CAN be measured now to confirm the bound.
- **Revert criterion:** if the pad does not drop FP4 conflicts (the FP4 sWq/sAq are u32-packed, 8 u32/row — verify the pad lands them in distinct banks; may need PAD=4 u32 → 12 u32/row `%8==4`), revert and reconsider a 3-bit swizzle.

### Phase 4 — (optional, gated) TMA probe + SSM/attn cluster
- **TMA probe:** only if Phases 1-3 land and the FP4 kernel is then load-issue-bound (not conflict-bound) per ncu. Swap cp.async→`cp.async.bulk` on the existing cooperative kernel (NOT a CUTLASS rewrite, NOT warp-specialization). Expected incremental (single-to-low-double-digit %). Revert if net-neutral.
- **SSM/attn cluster (Task #9):** `fa_prefill` 1-warp→multi-warp is the largest non-GEMM lever; required to push the stack past ~5559 toward beating 6139. Separate workstream, separate gates.

---

## 5. Key file references
- bw24 GEMM (edit target): `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec_gemm.cu`
  - int8 smem tiles `:324-325`; `ld_A_s8`/`ld_B_s8` (stride-parameterized) `:100-114`; int8 inline decode `:410-419`; fetch_activation `:365-384`; FIX-A pre-decode store `:388`; mma atom `:70-76`.
  - FP4 path: tiles `:994-1001`; insufficient 1-bit swizzle `SWZ_CHUNK` `:907`; ld_A/B_mxf4 `:916-938`; mma atom `mma_mxf4_m16n8k64` `:889-898`; fetch_act swizzle `:1034`.
  - the dismissed-as-no-op pad comment to overturn: `:56-60`.
- llama MMQ reference: `/data/projects/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`
  - conflict-killing pad rule `:177-178`; tile constants + static_asserts `:219-234` (note FP4 also pads, `:220`,`:232`); 8-warp `:307-313`; bulk decode-to-smem `load_tiles_q8_0` `:317+`; minitile accumulator `granularity`/`ntx` `:1222-1224`; pure-mma vec_dot tiles `:1218-1243`.
- FP8 next step (NOT built, separate task): `/home/avifenesh/projects/bw24/research/basics/FP8-GEMM-PLAN.md` (Task #22).
- Superseded framing: `/home/avifenesh/projects/bw24/research/basics/PREFILL-LEVERS-RANKED.md` — its occupancy-carveout Rank-1 is disproven by ncu (bound is bank conflicts, not CTA count); redirect to the smem-tile pad.
- DIAGNOSIS + adversarial verdicts (this workflow): the conflict bound and the 6139-is-NVFP4 baseline correction are on-device-verified; every pp512 projection is Amdahl arithmetic and must be re-measured.

## 6. Verified-vs-must-measure ledger
- VERIFIED (source + on-device ncu, this laptop): bw24 int8 5.3% tensor / 41.8M conflicts / 28.4% warps; llama Q8_0 38.2% / 0.13M / 16.7% warps; llama uses `%8==4` pad (mmq.cuh:227-234); bw24 sW/sA unpadded `%8==0`; FP4 MMQ also pads; no FP8 kernel in bw24; build flag `-gencode arch=compute_120a,code=sm_120a`; CUTLASS sm120 collective WOULD compile (no wgmma/tcgen05); llama Q8_0 pp512 ~2329 (6139 is the NVFP4 model).
- [MEASURE] (NOT yet proven on this silicon): pp512 after the pad; whether 4→3 CTAs/SM regresses; FP4 pp512 (needs an NVFP4 GGUF — none on disk); TMA incremental %; SSM/attn cluster speedups; that the int8+FP4+SSM stack actually clears 6139.
