# bw24 — Session Handover

_Written 2026-07-03. Read this cold, then continue. bw24 = from-scratch Rust+CUDA LLM inference engine, target rig RTX 5090 Laptop (sm_120a, Blackwell consumer, 24GB, 847 GB/s wall). AWS mirror = L40S (sm_89, Ada) on branch `arch/sm89-l40s`._

---

## THE GOAL (session-scoped `/goal` directive — three arms)

**Arm 1 — PRIMARY.** Take bw24 to its absolute edge on THIS EXACT rig. Vendor the best kernel pieces from ALL inference engines and their libraries (llama.cpp, vLLM, SGLang, flashinfer, TensorRT-LLM, CUTLASS). Mark those best-in-class kernels as the *floor* — then tune ON TOP of that floor for this specific rig: "every little drip of memory, every little compute power." Fuse kernels, rework every line, no ns unused, no missed parallelism, hide ops behind other work, saturate any idle memory/compute.
  - **Announcing a limit requires empirical measured proof across ALL directions** — not just the one being worked when the wall was hit. No hand-waving "this is as far as it goes."
  - Also surface: what I (the model) know that the user doesn't that yields big wins, and what the web knows.

**Arm 2 — secondary.** Drive the same work on the L40S AWS box. Branch `arch/sm89-l40s` (worktree `/home/avifenesh/projects/bw24-sm89`). Ada has NO FP4/block-scale but HAS int8 MMA + dp4a + FP8-plain + cp.async — so the k-quant int8 MMA path IS portable there.

**Arm 3 — secondary.** Every grind step becomes labeled training material for a **diffusion-autotuner**: multi-arch config→perf JSONL corpus (`research/tune-data/rig5090.jsonl`, mirror for sm_89). Every win positive, every loss positive.

**Support scope (roadmap, not all now):** NVFP4/mixed dense (Qwen3.5/3.6, Gemma-4) AND MoE (Qwen3.6-35B-A3B, Gemma-4 MoE, MiniMax-M2.7, DeepSeek-4-flash) across VRAM-full / VRAM+CPU-split / NVMe-spill tiers; MoE hot-expert caching; quantized KV cache w/ reuse+eviction; MTP + striped-vocab MTP spec decode; best GEMM/GEMV/matvec usage.

**User works the local grind WITH me directly.** Not a hands-off delegation.

---

## OPERATING RULES (hard constraints)

- **Model orchestration:** sonnet 5 for small tasks, opus 4.8 for medium, big tasks are mine (Fable) with updates to user. "no too much fable in parallel you get throttled" — do NOT fan out many Fable subagents.
- **Caveman mode ACTIVE (full):** terse output, drop articles/filler/hedging, fragments OK. BUT code/commits/PRs written normally, and security/irreversible-action confirmations written normally.
- **CLAUDE.md banned behaviors:** (1) No overstating feature scope — estimate by actual code change, not surface area. (2) No "call it a day" suggestions — user decides session boundaries.
- **Bench protocol (non-negotiable):**
  - Clock-lock for defensible ratios: `sudo -n nvidia-smi -lgc 1860,1860` … `-rgc` to release.
  - Peak numbers matching serve scripts: `gpu-full-power on` (`~/.local/bin/gpu-full-power`, boost=25, 175W).
  - N=5 median. Monitor thermal sag (clocks.sm should hold ~1860).
  - `run_gen.rs` prints wrong tok/s (known timing bug — primes prompt inside timed region). **Trace internals are truth**, not the printed number.
  - Numeric-token prompt for clean pp512: IDs 101..612, **argmax must == 82**.

---

## WHERE THINGS STAND (measured, 9B-NVFP4) — UPDATED 2026-07-03 after Q4_K/Q5_K MMQ landed

| metric | bw24 | llama.cpp | ratio |
|---|---|---|---|
| pp512 INTERLEAVED A/B (honest protocol) | **4655** | 5092 | **0.91x** |
| decode tg128 graph @ctx128 (clock-locked, NOT interleaved) | **109.6** | 117.8 | **~0.93x** |
| decode tg128 graph @512 / @2048 | 109.5 / 106.3 | — | — |

**MEASUREMENT LESSON (2026-07-03): cross-session pp512 numbers lied.** Sequential runs gave bw24 5531 vs llama 5451 ("parity") — but interleaved same-minute A/B gives 4655 vs 5092 = 0.91x. Root cause: llama holds 1852MHz during its run; bw24 sags to 1710MHz under the same clock lock (bw24 draws more power for the same work → worse perf/W → thermal sag). ALL ratio claims must be interleaved A/B from now on. bw24 has BOTH a remaining kernel gap (~9%) AND a power-efficiency gap (clock sag under load). Decode ratio not yet re-measured interleaved.

**Decode session 2026-07-03 (three landed levers, all bit-identical + gated):**
1. warp-per-block coalesced q8_1 epilogues (silu_mul_scaled_q8_1, quantize_q8_1, norm pass-2): 88.4→95.1 graph.
2. float4 warp-per-4-blocks norm pass-2 + 1024-thread CTA: 95.1→104.4.
3. adaptive FA split (32 keys ≤1024 ctx, 64 above; `BW24_FA_SPLIT` seam): →109.8/109.5/106.3.

**ncu parity probes (measured, clock-locked):** bw24 NVFP4 matvec = llama mul_mat_vec_q<40> EXACTLY (42% DRAM, ~74us both). Q6_K lm_head 1.07ms/tok — llama same (1.07ms). Matvec is NO LONGER the gap. Remaining 0.93→1.0x gap: FA decode kernels (bw24 fa_decode_vec_q 126us vs llama flash_attn_ext_vec 10.5us/launch — llama fuses whole-layer attention, bw24 splits+combines), graph-tail/launch overhead, elementwise remnants (l2_norm, scale, sigmoid still separate launches; llama has them too). MR=4 multirow kernel crashes ILLEGAL_ADDRESS (pre-existing, non-default — fix or remove).

Commits: **16e896c** (`feat(prefill): vendor llama Q4_K/Q5_K int8-MMA MMQ GEMM — pp512 1874->4576 clock-locked (2.4x)`) on top of **851e80f** (NVFP4 MMQ) + **e04c5f0** (sweep tune-seams, harness-agent work committed). Mirrored to `arch/sm89-l40s` as **7561a06** (BW24_CUDA_ARCH=89 build green; Q4_K/Q5_K int8-MMA path has NO FP4 gate — portable). Training record appended to `research/tune-data/rig5090.jsonl` (**0728173**).

**kernel-check:** ALL GREEN incl. new MMQ-q45k oracle gate (rel 5e-3..7e-3, gate 2e-2). argmax==82 on/off MATCH.

**PREFILL kernel-diff vs llama (nsys, same prompt, 2026-07-03) — the honest 9% decomposed:**
| kernel | bw24 | llama | delta |
|---|---|---|---|
| NVFP4 MMQ | 27.1ms | 25.8ms | +1.3ms (llama stream-K) |
| Q4_K+Q5_K MMQ | 23.4ms | 19.0ms | **+4.4ms** (llama stream-K + fixup; bw24 vendored xy-tiling only) |
| SSM prep (repack 3.6 + transpose 2.4 + conv_pad 1.1 + conv_silu 1.0) | 8.1ms | ~2.2ms (one concat_non_cont + ssm_conv) | **+5.9ms** |
| FA prefill | 1.9ms | 0.66ms | +1.2ms |
| gdn_scan | 17.6ms | 17.9ms | parity |
| scale_f32 (NVFP4 macro-scale bcast) | 4.2ms | ~2.9ms (k_bin_bcast mul) | +1.3ms |

Next prefill levers in order: (a) port stream-K to the q45k MMQ (llama's mul_mat_q_stream_k_fixup — biggest single delta), (b) fuse the SSM prep chain (transpose+repack+pad+conv into 1-2 kernels — pure bw24 self-inflicted, llama does ONE concat), (c) FA prefill tile config, (d) fold macro-scale into MMQ epilogue.

**Ranked levers (updated post-decode-session):**
1. **FA decode port** (llama fattn-vec structure: q8_1 Q + dp4a on raw K bytes, no smem staging) — agent running in background; bw24 FA ~1.3ms/tok vs llama ~0.6ms.
2. **MTP K=4 exactness fix** — debug agent running (worktree); K=1/2 PASS, K=4 diverges on ALL kernel paths (garbage special tokens ~idx 25 => indexing/state bug, not numerics). MTP is the profit lever: K=1 already 0.85x of plain at 81% acceptance.
3. Prefill: close the honest 9% interleaved gap + the clock-sag perf/W gap (bw24 draws more power per unit work — audit which kernels burn ALU needlessly; the MMQ tune-seams sweep is the tool).
4. Web-sweep items FOLDED into ROADMAP.md items 11-15 (DFlash, TCQ, FR-Spec, tensor-split=DEAD, ST-MoE prefetch). DONE.
5. L40S box i-TERMINATED is GONE (terminated, not in any account/region). Arm 2 = sm89 branch compile-mirrors only until a new box is provisioned.

---

## CURRENT TASK — DONE (2026-07-03). vendor Q4_K/Q5_K MMQ ✅

Landed as `cu/llama_mmq_q45k.cu` (new TU, self-contained), unified `qmatvec_mmq` dispatch in
`mmq_ffi.rs`, `mmq_supports` extended to QT_Q4_K/QT_Q5_K. All 6 NEXT STEPS below executed, all
gates green. Harness-agent's lib.rs/qmatvec_gemm.cu tune-seam WIP was committed separately
(e04c5f0). Section below kept for reference.

### Key correctness facts already established (do not re-derive)
- On Blackwell (`BLACKWELL_MMA_AVAILABLE`), **Q4_K and Q5_K both dequantize to int8 at tile-load, then run the shared int8 MMA inner loop.**
- Their `vec_dot_mma` is **`vec_dot_q8_1_q8_1_mma`** (mmq.cuh:1330), NOT q8_0 — because K-quants carry BOTH a per-subblock scale AND a min-offset, matching the q8_1 dual (d, m) layout.
- Both map to tile size `MMQ_MMA_TILE_X_K_Q8_1` (mmq.cuh:254-255).
- int8 MMA op: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (mma.cuh:946).
- Needs a **q8_1 activation quantizer** (distinct from the FP4 activation path already vendored).

### llama.cpp source map (`/data/projects/llama.cpp/ggml/src/ggml-cuda/`)
- `mmq.cuh`:
  - `#define MMQ_TILE_NE_K 32` :179
  - `MMQ_MMA_TILE_X_K_Q8_0 = (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4)` :219
  - `MMQ_MMA_TILE_X_K_Q8_1` = same expr :222
  - `MMQ_TILE_Y_K = (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)` :270
  - `load_tiles_q4_K` :2093
  - `load_tiles_q5_K` : follows q4_K (~2230+, the sed at 2240-2270 showed its body)
  - `unpack_scales_q45_K` :2083 — code: `return ((scales[(ksc%2)+(ksc!=0)] >> (4*(ksc&(ksc/2)))) & 0x0F0F0F0F) | ((scales[ksc/2] >> (2*(ksc%2))) & 0x30303030);`
  - `vec_dot_q8_1_q8_1_mma` :1330  ← THE shared int8 MMA vec_dot for both k-quants
  - `vec_dot_q8_0_q8_1_mma` :1159 (reference; q8_0 variant, read for the tile_A/tile_B/tile_C + load_ldmatrix + mma pattern)
  - traits: Q4_K :3358, Q5_K :3368
- `vecdotq.cuh`: `vec_dot_q4_K_q8_1_impl_mmq` :530 (VDR_Q4_K_Q8_1_MMQ=8 :502), `vec_dot_q5_K_q8_1_impl_mmq` (VDR_Q5_K_Q8_1_MMQ=8 :558)
- `mma.cuh`: int8 mma :946
- `ggml-common.h`: `block_q8_1` :258, `block_q4_K` :327, `block_q5_K` :345, `K_SCALE_SIZE=12`, `QI4_K`, `QR4_K=2`, `QR5_K=2`, `QK8_0=32`

### bw24 framework to reuse (`crates/bw24-engine/`)
- `cu/llama_mmq_nvfp4.cu` (593 lines) — the vendored NVFP4 MMQ framework. Has: `tile<>`/`load_ldmatrix`/`load_generic`/`mma` machinery, `load_tiles_nvfp4_nvfp4`, `vec_dot_nvfp4_mma`, `mmq_write_back_nvfp4`, `mul_mat_q_nvfp4`, `quantize_mmq_nvfp4_kernel`, C-ABI `bw24_mmq_nvfp4(w_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch, stream)`. Constants MMQ_NWARPS=8, MMQ_Y=128, MMQ_X=128, MMQ_TILE_NE_K=32. **Refactor shared tile/mma/write-back pieces into `cu/llama_mmq_common.cuh` if cleaner.**
- `src/mmq_ffi.rs` (80 lines, from 851e80f) — FFI to `bw24_mmq_nvfp4` + `bw24_mmq_nvfp4_act_bytes`. Has `mmq_supports(&self, w)` → true if `QT_NVFP4 && in_features()%64==0`. Declared `pub mod mmq_ffi;` at lib.rs:23.
- `build.rs` — compiles .cu as fatbin AND `llama_mmq_nvfp4.cu` as static lib (~lines 60-100), flags `-gencode arch=compute_120a,code=sm_120a`. **Must register the new .cu here.**

### QT tags (weight quant type ids in bw24)
`QT_Q8_0=0, QT_Q4_K=1, QT_Q6_K=2, QT_Q5_K=3, QT_NVFP4=7`

### Uncommitted work by harness-agent (do NOT stomp — coordinate)
- `src/lib.rs` (WIP): added `gemm_fatbin_path()` (reads `BW24_GEMM_FATBIN`, default `GEMM_FATBIN_PATH`), `k1_launch_override()` (OnceLock parsing `BW24_GEMM_K1_LAUNCH="BM,BN,NWARP"`). Launch sites ~1309/~1347 use `k1_launch_override().unwrap_or((128,128,8))` for is_k1 qtypes (QT_Q8_0|Q4_K|Q5_K). Matmul dispatch gates on `BW24_MMQ` + `mmq_supports` at ~855 and ~964. FA dispatch: fa_prefill ~1400, fa_decode_vec_q ~1501/1556.
- `cu/qmatvec_gemm.cu` (WIP): adding `#ifndef` guards around tunables (BM=64,BN=256,BK=32,NWARP=8,NSTAGE=3,K1_BM=128,K1_BN=128,K1_NSTAGE=2 — lines ~70-128). GQT tags lines 45-49.

### Runtime tune-seams (no Rust rebuild needed)
- `BW24_GEMM_FATBIN=<path>` — swap swept fatbins.
- `BW24_GEMM_K1_LAUNCH="BM,BN,NWARP"` — override k1 launch geom.
- Env gates: `BW24_FAST, BW24_GEMM, BW24_MMQ, BW24_MMVQ, BW24_FA_VEC`.

---

## NEXT STEPS (in order)

1. **Write Q4_K/Q5_K MMQ port** as new CUDA file(s) reusing the `llama_mmq_nvfp4.cu` framework (refactor shared → `cu/llama_mmq_common.cuh` if cleaner). Add:
   - `bw24_mmq_q4_K` + `bw24_mmq_q5_K` C-ABI launchers.
   - a q8_1 activation quantizer kernel (dequant weights→int8-with-(d,m), quantize activations→q8_1, shared int8 MMA inner loop = port of `vec_dot_q8_1_q8_1_mma`).
   - `load_tiles_q4_K`/`load_tiles_q5_K` ports incl. `unpack_scales_q45_K`.
2. Register new .cu in `build.rs`.
3. **After harness-agent's lib.rs lands**, extend `mmq_supports()` to Q4_K(tag 1)/Q5_K(tag 3) and route `BW24_MMQ` dispatch at lib.rs ~855/~964 to the new FFI.
4. **Gates:**
   - G1 build: `cargo build --release -p bw24-engine --bins`
   - G2 correctness: numeric prompt 101-612, argmax==82, `BW24_MMQ` on/off both MATCH.
   - G3 clock-locked pp512 median-of-5 vs 3332 baseline (expect ~4500-5000).
   - G4 no-regression with MMQ off.
5. **Mirror on sm_89 branch** (int8 MMA path IS portable to L40S — no FP4 gating needed for this one).
6. **Log pp512 delta as a training-data record** in `research/tune-data/rig5090.jsonl`.

---

## BACKGROUND AGENTS (may still be running / need resume)
- **harness-agent:** sweep-harness — `#ifndef` macro guards + JSONL sweep tool → `research/tune-data/rig5090.jsonl` + `record-manual.sh`. Owns the uncommitted lib.rs/qmatvec_gemm.cu edits above.
- **sm_89 port agent:** worktree `bw24-sm89`, branch `arch/sm89-l40s`, CUDA 12.8 + rust install, gate to green build + kernel_check. Already has commit 4902e68 (configurable `BW24_CUDA_ARCH` + FP4 gating).
- **web-sweep agent (completed):** ranked technique list — headline: DFlash spec-decode, TCQ KV quant, FR-Spec vocab trim, NVFP4 tensor-split fix, ST-MoE prefetch. **Output not yet fully consumed into roadmap — read + fold in.**

## RIG FACTS (sm_120a)
block-FP4/FP8 762/381 TFLOPS, NO wgmma/tcgen05, 847 GB/s mem wall, `compute_120a` trap (must use `120a` not `120`). Laptop 5090 = 24GB (not desktop 32GB), Intel Core Ultra, 60GB system RAM. Shares RAM with local LLM servers — check `free` before write-heavy benches.
