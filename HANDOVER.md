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
| pp512 INTERLEAVED A/B (honest protocol) | **5049** | 5072 | **0.995x — PARITY BAND** (was 0.91 at session start; conv-tm fuse -> 0.936, scale-fold -> 0.95, conv+GDN-repack fuse -> 0.995) |
| decode tg128 graph @ctx128 INTERLEAVED A/B | **110.5** | 106.3 | **1.04x — ABOVE llama** |
| decode tg128 graph @512 / @2048 | 109.5 / 106.3 | — | — |

**MEASUREMENT LESSON (2026-07-03): cross-session pp512 numbers lied.** Sequential runs gave bw24 5531 vs llama 5451 ("parity") — but interleaved same-minute A/B gives 4655 vs 5092 = 0.91x. Root cause: llama holds 1852MHz during its run; bw24 sags to 1710MHz under the same clock lock (bw24 draws more power for the same work → worse perf/W → thermal sag). ALL ratio claims must be interleaved A/B from now on. bw24 has BOTH a remaining kernel gap (~9%) AND a power-efficiency gap (clock sag under load). Decode ratio not yet re-measured interleaved.

**Decode session 2026-07-03 (landed levers, all gate-passing):**
1. warp-per-block coalesced q8_1 epilogues (silu_mul_scaled_q8_1, quantize_q8_1, norm pass-2): 88.4→95.1 graph.
2. float4 warp-per-4-blocks norm pass-2 + 1024-thread CTA: 95.1→104.4.
3. adaptive FA split REVERTED (broke run-spec exactness — split count changes combine FP order). Fixed 64 + BW24_FA_SPLIT seam kept. ~109.6 stands.
4. gdn_prep_decode_f32: repack+2xL2+sigmoid+glog fused 5→1 launches (perf re-measure pending GPU free).

**Prefill session 2026-07-03 part 2 — INTERLEAVED PARITY (0.995x):**
1. ssm_conv1d_tm_f32: transpose+zeros+pad+conv → 1 token-major kernel (0.91→0.936x).
2. NVFP4 macro-scale folded into MMQ write-back — bw24 now does strictly LESS work than llama here (0.936→0.95x).
3. ssm_conv1d_gdn_f32: conv+SiLU scatters straight into GDN q/k/v layout, SSM prep 5 kernels→1 (0.95→0.995x).
Background agent results:
- **K=4 spec-divergence: FIXED** (befc9d0). Root cause: verify used fa_prefill_view (different FP accumulation order than fa_decode) — at tight logit margins argmax flips. Fix = per-row fa_decode in full_attn_verify. K=1..4 ALL PASS at N up to 512. LESSON (2nd instance): greedy-spec exactness requires the SAME kernel FP order, not equivalent math.
- **stream-K q45k MMQ: NEGATIVE, gated off** (fbc5f46, BW24_MMQ_STREAMK=1 to enable). Per-GEMM 1.11x real (712.9→638.4us incl fixup, per-GEMM rel ≤1.2e-6 vs conventional) but model-level argmax deterministically flips 82→68 (top-2 margin 0.14, 1e-6 reorder noise amplifies over 33 layers). 3rd instance of the FP-order lesson. Warps-active unchanged 16.7% — the q45k occupancy ceiling is 57KB smem 1-CTA/SM, NOT the tail; next q45k lever = smem diet (2-stage y-tile or smaller MMQ_X) to reach 2 CTA/SM.
- **FA-decode port: DONE INLINE** (after agent died twice). Register-dequant rewrite of fa_decode_vec_q/_dc: no smem staging, no syncs, coalesced byte-per-lane K reads, GQA reuse via L2. **+48% tg at 8k ctx (59.8→88.8)**, parity at short ctx. CRITICAL: had to KEEP one bf16 round-trip on dequanted K/V — raw f32 dequant flipped run-spec K=2 (FP-order lesson instance #4). Old-vs-new direct binary A/B, all gates green.

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

**FA-decode grid starvation (27B ncu, 2026-07-03):** register-dequant kernel at short ctx runs grid (n_head_kv=4, n_splits=2) x (32,6) = 8 CTAs on 82 SMs; long-scoreboard stall 77%, warps_active 12.5%, DRAM 0.08%. It's LATENCY-bound (serial dependent K loads per warp), not BW. Split=32 helps 27B +3.4% but BREAKS 9B spec exactness (even with same-kernel per-row verify — combine order still shifts) so default stays 64. Structural fixes evaluated: (a) 2-key ILP — BUILT AND MEASURED NEGATIVE (63.6 vs 96.8 @8k direct A/B: doubling in-flight K/V registers at dpl=8 collapsed occupancy; the 77% stall wants MORE WARPS, not more per-warp work). (c) dp4a int8 K-dot: BUILT AND MEASURED NEGATIVE — no perf gain (81.2 vs 82.2 @8k, latency-bound on K loads either way; dp4a cut ALU not loads) AND K=1/2 spec exactness broke (FP-order lesson #5 REFINED: same-kernel is necessary but NOT sufficient — int8-Q accuracy change flips tight-margin argmax in the DRAFT chain). **FA-decode dot direction now empirically CLOSED with measured proof on all four variants: smem-broadcast (126us, slow), register-dequant f32 (winner, shipped), 2-key ILP (occupancy collapse 0.66x), dp4a (no gain + exactness break).** Untried: (b) combine-fold into last CTA, (d) 2-kv-heads/CTA packing — both only reshuffle the same latency-bound loop; expected value low after (a)/(c) results.

## EMPIRICAL WALL LEDGER (per the goal's proof standard — every direction measured, 2026-07-03)

**CLOSED with measured proof (do not retry without new information):**
| direction | proof |
|---|---|
| FA-decode dot variants | 4 variants A/B'd: smem-broadcast 126us (slowest), register-dequant f32 (WINNER, shipped, +48% @8k), 2-key ILP 0.66x (occupancy collapse), dp4a K-dot 0.99x + K=1/2 exactness break |
| stream-K on q45k MMQ | 1.11x per-GEMM but argmax 82→68 deterministic (1e-6 reorder amplifies over 33 layers); gated BW24_MMQ_STREAMK=1 |
| q45k MMQ_X=64 smem diet | -2% pp512 (tile efficiency > 2-CTA/SM occupancy at T=512); seam BW24_MMQ_X_Q45K kept |
| adaptive FA split | +1.2% decode but breaks spec exactness; BW24_FA_SPLIT seam kept (27B non-spec +3.4% at 32) |
| q4_K multi-row (27B) | ±0% (measured again this session) |
| matvec DRAM% | at exact llama parity (42% both, same kernel shape) — SOL-bound |
| decode elementwise | all coalesced+fused now; graph 1.04x ABOVE llama interleaved |

**FP-ORDER RULES (5 instances, now law):** (1) any reduce-order change flips tight-margin greedy argmax; (2) same-kernel between decode and verify is NECESSARY; (3) …but NOT SUFFICIENT — accuracy changes (int8 Q) flip the draft chain too. Full gate battery = kernel-check + argmax + run-spec K=1..4.

**OPEN directions (not walls — unmeasured or needs infra):**
- **MTP >1.0x: DONE (2026-07-03).** Batched linear-attn verify (linear_attn_verify_t: T-token projections + carried-state conv ssm_conv1d_tm_state + one gdn_scan(T), bit-identical) crossed it: K=4 = 1.03x plain (48.2 vs 47.1 tok/s), all K exact. K sweep 1..8: K=4 optimum, acceptance decays 8-10pts/K (98→46%). Further spec profit needs acceptance lift (draft chaining quality), not deeper K — the verify is no longer the cost. Web context (zolotukhin.ai 2026-05-08 + llama PR 22673): MTP c≈1/L makes it the right draft for hybrid models; llama's 2.4x is against a 7 tok/s launch-bound baseline, ours is BW-bound at 47 — Leviathan-consistent.
- 27B decode 0.92x: NVFP4 matvec 81% of its decode at 41% DRAM (= llama parity — SOL-bound); dual-matvec neutral there. Remaining 27B decode delta is llama's fused mul_mat_vec_q<40,1,true> variant (its 52.8% kernel — the fusion flag folds MORE than gate+up; investigate what the `true` template arg fuses). FA split=32 = +3.4% for non-spec serving (env). **27B spec: UNPROFITABLE at any K (0.45-0.80x) — MTP head acceptance is 54% @K=1 vs 9B's 98%; that's model quality, not engine cost (batched verify ruled out via worktree A/B). Also found: 27B long-gen (NGEN=128) spec exactness FAIL pre-existing on all binaries — separate margin-divergence bug, repro recorded.** FR-Spec GGUF variants on disk are the 27B draft-cost lever if acceptance can't move.
- Prefill FA: **CLOSED with measured proof** (FA-PREFILL-OVERLAP-DESIGN.md, clock-locked table): fa share of prefill = 0.7% @512 / 3.7% @8k, grows only ~seq^0.59 — 10% share needs ~48K-token prompts. The producer/consumer overlap is a 5-20% speedup of a ~1-4% component = <1% e2e below 32k ctx. Not worth building until long-ctx prefill becomes a daily workload.
- **Spec-loop graph integration (NEW quantified lever):** run-spec's plain baseline is EAGER decode (9B eager 95.5 vs graph 110.5 = 16% launch-overhead gap); every draft MTP pass and verify batch pays eager launch costs. Graph-capturing the fixed-shape draft chain (T=1 MTP head) + bucketed verify shapes is the remaining big spec-throughput lever (llama's serve config runs graphs ON with spec). BLOCKED on the 27B divergence fix landing first (same file).
- MoE caching/spill, KV reuse/eviction, striped-vocab MTP, safetensors loaders for gemma4/minimax/deepseek: ROADMAP items 6-9/11-15 — feature work, untouched this session by priority (arm 1 = perf on daily 9B/27B first).
- L40S benches: blocked on hardware (box terminated). All commits compile-mirrored to sm89.

## USER'S OWN LLAMA WORK (2026-07-04 — read these before touching spec/quant/MoE)
- **Issue 25187 (his)**: FR-Spec draft-vocab trim for native MTP — HIS branch `avifenesh/llama.cpp/frspec-mtp-vocab-trim` (047bfa508), HIS trimmed GGUFs on disk (frspec32768 variants + d2t tensor). llama measured: draft lm_head -85%, e2e 83.9→85.1 (public map) / 86.5 (code map) tok/s on the 27B daily config. bw24 FR-Spec consumer = agent in flight.
- **PR 25153 (open, his)**: imatrix-aware NVFP4 quantization (scale search) — the NVFP4 quality side.
- **PR 23170 (closed, his)**: MoE experts as cache residents during offloading — EDGE-1 ancestor.
- **Serve script daily config** (~/.local/bin/serve-qwen36-27b): NVFP4 trunk + SEPARATE Q4_K_M MTP draft GGUF, `--spec-draft-n-max 3 --spec-draft-p-min 0.1`, KV q8_0(rotated)/q5_1, graphs ON, 175W, 128k ctx. **llama 27B e2e ≈ 84 tok/s WITH spec — THE number to beat, not plain tg128 42.4.** bw24 27B spec unprofitable (54% blind-draft acceptance) because it lacks p-min CONFIDENCE GATING: llama stops the draft chain when token confidence < p-min, converting low-acceptance rounds into cheap short drafts. That + FR-Spec are the two 27B spec unlocks. Note llama accept ~0.75 at n-max=3 with p-min — same head, gated drafting.

**Vendor-from-everything directive (user, 2026-07-04):** edges can come from ANY tool — sglang, vllm, ktransformers, lmcache, flashinfer, cuBLAS, TensorRT-LLM, ollama, DeepSeek-4 stack, papers. research/inference-maps/ already maps vllm/sglang/ktransformers/lmcache/flashinfer/trt-llm/cutlass-marlin/exllamav3 — USE them per component. E2E tok/s vs llama at the daily serve config (spec+KV-quant on) is the headline bench, not kernel microbenches.

## SPEC SCOREBOARD (2026-07-04, 9B, exact greedy)
plain 47.1 → K=4 ungated 1.03x → p-min 1.11x → **bonus-fold + pmin 0.2 @K=3: 52.17 tok/s (1.11x), K-curve flattened (K=4/6 hold 1.05/1.06 under pmin 0.3), K=1..8 all exact.**
Bonus-fold trade measured: 1 trunk read/round saved vs ~10pts acceptance (pseudo-hidden draft seed); STRUCTURAL OPTIMUM — a true-hidden fold requires knowing the bonus pre-verify, which IS the pseudo-seed. Recorded in JSONL.
Landed chain: batched linear verify → GPU-argmax draft → persistent snapshots → FR-Spec consumer (BW24_MTP_DRAFT + d2t) → p-min gate (BW24_SPEC_PMIN).
27B: 0.85x best (pmin 0.4) — acceptance-bound (45% native, 31% with the Q4_K draft file whose NextN block loses 15pts). 27B needs a target-quality trimmed draft (NVFP4 block + trimmed head) — the agent's HEAD_ONLY probe failed exactness on shared_head_norm mismatch; producer work.
**NEXT quantified spec levers:** (1) bonus-token fold — every round pays a full T=1 weight read for the bonus (decode_step_h); llama folds it into the next verify batch. (2) 9B FR-Spec trimmed head (draft lm_head=750MB Q6_K, c≈0.13/token → 87% cut; user's gguf-py producer recipe in his llama issue 25187). Both together should push toward the accept-rate-implied ~1.5-2x.

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
