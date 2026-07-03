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

## E2E SCOREBOARD (real-prompt protocol, 27/1845/6257-tok prompts; llama = serve-script config)

**BATCHED PROMPT PRIME LANDED (03787d9, 2026-07-03) — gap 1 of e2e-image-1 CLOSED.** generate/generate_spec now prime via `prime_cache` (forward_last's prefill body + cache side-effects: batched quantized KV append `_rows` kernel + stateful linear layers via carried-ring conv + one gdn_scan from cache state) instead of the ~102/38 tok/s tokenwise decode_step loop. Seams: `BW24_PRIME_TOKENWISE=1` (escape), prompts <16 tok stay tokenwise, `BW24_PRIME_APPEND_LOOP=1` (per-row append A/B — measured equal, launch overhead hides in the async queue).

| metric (27/1845/6257 tok) | bw24 BEFORE (e2e-image-1) | bw24 NOW | llama |
|---|---|---|---|
| 9B prime/TTFT | 0.26s / 18.1s / **61s** | 0.10s / 0.80s / **2.66s** (~2350 tok/s) | 0.20s / 0.32s / 1.05s (5934 tok/s) |
| 27B prime/TTFT | 0.7s / 48s / **163s** | 0.32s / 2.72s / **8.74s** (~716 tok/s) | 0.17s / 0.94s / 2.96s (2114 tok/s) |
| 9B gen plain / specK3 | 106.3/100.2/99.2 / 116.6/81.5/73.1 | 104.9/99.3/98.9 / 114.1/84.8/71.9 (noise) | 121.3/120.3/116.5 |
| 27B gen plain / specK3 | 39.4/37.6/37.7 / 53.4/39.7/35.5 | 39.4/38.1/37.9 / 49.4/43.0/35.8 (noise) | 87.6/92.7/75.9 (spec) |

Remaining prime gap vs llama = the PREFILL gap itself (9B 0.40x, 27B 0.34x at 6k) — prime rides forward_last, so every prefill kernel win now transfers 1:1 to TTFT. Gates: kernel-check ALL GREEN (incl new rows-vs-loop byte-identity), run-gen 82==82, run-spec K=1..8 PASS x {9B synth, 9B text, 27B real} all with batched prime, first-token A/B batched-vs-tokenwise agrees 6/6 (full-16 identical 5/6; 9B p2 flips at index 11 — allowed cache-state FP, argmax-level agreement holds where it matters).

**GAP 2 (spec degrades with ctx) — CLOSED 2026-07-03, three stages, nsys-decomposed first (JSONL `spec-longctx-profile`).** The 64-tok steady loop at 9B/6.3k was: 44% MMVQ trunk weights (verify + the partial-accept REPLAY duplicate pass, ~1.54 trunk reads/round), 21% per-row verify FA (fa_decode_vec_q 201us x T rows x 8 full layers — 392-CTA launch = 4.8 CTA/SM vs 12 achievable), 17% draft graphs (~58% of that = q6_K lm_head), 3.5% host gaps (loop already async-clean; no free host win). Design B (key-major shared walk) buried: per-row split boundaries derive from t_kv_r ⇒ never partition-exact.
- **Stage 1 — MULTI-ROW FUSED VERIFY FA (design A):** `fa_decode_vec_q_rows` + `fa_decode_combine_rows`, grid.z=row, per-row t_kv_r/n_splits_r derived in-kernel from the same fa_split_keys formula ⇒ bit-identical per row (kernel-check pins rows-vs-loop bitdiff=0 incl a 64-boundary crossing). Engages when base_len+1 ≥ FA_VEC_MIN_TKV; `BW24_FA_ROWS_OFF=1` seam. A/B: 9B p2 +13.8% / p3 +5.7%; 27B p2 +13.7% / p3 +3.3%. Fused per-row cost 140us vs 201us — floor = single-warp 64-key walk latency (split=64 exactness-pinned).
- **Stage 2 — REPLAY-FREE PARTIAL ACCEPT (the big one):** the verify's first j=base+n_acc columns ARE the committed state (bit-identical to eager, verify-probe contract) — KV truncates to pos+j, recurrent state rebuilds from a per-round `VerifyCkpt` (batched layers: prefix re-run of the SAME gdn_scan kernel t=j from the snapshot state over retained inputs — its t-loop is prefix-stable; conv ring: pure-copy `ssm_conv_ring_rebuild_f32`; per-column layers: dtod state clones), bonus rides PENDING like the full-accept fold. Duplicate trunk replay GONE. `BW24_SPEC_REPLAY=1` seam. A/B: 9B p3 +14.6%, p2 +10.7%; 27B p3 +32%, p2 +19%.
- **Stage 3 — draft side:** (3a) TRUE-HIDDEN REFRESH — batched mtp_kv_fill of ALL committed positions from the verify hidden stack each round (`BW24_SPEC_NOREFRESH=1` seam): 9B p2 +6.2%, 27B p3 +4.0%. (3b) HEAD-LESS pseudo-seed graph (second capture, no lm_head/argmax/prob — identical h_nextn): 9B p3 +4.2%; draft parity graph-vs-eager still bit-identical.
- **RESULT (protocol K=3 pmin0.2, clock-locked, N=5 medians on headline cells): 9B 130.8/100.9/80.2 tok/s (1.41x/1.15x/0.91x, was 1.16/0.93/0.72 pre-work same-session); 27B 66.0/59.7/45.3 (1.86x/1.76x/1.33x — was 1.34/1.09-1.24/0.94). 27B spec now WINS at every size; 9B holds ≥1.0x through p2.** Config sweep (JSONL `spec-config-sweep`): post-replay-free the 9B optimum moved to **K=2 pmin=0.3 uniform: 137.7/104.1/88.5 = 1.48x/1.19x/1.00x — parity at 6k ctx** (was 0.72x); 27B K=2/0.2 = 1.37x at p3; 27B K=4 cliff (T=5 splits b4 MMVQ into b4+b1). Full battery green on the final tree incl all 5 seams. **NEXT levers (profiled, 9B p3 24.5ms/round): (a) MMVQ b4 efficiency — 42% of wall at ~1.5x the per-read cost of m=1 (which is DRAM-SOL); a per-row-order-preserving b4 pass = same exactness class as the rows kernel, worth ~+15%; (b) draft lm_head 22% (model-bound, FR-Spec 9B negative); (c) fa_rows 17% (walk-latency floor).**

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
- 27B decode 0.92x: NVFP4 matvec 81% of its decode at 41% DRAM (= llama parity — SOL-bound); dual-matvec neutral there. Remaining 27B decode delta is llama's fused mul_mat_vec_q<40,1,true> variant (its 52.8% kernel — the fusion flag folds MORE than gate+up; investigate what the `true` template arg fuses). FA split=32 = +3.4% for non-spec serving (env). **27B spec: UNPROFITABLE at any K (0.45-0.80x) — MTP head acceptance is 54% @K=1 vs 9B's 98%; that's model quality, not engine cost (batched verify ruled out via worktree A/B). 27B long-gen divergence: FIXED 2026-07-04 — three FP-order mismatches in verify (rms_norm blockDim 256 vs fused-1024, l2_norm 256 vs warp-tree, dp4a vs MMVQ at T>=5); decode-exact dispatch variants shipped (FP-order lesson #7: verify must be kernel-DISPATCH-identical). 27B real-prompt K=1..8 all exact.** FR-Spec GGUF variants on disk are the 27B draft-cost lever if acceptance can't move.
- Prefill FA: **CLOSED with measured proof** (FA-PREFILL-OVERLAP-DESIGN.md, clock-locked table): fa share of prefill = 0.7% @512 / 3.7% @8k, grows only ~seq^0.59 — 10% share needs ~48K-token prompts. The producer/consumer overlap is a 5-20% speedup of a ~1-4% component = <1% e2e below 32k ctx. Not worth building until long-ctx prefill becomes a daily workload.
- **Spec-loop graph integration (NEW quantified lever):** run-spec's plain baseline is EAGER decode (9B eager 95.5 vs graph 110.5 = 16% launch-overhead gap); every draft MTP pass and verify batch pays eager launch costs. Graph-capturing the fixed-shape draft chain (T=1 MTP head) + bucketed verify shapes is the remaining big spec-throughput lever (llama's serve config runs graphs ON with spec). BLOCKED on the 27B divergence fix landing first (same file).
- **Condition-scope audit (2026-07-04):** MoE SLRU cache + fused router: BUILT (e1f49ec, 35B decode 6→24-31 tok/s, EDGE-1). Tiered VRAM/pinned/disk spilling: BUILT (e44b89a, BW24_SPILL_DISK). KV quant q8_0/q5_1: BUILT (daily default). Safetensors loaders incl MoE gather + NVFP4 repack: BUILT. FR-Spec/trimmed-vocab MTP: BUILT this session. STILL NOT BUILT: KV prefix reuse/eviction across requests (bw24-server has no prefix cache — matters for the 2-4-agent serve pattern; the lmcache maps in research/inference-maps are the design source), MoE async prefetch stage (SLRU is sync-DMA on miss), 35B/gemma4 interleaved re-bench vs llama with cache on.
- L40S benches: blocked on hardware (box terminated). All commits compile-mirrored to sm89.

## USER'S OWN LLAMA WORK (2026-07-04 — read these before touching spec/quant/MoE)
- **Issue 25187 (his)**: FR-Spec draft-vocab trim for native MTP — HIS branch `avifenesh/llama.cpp/frspec-mtp-vocab-trim` (047bfa508), HIS trimmed GGUFs on disk (frspec32768 variants + d2t tensor). llama measured: draft lm_head -85%, e2e 83.9→85.1 (public map) / 86.5 (code map) tok/s on the 27B daily config. bw24 FR-Spec consumer = agent in flight.
- **PR 25153 (open, his)**: imatrix-aware NVFP4 quantization (scale search) — the NVFP4 quality side.
- **PR 23170 (closed, his)**: MoE experts as cache residents during offloading — EDGE-1 ancestor.
- **Serve script daily config** (~/.local/bin/serve-qwen36-27b): NVFP4 trunk + SEPARATE Q4_K_M MTP draft GGUF, `--spec-draft-n-max 3 --spec-draft-p-min 0.1`, KV q8_0(rotated)/q5_1, graphs ON, 175W, 128k ctx. **llama 27B e2e ≈ 84 tok/s WITH spec — THE number to beat, not plain tg128 42.4.** bw24 27B spec unprofitable (54% blind-draft acceptance) because it lacks p-min CONFIDENCE GATING: llama stops the draft chain when token confidence < p-min, converting low-acceptance rounds into cheap short drafts. That + FR-Spec are the two 27B spec unlocks. Note llama accept ~0.75 at n-max=3 with p-min — same head, gated drafting.

**Vendor-from-everything directive (user, 2026-07-04):** edges can come from ANY tool — sglang, vllm, ktransformers, lmcache, flashinfer, cuBLAS, TensorRT-LLM, ollama, DeepSeek-4 stack, papers. research/inference-maps/ already maps vllm/sglang/ktransformers/lmcache/flashinfer/trt-llm/cutlass-marlin/exllamav3 — USE them per component. E2E tok/s vs llama at the daily serve config (spec+KV-quant on) is the headline bench, not kernel microbenches.

## SPEC SCOREBOARD — GRAPH-GRADE SPEC LANDED (2026-07-03/04 graph-spec session; all numbers same-session interleaved, clock-locked 1860 w/ sag to ~1725)
**9B: plain eager 90.4 → spec K=3 pmin=0.2: 130.6 tok/s (1.44x plain eager, 1.29x over same-session graph-plain ~101) — SPEC IS THE OUTRIGHT DAILY WINNER. All exact: 9B synthetic + 9B TEXT + 27B real-prompt, K=1..8.**

**STAGE 4 — PERSISTENT MTP DRAFT KV LANDED (266620f, 2026-07-03, free clocks, interleaved x3): the acceptance lever paid.** The NextN scratch KV no longer resets per round — slot p holds the MTP block's K/V for committed token p (the reference engine's mtp_update design), so the draft chain attends over FULL history. Mechanics: scratch cap=max_ctx allocated once, len via len_d ⇒ the ONE round-0 graph capture serves every t_kv (zero recaptures; fa_decode_dc bucket_max=cap, empty splits self-skip); eager draft uses the SAME dc launcher ⇒ parity by construction (verified: accept/draft counts bit-identical NOGRAPH vs graph, both models, every K). Entry sources: chain appends (accepted positions KEPT — hidden chain-approximate, reference-endorsed), `mtp_kv_fill` K/V-only batched pass (no wq/attn/FFN/lm_head) for prompt positions (from prime-collected exact trunk hiddens) + the last-draft slot on full accept (from vh_seed). Draft-side rollback = round-start set_len truncation. Seam: `BW24_SPEC_KVLOCAL=1` = legacy round-local (verified to reproduce the old 58% 27B K=1).
- **27B real-prompt: acceptance K=1 58.0→70.7%, K=2 57.4→63.4%, K=3 47.1→56.3%; best spec 42.6 (K=2, 1.03x) → 46.8 tok/s (K=3, 1.13x) — new 27B optimum is K=3.** (Reference serve config ≈75% with the same head — most of the gap closed.)
- **9B TEXT: K=1 78.9→85.5%, K=2 61.6→75.2%; best 110.3 → 126.4 tok/s (K=2, 1.14x)** at free clocks.
- **9B SYNTHETIC seq: REGRESSES at K≥3 (154.3→140.2 tok/s; accepted counts EQUAL, drafted balloons — full-context confidence keeps p-min from firing on the toy distribution).** Real prompts are the serving verdict (tune-data README rule); synthetic bench comparisons must use BW24_SPEC_KVLOCAL=1 or compare persistent-to-persistent.
- Gates: kernel-check ALL GREEN, run-gen 82==82, run-spec K=1..8 PASS on all 3 configs, parity identical. JSONL row `commit:266620f`.
- Next acceptance levers (now 8-19pts below the K≥2 ceiling): refresh ACCEPTED entries from verify TRUE hiddens (currently chain-approximate), p-min re-tune for full-context confidence (0.2 was tuned for windowless drafts).

Session chain (each stage interleaved-A/B'd + gated + committed):
1. **FP-order lesson #8 fix (75b3e6b, exactness):** the verify's decode-exact dispatch must mirror eager PER LAYER — uses_q8_1_fast is per-tensor and the 9B GGUF stores ssm_beta/ssm_alpha as Float on layers 1/2/4, so eager takes the UNFUSED 256-thread rms_norm there while verify ran the 1024-thread norm; 1 ULP at layer 2 amplified through GDN to 2.3e-1 logits and flipped the 9B TEXT prompt at K=1..8 (pre-existing on main — synthetic + 27B had passed on margin luck). Also: batched linear verify now requires ALL-fast projections (matmul_decode_exact routes Float to cuBLAS GEMM at m=t vs eager's per-token GEMV). New **verify-probe** bin = the permanent gate (eager-vs-verify logits at T=1/2/3 + per-layer residual bisect + pair checks + fastness map); after fix maxdiff 0.000e0 everywhere.
2. **Stage 1 — device-argmax verify (23fbf9f):** verify logits stay ON DEVICE; per-column argmax (argmax_gate-validated kernels via column views) + one [T] u32 read replaces the 1-4MB dtoh + T host argmaxes per round; last_logits Vec → last_pred u32. 106.3 → 110.6 (+4.0%).
3. **Stage 2 — graph-captured draft chain (6502142):** the fixed-shape T=1 MTP forward captured ONCE, replayed per draft step — ONE bucket serves every draft index (scratch t_kv ≤ k+1 < 96 pins fa to scalar/n_splits=1; append/fa _dc twins read len_d). Token/pos/seed chain themselves in-graph; host reads 4B tok (+4B p-min) between replays. Bonus-fold pseudo-seed = one more replay. 110.5 → 130.6 (+18.2%); draft parity = acceptance counts BIT-IDENTICAL to eager at every K on 9B AND 27B. Seams: BW24_SPEC_NOGRAPH=1; auto-fallback for MoE/capture-failure.
**27B real-prompt: K=2 = 37.6 vs plain 36.8 = 1.02x — first 27B spec config above 1.0x** (still acceptance-bound at 57%; FR-Spec trim = the next 27B lever). → SUPERSEDED by Stage 4 persistent draft KV: 46.8 tok/s K=3 = 1.13x at 71%/63%/56% acceptance (K=1/2/3).
**Stage 3 (graph-captured verify) = MEASURED NEGATIVE as specified, do not build without a new unlock:** savings ceiling = per-trunk-pass launch overhead (11.06 − 10.41 ≈ 0.65-1.15ms) × ~1 verify pass/round ≈ +3-5% e2e; capture cost = 3 REAL verify passes (~33ms) per bucket, and the bucket key must be (T, per-row fa n_splits vector) because R1 forbids padded splits (measured argmax flips) — with p-min T varies 2..k+1 and n_splits churns every 64 ctx tokens + boundary vectors ⇒ ~10-13 captures per 128 tokens ≈ 330-430ms = 4-10x the savings, at ANY gen length. Unlocks that change the math: (a) cudaGraphExecUpdate (re-parameterize a baked graph instead of re-capturing — not exposed in cudarc 0.19), (b) make fa n_splits t_kv-INDEPENDENT in BOTH decode and verify (the exactness pair moves together; kills rebucketing AND adds CTAs at short ctx where fa-decode is grid-starved) — decode-wide policy change, needs its own full battery.
**ENV LAW for spec gates/benches: BW24_FAST=1 BW24_MMVQ=1 both required** — without BW24_MMVQ, decode m=1 falls to dp4a while the verify uses MMVQ-class kernels and exactness breaks (pre-existing dispatch split, cost this session an hour).
sm89/L40S status (ARM 3, subagent-only per user): first hardware run happened — argmax gate PASS but kernel-check/prefill CRASH (illegal address) and decode 0.23x of llama; the sm89 branch compiles but is NOT execution-ready on real Ada hardware. Debug = future subagent work; box artifacts at fpv-train:~/bw24-bench/.

## SPEC SCOREBOARD (2026-07-04, 9B, exact greedy — PRE-TIMING-FIX prints below, ratios valid)
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
