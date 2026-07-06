# bw24 — Session Handover

_Written 2026-07-03, standings updated 2026-07-06. Read this cold, then continue. bw24 = from-scratch Rust+CUDA LLM inference engine, target rig RTX 5090 Laptop (sm_120a, Blackwell consumer, 24GB, **858 GB/s measured read wall** — microbenched, not the 847 spec). Second box = bw24-g7e (RTX PRO 6000 96GB, sm_120-compatible). Repo PUBLIC: https://github.com/avifenesh/bw24 — both rigs sync via origin. L40S/sm_89 lane CLOSED (box terminated)._

## STANDINGS 2026-07-06 (E2E IMAGE 6, gen tok/s p1/p2/p3, same prompts, llama at serve-best)

- **27B NVFP4 (daily driver): 99.1/87.6/75.9 vs llama 86.6/91.9/75.3 = 1.14x WIN / 0.95x / 1.01x WIN.** >100 milestone crossed on code (102.6 N=3, CLI K=3). Serve 96-97 at 512-tok turns.
- **9B: 181.3/152.6/142.8 vs 121.7/120.5/116.8 = 1.49x/1.27x/1.22x clean sweep.** 256k-ctx edge proven (278k prompt exact on 24GB).
- **35B MoE: 112 tok/s vs llama 169.6 = 0.66x** (not a driver; MoE work feeds MiniMax).
- **DAILY 27B CONFIG: `BW24_SPEC_HPOST=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.15 BW24_FRSPEC_TRIM=<frspec-balanced32768>` + embedded MTP block + env law (FAST/GEMM/MMVQ/FA_VEC).** HPOST = post-norm h_seed (llama.cpp 166fe2949 convention); the pre-norm era was the low-acceptance era. Retrain-at-10k-corpus CLOSED negative (author block best).

## 27B OPEN GAPS (ranked by measured headroom, 2026-07-06)

1. **Prefill 1.65x** — p3 prime 4.96s vs llama ~3.0s (pp 2098 vs ~1265 effective). Next local arc.
2. **p2 gen 0.95x** (4.3 tok/s) — mid-ctx verify cost + mid-domain acceptance 72-74%; needs p2-depth anatomy.
3. **b4 verify tier on 5090** — 43% DRAM, 62% long_scoreboard, reg-limited. rpsc (smem scale prestage, f57f10e) fixed b8 tier only; k-split BANNED for verify (FP-order lesson x4: self-consistency FAIL) but LEGAL for MoE experts (softmax sums).
4. **Acceptance** — head levers closed (retrain negative, author block + HPOST is the ceiling); untested: code75 trim variant under HPOST.

## CLOSED DIRECTIONS 2026-07-06 (don't retry — proof in rig5090.jsonl)

spec fixed-sync-overhead (pmin/burst A/B null); m=4 MMA verify (grid starves, crossover needs token volume); forced-BV globals (auto optimal); b4 unroll-2 + rpca cp.async (reg-growth occupancy trap); p3 config space (K/pmin/split/KVLOCAL all swept — KVLOCAL costs 35 accept pts); MTP head retrain at 10k corpus; 35B trunk q8 (94% wall = done); 35B expert variant space (geometry flat — new-kernel design is the open lead, agent on it).

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

## NEXT KERNEL ARCS (2026-07-05, measured on the 40k trace — the 27B >100 base-path levers)

**ARC A — fa_decode_vec_q long-ctx (decode decay 43->25 tok/s over 1.8k->40k):** wall math says
646MB KV/token (17 layers x 40k x 928B) = 0.76ms at 847GB/s; measured decay +18ms/tok = ~24x off
wall. Kernel facts (cu/flash_attn.cu KERNEL 2b): register-dequant rewrite deliberately dropped the
smem broadcast because "L2 serves the reuse (KV@2048 = 2.2MB << L2)" — at 40k a LAYER's KV is
~37MB, L2 can't hold it, so the 4 GQA warps' re-read = 4x DRAM traffic. Candidates, in order:
(1) re-introduce the smem KV tile broadcast ABOVE a t_kv threshold (dispatch seam, keep the
register path short-ctx — it won at 2k by 12x); (2) the 34B/24B block stride breaks 32B-transaction
alignment — pack-on-append into a 32B-aligned layout (needs an append+decode pair change, gate
battery arbitrates); (3) split sizing at 40k (n_splits already retuned; check tail effects).

**ARC A candidate 2 (32B-aligned split-plane K): MEASURED NEGATIVE 2026-07-05 (g7e lane, JSONL
`arca-kv-align-probe-negative`) — probe-gated, nothing wired.** Standalone microbench
`probe/kv_align_probe.cu`: verbatim fa_decode_vec_q + _smem bodies with a split-plane K twin
(qs plane [max_ctx][nblk][32B] + separate scale plane — pure address remap, partials bit-identical),
27B geometry, 8 rotating layer buffers, clock-locked N=5. The straddle theory's MECHANISM is real —
sectors/request drops 1.22->1.07 (-11% L1 sector traffic, ncu) — but duration is UNCHANGED at
16k-64k (smem 32k: 256.06 vs 256.03us; reg 32k: 1.2% WORSE) because the kernel is LATENCY-bound
(DRAM 15% of peak, warps 34%), not transaction-bound: saved sectors buy zero time. Only reg@8k
gained (+4.5%), below the wire bar and not the blocker cell. Candidate 2 CLOSED with ncu evidence.
Side observation worth a cheap follow-up: in the 8-layer-rotating synthetic the smem twin beats the
register path 2.1-2.3x at EVERY depth incl 8k — the BW24_FA_SMEM_TKV=16384 crossover may sit too
high (real-trace L2 differs; sweep 4096/8192 on real prompts before believing it). Remaining ARC A
room is NOT KV byte addressing — the +18ms/tok decay must live in latency/occupancy structure
(combine chain, n_splits tail, launch overhead).

**ARC B — fa_prefill_q chunk-prime redundant dequant: DONE 2026-07-05 (g7e lane, JSONL
`arcb-prime-deqw`).** Was 30.5%/153ms-launch of the 32k prime wall (every T/64 q-block CTA x
n_head re-dequanted the whole quantized KV stream). Landed all three candidates as stacked
bit-identical rungs, DEFAULT ON: (1) `fa_dequant_kv_ws_bf16` dequants K/V ONCE per (layer,chunk)
into a resident grown-and-reused bf16 workspace (`Engine.prime_deqw_ws`, ~83MB at 40k) storing
exactly the `__float2bfloat16(dq_*)` values the old kernel staged to smem; `fa_prefill_qw` is the
byte-identical MMA/softmax/PV twin over it (32k prime 26.41→18.11s); (2) 16B-uint4 vectorized
workspace→smem staging (→17.10s); (3) `fa_prefill_qw_db` cp.async double-buffered staging twin
(+32KB smem, 1 CTA/SM — wins anyway, single-buffer ncu: mem 66%/SM 15%/DRAM 0.6% staging-stalled)
(→16.51s). **27B N=3: 32k prime 26.41→16.51s (1.60x), 16k 11.80→8.65s (1.36x).** Seams:
`BW24_PRIME_DEQW=0` (inline-dequant fa_prefill_q), `BW24_PRIME_DEQW_DB=0` (single-buffer twin).
Gates: kernel-check ALL GREEN incl new ws-vs-inline bitdiff=0 gate (both twins); argmax 1178/268
MATCH; run-spec K=1..8 PASS both models; p3 chunk-prime token-md5 identical across
deqw-on/deqw-off/monolithic; 16k+32k deep prompts PASS. NEXT prime lever: mul_mat_q_nvfp4_w4a8 is
now 52.5% of the 32k prime — the prime wall is the GEMM itself, not attention.

**27B K=4 SPEC CLIFF — KILLED 2026-07-05 (g7e lane, JSONL `mmvq-b8-k4-cliff-fix`).** ROOT CAUSE
CORRECTION: the remembered "T=5 splits b4 MMVQ into b4+b1" theory was WRONG — the batched dispatch
in matmul/matmul_pre/matmul_decode_exact was gated `(2..=4).contains(&m)`, so T=5..8 verify fell to
the per-row grid.y=m MMVQ (K=4 nsys: 6448x `qmatvec_nvfp4_mmvq_rp` @49us avg = m FULL weight reads
per launch, 8.3% of wall; acceptance held 54-55% — kernel path, not acceptance). Fix = `b8` tier:
mcols=8 instantiations of the EXISTING batched templates (c>=m masked, per-(token,row) dp4a order
verbatim) for NVFP4 (base/pf/r2/r2w8/rp/rpr2/rpr2w8) + k-quants (base/r2), dispatch widened to
2..=8, mcols map 2/4/8. b8 AUTO = the w8 twin everywhere — the b4 wave-crossing rule MISPICKS for
b8 register classes (measured: forced rpr2w8 97.9 vs b4-rule 93.3 tok/s e2e; DRAM-cold rp msweep
m=5/6/8 x 5 shapes: rpr2w8 wins/ties all but ffn_gate m=5/6 where rpr2 edges 1-3%). **27B p3 spec
pmin0.15 clock-locked N=3: K3 107.2 (unchanged, control), K4 75.4→98.0 (+30%, now 0.91x of K3 =
cliff GONE), K5 70.8→95.5, K6 66.4→91.5. Verdict: K=3 stays the 27B optimum (acceptance decay
67→55→50→44% outruns the kernel win); K>=4 now decays smoothly (-8.6/-10.9/-14.6% vs K3) instead
of cliffing — deeper-K becomes viable wherever acceptance holds >60%.** Seams: `BW24_B8=0` (m=5..8
back to per-row; m<=4 untouched), BW24_NO_BATCHED still the global reference, BW24_MMVQ_BV forces
b8 variants too. Gates: kernel-check 126 OK 0 FAIL incl new b8 m=5/6/8 cells (rel=0.00e0 all dtypes,
RP bit-bad=0); argmax 1178/268 MATCH; run-spec K=1..8 PASS both models; p3 real-prompt PASS K=3..6.
LESSON x2: (1) read the dispatch before trusting a remembered mechanism — the "split" was actually
a fall-through to the WORST path; (2) variant auto-rules don't transfer across register classes —
re-measure per MCOLS tier.

**Session/serve wiring (engine API done, server pending):** SpecSession + generate_spec_session
landed with the session-gate oracle (3-turn MATCH both models; 42.6x turn-start at 40k). bw24-server
still creates a fresh cache per request — wire sessions into worker.rs next serve slot.

## NEXT LOCAL ARC — MoE expert dp4a upgrade (specced 2026-07-06, measured on local 35B nsys)

Local 35B decode window (post launch-arc port, 19.7 tok/s): **47% of decode wall = Stage-A f32
dequant expert kernels** — qmatvec_f32 2377ms/272k (staged sequential path via qmatvec_view),
moe_gate_up_silu8_f32 1667ms/14.5k, moe_down8_fma_f32 1280ms/14.5k (per 48-tok window). Experts:
gate/up = IQ3_S (GGUF type 21), down = IQ4_XS (23). All three kernels dequant f32-elementwise
(`deq()`); the trunk already runs q8_1+dp4a everywhere.

THE WORK (matched-pair set — BW24_MOE_GATE byte-identity requires sequential+fused+_dev change TOGETHER):
1. q8_1-quantize the MoE input row `z` once per token (quantize_q8_1 exists; the trunk's rms_norm_q8_1
   fusion pattern applies) and the act row before down.
2. dp4a bodies: IQ4_XS lift from qmatvec_iq4_XS_dp4a (exists, :3007); IQ3_S port from llama
   vec_dot_iq3_s_q8_1 (vecdotq.cuh:1148 — sign-via-__vcmpne4/__vsub4 trick) against bw24's
   deq_iq3_s layout (block=110B: d@0, qs@2, qh@66, signs@74, scales@106; db=d*(1+2*sc_nib)).
   NOTE llama's block scale applies OUTSIDE the int dot — same separable structure as q4k/q5k mmvq.
3. Kernel set (all change in one commit): qmatvec_view -> qmatvec_view_q8 (dp4a),
   moe_gate_up_silu8_{f32,_dev} -> _q8 twins, moe_down8_fma_{f32,_dev} -> _q8 twins. Same
   grid/block/slot-order/silu expression; ONLY the dot arithmetic changes (int dp4a + separable
   scales instead of f32 elementwise).
4. GATES: 35B argmax 248046 (local) / 1178 (box) — the FP-order change SHIFTS logits, argmax +
   run-gen prefill==decode + 64-tok stream identity vs f32 seam (BW24_MOE_Q8=0 rollback) arbitrate;
   BW24_MOE_GATE byte-identity between new sequential and new fused (the pair contract);
   kernel-check oracle gates for both new dot bodies (int-band rel like MMQ-W4A8's).
5. EXPECTED: local 35B 19.7 -> ~26-28 (47% slice at ~2-3x dp4a speedup); G7e 112.9 -> higher
   (silu8/down8 are 41% of its GPU time too). Both regimes win — this is the rare shared lever.
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
- **RESULT (protocol K=3 pmin0.2, clock-locked, N=5 medians on headline cells): 9B 130.8/100.9/80.2 tok/s (1.41x/1.15x/0.91x, was 1.16/0.93/0.72 pre-work same-session); 27B 66.0/59.7/45.3 (1.86x/1.76x/1.33x — was 1.34/1.09-1.24/0.94). 27B spec now WINS at every size; 9B holds ≥1.0x through p2.** Config sweep (JSONL `spec-config-sweep`): post-replay-free the 9B optimum moved to **K=2 pmin=0.3 uniform: 137.7/104.1/88.5 = 1.48x/1.19x/1.00x — parity at 6k ctx** (was 0.72x); 27B K=2/0.2 = 1.37x at p3; 27B K=4 cliff (T=5 splits b4 MMVQ into b4+b1). Full battery green on the final tree incl all 5 seams. **NEXT levers (profiled, 9B p3 24.5ms/round): (a) MMVQ b4 efficiency — LANDED 2026-07-03 (JSONL `mmvq-batched-latency-fix`): ncu --set full on the real 27B verify showed the batched kernel is memory-LATENCY bound (long_scoreboard 18-30/issue, DRAM 41-51%, ONE 6-LDG weight wavefront in flight/warp = half the m=1 mr2 kernel's weight MLP — NOT bandwidth, NOT the break). Fix = `pf` (next-g weight-prefetch double-buffer, 48 regs) + `r2` (two rows/warp, 67 regs) NVFP4 variants with wave-aware per-shape auto dispatch (`BW24_MMVQ_BV` seam): b4 execution -11.5% on the 27B round -> e2e +7.2%/+6.0% (27B p2/p3 = 63.9/48.0 tok/s), +2.0% (9B both = 106.0/90.2), N=5 interleaved clock-locked, plain-decode control unchanged, all exactness gates green (variants bit-identical per (token,row) — only load issue time / row->warp mapping change). STAGE 2 same day (JSONL `mmvq-b4-r2w8-wave-crossing`): `b4_r2w8` __launch_bounds__(128,8) twin (64 regs, 8 blocks/SM) + integer-wave-crossing dispatch — deleting the straggler wave flipped ffn_down/ssm_out/qkv to r2w8 (down 112.5->81.6us DRAM-cold, beats pf); 27B cumulative +9.1%/+7.7% (p2 65.1 / p3 48.7 tok/s vs base 59.6/45.2), 9B dispatch-identical. Remaining b4 headroom: host-fusing the tiny GDN projections (out_f<=1024 launches still pure-latency 15-16us each), k-quant batched variants (~19% of 9B verify, same recipe). (b) draft lm_head 22% (model-bound, FR-Spec 9B negative); (c) fa_rows 17% (single-warp walk-latency floor, split=64 pinned).**

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
| graph-captured verify (spec stage 3) | RE-MEASURED NEGATIVE 2026-07-03 post-unlocks (JSONL `graph-spec-stage3-remeasure`): even with 642582a's in-kernel per-row splits (rebucketing churn GONE) the ceiling shrank below the build gate — steady-window nsys at the image-2 optima: 9B p2 verify 95.1% GPU-busy, gap 0.70ms of a 19.3ms round = 3.8% ceiling; 27B p2 97.2% busy, 1.11ms of 43.6ms = 2.8%; realizable ~half that (graphs pay ~0.2us/node). Capture cost still 3 real-shape passes/bucket => negative through NGEN~500. Stages 1-2 drained the launch pool the graph would have reclaimed. The "27B gap = llama's whole-trunk graphs" hypothesis is measured FALSE: the 27B round is 73.3% qmatvec_nvfp4_mmvq_b4 EXECUTION (~498 launches, 54.9us each) — the gap is b4 MMVQ efficiency + acceptance (0.58 vs ~0.75), not launch overhead |

**FP-ORDER RULES (5 instances, now law):** (1) any reduce-order change flips tight-margin greedy argmax; (2) same-kernel between decode and verify is NECESSARY; (3) …but NOT SUFFICIENT — accuracy changes (int8 Q) flip the draft chain too. Full gate battery = kernel-check + argmax + run-spec K=1..4.

**OPEN directions (not walls — unmeasured or needs infra):**
- **MTP >1.0x: DONE (2026-07-03).** Batched linear-attn verify (linear_attn_verify_t: T-token projections + carried-state conv ssm_conv1d_tm_state + one gdn_scan(T), bit-identical) crossed it: K=4 = 1.03x plain (48.2 vs 47.1 tok/s), all K exact. K sweep 1..8: K=4 optimum, acceptance decays 8-10pts/K (98→46%). Further spec profit needs acceptance lift (draft chaining quality), not deeper K — the verify is no longer the cost. Web context (zolotukhin.ai 2026-05-08 + llama PR 22673): MTP c≈1/L makes it the right draft for hybrid models; llama's 2.4x is against a 7 tok/s launch-bound baseline, ours is BW-bound at 47 — Leviathan-consistent.
- 27B decode 0.92x: NVFP4 matvec 81% of its decode at 41% DRAM (= llama parity — SOL-bound); dual-matvec neutral there. Remaining 27B decode delta is llama's fused mul_mat_vec_q<40,1,true> variant (its 52.8% kernel — the fusion flag folds MORE than gate+up; investigate what the `true` template arg fuses). FA split=32 = +3.4% for non-spec serving (env). **27B spec: UNPROFITABLE at any K (0.45-0.80x) — MTP head acceptance is 54% @K=1 vs 9B's 98%; that's model quality, not engine cost (batched verify ruled out via worktree A/B). 27B long-gen divergence: FIXED 2026-07-04 — three FP-order mismatches in verify (rms_norm blockDim 256 vs fused-1024, l2_norm 256 vs warp-tree, dp4a vs MMVQ at T>=5); decode-exact dispatch variants shipped (FP-order lesson #7: verify must be kernel-DISPATCH-identical). 27B real-prompt K=1..8 all exact.** FR-Spec GGUF variants on disk are the 27B draft-cost lever if acceptance can't move.
- Prefill FA: **CLOSED with measured proof** (FA-PREFILL-OVERLAP-DESIGN.md, clock-locked table): fa share of prefill = 0.7% @512 / 3.7% @8k, grows only ~seq^0.59 — 10% share needs ~48K-token prompts. The producer/consumer overlap is a 5-20% speedup of a ~1-4% component = <1% e2e below 32k ctx. Not worth building until long-ctx prefill becomes a daily workload.
- **Spec-loop graph integration: draft chain DONE (6502142); verify capture CLOSED with measured proof** — see the wall-ledger row + JSONL `graph-spec-stage3-remeasure` (2026-07-03 re-measure at the image-2 optima: ceiling 2.8-3.8% of round, verify already 95-97% GPU-busy, capture cost negative through NGEN~500). Ranked 27B spec levers from that trace: (1) qmatvec_nvfp4_mmvq_b4 efficiency (63% of the whole round) — FIRST TRANCHE CASHED 2026-07-03 (JSONL `mmvq-batched-latency-fix`: b4 execution -11.5%, 27B e2e +6-7.2%), (2) acceptance 0.58→0.75 (draft quality) — CASHED 2026-07-04 via the hidden-pairing fix (see SPEC SCOREBOARD, JSONL `spec-hidden-pairing-prev`), then nothing structural.
- **Condition-scope audit (2026-07-04):** MoE SLRU cache + fused router: BUILT (e1f49ec, 35B decode 6→24-31 tok/s, EDGE-1). Tiered VRAM/pinned/disk spilling: BUILT (e44b89a, BW24_SPILL_DISK). KV quant q8_0/q5_1: BUILT (daily default). Safetensors loaders incl MoE gather + NVFP4 repack: BUILT. FR-Spec/trimmed-vocab MTP: BUILT this session. STILL NOT BUILT: KV prefix reuse/eviction across requests (bw24-server has no prefix cache — matters for the 2-4-agent serve pattern; the lmcache maps in research/inference-maps are the design source). MoE async prefetch: CLOSED NEGATIVE 2026-07-05 (see MoE DECODE-CACHE scoreboard below — miss staging is 3.6% of decode wall, <5% bar). 35B decode re-bench vs llama recorded (74.9 vs 235.4 tg128); gemma4 re-bench blocked on the port gap list below.
- L40S benches: blocked on hardware (box terminated). All commits compile-mirrored to sm89.

## USER'S OWN LLAMA WORK (2026-07-04 — read these before touching spec/quant/MoE)
- **Issue 25187 (his)**: FR-Spec draft-vocab trim for native MTP — HIS branch `avifenesh/llama.cpp/frspec-mtp-vocab-trim` (047bfa508), HIS trimmed GGUFs on disk (frspec32768 variants + d2t tensor). llama measured: draft lm_head -85%, e2e 83.9→85.1 (public map) / 86.5 (code map) tok/s on the 27B daily config. bw24 FR-Spec consumer = agent in flight.
- **PR 25153 (open, his)**: imatrix-aware NVFP4 quantization (scale search) — the NVFP4 quality side.
- **PR 23170 (closed, his)**: MoE experts as cache residents during offloading — EDGE-1 ancestor.
- **Serve script daily config** (~/.local/bin/serve-qwen36-27b): NVFP4 trunk + SEPARATE Q4_K_M MTP draft GGUF, `--spec-draft-n-max 3 --spec-draft-p-min 0.1`, KV q8_0(rotated)/q5_1, graphs ON, 175W, 128k ctx. **llama 27B e2e ≈ 84 tok/s WITH spec — THE number to beat, not plain tg128 42.4.** bw24 27B spec unprofitable (54% blind-draft acceptance) because it lacks p-min CONFIDENCE GATING: llama stops the draft chain when token confidence < p-min, converting low-acceptance rounds into cheap short drafts. That + FR-Spec are the two 27B spec unlocks. Note llama accept ~0.75 at n-max=3 with p-min — same head, gated drafting.

**Vendor-from-everything directive (user, 2026-07-04):** edges can come from ANY tool — sglang, vllm, ktransformers, lmcache, flashinfer, cuBLAS, TensorRT-LLM, ollama, DeepSeek-4 stack, papers. research/inference-maps/ already maps vllm/sglang/ktransformers/lmcache/flashinfer/trt-llm/cutlass-marlin/exllamav3 — USE them per component. E2E tok/s vs llama at the daily serve config (spec+KV-quant on) is the headline bench, not kernel microbenches.

## MoE SCOREBOARD — A2 GROUPED PREFILL: RESIDENT + SPILL BOTH LANDED ON rig5090 (2026-07-04)
**Three points on the same change now recorded: resident-G7e (4.8-7.2x), resident-local, capped-spill-local (JSONL rig5090 records).** 35B-A3B IQ4_XS, `BW24_MOE_GROUPED=1` + `BW24_MOE_CACHE=1`, N=5 in-process medians via the new run-gen pp-only harness (`BW24_PP_REPS`/`BW24_PP_WARMUP`, per-rep tok/s + per-rep staged-GB prints).
- **Resident-local (auto cache = 10390 slots ~11.6GB):** pp501 85.8→126.9 (1.48x), pp1845 91.4→178.6 (1.95x), pp6257 95.1→179.4 (1.89x). Smaller than G7e's 4.8-7.2x for a GOOD reason: the local sequential baseline is 85-95 tok/s vs G7e 60-75 (961920 m=1 launches are latency-bound; the laptop clocks higher and G7e's 188 SMs sit idle on matvecs), so grouped's headroom is smaller. KEY FINDING: even the daily "resident" config is a partial-spill regime — the auto cache holds ~34% of the 30720-block expert set; grouping cut H2D 28-30→10-11 GB/forward (2.7-3.8x).
- **Spill-local (BW24_MOE_SLOTS=64, hit-rate ~0.6%, PCIe-dominant):** seq 48.1/47.9/46.8 tok/s at 112/416/1460 GB H2D per forward (linear in T); grouped 163.1/173.9/180.3 at 8.1/14.3/14.9 GB (bounded by active experts per layer, NOT tokens) = **3.4x/3.6x/3.9x pp, H2D 14x/29x/98x less**. nsys (cap64, T=501): seq 113.4 GB H2D / 4.37s memcpy time / 226876 copies vs grouped 10.2 GB / 0.48s / 33385 — the H2D critical-path collapse is the whole story; grouped is compute-bound again even at 0.6% hit-rate.
- **Expert order (measured, now DEFAULT desc):** processing experts by descending m_e admits the hot experts to the SLRU before the small tail pollutes it → residency converges in ONE forward: auto-cache T=501 first-forward 126.9→169.9 (1.34x), cap512 119.6→160.8 (and kills a rep-to-rep bimodal); wash (<2%) at cap64 and long prompts. `BW24_MOE_ORDER=id` restores id order. Slot scheme keeps byte-identity under ANY order.
- **Disk tier (BW24_SPILL_DISK, 31481/31488 blocks mmap'd):** argmax 1178 MATCH, grouped cap64 163.6 tok/s == pinned-tier speed (page-cache-warm; cold-NVMe untested).
- **Gates (all on the NEW desc default):** run-gen argmax 1178==1178 grouped+seq at auto AND cap64; BW24_MOE_GATE byte-identity PASS at auto AND cap64 (staging live); kernel-check ALL GREEN; 9B run-gen 268==268 + run-spec K=2 PASS (133.9 tok/s, 66.7% accept); 27B run-spec K=3 PASS (78.2 tok/s, 83.6% accept).
- **Decode untouched** (t=1 falls through to sequential). Remaining MoE levers: async copy-stream prefetch of the NEXT expert group during the current GEMM (moe_cache.rs §C.2 infra wired, `prefetch_active` barrier ready — at cap64 grouped the 0.48s H2D is serial with compute, worth ~15% pp); MiniMax-M3 REAP as the true doesn't-fit model.

## Q8 TRUNK-FUSION SCOREBOARD — DENSE-PROJECTION LAUNCH-FUSION SHIPPED: 112.9 → 116.1 tok/s (+3.4%) (2026-07-05, g7e kernel lane)
**The launch-arc's remaining q8_0 wall (qmatvec_q8_0_mmvq 64k launches/128-tok window = 250/token = 18.3% of GPU time in 2.4-14us kernels) attacked with horizontal fusion (JSONL `Q8 TRUNK-FUSION` row).** Call-site map (nsys grid-shape decomposition, verified vs GGUF dtypes — ALL 35B trunk dense projections are Q8_0, experts IQ3_S/IQ4_XS, lm_head Q6_K): per token = 30x(wqkv 8192 + wqkv_gate 4096) linear + 10x(wq 8192 + wk 512 + wv 512) full-attn + 30x ssm_out + 40x(gate/up/down shexp) + 10x wo. Fix = `qmatvec_q8_0_mmvq_fused2/fused3`: the dual-mr2 recipe with the same-out_f restriction LIFTED via a block-offset split (blocks [0,nb0) → tensor 0, rest → tensor 1/2; per (tensor,row) the body is `q8_0_mmvq_row1` = qmatvec_q8_0_mmvq VERBATIM → bit-identical). Fused sites: wqkv+wqkv_gate (both all_fast and the 35B else-arm where F32 beta/alpha blocked all_fast — that arm also folds the two redundant quantizes), wq+wk+wv triple, shexp gate+up at t=1 (sequential AND _dev MoE paths), 9B q8_0 ssm_beta/alpha fused2 twin of the NVFP4 dual. Spec verify T=1 mirrors EVERY site (decode==verify dispatch identity, FP-order lesson #8). Unfusable singles remain by data dependence (ssm_out/down_shexp/wo consume mid-layer activations).
- **Numbers (free clocks, N=3 interleaved, env law + BW24_MOE_CACHE=1): tg128 116.1 vs 112.3 off (+3.4%); tg512 115.9 vs 112.3 (+3.2%). nsys: q8_0 m=1 launches 64k→20.5k/window (+17.9k fused2 +2.6k fused3 = -160 launches/token), quantize_q8_1 56.6k→38.7k, cuLaunchKernel 269.6k→228.6k.**
- **27B/9B p3 spec battery: NEUTRAL as predicted** (27B p3 K=3 124.3 tok/s both arms — trunk is NVFP4/k-quant, zero q8_0 m=1 work; 9B ~204 both arms, only the tiny beta/alpha pair rides fused2). The dense-model trunk projection path shares the CODE but not the DTYPE — no cross-model win, none expected.
- Gates: kernel-check Q8-FUSED2 (8192/4096 + 512/512) + Q8-FUSED3 (8192/512/512) rel=0.00e0 bits=true, ALL GREEN × {35B, 9B, 27B}; 35B argmax 1178 t=1+t=4 with streams IDENTICAL on/off; 9B 268 / 27B 1178; run-spec K=1..8 PASS × {35B, 9B} + 27B real-p3 K=3 + 9B text-p3 K=2; BW24_MOE_GATE clean. Seam: `BW24_Q8_DUAL=0`.
- **Remaining 35B decode wall: MoE _dev pair 43.3+36.1us = 43% of GPU (m=1-matvec-bound — the k-quant r2 recipe may apply), fa_decode_vec_q 135us = 11%, then the ~80 unfusable q8_0 singles (5.2%).**

## DEV_Q8 MULTIROW/OCCUPANCY SCOREBOARD — down8 IDLE-LANE FIX SHIPPED: 129.9 → 134.8 tok/s (+3.7%) (2026-07-05, g7e kernel lane)
**The resident-experts dp4a pair (43% of decode GPU) attacked with launch-geometry twins — bit-identical outputs, only grid/block changed (JSONL `devq8-multirow-occupancy`).** 10 variants built + measured, all behind seams (`BW24_MOE_DEVQ8_GU` / `BW24_MOE_DEVQ8_DOWN` / `BW24_MOE_DEVQ8_WPB`), every one argmax-1178 + 64-tok-stream IDENTICAL (t=1 AND t=4).
- **WINNER, DEFAULT ON: `moe_down8_fma_dev_q8_w8h2`** (auto when in_f==512 & n_used<=8; `BW24_MOE_DEVQ8_DOWN=0` restores base). Two stacked ideas: (a) slot-parallel block (32,8) — warp j computes slot j's dot, partials via smem, warp 0 replays the slot-ordered __fmaf_rn chain (8 serial FMAs, FP order verbatim); (b) half-warp dual-row — down's nsb=16 left lanes 16..31 IDLE in every warp of the base kernel; now lanes 0..15 = row o, 16..31 = row o+1, row-B partials __shfl_down 16 into the base lane layout, then the SAME masked 32-lane tree → bit-identical per row. Kernel 28.3→20.3us (-28%), share 22.5→17.3%; tg128 129.9→134.8 (N=3, free clocks, env law).
- **gate_up: ALL NEGATIVE OR FLAT — base kernel stays.** RPW multirow r1/r2/r4 x WPB 1/2/4/8 (128-130, r4=-5%), s2 gate/up warp-split (129.4), s2z (128.9), gs4 4-warp g-split (129.1), u64 ILP unroll (129.8), j8 slot-packed blocks (133.9 — 8x fewer blocks and still -0.6 vs w8h2 default). down w8r1/2/4 full-warp slot-parallel (129.5-131.0), h2 alone (132.8), w8h2r2 mr2-stack (134.5) all below w8h2.
- **LESSON: "poor occupancy" was the wrong theory for gate_up — the warps are latency-bound individually, and neither fewer (multirow) nor more (dot-split) warps helps; the real hole was down8's DEAD HALF-WARP (nsb=16 vs 32 lanes) + its serial 8-slot loop.** nsb=64 gate_up has no idle-resource hole → geometry-inert.
- Gates: kernel-check ALL GREEN; 35B argmax 1178 t=1 + 198 t=4, streams identical vs pre-arc for all 10 variants AND the new default; run-spec SELF-CONSISTENCY PASS x {35B, 9B, 27B-NVFP4} K=1..8. NOTE: `BW24_MOE_GATE` t=4 grouped-vs-seq mismatch (maxdiff 3.4e-4) PRE-EXISTS at clean b91d5072 (verified via stash) — the dev_q8 commit's sequential-q8 vs grouped-f32 FP-order shift, NOT this arc (sequential path + expert_dot_g bodies untouched here).
- **Remaining 35B decode wall (post-arc nsys): gate_up 22.0us = 18.8%, down8_w8h2 20.3us = 17.3% (the pair now ~36%), fa_decode_vec_q 134us = 10.4%, fused2 8.7%. Next MoE lever must cut the pair's DRAM/latency itself (e.g. gate+up row merge with act-quant reuse, or L2-resident expert tiling), not its geometry.**

## MoE DECODE SCOREBOARD — LAUNCH-STRUCTURE ARC SHIPPED: 74.9 → 112.9 tok/s (+50.7%) (2026-07-05, g7e launch lane)
**The predecessor's measured wall (64% of decode in launch/bookkeeping, not kernels) attacked in 3 stages, each interleaved-A/B'd + gated + committed on main. 35B-A3B IQ4_XS tg128@ctx128, env law + BW24_MOE_CACHE=1, free clocks, N=3.**
**STAGE 1 — router DtoH elision (185f0a72, +1.4% on top of stage 2):** ROOT CAUSE of the old "fused router 2% WORSE" verdict found — the `=1` arm paid TWO full stream syncs (dtoh_i32(sel) + dtoh(w), each clone_dtoh+synchronize) + 2 alloc_zeros memsets per MoE layer, vs ONE sync for the host route's single 1KB logits dtoh. New `moe_router_topk_host`: uninit outputs, both DtoH issued ASYNC into a persistent PINNED host stage (flags=0 CACHEABLE — cudarc's `alloc_pinned` is WRITECOMBINED, pathological for host reads), then ONE synchronize. **Fused router is now DEFAULT-ON at all t** (`BW24_FUSED_ROUTER=0` rollback) so decode and spec-verify route identically (FP-order lesson #8 class). Selection exact (kernel-check tie gate); w rel 2.2e-7 (3 ULP, expf-vs-libm).
**STAGE 2 — memset/alloc churn (185f0a72, 74.9 → 80.9 alone):** moe_out alloc_zeros→uninit when gdec can fire (moe_down8_fma fully overwrites its row; sequential-fallback tokens lazily zero ONLY their row); the per-layer scratch_g/u/d trio (3× ~1MB alloc_zeros+memset+free per MoE layer per token — DEAD code under BW24_MOE_CACHE) now lazy+uninit; act/sa/g and router sel/w zeros→uninit (full-overwrite producers). nsys: memsets 149.5k→60.6k/window.
**STAGE 3 — zero-DtoH device-dispatch MoE (ecba3bdf, 82.0 → 112.9, the big one):** design option (c) indirect-kernel from the arc brief, no graph needed. When a layer's expert blocks are ALL SLRU-resident: router sel/w STAY ON DEVICE; `moe_gate_up_silu8_dev`/`moe_down8_fma_dev` twins read expert ids (+w) from the router's device output and weight pointers from a per-layer device table [3, n_expert] of fixed slot addresses (bit-identical FP chains to the gdec param twins — only the pointer/id SOURCE changes). MoeSlotCache grew per-layer residency counts + lazily-uploaded device rows (invalidated on eviction) + **one-shot `prewarm_layer`** (force-admits while FREE slots cover the layer, never evicts — 15.2GB one-time first-forward H2D on 96GB, hit-rate 100% from token 0; spill rigs skip automatically; `BW24_MOE_PREWARM=0` organic). Kills the per-layer router DtoH + stream sync (~40 host round-trips/token): nsys cuMemcpyDtoHAsync 20.7k→256 calls, cuLaunchKernel 408k→269k. **pp512 also rides it: 244→338 (+38%)** (prefill routes through the same path at t>1). Seams: `BW24_MOE_DEV=0`; `BW24_FUSED_ROUTER=0` disables dev too.
- **Numbers: tg128 74.9 → 82.0 (stages 1+2) → 112.9 (stage 3); tg512 112.3, tg1024 113.8; vs llama.cpp b9863 same box (tg128 235.4): 0.32x → 0.48x.**
- **Graph replay RE-MEASURED post-stage-3: NEGATIVE (JSONL `moe-graph-decode-negative-post-stage3`), 0.931x tg128 / 0.933x tg512** — same verdict class as graph-spec stage-3: once launches are structurally removed, capture/recapture overhead (3 warmup runs per t_kv bucket, recapture every 64 tokens) exceeds what batching reclaims. Do not re-open unless decode becomes launch-bound again.
- Gates (every stage): 35B run-gen argmax 1178 MATCH (t=1 [55] AND t=4 [9419,11,1814,0] prompts; 64-tok greedy stream IDENTICAL dev on/off AND fused on/off); 9B 268 / 27B 1178 MATCH; kernel-check ALL GREEN; run-spec K=1..8 self-consistency PASS × {35B, 9B, 27B} with accept counts unchanged.
- **Remaining 35B decode wall is now KERNEL time, not launches:** qmatvec_q8_0_mmvq 64k launches/window (the 40 trunk layers' dense attn/GDN m=1 projections — 18% of GPU time in 2.4us launches, a batching/fusion target), fa_decode_vec_q 135us/call (11%), and the _dev MoE pair 41%. Next levers by size: (1) fuse/batch the per-layer dense projection chains (same indirection trick or megakernel), (2) fa_decode split tuning at short ctx, (3) MoE _dev kernel efficiency (43us gate+up for 8 experts = m=1 matvec bound, the k-quant r2 recipe may apply).

## MoE DECODE-CACHE SCOREBOARD — ASYNC-PREFETCH ARC CLOSED NEGATIVE + GEMMA4 GAP LIST (2026-07-05, g7e lane)
**Arc 1 (async prefetch on decode misses): NEGATIVE-NOT-WORTH-IT, measured before building (JSONL `moe-async-prefetch-negative`).** 35B-A3B IQ4_XS decode tg128@ctx128, env law + BW24_MOE_CACHE=1, N=3: **74.9 tok/s** (up from the 69.6 record — drift from main-sync kernel work, not this arc); tg512 80.4, tg1024 83.4. nsys decode-window profile: ALL H2D staging = 70.5ms of a 1944ms window = **3.6% of decode wall**, and warmup-concentrated — last-quarter H2D is 1.4%. Steady-state hit-rate self-extinguishes misses on 96GB: 90.8%@64tok → 96.3%@512tok (16.6 MB/tok vs 454 stage-1). Perfect copy_stream overlap buys ≤3.6% for the store-before-evict hazard set — below the bar. The REAL 35B decode wall per the same window: cuLaunchKernel 573ms/185k calls + router DtoH 306ms/5.2k + memset/alloc/free ~370ms + kernel time — launch structure, not PCIe. `prefetch_active` seam stays wired for the SPILL rigs where misses never extinguish (24GB local; revisit there, not here). Side A/B: BW24_FUSED_ROUTER=1 is 2% WORSE at t=1 (73.4 vs 74.9) — stays default-off. llama.cpp b9863 same box: 35B tg128 235.4, pp512 8527 — bw24 decode 0.32x, gap = launch/bookkeeping structure (185k launches per 128 tokens; llama runs graphs).
**Arc 2 (gemma4-26b-a4b MoE bench): loader can't take it — EXACT 9-item port-gap list recorded (JSONL `gemma4-26b-moe-gap-list`), the MiniMax-plan-style deliverable.** Full metadata+tensor audit of gemma4-26b-a4b-q4km.gguf (15.63GiB, 30L, 128ex/top-8) found 3 NEW hard gaps beyond the 2026-07-04 STAGE-4 six: (7) GELU everywhere — bw24 only has silu; (8) Q5_0 dequant missing (blk.3.ffn_down{,_exps}) — enum exists, no dequant path, no kernel; (9) tokenizer.ggml.model=gemma4 sentencepiece — bw24-tokenizer hard-errors on non-gpt2. Plus the sharpened attention picture: per-layer head_count_kv [8×5,2]×6 with 25 SWA layers (window 1024, hd 256, rope 10k) / 5 global layers (blk 5,11,17,23,29: hd 512, kv 2, rope 1M + rope_freqs), and the global layers are V-LESS — llama gemma4.cpp:247 uses Vcur=Kcur + rms_norm(V). Router prologue: logits = mm(gate_inp, rms_norm(attn_out)/sqrt(n_embd) * gate_inp_s). Block wiring: parallel shared-MLP + MoE both from attn_out, summed, dual post-norms, layer_output_scale, final_logit_softcapping 30. llama reference numbers banked for the eventual vs-dense comparison: gemma4 pp512 11234 / tg128 224.3. STAGE-4 alternative stands: a qwen3moe/olmoe-shaped second MoE validates shape breadth with zero port.

## C1 SCOREBOARD — K-QUANT BATCHED r2 PORT SHIPPED (2026-07-04, closes the "k-quant batched variants ~19% of 9B verify" open item)
**The NVFP4 batched-latency recipe ported to q4_K/q5_K/q6_K `_b2/_b4` (JSONL `kquant-batched-r2-port`).** ncu-first on the DRAM-cold msweep (9B real shapes, m=4): q4_K long_scoreboard 19.6/issue DRAM 47.7%, q5_K 16.4/38.2% = memory-LATENCY (NVFP4 pre-fix class); q6_K lm_head the outlier at DRAM 90-91% = wall-bound. Shipped `_r2` twins (two rows/warp) + `_r2w8` (seam-only) + per-dtype wave-aware auto: **q4_K r2 whenever blocks>=4*SMs** (incl. the straggler window where NVFP4 r2 lost — qkv -15%, ffn_down in_f=12288 -35%, all measured shapes -11..-35% b4 / -6..-22% b2), **q5_K/q6_K r2 only at waves>=2** (the 248320-row lm_heads: q6_K -8% DESPITE 90% DRAM — the halved grid's deeper MLP raises achieved DRAM; q5_K 27B lm_head -2.1% b4 / -2.9% b2; mid q5_K shapes base-or-flat — the 5/6-bit two-stream unpack makes r2's staging pricier). **r2w8 NEVER in auto for k-quants** — the 72->64 reg squeeze adds a stack spill and loses on every measured cell incl. wave-crossing lm_heads (mirror of the NVFP4 positive: residency crossing only pays spill-free). NO pf port (a k-quant group stages 10+ words vs NVFP4's 5), NO q8_0 r2 (only real batched shapes are out_f=32 ssm_alpha/beta), NO q6_K split-plane repack (recipe step 3 measured before building: lm_head is bandwidth-bound, the A6 latency prerequisite is absent). New seam `BW24_KQ_BV=base|r2|r2w8` (k-quant-only force, leaves NVFP4 dispatch alone; `BW24_MMVQ_BV` still maps globally).
- **E2E (free clocks, N=3 interleaved base-vs-auto medians, NGEN=256): 9B 182.8/145.5/122.3 vs 179.1/145.1/120.0 (+2.1/+0.3/+2.0%); 27B noise-level (+0.1/+0.0/+0.5%) — EXPECTED: the 27B round has ~no k-quant batched work (trunk NVFP4 already r2/r2w8, draft lm_head is m=1).**
- Gates: kernel-check ALL GREEN x {9B, 27B} x {auto, KQ_BV=r2, KQ_BV=r2w8 forced} (k-quant BATCHED rel=0.00e0); msweep bit-exact-bad=0 all cells; run-gen 82==82 both models; verify-probe byte-identical to clean-main baseline (3.815e-6, argmax exact T=1/2/3); run-spec K=1..8 PASS x {9B synth, 9B text p2, 27B real p3} on the final tree.

## C1 SCOREBOARD — A5 cp.async NEGATIVE / A6 SPLIT-PLANE REPACK SHIPPED (2026-07-04, SOTA-ADOPTION items 5+6)
**A5 (cp.async smem weight ring for batched MMVQ): MEASURED NEGATIVE — wall-ledger it.** Mandated ncu-first gate passed (residual stall post-pf/r2w8 is STILL memory latency: long_scoreboard 8.8-16.4/issue vs FMA-dep wait <=1.9, DRAM 48-69%), but the 3-4-stage cp.async.cg ring LOSES 15-32% on every real 27B b4 shape (DRAM-cold msweep): occupancy fine (70% > r2w8's 65%), DRAM fell 61.5->49.9%, stall stayed long_scoreboard = the wait itself. Mechanism: at m<=4 there is NO cross-thread weight reuse to amortize smem staging; the register variants' L1-amplified 2-row MLP is strictly better. Kernels kept behind `BW24_MMVQ_BV=ca|car2` (pfr2 precedent). Do not retry without a new mechanism (TMA multicast / m>=8 regime). JSONL `mmvq-b4-cpasync-ring-NEGATIVE`.
**A6 (Marlin-style walk-order repack): SHIPPED, default ON (`BW24_RP=0` rollback).** All NVFP4 matmul weights repack at load into [quant plane out_f x in_f/64 x 32B][scale plane x 4B] (model.rs `repack_nvfp4_split`, host-side pre-htod, zero VRAM spike, per-tensor `rp` flag): a lane's per-group read becomes ONE aligned LDG.128 + a dense 4B scale word instead of 5 scattered 4B LDGs at 36B stride. ONE copy serves every consumer — batched b2/b4 (`rp/rpr2/rpr2w8` under the same wave-aware rule), m=1 mmvq/mr2/dual `_rp` twins, dp4a `_rp`, Stage-A tag 9, prefill int8 GEMM kernel2 (RP=pre-decode: the cp.async raw ring that lost -3% on GGUF's 36B straggle WINS on the aligned layout). Embeddings (gather-only) stay GGUF. NOT ported (fall to rp int8 GEMM, recorded): W4A4 MMQ `BW24_MMQ` + `BW24_FP4` opt-ins. (2026-07-05: the W4A8 MMQ IS rp-ported — see STAGE 2b — and is now the default prefill.)
- **E2E (interleaved rp on/off, N=3 medians, free clocks, NGEN=256): 27B 94.2/87.4/69.1 tok/s (+2.2/+2.1/+2.0%); 9B 180.3/148.0/119.2 (+0.9/+0.6/+0.7%). PRIME/TTFT also wins: 27B -6.1..-8.1% (p3 8.75->8.22s), 9B -5.5..-7.0%. rp wins every interleaved pair (27B 15/15, 9B 9/9).** vs llama serve cross-session: 27B ~0.98x p2 / ~0.94x p3 (llama re-run needed for an honest image).
- Kernel-level (DRAM-cold msweep, 27B m=4): b4 trunk sum 366.0 -> 352.5 us/layer (-3.7%, wins all 5 shapes: ffn_down -6.3%, gate/up -3.7%, attn_gate -2.7%, ssm_out -2.0%, qkv -1.1%); m=1 -2.2..-4.5%.
- Gates: kernel-check ALL GREEN x {9B, 27B} incl the new RP battery (repack-roundtrip 0 bytes + bit-bad=0 for every consumer kernel: MMVQ m=1/2, BATCHED m=2/3/4, DP4A m=1/5, GEMM T=128, STAGE-A); run-gen 82==82 rp on AND off; verify-probe 0.000e0; run-spec K={1,2,3,4,6,8} PASS x {9B synth, 9B text, 27B real p2, 27B real p3} on the final tree. JSONL `mmvq-b4-splitplane-repack-prototype` + `nvfp4-splitplane-repack-shipped`.
- FP-order lesson (A5+A6 pair, for the corpus): cp.async staging pays where cross-thread smem reuse exists (GEMM tiles) and loses where it doesn't (m<=4 matvec); layout alignment flips the sign of the SAME staging mechanism.

## C3 SCOREBOARD — A4 CHUNKED WY GDN PREFILL SHIPPED, default ON (2026-07-04, SOTA-ADOPTION item A4; `BW24_GDN_CHUNKED=0` rollback, `BW24_GDN_CHUNK=32` default)
**gdn_scan_s128's sequential per-token recurrence replaced FOR PREFILL ONLY by the chunk-parallel WY/blockwise-inverse matmul form (flashinfer/fla chunked delta-rule; full derivation in the cu/hybrid.cu K1-K5 header).** 5 kernels: cumgate cumsum -> A/P chunk matrices (2x2 register tiles, P upper zero-filled) -> (I+A)^{-1} both-RHS forward substitution (thread-private column history, register templates C=32/64) -> the ONLY chunk-sequential kernel (Y=U-WS + rank-C state update, smem col-split state, chunk-start snapshots move the o_inter dot off the serial path) -> output assembly (4x4 register tiles). Dispatch `gdn_scan_prefill` in linear_attn + linear_attn_prime ONLY — decode and the spec verify still call gdn_scan_s128 directly (decode==verify dispatch identity untouched); prime rides prime_cache so the win lands in TTFT.
- **Kernel (gdn-bench, clock-locked 1852): T=512 759->416us (1.82x), T=6257 10.69->4.93ms (2.17x — grows with T; the scan is O(T) serial).** Chunk sweep: C=32 wins EVERY interleaved pair (chunk matrices cost O(T*C); the state pass is C-flat); C=64 close; C=128 NEGATIVE (gated to generic fallbacks).
- **E2E (free clocks, N=3 interleaved off/on medians): pp512 9B 201.7->192.9ms (+4.6%, 3/3), 27B 650->629ms (+3.3%, 3/3). PRIME/TTFT: 9B p2 -6.9% / p3 2.457->2.334s (-5.0%); 27B p2 -2.9% / p3 8.098->7.832s (-3.3%) — wins 12/12.** Spec-optima gen tok/s moves ±4% in OPPOSITE directions per model (9B +4.0, 27B -3.8) = acceptance on drifted sequences (content), not engine cost.
- **EXACTNESS (the A4 risk class — chunked FP accumulation order != sequential, NOT bit-identical by design):** kernel-check ALL GREEN x {9B, 27B} incl new f64-truth gates (chunked out <=4.2e-5 / state <=1.1e-4 rel vs truth; sequential itself is 3.9e-6/1.2e-5 — the (I+A)^{-1} conditioning noise class); real-weights per-layer diff (9B T=512, all 24 linear layers, BW24_GDN_DIFF=1 dual-run mode): out max_rel <=4.5e-5, state ~2e-4 typical / 1.1e-3 worst (layer 0). run-gen argmax 82==82 on/off x both models (24/24). run-spec K={1,2,3,4,6,8} PASS x {9B synth, 9B text, 27B p2, 27B p3} chunked ON — plain and spec prime through the SAME path, exact by construction, verified.
- **REPORTED PROMINENTLY (user judges): e2e greedy token agreement off-vs-on, 3 prompts x 2 models: first-16 IDENTICAL 6/6, but full-256 forks on 5/6 (index 47/61/110/118/125; 27B p1 fully identical).** Prefill-state FP drift flips one tight-margin argmax and the continuations legitimately diverge — same accepted class as the batched-prime precedent (which forked 9B p2 at index 11), just measured deeper here. FP-order lesson #9: prefill-side reorder does NOT break spec exactness (verified) but forks greedy text at depth ~50-125.
- Wall-ledger (JSONL `gdn-chunked-wy-prefill-prototype` notes): K4-state optimization ladder — smem tiles + accumulator-innermost register tiling = the win recipe (K5 3.1x, K3 3.6x); MEASURED LOSERS: step-B 4x4 reg tiles (reg pressure), NSPLIT=8, cp.async double-buffered staging pipeline, 512-thread blocks — the serial chain's walls are the U/Y global round-trips, occupancy capped by smem regardless of split. Next lever if reopened: per-chunk grid-wide launches / cooperative grid.sync, or cublasLt strided-batched f32 for the parallel matrices.

## SPEC SCOREBOARD — HIDDEN-PAIRING FIX LANDED (2026-07-04, the 27B acceptance lever CASHED)
**ROOT CAUSE of the 0.58-vs-0.75 acceptance gap: hidden-PAIRING convention, found by reading the reference draft-mtp impl (speculative.cpp "shift the tgt embeddings to the right by one position").** The NextN head is trained on rows (token x_p, trunk hidden h_{p-1}); bw24 paired SAME-ROW (x_p, h_p) in mtp_kv_fill and seeded chain step 0 via the pseudo-hidden pass. Fix (spec.rs, no kernel change): fill hiddens shifted by one (row 0 zeros at prompt / carried `fill_prev` in-loop) + step-0 seed = predecessor's TRUE verify hidden directly — the per-round pseudo-seed pass (and its head-less graph) is DELETED on the default path. Default flipped; `BW24_SPEC_HSAME=1` = legacy seam. `BW24_SPEC_STATS=1` = per-slot accept + draft-len histograms.
- **Metric normalization vs reference: definitions IDENTICAL** (accepted/drafted, p-min-stopped chain, sub-threshold token discarded uncounted — verified in server-context.cpp + speculative.cpp). The gap was real. Caveat: their p-min gates on TOP-10-renormalized p (laxer than our full-softmax p at the same value).
- **Acceptance (real prompts, deterministic): 27B K3/pm0.2/trim p2 0.569→0.731, p3 0.445→0.614; 27B K1/pm0 p2 0.783→0.855, p3 0.667→0.816; 9B K2/pm0.3 p2 0.604→0.749, p3 0.498→0.594; 9B synth K2/3/4 0.73/0.62/0.47→0.93/0.83/0.75** (also fixes the persistent-KV record's synth K≥3 regression). Per-slot p2: [.755 .536 .395]→[.843 .725 .613] — the "late slots collapse" was seed corruption, not chain depth.
- **E2E (interleaved old-vs-new, N=3 medians, free clocks, NGEN=256): 27B 90.0/83.4/66.0 tok/s (+13.5/+18.4/+25.0% vs 79.3/70.4/52.8); 9B 174.9/144.1/115.4 (+10.7/+21.1/+14.2%).** vs llama serve cfg same prompts: 27B p2 0.94x (88.8), p3 0.90x (73.4) — was 0.79x/0.72x. Acceptance vs ref: p2 0.731 vs 0.826, p3 0.614 vs 0.640.
- **Hypothesis battery (JSONL `spec-hidden-pairing-prev`):** H1 pseudo-vs-true seed DISSOLVED (true seed now free — it IS the last verify column); H3 p-min sweep post-fix: 27B K=3 pm0.2 STAYS optimal (pm≤0.15 no gain, pm0.3 flat), 9B K=2 pm0.3 stands (K=3 pm0.3 ties); H4 FR-Spec trim KEEP (acc unchanged, +10.8%/+7.5% tok/s); H5 external Q4 draft file NEGATIVE (acc .703 < native .731, tok/s 74.9 < 84.4 — the ref's 0.75 rode the pairing, not the draft file); true-hidden refresh still positive (.685/79.9 without vs .731/84.4 with) — stays default.
- Gates: kernel-check ALL GREEN, run-gen 82==82, run-spec K=1..8 PASS x {9B synth, 9B text, 27B real}, verify-probe 0.000e0, all 5 seams (KVLOCAL/HSAME/NOREFRESH/REPLAY/NOGRAPH) exact.
- **Remaining acceptance gap (p2 .731 vs .826) is NOT the next lever** — plausibly NVFP4 trunk-hidden quality + their laxer p-min chain mix (their mean len 3.48 vs our 3.10). Next 27B spec levers by cost: draft/verify kernel time (b4 MMVQ tranche 2: host-fused tiny GDN projections, k-quant batched variants), not acceptance.

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
**Stage 3 (graph-captured verify) = MEASURED NEGATIVE as specified, do not build without a new unlock:** savings ceiling = per-trunk-pass launch overhead (11.06 − 10.41 ≈ 0.65-1.15ms) × ~1 verify pass/round ≈ +3-5% e2e; capture cost = 3 REAL verify passes (~33ms) per bucket, and the bucket key must be (T, per-row fa n_splits vector) because R1 forbids padded splits (measured argmax flips) — with p-min T varies 2..k+1 and n_splits churns every 64 ctx tokens + boundary vectors ⇒ ~10-13 captures per 128 tokens ≈ 330-430ms = 4-10x the savings, at ANY gen length. Unlocks that change the math: (a) cudaGraphExecUpdate (re-parameterize a baked graph instead of re-capturing — not exposed in cudarc 0.19), (b) make fa n_splits t_kv-INDEPENDENT in BOTH decode and verify (the exactness pair moves together; kills rebucketing AND adds CTAs at short ctx where fa-decode is grid-starved) — decode-wide policy change, needs its own full battery. **UPDATE 2026-07-03: unlock (b) effectively ARRIVED via 642582a (fa rows derives per-row splits in-kernel from fixed 64-key chunks) and the question was RE-MEASURED — still negative, ceiling shrank to 2.8-3.8% because stages 1-2 removed the launches the graph would have reclaimed. See wall ledger + JSONL `graph-spec-stage3-remeasure`. Do not re-open on launch-overhead grounds; only if the verify becomes launch-bound again (much faster kernels or different T regime).**
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

## 27B PREFILL/TTFT ARC — LOCAL LANE (Fable, main branch, 2026-07-05)

North-star: close the 27B prefill/TTFT gap (vLLM prefills the same 27B NVFP4 5.6-6.8x faster; local vLLM 27B prefill 4.9k tok/s vs bw24 ~0.8k). Goal 27B TTFT p3 <1s, e2e >100 tok/s.

**STAGE 0 — baseline re-measured (commit 1e5bfab, N=3, free clocks, JSONL `1e5bfab (baseline re-measure...)`):** 27B prime .289/2.419/7.864s p1/p2/p3 (matches HANDOVER ref within 1.1%, no thermal sag). pp-only tok/s FLAT vs T: 810/773/804 @ T=512/1845/6257. nsys p3: qmatvec_gemm_nvfp4_rp (int8 W4A8) = **79.6% of prefill**, fa_prefill 4%, gdn cluster 4.9%, quantize 4.5%. Peak VRAM 9.5GB. First-16 reference tokens captured for all 6 (model,prompt) cells (the arc's agreement gate).
- **CHUNKED-PREFILL SCHEDULING = NO-GO (measured, cheap-stage-first per the plan):** pp tok/s is already flat in T with no long-T degradation, and bw24 prefills one request in one batched pass with no inter-request scheduler. vLLM's chunked-prefill is a *multi-request* scheduling win, not a single-request kernel win. The 6x gap is in the GEMM. Zero-accuracy-risk stage yielded zero — do not build it for single-request TTFT.

**STAGE 1 — vendored NVFP4 W4A4 MMQ A/B = NEGATIVE (accuracy gate), commit 1e5bfab, JSONL `A/B ... vendored llama NVFP4 W4A4 MMQ`:** BW24_MMQ=1 (needs BW24_RP=0; the A6 repack silently bypasses it). 3-arm interleaved, N=3. **Perf is real: 27B pp 2.7-3.0x (810→2414/2295/2202 tok/s), p3 prime 7.86→2.93s; 9B 2.2-2.4x. mul_mat_q_nvfp4 = 43.6% of ARM_B p3 prefill (engaged).** BUT the W4A4 activation quant is the gate-breaker: 27B argmax 82==82 MATCH yet **p3 first-16 FORKS AT INDEX 0** (emits think-token prefix 248046/248045); 9B **argmax FAILS 78!=82** + p1 forks at index 8. kernel-check ALL GREEN + run-spec K=1..8 self-consistent (internally exact) — so it's the historical W4A4 accuracy class (maxdiff ~1.0) reproduced at e2e, NOT a bug. **CUTLASS W4A4 (cu/cutlass_fp4_sm120.cu, the OFF seams) inherits this exact class — do NOT build resident/OTF CUTLASS until the ACTIVATION side changes.** Keep BW24_MMQ W4A4 as an opt-in speed/accuracy tradeoff.

**STAGE 2b — RP TILE LOADER + DEFAULT-FLIP = SHIPPED (2026-07-05, JSONL `w4a8-rp-loader+default-flip`).** The W4A8 MMQ now COEXISTS with the A6 split-plane repack: `load_tiles_nvfp4_w4a8` gained an `is_rp` template arm that reads the split-plane layout (quant plane `W + ib*32`, scale plane `W + out_f*(in_f/64)*32 + ib*4`, flat block index `ib = row*stride + kbx` — the repack copies qs/d bytes verbatim so the flat index serves both planes). PURE ADDRESS REMAP: same LUT dequant, same smem write order, same FP ops in the same order → **bit-identical to the GGUF loader, proven by the new hard `MMQ-W4A8-RP` kernel-check gate (0 mismatched bits over T=16/64/128/512, up to 6.3M f32 values)**. C-ABI `bw24_mmq_nvfp4_w4a8` takes an `rp` flag; dispatch routes per-tensor off the model `rp` flag. **DEFAULT-FLIP taken: the vendored MMQ prefill suite (NVFP4 W4A8 + Q4_K/Q5_K int8-MMA — the exact pair the old `BW24_MMQ_W4A8=1` arm engaged) is now DEFAULT-ON** via `mmq_w4a8_enabled()` in `mmq_supports`; `BW24_MMQ_W4A8=0` = escape hatch to the int8 GEMM; `BW24_MMQ=1` still opts GGUF-layout NVFP4 into W4A4; lib.rs matmul/matmul_pre MMQ gates collapsed into `mmq_supports` (policy in one place). **Perf (N=3 interleaved a/b/c): W4A8+rp pp512 27B 3140 / 9B 8848 (1.91x/1.80x vs int8+rp 1648/4928), 27B prime p1/p2/p3 0.082/0.612/2.083s (2.51x/1.72x/1.54x) — within 1.6% of the rp0 predecessor arm (3160/8901, 0.082/0.602/2.050) — AND rp-on decode fully retained: 27B gen 67.9/61.0/60.1 tok/s == the int8+rp arm, where rp0 pays -1.7..-2.5% (66.2/59.7/58.9).** Gates on the final flipped default (no env): kernel-check ALL GREEN incl MMQ-W4A8 oracle (rel 4.2e-3..7.4e-3) + MMQ-W4A8-RP bit-identity; run-gen argmax 27B 1178==1178 + 9B 268==268 MATCH; first-16 IDENTICAL to the int8 reference on ALL 6 cells (also on the =0 escape hatch — 12/12 arms); run-spec K=1..8 PASS x {9B, 27B p3}. NEXT: FP8-activation CUTLASS at large m (W4A4 CUTLASS stays closed), stream-K for W4A8 (needs the fixup FP-order care), or the SSM-prep/FA-prefill levers from the kernel-diff table.

**STAGE 2 — THE ACCURACY-SAFE RUNG = BUILT + POSITIVE (commit stage2-w4a8-mmq, JSONL `stage2-w4a8-mmq`).** Vendored llama non-Blackwell NVFP4 W4A8 MMQ as a new self-contained TU `cu/mmq_nvfp4_w4a8.cu`, opt-in `BW24_MMQ_W4A8=1` (requires `BW24_RP=0` — A6 repack bypasses MMQ, same as W4A4). **pp512: 27B 1647->3109 tok/s (1.89x), 9B 4854->8670 (1.79x); 27B prime p1/p2/p3 0.206/1.038/3.136s -> 0.081/0.582/1.971s (2.53x/1.78x/1.59x). N=3 interleaved, free clocks, spread <0.2%.** ALL FOUR GATES GREEN: (1) kernel-check MMQ-W4A8-vs-f32-oracle rel 3.3e-3..4.7e-3 = INT8 BAND (hard gate 2e-2), NOT the 0.08-0.11 W4A4 band — new MMQ-W4A8 oracle gate added to kernel_check; (2) run-gen argmax 27B 82==82 + 9B 82==82 MATCH with W4A8 engaged e2e; (3) first-16 IDENTICAL to the default-int8 reference on ALL 6 (model,prompt) cells — including 27B p3 index 0, THE cell where W4A4 forked at index 0; (4) run-spec K=1..8 self-consistency PASS. **This is the honest fast-AND-exact prefill: W4A8 keeps bw24 default-GEMM int8 math (weight FP4->int8 dequant is bit-exact; q8_1 D4 symmetric-scale activation is the same int8 class), so argmax and prefix hold where W4A4 broke them. Half the W4A4 throughput (3109 vs 5379 on 27B) is the price of the accuracy class.** Wiring: both `BW24_MMQ` and `BW24_MMQ_W4A8` now enter the matmul/matmul_pre MMQ dispatch. NEXT: rp tile loader (W4A8 with the A6 repack instead of forcing rp-off), or default-flip after soak; then reconsider whether an FP8-activation CUTLASS variant beats W4A8-MMQ at large m (W4A4 CUTLASS stays closed per Stage 1).

--- ORIGINAL PLAN (kept for the line refs, now DONE) ---

**STAGE 2 plan:** W4A8 on the SAME fast MMQ tile. llama's NON-Blackwell NVFP4 MMQ path is exactly this and vendors 1:1:
  - `mmq.cuh:1069 load_tiles_nvfp4` (the `#else`/non-BLACKWELL branch) LUT-dequants FP4→int8 via `get_int_from_table_16(src_qs, kvalues_mxfp4)` into the int8 x_qs tile + `x_df` = `ggml_cuda_ue4m3_to_fp32(bxi->d[sub])` per-16 scale (NVFP4 is symmetric — no min-offset, simpler than k-quants).
  - vec_dot = `mmq.cuh:1495 vec_dot_q8_0_16_q8_1_mma` (int8 m16n8k32 mma, W4A8), traits at `mmq.cuh:3337` `#else` arm; tile size `MMQ_DP4A_TXS_Q8_0_16` / `MMQ_MMA_TILE_X_K_NVFP4` (`:221`, `%8==4` padded).
  - Activation quant = `quantize_mmq_q8_1` DS4 — ALREADY vendored in `cu/mmq_q45k.cu` (`quantize_mmq_q8_1_ds4_kernel`); reuse it. The whole q45k MMA/writeback/xy-tiling scaffold (`mul_mat_q_q45k`, `mmq_write_back_q45k`) is reusable — add a `load_tiles_nvfp4_w4a8` + a `bw24_mmq_nvfp4_w4a8` C-ABI launcher alongside the existing bw24_mmq_nvfp4 (W4A4) in `cu/mmq_fp4.cu` or a sibling TU.
  - EXPECTED: keeps bw24's current int8-W4A8 accuracy class (which passes ALL gates — it IS the default GEMM's math) at ~Q4_K-MMQ throughput (llama's k-quant MMQ pp is in the 4500-5000 band). This is the honest path to a fast AND argmax-exact prefill.
  - GATE IT: kernel-check MMQ-vs-oracle rel in the int8 band (~1e-3, NOT the 0.1 W4A4 band); run-gen 82==82; first-16 vs the Stage-0 reference lists MUST match on all 6 cells; run-spec K=1..8 PASS. Only then wire into matmul/matmul_pre and A6-repack (needs an rp tile loader OR keep rp-off for the MMQ path like today).
  - THEN reconsider CUTLASS: only worth it if W4A8-MMQ leaves a large-m gap CUTLASS's native-FP4 tensor cores close AND an FP8-activation variant (sm_120 block-FP8 mma, 381 TFLOP) can hold the gate — W4A4 is closed.

**NO_EVT DEFAULT-FLIP — SHIPPED in this arc (lib.rs):** event tracking now DEFAULT OFF, `BW24_EVT=1` = escape hatch (legacy BW24_NO_EVT is a no-op alias). Cross-stream hazard audit (in the code comment): copy_stream touched by only stage_expert_async (ZERO callers) + the moe_cache admit barrier (gated on `prefetch_active`, setter never called); graph-capture sites use live-state `was_tracking` guards → degrade to no-ops. +4.6% measured 27B decode. Full gate battery running to confirm flip-safe before commit.

**DECAY-STUDY STAMPS (research/e2e/g7e-decay-study-stamps.jsonl):** independent second opinion on lane/g7e 6bcf8e1/438db11 from raw artifacts (/tmp/g7e-decay). (a) acceptance content-driven not ctx-driven = **CONFIRMED** (prose control -0.5pp over 4400 ctx vs code -12pp; decomposition arithmetic re-derived 41.8/58.2 vs claimed 41.1/58.9). (b) FA-verify-cost = **CONFIRMED** (independent sqlite: p1 54us vs p3 373.5us/instance, +2.56ms/round == claim). Main thread landed its own stamp in g7e-rtx6000.jsonl (aggregates FA as 4.64ms/16.9%-of-round total vs my growth-only 2.56ms — same direction, complementary framing). Ranked lever (fa rows shared-K reuse) is a G7e-lane priority.

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
