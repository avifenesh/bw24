# perf-frontier: the next absolute-performance levers for the 5090 (2026-08-02)

Lane `lane/perf-frontier` (from `restructure/public-split` b8ca4e2e). Reading/mapping lane:
each lever ships with a measure-first experiment for its own future lane; nothing here is a
measured claim about memra unless it carries an internal receipt path. Mission (owner posture
2026-08-02): the comparison era is over — find the next absolute-performance levers for the
RTX 5090 (sm_120a) on the ALREADY-SUPPORTED model set, from (a) other engines' recent work
and (b) papers. No new model ports, no H100 focus.

Method: three parallel web surveys (engines / papers / sm_120 hardware; citations dated
in-line), judged against internal evidence: `research/fa-decode-deep-20260802/`,
`research/depth-decode-20260802/`, `research/verify-economics-20260802/`,
`research/tune-data/current-board.json` + `rig5090.jsonl`, `docs/FLAGS.md`,
`research/SOTA-SWEEP-2026-07-13.md`, `research/upstream-sweeps.md` (through 2026-07-27),
`research/sm120-empirical-capabilities.md`, the ARCHITECTURE.md refutation ledger, and the
prefill-GEMM memory ledger. GPU disclosure: the hardware survey ran seconds-long on-device
capability probes (ptxas assembly checks, one CLC correctness kernel, one L2-window
microbench, cluster-launch checks) — sources preserved in `ptxprobe/` here; these are
capability facts, not perf lanes, and every perf number from them is labeled single-run
microbench. No model benchmarks were run in this lane.

## 0. Where the wall is TODAY (the internal map the levers must hit)

| front | state (2026-08-02 board) | residual, mechanism-named |
|---|---|---|
| plain decode 512-ctx | 1.09–1.13x llama (q9/q27/q35) | m=1 DRAM wall; k-quant instruction diet refuted at locked clock — remaining lever class is J/token → DVFS clock residency (rig5090.jsonl 2026-07-08) |
| decode at depth | deep fa landed (vec kernel 1.43x @d6144); leads hold 1.11–1.13x at 6.3k | in-kernel residual: long-scoreboard 26% / MIO / order-pinned B3 chain vs the 3-blocks/SM smem cliff; PLUS the stale sp8→sp64 rung at 3072 (combine 19.0µs > deep vec 11.5µs at d2048 — fa-decode-deep §8, priced not built) |
| spec decode | 1.06–2.30x llama; q27 K=3 = 2.00–2.18x its own plain | verify-tier premium (vT2/3/4 = 1.13/1.19/1.30x one decode step) pins K at 3; counterfactual v=1.05 → 3.05x at K=4–5 (verify-economics §3); b4 trunk matvecs = ~78% of the T=4 verify pass; no b16 twin for NVFP4/Q4_K/Q5_K (K=8 cliff 0.77x); mmvq_b8 at 30–35% of peak BW (ncu mapped, not attacked) |
| prefill NVFP4 dense | 0.59–0.78x llama | llama benches W4A4 (e2m1 activations) — exactness-REJECTED twice here (p3 argmax fork 2026-07-07/08); our W4A8 int8 MMQ config space is CLOSED (2026-07-06); W4A8-FP8 (`MEMRA_MMQ_F8F4`) is per-model only |
| KAT-Coder prefill | pp512 0.48x llama (2060 vs 4254) post IQ-MMQ port | bar-binding bring-up gap; residual = Q8_0-trunk-MMQ-vs-IQ4_XS tile decode + ssm prefill share |
| gemma hd512 globals decode | g12-plain-d1736 0.97x | llama packs gqa=8 q-heads as mma columns (tb512 30.8µs/layer vs their ~14.7) — gqa-packed mma hd512 decode kernel: OPEN TARGET (next-targets-20260730 §10) |
| Hy3 spill | 5.13 tok/s served (stated target 10) | I/O wall; artifact-demotion axis exhausted — needs a mechanism |
| o35b prefill short | pp512 0.415x llama | k-quant dequant amortization; direct tile loaders landed for the sk forms — short-prompt residual remains |

Exactness contract (binds every lever): FP summation order is part of the contract;
spec/graph/batched output token-identical to plain decode. W4A4 e2m1 activations are the
standing counter-example.

## 1. Survey I — engines (Feb–Aug 2026)

Versions verified 2026-08-02: llama.cpp b10229 (2026-08-02), vLLM v0.26.0 (2026-07-27),
FlashInfer v0.6.16 (v0.6.17rc1 tagged), TensorRT-LLM v1.2.1 (v1.3.0rc train), exllamav3
v1.3.0 (2026-07-31), SGLang v0.5.16 (2026-07-25).

**The consumer-Blackwell ground truth** (load-bearing): NVIDIA's first-party kernel
investment in sm_120/121 is thin — trtllm-gen FMHA cubins do NOT exist for SM120/121
(TRT-LLM issue #11799, 2026-02-28: SM120 has no tcgen05/TMEM; SM100 source "cannot simply
be recompiled"); FlashInfer's sm_120 attention today = FA2-class paths + the
NVIDIA-authored XQA decode kernels (`csrc/xqa/`, supported_major_versions=[9,10,12], incl.
`mla_sm120.cu`); SM120 MoE runs on CUTLASS sm120f / FlashInfer B12x backends (v0.6.16
added SM120/121 NVFP4 + W4A16 MoE). Everything competitive on sm_120 is warp-level
mma.sync + cp.async/TMA — exactly memra's design space; nobody has a moat there.

New-candidate highlights (full ranking in §4):

- **llama.cpp NVFP4 W4A4 activation-quant quality work** — PR #25730 (merged 2026-07-22):
  improved activation quantization for the native NVFP4 MMQ (per-channel amax alternative,
  fused amax kernel, 256-bit loads, hardware `cvt.rn.satfinite.e2m1x2.f32` via sm120f-class
  ISA — FlashInfer PR #2460 doubled NVFP4 throughput from the same convert instructions);
  they explicitly traded ~5% e2e to keep scale-search for quality. Local fork HEAD
  (bb090d1f1) already carries the NVFP4 series. This is the competitor's answer to the
  exact problem that blocks our fastest prefill path.
- **DFlash/DSpark block drafting went mainstream** — llama.cpp merged DFlash (PR #22105,
  2026-06-28) and DSpark (PR #25173, 2026-07-28); SGLang v0.5.16 ships DSpark
  (confidence-scheduled ragged verify, graph tiers keyed on packed token count); vLLM
  v0.26.0 ships DSpark drafters + runtime draft-weight update (#46725). Internal status:
  the DFlash lane CAPSTONE (2026-07-13, rig5090-gemma4.jsonl) measured DFlash LOSING on
  this rig — 55.5 vs MTP 89.7 at depth; root cause is a rig regime (the ~20ms/round q4
  drafter forward cannot be free on 24GB; published wins ride bf16 drafters on 96GB parts),
  mapped cuts cap ~0.72x of MTP. Verdict unchanged by the survey — but llama's spec-best
  baseline WILL move; re-sweep their flags at the next board pairing (standing rule,
  next-targets §10).
- **Fireworks KV-outer block-sparse attention** (blog 2026-07, open-sourced
  fw-ai/minimax-kernels; FlashInfer #3655 shipped MSA for SM120/121 in v0.6.16): loop
  inversion (KV-stationary) + tile-ordered partial stores + deterministic atomic-free split
  scheduler. 1.9–2.4x over query-stationary FlashInfer on B200 — but for top-k
  block-SPARSE attention (a model property none of our supported models have) and the
  LSE-merge reorders FP. Scheduling patterns transferable; not a lever for this model set.
- **llama.cpp CUDA-graph policy** — delayed capture until a graph repeats (#19754,
  2026-02-21: 1.11–1.12x mixed pp+tg on RTX 6000 Ada) + LRU multi-graph cache (#21611).
  Internal graph paths already went a different, measured way (qwen dc-eager beats graphs
  on the 5090; H100 graph door +16%); the multi-graph-LRU idea is a small candidate for
  spec-verify shape alternation only.
- **exllamav3 v1.0–1.3** (2026-07): coop GEMM + INT8 GEMV rework claiming RTX 5090 decode
  +11–109% (model-dependent) — the framing "high-BW/low-ALU favors table/INT8 GEMV over
  dequant-to-half" matches sm_120; our dp4a family already lives there (their gains are
  vs their own prior kernels, not vs a dp4a-class baseline). New: second-tier CPU KV cache
  with page/checkpoint eviction (v1.3.0) — the KV analog of our expert spill, relevant
  only past ~8k ctx.
- **Router path fusions** — vLLM DSv4 fused routing kernel (2.94% E2E TPOT, #48660),
  `fused_topk_bias` (1.5–2x kernel, #47463); llama.cpp topk-moe fusion (#25267), MoE
  gate/up activation-quant dedup (#25441). memra's in-house router (w8 GEMV + batch twin +
  fused sigmoid-dot + device dispatch) already covers this class; the residual named by
  verify-economics is q35's router at 1.2ms/verify-pass — a re-check, not a port.
- **Zero-sync spec loop** (vLLM MRV2 default since v0.25, SGLang Spec V2 overlap default):
  device-fed accepted counts, no host readback between rounds. memra's device accept walk
  exists (`MEMRA_SPEC_DEVACC`, token-identical); the stage-c zero-readback burst measured
  NEGATIVE on this rig (fixed-K waste > round-trip savings, 2026-07-10). Their win is at
  scheduler/Python layers memra doesn't have.
- **XQA ragged-Q spec verify + mask-derived KV-max** (FlashInfer #4137, 2026-07-28;
  llama.cpp `flash_attn_mask_to_KV_max`): per-row draft-token windows + device-side KV
  bounds — the July-13 sweep's "shape-stable graph windows" item, still unbuilt, now with
  two reference implementations.

## 2. Survey II — papers (2025–2026)

Ranked-relevant (full agent list preserved in the lane transcript; arXiv ids verified):

- **ARCQuant** (arXiv:2601.07475, Jan 2026, ACL 2026, code actypedef/ARCQuant): W4A4 NVFP4
  with augmented residual channels — error compensation rides INSIDE the GEMM reduction
  dim on stock NVFP4 block-scale kernels; worst-case error bound comparable to MXFP8;
  deployed on RTX 5090, up to 3x over FP16. The residual-channel count is a monotone
  exactness dial — the first principled mechanism to buy argmax stability for a W4A4 path
  since our two rejects.
- **MR-GPTQ / QuTLASS** (arXiv:2509.23202, Sep 2025, ICLR 2026, IST-DASLab): proves NVFP4's
  16-element groups NEUTRALIZE classic global Hadamard rotations — rotations must be
  block-size-matched (fused micro-Hadamard); 6x layer / 4x e2e vs FP16 on RTX 5090.
  Governs any W4A4 attempt: do not port a global-rotation scheme.
- **DFlash** (arXiv:2602.06036, Feb 2026, ICML 2026) + **DDTree** (arXiv:2604.12989) +
  **FastEagle** (arXiv:2509.20416) + **Domino** (arXiv:2605.29707): the one-pass drafter
  class, >2.5x over EAGLE-3 claims. Internally refuted ON THIS RIG for the measured reason
  above; watch for a drafter whose forward is <5ms-class on 24GB quantized.
- **Graft** (arXiv:2605.20104, May 2026): training-free — prune marginal draft positions,
  graft retrieval (n-gram/suffix) candidates into the freed verify slots; +21.8% avg over
  EAGLE-3, up to 5.41x. Composes with the own-trim MTP chain; targets exactly our weakest
  spec cells (p2/p3 acceptance 52–58% — the content-class gap depth-decode §3 isolated).
  Lossless by construction (verify arbitrates).
- **VeriCache** (arXiv:2605.17613) + **QuantSpec** (arXiv:2502.10424): draft-on-compressed
  KV / verify-on-full — the exactness-preserving verify-cost split (already on the July-13
  watch list; VeriCache's swap engine maps onto our spill machinery). Pays mainly ≥8k ctx.
- **MoE-SpeQ** (arXiv:2511.14102, Nov 2025): spec drafts predict the expert sequence for
  FUTURE tokens → prefetch experts ahead of routing; adaptive governor via an amortization
  roofline. Up to 2.34x over SOTA offloading on memory-constrained MoE. Exactness-clean
  (prefetch never changes routing). Direct hit on the Hy3 spill target — and memra already
  has the prediction-guided prefetch worker (`start_moe_prefetch_predictor`,
  hybrid_forward.rs) to graft the draft-token oracle onto.
- **BitDecoding** (arXiv:2503.18773, v3 Jan 2026, code OpenBitSys/BitDecoding): low-bit KV
  decode attention co-op CUDA+tensor cores, Blackwell NVFP4 paths, 3x single-batch at
  128K. Win regime is ≥32k ctx — our board lives ≤8k where the deep kernel just landed;
  banked for a long-context front, plus its query-transformation trick is a reference for
  an mma-based score phase.
- **SageAttention3** (arXiv:2505.11594, NeurIPS 2025): FP4 attention, 1038 TOPS on RTX
  5090 — LOSSY; blocked by the exact-math laws (same verdict as 2026-07-30 sweep).
- **FlashFormer** (arXiv:2505.22758) megakernels: internal verdict stands (not now — our
  GEMV layer ≥ published SOTA at batch 1).
- **Four Over Six** (arXiv:2512.02010, MIT-HAN): adaptive per-block NVFP4 scale target —
  artifact-side quality improvement for NVFP4 quantizer tooling, zero kernel change.
- **EQSPEC** (arXiv:2510.22876): every public batched-spec implementation violates output
  equivalence — read-before-building any batched spec; memra's contract is stricter than
  anything in the paper.

## 3. Survey III — sm_120a hardware (NVIDIA 2025–2026 materials + on-device probes)

Probes: sources in `ptxprobe/` (ptxas 13.1.115, RTX 5090 Laptop cc 12.0).

1. **Cluster Launch Control (CLC) WORKS on sm_120a — the headline find.** CUDA 13.3
   Programming Guide §4.12; PTX ISA 9.3 (`clusterlaunchcontrol.try_cancel`, the
   `.multicast::cluster::all` variant explicitly lists sm_120a); Colfax CLC tutorial
   (2026-05-11: "CLC can be used on SM12x"). ON-DEVICE: the try_cancel instruction
   assembles for sm_120a AND a full work-stealing kernel (4096 blocks, mbarrier cancel
   loop) ran correctly — every index executed exactly once (`ptxprobe/clc_test.cu`).
   Hardware work-stealing = dynamic persistent scheduling that kills tail-wave loss
   WITHOUT megakernel machinery and WITHOUT changing any tile's math (no FP reorder —
   the block-level answer to the stream-K reorder that failed our gates). CUTLASS ships
   CLC schedulers in its SM120 pingpong kernels.
2. **Thread-block clusters + DSMEM partially CONTRADICT the internal "broken" note.**
   ON-DEVICE: dynamic cluster sizes 2/4/8 launch and sync; `cluster.map_shared_rank()`
   cross-block smem reads return correct peer values; size 16 fails (portable max 8 —
   B200-only opt-in beyond). The vLLM #47164 "broken at 4/8/16" report does not reproduce
   for 4/8 on this rig/driver. UPDATE the sm120 capability ledger. Caveat: DSMEM
   bandwidth on GB203 is undocumented — probe before building.
3. **TMA multicast is a TRAP on sm_120.** ptxas 13.1 advisory verbatim: multicast "should
   be used on sm_90a/100a/... instead of sm_120a as this feature is expected to have
   substantially reduced performance"; CUTLASS Blackwell docs: "On GeForce ... no
   multicast feature, therefore the cluster shape is fixed to 1x1x1". L2 is the sharing
   fabric on this chip. (Two NVIDIA sources — the exact datacenter/consumer poison-pill
   this lane was told to hunt.)
4. **TMA otherwise fully available**: `cp.async.bulk[.tensor]` loads (incl. 4d im2col),
   TMA stores, `cp.reduce.async.bulk.tensor.add` (global reduction from smem — split-K
   relevance), and `cp.async.bulk.prefetch.tensor.2d.L2` all assemble for sm_120a.
5. **`setmaxnreg` IS available on sm_120a** (PTX ISA 9.3 target notes; assembles
   on-device): producer warps can shrink to ~24–40 regs, consumer mma.sync warps grow
   toward 255 — the missing half of the no-wgmma warp-specialization that the internal
   register-resident prefill-GEMM rewrite (#49, the priced crossing lever) needs. The
   2026-06-28 warp-spec attempt predates this insight (producers couldn't shed registers).
6. **L2 persistence window pays on the 64MB L2.** Device reports
   maxPersistingL2CacheSize=40MB, maxAccessPolicyWindowSize=128MB. ON-DEVICE single-run
   microbench (`ptxprobe/l2win.cu`): 32MB hot buffer re-read after a 2GB streaming
   polluter: 0.063→0.039ms (1.63x) with a persisting window. Distinct mechanism from the
   internally-refuted prefetch (adds no bandwidth) and evict-first-on-weights (−8%): the
   window PROTECTS re-read bytes from streaming eviction. Internal fit: KV at d6144 is
   ~57.5MB re-read every token across the 10 attn layers — a 40MB window covers ~70%.
7. **Eviction-policy loads**: `ld.global.L2::evict_last` exists on sm_120 but requires
   256B vectors (`.v8.b32`); `createpolicy.fractional` assembles. Weights-side evict-first
   already measured −8% internally; KV-side evict_last is the untested half — fold into
   the §6 experiment.
8. **CUTLASS 4.x SM120 catalog** (docs.nvidia.com/cutlass/4.6.0/CHANGELOG): TN-only,
   cluster fixed 1x1x1, pingpong + cooperative schedules, 128x32/64 tiles "+30% on
   SM121-related kernels" (4.5.x), **tileN=8,16 for SM120 blockscale GEMM** (4.6.x —
   NVIDIA itself concluded decode-shaped skinny-N block-scale tiles matter on consumer),
   ptr-array grouped FP8 GEMM for MoE (`MainloopSm120ArrayTmaWarpSpecialized`). Copy-
   references only (zero runtime dependency), incl. the "asymmetric DMA" kernel (different
   stage counts for A vs B). Warning: CUTLASS #3096 — SM120 NVFP4 grouped GEMM produced
   garbage on desktop Blackwell under compute_120; verify correctness against any CUTLASS
   comparison arm.
9. **XQA sm120 is the NVIDIA-authored decode-attention reference** (flashinfer csrc/xqa/,
   incl. mla_sm120.cu). There is no trtllm-gen "warpspec_sm120 FMHA" — that was SM100-only
   (corrects the July-13 sweep's §2c pointer). Read XQA for KV layout, cp.async
   pipelining, and warp mapping on this exact ISA — the direct reference for the
   gqa-packed hd512 open target.
10. **PDL advanced**: `cudaLaunchAttributeLaunchCompletionEvent` in 13.1 headers; PDL
    edges inside graphs via `cudaGraphDependencyTypeProgrammatic` ports;
    `griddepcontrol.*` assembles for sm_120a. Internal PDL waves already shipped; the
    graph-native PDL edges are the unused remainder (low ceiling per the +0.4% ledger).
11. **Green contexts** (CUDA 13.1 runtime API; CUTLASS ex 95): SM-partition co-location.
    openinfer measured prefill/decode co-location +60% on a 5070 Ti but flags the 5090
    power wall — on this 150–175W laptop, worse; decode is DRAM-bound and co-located
    compute steals bandwidth. Watch-list only for single-user.
12. **CompileIQ/ACF**: ptxas 13.1 accepts `--apply-controls` on-device; internal verdict
    stands (H100 kernels −2.0/−1.3%, no cross-TU transfer) — per-TU search only, low EV.
13. **sm_120f family target**: dense block-scale MMA is sm_120f-portable (only sparse MMA
    stays `a`-only; PTX ISA 8.8+, FlashInfer #3170) — build-insurance flag decision, zero
    perf now.
14. **cc 12.0 = 48 warps/SM max** (Blackwell tuning guide; consistent with the measured
    1536 threads/SM) — occupancy targets tuned against 64-warp intuition should re-check.
    GDDR7 read/write turnaround asymmetry: NOT documented anywhere official — if KV-append
    or epilogue stores underperform, that would be memra-original evidence.

## 4. The ranked lever list

Rank = (magnitude x breadth) / (size x risk), all magnitudes scaled to OUR baseline (the
source's own numbers stated separately). "Models touched" from the supported set.

| # | lever | mechanism | models touched | est. magnitude (ours) | size | exactness risk |
|---|---|---|---|---|---|---|
| 1 | **CLC work-stealing prefill scheduler** | hardware block-level work stealing (`clusterlaunchcontrol.try_cancel`) replaces static grids on prefill GEMMs — kills tail-wave/wave-quantization on 82 SMs; each tile's math unchanged | ALL (every prefill GEMM family: W4A8 MMQ, IQ4_XS dense MMQ, f16g sk visitor + tail, KAT trunk) | +8–15% pp on tail-wave shapes (internal wave-quant factor priced at 1.22x in the prefill-GEMM ledger; Colfax saw ~40% loss on one 1.36-wave shape) | S–M: ~30 lines/kernel + mbarrier machinery, per family | LOW — no FP reorder (whole-block steal; the legal version of what stream-K wasn't) |
| 2 | **W4A4 exactness rescue** (llama #25730 activation-quant recipe + ARCQuant residual channels, arXiv:2601.07475) | vendor llama's improved NVFP4 activation quantization (per-channel amax + hw cvt + scale-search) into the existing `MEMRA_MMQ` door; if p3 still forks, add ARC residual channels — a monotone exactness dial on stock block-scale mma — until argmax holds | q9, q27 (NVFP4 dense) | prefill 0.59–0.78x → ~1.0x llama class (+40–70% pp; our own 2026-07-08 measurement of the blocked path: 1.40–1.76x our default) | S–M: door + quantizer exist; vendor quantize path, optional residual-channel epilogue | HIGH (two prior p3 rejects) — but the experiment is cheap and the dial is a new mechanism; ships ONLY if the full battery greens |
| 3 | **Verify-tier flattening toward v≈1.05** | attack the b-tier premium that pins spec K at 3: ncu-name the b4 per-column premium (~78% of the T=4 pass), build b16-class exact twins for NVFP4/Q4_K/Q5_K (kills the T=5 cliff + K=8 falloff), rework mmvq_b8 (30–35% of peak BW) | q27, q9, o35b, KAT, gemma spec cells (the product headline) | +10–20% spec e2e at the re-opened K=4–5 optimum (bounded by the measured +40% v=1.05 counterfactual, verify-economics §3) | M–L: batched-MMVQ kernel family work; dual-twin precedent shows the bit-identity pattern | LOW–M — per-(token,row) chain identity by construction is the house pattern |
| 4 | **gqa-packed mma decode attention** (XQA sm120 as the NVIDIA-authored reference) | pack the gqa q-heads of one kv head as mma rows/columns in the decode/verify attention kernels — llama's hd512 trick (30.8 vs ~14.7µs/layer); same idea applies to the deep vec kernel's dp4a score phase (nkv=2, gqa=8) as the residual attack | gemma 12B/31B/E4B (hd512 lane), then KAT/q35/o35b depth class | gemma: closes the 0.97x depth cell (+2–3% those cells); depth class: part of the remaining ~2% vec-kernel headroom | M: one kernel family per geometry | M — new numeric config (score/accumulate order moves); full battery per model |
| 5 | **Draft-driven expert prefetch for the spill tier** (MoE-SpeQ, arXiv:2511.14102) | the spec chain's drafted tokens ARE a future-expert oracle: route drafted t+1..t+K on the host copy of the router, prefetch those experts' NVMe/RAM reads behind token t's compute; graft onto the existing `start_moe_prefetch_predictor` worker + `MEMRA_MOE_PREFETCH` machinery | Hy3 Layer103.5, MiniMax-M3 (spill regime) | +30–60% on Hy3 serve (paper: 2.34x on offloaded MoE; our stated target 5.13→10 needs a mechanism exactly like this) | M: host routing on drafted ids + prefetch enqueue; no new kernels | ZERO — prefetch never changes routing/bytes served |
| 6 | **L2 persistence window for KV at depth** (rider — attach to any depth lane) | `cudaAccessPolicyWindow` carve-out (40MB persisting max, measured 1.63x hot re-read microbench post-polluter) pins the KV region streaming weights otherwise evict; KV at d6144 = ~57.5MB re-read every token | KAT/q35/o35b/gemma at depth | +1–3% depth decode cells (attention = ~5% of token wall at d6144, ~2x KV-read speedup on the covered 70%) | XS: stream attribute + phase reset, zero kernel change | ZERO — caching hint only |
| 7 | **Graft-style retrieval splice into rejected chain slots** (arXiv:2605.20104) | keep the own-trim MTP chain; fill pruned/low-confidence draft slots with suffix/n-gram retrieval candidates from the session text — verify width unchanged, acceptance rises on repetitive/agentic content | all spec cells; targets p2/p3 (acceptance 52–58%, the content-class gap) | +5–15% spec e2e on p2/p3 classes (paper: +21.8% over EAGLE-3 — their baseline lacks our trim+adaptive stack, so scale down) | M: host-side Rust (suffix index + slot fill), no kernels | ZERO — verify arbitrates every token |
| 8 | **setmaxnreg + TMA register-resident prefill GEMM rewrite** (the #49 crossing lever, newly tractable) | producer warps shed to ~24–40 regs via `setmaxnreg`, consumers grow toward llama's 255-reg profile; TMA bulk loads free address-gen warps; CUTLASS sm120 pingpong/cooperative + asymmetric-DMA kernels as copy references | q9/q27 (W4A8 MMQ), o35b/q35/KAT (sk visitor forms) | the internal ledger's own pricing: llama's structure = 40% SM vs our 22–27% → +50–80% prefill GEMM if fully landed | L (multi-lane; the known hard core) | LOW–M — same math, new schedule; per-tile FP order must be pinned |

**Recommended queue = #1 → #5** (with #6 riding whichever depth lane runs first). #2 runs
early because its de-risking experiment is a day-class measurement on an existing door;
its SHIP decision remains gated on the full exactness battery — a third p3 reject parks it
again without appeal. #8 is the biggest number on the board but multi-lane sized; it
should start only after #1 lands (CLC composes with it and de-risks the grid design).

Measure-first experiments (one line each):
1. CLC-wrap ONE kernel (the sk 32x64 tail or W4A8 MMQ), pp512/pp2048 interleaved A/B + bit gate vs the static grid.
2. Vendor llama's #25730 quantize path behind `MEMRA_MMQ=1`, rerun the exact 2026-07-08 p3 reject battery (q27 p3 argmax + q9 p3 greedy tail); on fail, +ARC channels sweep {1,2,4} and re-gate.
3. ncu one b4 shape (`attn_qkv _b4_rpr2`) vs 4x its m=1 twin — name the per-column premium mechanism before building anything.
4. Read XQA sm120 source; prototype the hd512 gqa-packed kernel against tb512 on the g12 d1736 cell, N=5 interleaved.
5. Offline: replay a Hy3 `MEMRA_MOE_TRACE` + drafted-token routing to measure expert-set hit rate of draft-predicted vs actual (>70% ⇒ build).
6. One flag-guarded run of q35/KAT d6144 with a 30MB KV window vs naked, N=5 interleaved.
7. Offline: suffix-index the p3 agentic session text, measure would-have-accepted fill rate on the recorded reject slots.
8. Micro-probe `setmaxnreg` + TMA producer loop on one 128-row tile vs the current cp.async ring (harness, not e2e).

## Appendix A — already absorbed (honesty clause): 18 items

Surveyed items memra already ships (or already measured and reverted — the JSONL is the
record), NOT padded into §4:

1. PDL on decode chains (llama #24087/#25185, FlashInfer #3756 — memra: three PDL waves default-on; vLLM lists PDL as future work).
2. CUDA-graph decode with exec-update + segment recapture (universal; memra's dc/graph doors are measured per-model).
3. MTP/EAGLE-class trimmed draft heads + adaptive draft length + confidence cuts (DSpark scheduler class, exl3 dynamic draft window, ATLAS adaptive speculators ≈ own-gen trim + adaptive-K floor + in-round p-min; the DSpark window arm itself was built and deleted FLAT 2026-07-30).
4. FR-Spec vocab-trimmed drafters (llama.cpp just landed it upstream — commit on our own fork HEAD).
5. Device-side sampling/argmax (llama backend-sampling series; memra: device argmax + seeded gumbel serving rows).
6. Quantized KV + fused FA incl. online quantize-on-write (exl3 v1.0 ≈ our q8_0/q5_1 fused appends).
7. Flash-decoding split-KV with tuned ladder + stream-k fixup class (llama #20586/#21159; memra: fa_split_keys ladder, re-swept 2026-08-02).
8. int8-MMA MMQ prefill + stream-K + fastdiv tile decomposition (llama #22298/#24127).
9. Grouped/batched expert GEMM MoE prefill with direct-from-quant loaders (llama OpenCL ragged tiles #25433; memra's sk visitor + direct tiles are ahead on this rig).
10. SLRU/tiered expert cache with VRAM→RAM→NVMe spill (llama has only feature request #20757 — memra ahead; vLLM offload v2 is the datacenter analog).
11. Resident-if-fits expert residency (no external equivalent surfaced; internal 2026-08-02 default).
12. fp8 KV (XQA fp8 MLA exists on sm120; field report calls it a dead end — matches our measured revert).
13. W4A4 FP4 MMA prefill existence (ours built 2026-07; blocked on exactness — the UNBLOCK recipe is lever #2).
14. Cross-request prefix caching with cache_salt namespacing (vLLM-style; ours shipped 2026-08-02).
15. Batched z-dim attention + KV append for serving (vLLM/SGLang batched runners; ours = seqs kernels + pointer tables).
16. MoE gate/up activation-quant sharing (llama #25441 ≈ our fused dual/act-quant epilogue arms).
17. Zero-sync spec loop machinery (vLLM MRV2/SGLang overlap; ours = MEMRA_SPEC_DEVACC token-identical, stage-c burst measured negative on this rig).
18. Suffix/n-gram draft sources as a concept (vLLM suffix decoding — internal watch item; subsumed into lever #7's splice form).

## Appendix B — refuted/blocked internally (do not re-propose without new evidence)

- W4A4 e2m1 activations as-is: p3 argmax fork, reproduced 2026-07-07 + 2026-07-08 (lever #2 is the gated rescue, not a re-proposal).
- L2 prefetch during saturated phases: WASH at 92%+ DRAM; evict-first on weights −8% (2026-07-10 ledger). The §4 #6 window is a different mechanism (protection, not prefetch).
- Exact-MIPS head pruning: skip rate 0.0% (bounds 46.8x looser than logits).
- Lossless KV entropy recode: 94–99% of stored entropy — nothing to win (SplitZip-class ideas re-confirm the ceiling: ~1.1–1.3x on BF16, less on already-quantized KV).
- fp8 K-cache: spec self-consistency FAIL (74%→20.5% acceptance); fp8 KV e2e flat-to-negative, reverted 2026-07-28.
- Megakernels: 2026-07-13 verdict stands; FlashFormer/Lucebox numbers don't change the "constituent GEMVs already ≥ SOTA" analysis (Lucebox's 2x-vs-llama claim is self-reported, unverified).
- Whole-round spec graphs at fixed shapes: −10/−11%; enqueue-ahead law (4x confirmed).
- DFlash block-diffusion drafter ON THIS RIG: 0.62x of MTP at depth, mapped ceiling 0.72x (capstone 2026-07-13) — rig regime, not a bug; re-open only if a <5ms-forward drafter for a supported target appears.
- Adaptive-K EMA + accepted+1 on qwen: flat-to-negative (2026-08-01).
- DSpark marginal-rate verify window: FLAT, arm deleted (2026-07-30).
- FA4 reversed-x causal swizzle: FLAT (grids already wave 82 SMs).
- ACF/CompileIQ: −2.0/−1.3% on tuned H100 kernels, no cross-TU transfer — per-TU only, low EV.
- Stream-K k-split on exact paths: FP-order lesson #3 (CLC is the block-granularity replacement).
- SageAttention3/FP4 attention + sparse verify attention (SpecPV/Dustin class): lossy — outside the exact verify lane by law.
- Thread-block clusters "broken on SM120": SUPERSEDED — sizes 2–8 + DSMEM verified working on-device this lane (ptxprobe/); TMA multicast remains a trap (ptxas advisory + CUTLASS GeForce note). Update the capability ledger; DSMEM bandwidth still unmeasured.

## Competitive re-pin notes (not levers)

- llama.cpp b10229 carries DFlash + DSpark + FR-Spec trim + NVFP4 activation-quant fix +
  backend sampling (~7% e2e claim on 2x5090): the llama spec-best arm must be re-swept at
  the next board pairing (standing rule).
- vLLM v0.26.0 / SGLang v0.5.16 / FlashInfer v0.6.16 / exllamav3 v1.3.0 / TRT-LLM v1.2.1
  pinned above for the next H100 or fleet head-to-head.
