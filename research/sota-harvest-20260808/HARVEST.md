# SOTA harvest — 2026-08-08 (aimed at the NEXT floor-raise)

Owner doctrine (2026-08-08, verbatim): *"we need to keep bring the sota home and create sota at
home."* The trimmed spec head was the owner's own research — proof the create-at-home lane pays.
This harvest is aimed past today's numbers, ranked by projected floor-raise per engineering-week.

Method: three parallel web sweeps (spec-decoding papers; attention/KV papers; engine capability
survey) + the local receipt base below as the diff base for every projection. Prior sweeps
consulted so nothing is re-reported at unchanged priority: `research/upstream-sweeps.md`
(2026-08-05 and 2026-08-07 sections), `research/SOTA-SWEEP-2026-07-13.md`. Everything cited
below is dated; projections are stated against OUR receipts, never against vendor baselines.

## The diff base (today's receipts, all same-repo)

| surface | number | receipt |
|---|---|---|
| Step-3.7-Flash PP-2 prefill, grouped+pipelined | 497.5 / 639.2 / **697.6** tok/s at pp512/2048/4096 (N=5) | `research/leverC-20260808/PROGRESS.md` (merged 9a264c76) |
| Step-3.7 PP-2 4k streaming TTFT | **11.009 s** p50 (serial 17.9, Lever-B 15.5) | `research/pipeprime-20260808/PROGRESS.md` Inc 14 |
| Step-3.7 PP-2 batched decode | 81→**130** agg tok/s c=1→8 (3.8x vs 34-flat), both boxes | `research/step35-batch-20260808/PROGRESS.md` |
| spec placement policy | PP-2 spec OFF everywhere (0.19–0.50x); single-card spec ON c≤2 (1.67x c=1, 1.08x c=2, 0.61x c=4) | `research/specplace-20260808/PROGRESS.md` |
| spec concurrency scaling | spec path SERIALIZES sessions: flat 346.5→345.2 agg c=1→8 single card | `research/pp2-spec-20260806/RESULTS.md` §2 |
| segmentation exactness | chunk/tick/LCP-split/off-grid-resume all bit-exact; residual = extent-classed prefix entries | `research/tick-seg-20260807/PROGRESS.md` |
| prefix-cache metering | hit-ratio + LCP histogram + per-tenant on /metrics, live | `research/cache-meter-20260807/PROGRESS.md` |
| darklane valley machinery | idle detect + SIGSTOP yield 19.4 ms median (5090), 37.2 ms (PP-2 box) | `research/darktrain-20260807/PROGRESS.md` |
| serve isolation gap | staggered-depth batches NOT bit-identical (ladder-rung straddle class), open | `research/iso-gap-20260807/PROGRESS.md` |
| KV quant | q8_0/q5_1 defaults; fp8-K FLIP-BLOCKED (acceptance 74%→20.5%) | `docs/FLAGS.md` MEMRA_KV_K |
| MoE decode dispatch | m=1 expert launch pairs below B=8 (`b1_stage_fast` eager chain); leverC grouped only PREFILL | `research/pp-prefill-20260807/PROGRESS.md` (28% share), leverC scope note |
| acceptance is prompt-shaped | short-ctx sampled acceptance 0.55 vs 0.73 | upstream-sweeps 08-05 (dogfood receipt) |

(Sections below are filled per-basket; the ranked top-8 closes the file.)

## Basket 1 — BRING HOME (engines)

### Prior-sweep delta first (per the do-not-re-report rule)

Status changes against `research/upstream-sweeps.md` (08-05/08-07 shortlists) with THIS
week's numbers:

- **Chunk-boundary invariance (vLLM #38561/#40372/#51113)** — ADOPTED AND EXCEEDED. The
  tick-seg + chunkfix lanes shipped bit-exact segmentation invariance incl. off-grid
  resume, with canary teeth. Drop from the queue; the residual is the extent-class item
  (own lane, basket 3 note under 3.d).
- **PP relay-starvation laws (TRT-LLM #16170)** — RESOLVED here: #87 closed 2026-08-08
  (reverse-publication fence). The checklist entry stands as PP-2 regression armor only.
- **TMA OOB odd-M (FlashInfer #4210)** — verdicted N/A 2026-08-05 (memra never stages;
  `research/tma-oddm-20260805/VERDICT.md`). Stays closed.
- **W4A8 (llama.cpp #24364/#26675)** — actioned better than the ask: the w4a8-prefill
  lane found the mxf8f6f4 one-line PTX form swap (1.2153x pp512 q27) for one day of work
  (`research/w4a8-prefill-20260806/VERDICT.md`). Track #26675 (ggml_prec contract) as
  FP8-ST convergence signal only.
- **Spec acceptance telemetry (llama.cpp #26389)** — SHIPPED (per-position counters +
  accept-gate battery). The follow-on is now the K-policy that CONSUMES it (basket 3.3).
- **PRIORITY RAISED: vLLM #48341 (async scheduling default-on for draft-model spec) and
  the DSpark/EAGLE-3.1 serving receipts** — the specplace verdict (PP-2 spec OFF
  everywhere; single-card flat-in-c) makes "spec that survives batching" the largest
  known gap vs upstream. Upstream holds 1.71x at c=4; we hold 0.61x. This is the week's
  headline diff and it feeds baskets 2A.1/2A.4 and 3.1.

### Engine mechanism diffs (four axes) — see engines survey

(Filled after the engines sweep lands.)

## Basket 2 — BRING HOME (papers, last ~6 months)

### 2A. Speculative decoding beyond fixed-K MTP

The literature converged on our two measured failure modes: fixed K dies under batch
(our c=4 0.61x), and serial draft chains die under pipelining (our PP-2 0.19–0.50x).
Items ordered by projected floor-raise on our cells.

**2A.1 Dynamic per-request verify budget — DSpark class.**
DSpark (DeepSeek, arXiv 2607.05147, 2026-07-06; SGLang integration
lmsys.org/blog/2026-07-06-dspark-sglang): a confidence head scores each drafted token's
survival; a scheduler sizes each request's verify window per step from an offline-profiled
cost model `T(bs,K)`, argmaxing expected accepted tokens per real marginal cost. Deployed in
DeepSeek-V4 production: +60–85% per-user speed at matched throughput vs MTP-1; the win is
primarily a HIGH-BATCH effect (ties at bs=1, pulls ahead as throughput plateaus). Mixed
traffic: windows contract with difficulty (5.24/3.78/2.91 tokens on gsm8k/arena-hard/poetry).
The full semi-autoregressive drafter is a training project, but the SCHEDULER is separable —
and SGLang's engineering answer to fixed-shape CUDA graphs is directly reusable: front-pack
the ragged per-request verify into one buffer, key the graph on TOTAL token count rounded to
captured tiers, so a trimmed batch replays a genuinely smaller graph.
*Projected on our numbers:* the single-card c=2 cell (1.08x, nearly wash) and the c=4 cell
(0.61x, gated off) are exactly the "verify budget exceeds what the batch can afford" regime;
K-shrink-under-load reopens them without the batched-drafting lane. Composes with 3.3
(the policy) and 3.1 (the batched rounds). *Cost:* days for the policy skeleton on our
existing telemetry; weeks if the tiered ragged-verify graph capture is included.
*Lane:* codex for policy, fable for the ragged-graph capture design.

**2A.2 Full-information online K selection — "Not-a-Bandit" (arXiv 2510.20064 v2, ICLR
2026).** The verify pass's target logits score ALL counterfactual K choices for free —
no extra target queries, provably no-regret online selection. This is the theoretical
backbone for basket 3.3 and costs almost nothing to adopt. SpecDec++ (arXiv 2405.19715,
COLM 2025: threshold-on-predicted-rejection is the optimal stop rule, 7.2–11.1% over fixed
K) and SVIP (EMNLP 2025, arXiv 2411.18462: draft-entropy early exit, training-free, +17-22%
over fixed lengths) are the same family; SVIP's entropy cutoff is the cheapest first arm.
*Cost:* days. *Lane:* codex (folded into 3.3).

**2A.3 Suffix/ngram draft source — training-free, batch-safe, CPU-side.**
Suffix Decoding (Snowflake, arXiv 2411.04975; vLLM `method: suffix`, merged via #25784):
suffix-tree over prompt + prior generations drafts without any GPU drafter pass — 1.8–4.5x
on SWE-Bench agentic subtasks, ~1.45x on repetitive coding. vLLM's own guidance: ngram/
suffix are the batch-safe option because drafting adds zero GPU load. SSSD (arXiv
2411.05894, ACL 2026) adds the roofline formula for how much speculation a batch can afford
(`s = I_knee / batch_size`) — the first-principles derivation of the K(c) curve we measured
empirically. *Projected on our numbers:* composes with MTP as a zero-cost draft source; on
the owner's dogfood traffic (agentic, repetitive tool output) this is the highest ROI per
line of code in the spec family — the drafts feed the EXISTING verify path unchanged, so
fixed-shape graphs are untouched. On PP-2 where the drafter forward is the poison, a
CPU-side draft source is the one spec form with no placement penalty — it may reopen PP-2
spec despite the specplace OFF verdict, because the verdict's mechanism (serial GPU draft
chain) does not apply. *Cost:* ~1 week (suffix tree + draft-slot injection + gates).
*Lane:* codex worker — well-specified, isolated.

**2A.4 One-pass parallel drafter — P-EAGLE class (arXiv 2602.01469, 2026-02; vLLM since
v0.16.0).** Replaces K sequential drafter passes with ONE forward emitting all K tokens
(learnable MASK embedding for positions 2..K). Blog receipts (B200, GPT-OSS-20B): 1.55–1.69x
over EAGLE-3 at c=1, and — the part that matters for us — one drafter launch per round
batches trivially across requests. EAGLE-3.1 (vllm.ai/blog/2026-05-26-eagle-3-1) holds
1.71x at c=4 in vLLM serving with LINEAR chains (trees are dead for serving — Red Hat
receipt: tree verify cost dominates at load; vLLM ships chain-only). DFlash (arXiv
2602.06036, ICML 2026; SGLang Spec V2) is the higher-ceiling block-diffusion version.
*Projected on our numbers:* this is the drafter ARCHITECTURE that makes 3.1 cheap — a
one-pass drafter turns the batched draft step from B serial chains into one B×K forward.
Requires retraining the drafter head (TorchSpec exists; our make-trimmed-draft pipeline is
the in-house analog). The attention-drift finding (arXiv 2605.09992: drafter hidden-state
magnitude grows along the chain, acceptance decays — fix is post-norm feeding) is a free
architecture note for any drafter we train (feeds 3.5). *Cost:* weeks (training + bring-up).
*Lane:* fable for the training recipe, codex for bring-up.

**2A.5 Spec × pipeline parallelism.** SpecPipe (arXiv 2504.04104 v2): fill PP bubbles with
speculative tokens, 4.19–5.53x TBT vs standard PP at 8 stages; PipeSpec (arXiv 2505.01572,
ACL Findings 2025): async draft/verify across devices with rollback, >sequential for any
nonzero acceptance; Saguaro (arXiv 2603.03251, ICLR 2026): pre-draft round N+1 against
predicted verify outcomes, removing draft latency from the critical path (~30% over
optimized spec baselines). Evidence basket 3.2 builds on — see there for the 2-stage
adaptation. Negative worth keeping: component-aware self-spec collapses on sequential
hybrids like Qwen3.5 (α=0.038, arXiv 2605.01106) — do not try SSM-subgraph self-drafting
on the q-family. Self-spec generally caps at ~1.4–1.5x (KnapSpec, ICML 2026) — below our
existing c=1 drafter spec; fallback-only, no lane.

### 2B. Attention/prefill + KV (sm_120a-portable)

The FA3/FA4 hand-port thesis HOLDS and now has named, dated artifacts. Items ordered by
projected floor-raise per week on our receipts.

**2B.1 SGLang dynamic chunk sizing for pipelined prefill (LMSYS blog 2026-01-15,
lmsys.org/blog/2026-01-15-chunked-pipeline).** Their chunked-pipeline-parallelism work hit
our exact failure mode at P=2: fixed-size chunks mean per-chunk time GROWS with prefix
length (attention is quadratic in prefix), so stages misalign and the later stage bubbles.
Their fix is scheduler-only: model cumulative runtime as a quadratic in prefix length and
SHRINK successive chunk sizes so per-chunk time stays constant (`Runtime(L+ΔL) −
Runtime(L) = Runtime(chunk₀)`), smoothing 0.6–0.85, page-aligned. Receipts: DeepSeek-V3.1
H20 PP4 = 3.31x prefill vs TP8; TTFT −67.9% at 128K.
*Projected on our numbers:* pipeprime's auto policy is ≤8 EQUAL microchunks with a
128-token floor — the exact fixed-size shape their analysis says leaves fill/drain bubbles
that grow with T. Our own geometry sweep already shows the appetite (16 chunks beat 8 by
1.5% at pp4096 despite double the overhead — equal-size is fighting the skew). A
decreasing-chunk schedule attacks the same bubbles for free; on the 697.6 receipt even
5-10% is found money, and TTFT at long prompts is where the 11.0 s p50 lives. *Cost:*
days — pure host-side schedule in `prime_chunk_tokens()`, gated by the existing
ppsplit/tickinv batteries (chunk boundaries change ⇒ the tick-seg invariance gates are the
safety net, and they are ALREADY GREEN for arbitrary segmentation — the fix lane was built
for exactly this freedom). *Lane:* codex worker — the gate battery makes this mechanical.

**2B.2 FA4 conditional softmax rescaling (arXiv 2603.05451, 2026-03-05;
tridao.me/blog/2026/flash4).** The one FA4 piece that is pure algorithm: rescale the
online-softmax accumulator only when the new row-max exceeds the old by a threshold τ
(final normalization stays exact) — ~10x fewer correction ops, called out by Modal's
dissection (modal.com/blog/reverse-engineer-flash-attention-4, Sept 2025) as "a good, and
very portable, idea." Also portable from the paper: LPT tile scheduling for causal masks +
varlen batch sorting (paper states these are GPU-independent, used in FA3 too).
*Projected on our numbers:* prefill attention share on the Step SKU is modest (the anatomy
puts MoE+peer-read at ~50% of prime), so honest projection is the 2-5% class on prefill
cells — but it also applies to the decode FA twins where softmax correction is per-step
overhead. *Cost:* days per kernel family, exactness-gated (the correction is
numerics-changing ⇒ run the full argmax/self-consistency battery; if bit-identity is
required, this becomes a new numeric class and needs the chunk-invariance treatment).
*Lane:* codex, one kernel family at a time.

**2B.3 CUTLASS PR #3030 — the SM120 warp-specialization template
(github.com/NVIDIA/cutlass/pull/3030, 2026, Second Nature Computing).** FA2-forward on
SM120 with FA3 structure grafted on: TMA `cp.async.bulk` loads via a dedicated DMA warp,
mbarrier producer/consumer (`PipelineTmaAsync`), multi-stage KV double buffering, separate
K and V pipelines — all on `mma.sync.aligned.m16n8k16`. Validated on GB10 (SM121a). Their
FP8 variant documents the operand-layout tax honestly (0.88x vs BF16-TMA at small S).
Binding constraint documented: 97.0 of SM120's 99 KB SMEM — deep FA4-style pipelines do
NOT transplant. *Projected:* this is the reference design for any memra FA rewrite — read
before writing, per the July sweep's TRT-LLM warpspec_sm120 note (this PR is the fuller
artifact). Paired with SageAttention3 (NeurIPS 2025 Spotlight, arXiv 2505.11594: FP4
microscaling attention at 1038 TOPS on RTX 5090, ~5x vs FA — with the honest caveat that
FP4 attention needs per-model quality gates and some models need INT8/FP8 hybrid fallback),
it defines the sm_120a attention ceiling. *Cost:* weeks-class kernel program; only worth
opening when prefill attention becomes the wall (post-2B.1/leverC it may). *Lane:* fable.

**2B.4 NVFP4 KV cache — receipts on OUR exact rig profile (SGLang PR #21601, merged
2026-07-17; independent replication hikarioyama/sglang-nvfp4-kv-sm120, June 2026).**
E2M1 + per-16 E4M3 block scales + per-layer FP32 global scale. The replication is
literally our deployment shape: RTX PRO 6000 pair, TP=2, CUDA-graph decode, hybrid SWA-512,
198B MoE — 1.778x KV capacity vs FP8, ~4% decode cost, large-model generation
byte-identical to FP8. Two law-grade findings: (1) small models break (Qwen2.5-7B
incoherent under FP4 KV — massive-activation K channels; our own fp8-K flip-block, 74%→20.5%
acceptance, is the same disease one rung up); (2) the per-layer global scale MUST be
calibrated by an eager forward BEFORE graph capture or block scales underflow to garbage.
*Projected on our numbers:* KV capacity is the PP-2 concurrency ceiling on big-trunk SKUs
(the 192GB assessment killed Hy3 on 145KB/token); q5_1→NVFP4-V (or K8V4-style splits per
KVTuner, arXiv 2502.04420: keys need more bits than values) buys ~1.3-1.8x session
capacity where sessions are KV-capped. Quality-gated per model, spec-acceptance gate
mandatory (our fp8-K history says acceptance is the sensitive detector). *Cost:* ~2 weeks
(new KV block format in the FA twins + calibration pass + battery). *Lane:* fable for the
format decision, codex for the twins.

**2B.5 Layered prefill for MoE — the chunk tax we currently pay (arXiv 2510.08055,
Oct 2025).** Chunked prefill re-reads the ENTIRE expert bank per chunk; on MoE this
inflates memory traffic up to 39%. Their alternative interleaves prefill/decode across
LAYER groups instead of token chunks (TTFT −70% max vs chunked). *Why it matters here:*
pipeprime just moved us to MORE chunks (8 microchunks × every expert bank touched again),
and leverC's grouped prefill re-reads each expert's weights once per chunk per layer. The
per-chunk expert-reload term is now multiplied 8x — the two lanes' wins already net out
hugely positive, but this paper says there is a measurable tax to reclaim (weight-stream
counters would show it directly; our anatomy infra makes the verification a day). Full
layered-prefill is a scheduler rearchitecture — the cheap version is expert-weight-reuse
ACROSS microchunks within a stage (the pipeline already holds chunk N and N+1 in flight on
the same stage back to back). *Cost:* days to measure, weeks to act. *Lane:* codex for the
counter receipt, decision after.

**2B.6 POD-Attention — fused prefill+decode on the same SM (ASPLOS 2025, arXiv
2410.18038, FA2-based, no Hopper features).** One kernel runs compute-bound prefill and
bandwidth-bound decode CTAs concurrently with SM-aware scheduling: attention up to 75%
faster (mean 28%) vs separate kernels in hybrid batches, e2e up to 22%. Our tick scheduler
already forms hybrid prefill+decode ticks, so the batch shape exists. *Honest note:* this
is the biggest-effort kernel item in the basket and its win depends on how often our ticks
are genuinely hybrid (the metering exists to answer that). *Cost:* weeks. *Lane:* fable,
only after 2B.1/2B.5 receipts say the scheduler-level overlaps are exhausted.

**2B.7 KV eviction (TriAttention) and sub-4-bit KV — watch, don't build.** TriAttention
(preprint + TRT-LLM PR #16957 merged; ICML 2026 acceptance NOT verified) scores keys in
pre-RoPE space — clearly better than SnapKV-class (32.9 vs 20.0 at budget 2048) but still
below full attention; lossy, per-model calibrated. RateQuant (arXiv 2605.06675) shows
sub-4-bit uniform KV stays badly lossy even with optimal allocation (2.5-bit best case
PPL 14.9 vs FP ~8). Neither clears our exactness doctrine for defaults; both are
long-context research files. YOCO/CLA cross-layer sharing is a training-time choice —
model-selection intelligence only. *Cost:* zero now. *Lane:* none.



## Basket 3 — CREATE AT HOME (original research proposals)

The seeds were evaluated against the receipt base and this week's web sweeps. Verdicts first:

| seed | verdict |
|---|---|
| (a) placement-aware hybrid spec (drafter in pipeline valleys) | **KEEP** — now with independent literature confirmation (SpecPipe class), still nobody ships it at 2 stages |
| (b) cache-aware K policy | **KEEP, widened** — the stronger form is full-information counterfactual K (see below); cache-hit signal is one feature of it |
| (c) grouped-expert DECODE | **KEEP** — the decode analog of leverC; receipts already locate the cost |
| (d) extent-class always-windowed numeric class | **DEMOTE to engineering lane** — it is a known fix with a named mechanism (tick-seg residual), not open research; belongs as a measured default-flip lane, not this basket |
| (e) cross-request drafter batching | **KEEP — ranked first** — it is the measured root cause of both spec losses (flat-in-c AND PP-2), and the fix shape is the same one that took decode from 34-flat to 130 |
| (f) darklane drafter self-distillation (added) | **KEEP** — the trimmed-spec-head thesis extended: the serve box trains its own drafter in its own valleys |

### 3.1 (e) Cross-request drafter batching — batch the DRAFT chains like we batched decode

**What it is.** Today every live spec session runs its own serial draft-verify round; the
pp2-spec lane measured the consequence directly: aggregate spec throughput is FLAT in c
(346.5→345.2 tok/s, c=1→8, single card, arm A door-shut) while plain batched decode scales
3.9x over the same load. Spec at c=4 loses (0.61x) not because drafting is slow but because
rounds serialize across sessions. The proposal: one BATCHED draft step per tick — all live
spec sessions' draft position j runs as one B×1 drafter forward (the drafter is a tiny MTP
head; its batched matvec is the same b16-class kernel family the decode tier already gates),
then one batched verify at m=B×(K+1). The scheduler already forms decode batches; the spec
round becomes two batched calls instead of B serial rounds.

**Evidence base.** Our receipts: `research/pp2-spec-20260806/RESULTS.md` §2 (the flat-in-c
finding, named "worth its own lane"); `research/step35-batch-20260808/` (the identical fix
shape on plain decode: 34-flat → 130 agg). Literature converged on the same diagnosis this
year: "Batch Speculative Decoding Done Right" (arXiv 2510.22876 v3, 2026-02-15) shows every
existing batch-spec implementation violates output equivalence via ragged-tensor desync and
fixes it with same-length grouping (EQSPEC/EXSPEC, up to 3x at batch 8); vLLM's EAGLE-3.1
serving numbers hold 1.71x at c=4 (vllm.ai/blog/2026-05-26-eagle-3-1) — spec surviving c=4
is table stakes upstream, with linear chains, not trees.

**Projected floor-raise on our numbers.** Single-card q9: spec c=1 is 374.8 vs plain 224.5.
If batched rounds hold even 60% of the c=1 spec advantage at c=4, the c=4 cell moves from
377 (spec, flat) / 617 (plain) toward ~900 — the concurrency gate (LOW=2/HIGH=4) stops
being a concession and the single-card c=4/c=8 cells re-open. This is the largest single
projected win in this harvest because it multiplies an already-measured 1.67x by an
already-measured 3.8x scaling mechanism instead of buying a new one.

**Honest port cost.** Weeks-class (2-3): the ragged problem is real — per-session accepted
lengths diverge every round, so the batch must regroup or pad every round (2510.22876's
same-length grouping is the cheap shape); the iso-gap ladder-rung law applies in full (any
split/tier selection keyed on batch aggregates breaks per-session bit-identity — the gate
battery for this lane must include the staggered-depth arm from day one); the drafter graph
is per-session captured today (capture-retain keepers, `DraftGraphCtx`) and would need
B-bucketed capture like decode's graph buckets.

**Lane shape.** Fable lane for the mechanism + exactness contract (the ladder-rung/ragged
hazards need judgment); codex worker lanes for the bucketed-capture plumbing and the gate
battery once the contract is written.

### 3.2 (a) Placement-aware hybrid spec — the drafter rides the pipeline valleys

**What it is.** The specplace verdict is placement-aware spec OFF on PP-2 because the serial
draft chain forfeits batched, stage-split plain decode. But PP-2 decode has structural
bubbles: with one microbatch in flight, each stage idles roughly half of every step, and the
darktrain lane already built the valley machinery (idle detection from worker truth, 37.2 ms
yield on the PP-2 box). The hybrid: keep plain batched decode as the committed path, and run
DRAFT chains opportunistically on the non-head stage's idle windows — drafts that complete in
time convert the next step into a verify (accepted tokens ride free); drafts that don't are
dropped with zero cost to the committed path. This is darktrain's yield-first contract
applied INTRA-request at microsecond scale instead of inter-request at second scale.

**Evidence base.** Ours: specplace matrix (PP-2 spec 0.19–0.50x — the loss to beat);
pipeprime (stage-owned host walkers + per-stage streams exist and are soak-proven, 200/200);
darktrain (the yield contract). Literature: SpecPipe (arXiv 2504.04104 v2, 2025-08-29) fills
PP bubbles with speculative tokens — 4.19–5.53x TBT vs standard PP at 8 stages, 1.64–2.08x
vs vLLM multi-request; PipeSpec (arXiv 2505.01572, ACL Findings 2025) proves async
draft/verify across devices beats sequential for any nonzero acceptance. Nobody ships a
2-stage workstation version; the papers target 8-stage clusters.

**Projected floor-raise.** The PP-2 c=1 cell (223 plain vs 112 spec today): a bubble-hosted
drafter that achieves even half the single-card acceptance-driven gain would put c=1 in the
~300 class — and c=1 latency is the felt number for the interactive lane. At c≥4 the bubbles
shrink (batched decode fills them), so the policy naturally degrades to today's OFF — the
hybrid needs no new gate, only a "draft only in measured valleys" admission.

**Honest port cost.** Weeks-class (3-4, the most speculative item kept): needs a device-side
or cheap host-side valley signal at step granularity (the darktrain signal is seconds-scale);
the drafter must run on the non-head stage while its weights live on the head stage today
(drafter placement/replication is a real design decision, ~1-2 GB class for the trimmed
head); abandoning a late draft must not perturb the committed stream (stream isolation +
the #87 fence discipline). Prereq: 3.1's batched verify, or the wins cap at c=1.

**Lane shape.** Fable research lane end-to-end; this is a mechanism-invention lane with a
crisp kill criterion (if the step-scale valley signal costs more than the drafted tokens
earn at c=1, kill it — the receipts will say so in week one).

### 3.3 (b→widened) Full-information K policy, cache- and prompt-conditioned

**What it is.** Today K is fixed per config and spec admission is a binary gate. Two of our
receipts say the policy is leaving throughput on the floor: acceptance is prompt-shaped
(0.55 vs 0.73 short-ctx), and the cache-meter lane now publishes the prefix-cache hit signal
(LCP histogram) per request at zero new cost. The widened proposal: the verify pass's target
logits already score EVERY counterfactual K for free — after each round, compute "would
position j have been accepted?" for j beyond the chosen K from the logits already in hand,
and run per-session online K selection on the full-information estimates. Condition the
prior on the free request features: prefix-cache hit length (a hit means the prompt is
template-like — measured-higher acceptance), prompt length class, and lane (interactive vs
dark). K→0 IS the spec-off gate, so the binary admission gate becomes the degenerate case of
one continuous policy.

**Evidence base.** Ours: accept-gate lane (acceptance sign follows model × drafter × prompt —
the law is already written down, `research/accept-gate-20260806/DESIGN.md` §2); per-position
acceptance telemetry live on /metrics; cache-meter LCP histogram live. Literature:
"Not-a-Bandit" (arXiv 2510.20064 v2, ICLR 2026) — full-information online K/drafter
selection from verify logits, provably no-regret, no extra target queries; SpecDec++ (arXiv
2405.19715, COLM 2025) — the optimal stop rule is a threshold on predicted rejection
probability; DSpark (arXiv 2607.05147) productionizes per-request budgets from a calibrated
confidence + profiled cost table (+60–85% per-user at matched throughput, DeepSeek prod).

**Projected floor-raise.** Single-card c=2 today is 1.08x — nearly a wash because K tuned
for c=1 over-drafts under load. A K(c, cache-hit, len) policy is the difference between
"spec wins c≤2" and "spec never loses": the c=2 cell should recover toward its c=1 ratio on
cache-hit traffic, and the marketplace traffic shape (shared system prompts → high hit rate)
is exactly the favorable case. Low ceiling per cell (~5-15%) but it moves EVERY spec cell
and it is nearly free.

**Honest port cost.** Days-class (3-5): the counterfactual-acceptance readout is a small
change inside the verify epilogue (logits are resident); the policy itself is a table +
EMA per session; gating via the existing accept-gate battery (integer-exact at temp 0 —
the policy must be deterministic given the telemetry stream, which it is).

**Lane shape.** Codex worker lane — the mechanism is fully specified by the receipts + the
two papers; the gates already exist.

### 3.4 (c) Grouped-expert DECODE at small batch

**What it is.** leverC grouped the step35 prefill expert loop (+53–63%, the 697.6 receipt)
but scoped itself to prime. Decode still dispatches m=1 launch pairs per (token, layer)
below the graph tier — the pp-prefill anatomy measured that class at 28% of prime GPU time,
and at decode B=8 with top-k≈8 the same layer sees ~64 token-expert routings that today run
as ~64 serial m=1 pairs. The proposal: the leverC bucketing (group routed rows by expert,
run each expert's rows at m=m_e, scatter back in slot order) applied to the batched decode
walk at B∈{2..8}. Same q8 kernels, same slot-ordered FMA reduction — the exactness argument
transfers verbatim from leverC's oracle.

**Evidence base.** Ours: leverC PROGRESS (the mechanism + the bit-exact grouped oracle);
step35-batch (the batched decode walk the grouping would live in; its c=8 cell is 130 agg);
pp-prefill anatomy (the m=1 dispatch cost class). Upstream: every engine's fused-MoE decode
path does exactly this (FlashInfer/vLLM grouped MoE kernels) — for us the novelty is doing
it inside the sigmoid-router-legal host-routing family without touching the uniform-only
fused kernels (the CLAUDE.md contract).

**Projected floor-raise.** At B=8 the expert m_e averages B×topk/n_active_experts — small
(often 1-3), so the win is launch-count and weight-reread amortization, not GEMM shape:
honest projection is the 5-15% class on the c=4/c=8 decode cells (a repeat of leverC's
mechanism at 1/500th the m). Worth it because the c=8 cell IS the serving bill number.

**Honest port cost.** Days-class (4-7): the walker exists (step35_decode_batch_layers), the
bucketing code exists (leverC), the gate exists (decode-batch-gate --plen 520 bit-identity).
Risk: at m_e∈{1,2} the grouping overhead (gather/scatter) can eat the win — the lane needs
the same paired N=5 A/B leverC ran, with a pre-registered kill bar.

**Lane shape.** Codex worker lane with the leverC PROGRESS as the template.

### 3.5 (f, added) Darklane drafter self-distillation — the serve box trains its own drafter

**What it is.** The trimmed spec head proved create-at-home pays once, offline. The serve
stack now has all the pieces to make it CONTINUOUS: the darktrain runner executes
checkpointable background jobs in serve valleys with a proven yield contract; the accept
telemetry publishes per-position acceptance per served config; and every verify pass
produces (prompt, target-token, draft-token, accepted?) tuples — free, perfectly on-policy
training data for the drafter, from the exact traffic distribution the box serves. The
proposal: a standing MEMRA_BG_JOB that distills the drafter head against the logged verify
stream (LoRA-scale updates on the trimmed head), gated by the accept-gate battery before
any weight swap, with the swap itself using the runtime draft-weight-update seam vLLM just
shipped (#46725 — receipt that hot draft-swap is a solved interface upstream).

**Evidence base.** Ours: darktrain (runner + checkpoint/resume + VRAM budget, PP-2 receipt
7/7); accept-gate (the integer-exact acceptance assertion that makes a swap gateable);
owner doctrine (the trimmed head IS the precedent). Literature: online/continual drafter
adaptation exists as scattered papers (drafter staleness under distribution shift is the
known failure of trained drafters — EAGLE-3.1's attention-drift analysis, arXiv 2605.09992,
is adjacent), but no engine ships drafter self-distillation from its own verify stream on
the serving hardware. This is the most "create" item in the basket.

**Projected floor-raise.** Acceptance is the spec multiplier: the 0.55→0.73 short-ctx gap
is the measured headroom class. Closing half of a 0.15-0.20 acceptance gap on the traffic
the box actually serves is worth ~10-20% on every spec-win cell, compounding with 3.1/3.3.
Strategic value exceeds the number: it turns every serve-hour into drafter R&D — the
darklanes operating thesis made mechanical.

**Honest port cost.** Weeks-class (2-4 for v1): needs a training loop that fits the
VRAM-budget contract (the darktrain follow-up already names "first real GPU training job"
as the next consumer); the data logger (verify tuples → disk) is days; the gate discipline
is already built. The honest risk: LoRA-scale updates may not move acceptance enough —
week-one receipt is a one-shot offline distill on logged traffic before any online loop.

**Lane shape.** Fable lane for v1 (training-loop judgment + gate design), handing the
steady-state job to codex lanes once the recipe is frozen.

