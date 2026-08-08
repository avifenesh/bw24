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

