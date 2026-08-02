# Fine-tune SKU assessment — Qwen3.6-35B-A3B agentic/coding post-train as a darklanes model

Date: 2026-08-02. Lane: `lane/finetune-sku`. No GPU used. Every external fact cited with
source + date; raw API receipts and ToS HTML snapshots in this directory. Three web-research
threads (quality bar, prune-and-distill, licensing/cost) were run 2026-08-02; their sources are
folded into the source index.

**Honesty clause / kill criteria, stated up front:**

1. If cold-start distribution has no credible path for an unknown provider, the verdict is KILL
   regardless of model quality.
2. If the best public evidence says trace-SFT cannot beat the vanilla backbone on honest
   cross-harness evals, the "differentiated model" premise dies and only variants with a
   different economic engine (pruning) survive.
3. If teacher-trace generation cannot ride the harvest lane at zero marginal cost, the cost
   section must say so and price the real spend.

All three criteria fired at least partially. Verdict at the end; short version: **the
plain-SFT differentiated-SKU idea is dead as a near-term revenue bet; a narrow GO-later
survives with concrete triggers, and the interesting variant is prune-first, not SFT-first.**

---

## Receipts in this directory

| File | What |
|---|---|
| `or-models-20260802.json` | OpenRouter `/api/v1/models` full catalog (337 models), fetched 2026-08-02 |
| `or-rankings-20260802.json` | OR frontend rankings API; full-day board 2026-08-01, N=440 slugs |
| `qwen36-35b-endpoints-20260802.json` | All 9 endpoints of `qwen/qwen3.6-35b-a3b` |
| `nex-n2-mini-endpoints-20260802.json` | Nex-N2-Mini endpoint (nearest precedent) |
| `kimi-k3-endpoints-20260802.json` | Kimi K3 provider pricing (teacher cost) |
| `ep-*.json` | Endpoints for community fine-tunes (sao10k, thedrummer, KAT, Arcee, Hermes) |
| `kimi-com-modeluse-agreement-20260802.html` | Kimi product ToS raw HTML (anti-distillation clause) |
| `alibaba-product-terms-4.48-modelstudio-20260802.html` | Alibaba Product Terms Art. 4.48 raw HTML |
| `alibaba-coding-plan-page-20260802.html` | Coding Plan docs page (interactive-only policy, quotas) |
| `dsv4-flash-endpoints-20260802.json` | All 21 OR endpoints of deepseek-v4-flash w/ pricing |
| `openrouter-tos-20260802.html` | OpenRouter ToS raw HTML (Last Updated 2026-07-27) |
| `deepseek-open-platform-tos-20260802.html` | DeepSeek platform ToS (distillation-permissive §4.2) |
| `deepinfra-tos-20260802.html` / `novita-tos-20260802.html` | Allowlisted host ToS |
| `baidu-intl-ai-terms-20260802.html` | Baidu AI terms (3A.5 non-compete -> DENY) |

---

## Q1 — Distribution (the decisive axis)

### The demand pool being targeted

Measured on the full ranking day 2026-08-01 (receipt: `or-rankings-20260802.json`):
`qwen/qwen3.6-35b-a3b` did **76.8B tokens / 8.9M requests in one day** (69.9B prompt / 6.9B
completion), served by **9 providers** (receipt: `qwen36-35b-endpoints-20260802.json`), price
floor DeepInfra fp8 at $0.10/M in, $0.95/M out. At floor prices the entire model's daily gross
pool is **~$13.6k/day, split across 9 providers** (own arithmetic from the receipts). The whole
OR network moved 57.0T tokens that day (top-440 board).

A fine-tune is a **new model listing**: none of that routing share is inherited. The question is
what new listings by non-lab authors actually capture.

### What community/non-lab models actually capture (full-day census, 2026-08-01)

Of 337 models in the OR catalog, 57 are from non-lab authors. Aggregate traffic of the classic
community fine-tune authors (thedrummer, sao10k, NousResearch, gryphe, undi95, anthracite-org,
cognitivecomputations, aion-labs, mancer): **28.8B tok/day across 19 slugs = 0.051% of the
network**. Task-specific startups (Morph, Relace): 0.34B/day = 0.001%. New-lab post-trainers
(Nex AGI, Kwaipilot, Arcee): 25.0B/day = 0.044%. Receipts: catalog + rankings JSON.

The precedent table — every non-lab model with meaningful traffic, with the engine that drove it:

| Model (created) | Backbone | Tok/day 08-01 | Est. gross/day at listed prices | What drove it |
|---|---|---:|---:|---|
| nex-agi/nex-n2-mini (2026-06-24) | **Qwen3.5-35B-A3B post-train** | 16.0B | ~$545 (at $0.025/$0.10) | Funded lab, 397B flagship sibling, HF/GitHub/ModelScope launch, 2-week free window on OR, vendor SOTA claims (buildmvpfast.com 2026-06-13; HF cards fetched 2026-08-02) |
| sao10k/l3-lunaris-8b (2024) | Llama-3 8B | 7.1B | ~$290 | Years of RP-community brand; 3 providers picked it up |
| nousresearch/hermes-4-70b (2025-08) | Llama-3.1 | 3.0B | ~$385 | Nous brand, multi-year |
| kwaipilot/kat-coder-air-v2.5 (2026-07-10) | **Qwen3.6-35B-A3B post-train** | 2.3B | ~$369 (StreamLake $0.15/$0.60) | Kuaishou-backed lab, honest cross-harness evals, corporate cloud serves it |
| thedrummer/cydonia-24b-v4.1 | Mistral | 1.3B | ~$401 | RP brand (TheDrummer), Parasail hosts |
| arcee-ai/trinity-large-thinking (2026-04) | — | 1.3B | ~$330 | Model company serving itself + Parasail |
| morph/morph-v3-large (2025-07) | proprietary | 0.26B | ~$354 | **Task-specific apply model wired directly into coding tools** |
| deepseek/deepseek-r1-distill-llama-70b (2025-01) | Llama-70B | 0.15B | negligible | The R1-distill brand moment — **decayed to noise 18 months later** |

Read of the evidence:

1. **There is no precedent of an inference host's own no-name fine-tune capturing meaningful
   traffic.** Every traffic-bearing community model has one of: a funded lab brand with a launch
   push (Nex, Kwaipilot), a multi-year community brand (sao10k/TheDrummer/Nous — all RP, a
   different market), or a direct tool-integration engine (Morph/Relace apply models). The
   author category "provider that fine-tuned a model" is empty at every traffic level.
2. **Even the successes are small money.** The two direct analogs on our exact backbone-class —
   Nex-N2-Mini and KAT-Coder-Air — gross ~$369-545/day *for the entire model across all
   providers*, at loss-leader prices, with real labs and real launches behind them. That is the
   realistic *ceiling* for a well-executed unknown-lab 35B agentic SKU, not the expectation.
3. **Brand moments decay.** R1-distills were the loudest fine-tune listing event ever;
   the 70B distill now does 0.15B tok/day.
4. Traffic on this model class is ~90% prompt tokens (agentic workloads), so revenue rides the
   *input* price — the cheap side. Nex's 16B tok/day is only $545 because 14.1B of it is prompt
   at $0.025/M.

### OR's process for listing a NEW model

From `openrouter.ai/docs/guides/community/for-providers` (re-fetched 2026-08-02) and the
provider-apply page (fetched 2026-08-02, receipts in `research/or-provider-20260802/`):

- Listing a new model is **not a separate review** once you are an approved provider: OR's
  provider monitor watches your `/v1/models`; a new model is auto-staged, baseline-tested, and
  auto-unhidden when tests pass and pricing is set (`is_ready` controls launch timing). The gate
  is **becoming a provider at all** — application-reviewed, explicit backlog.
- The backlog quote cuts both ways for us: *"We currently have a large backlog of provider
  applications and are prioritizing providers with **proprietary models**."* A darklanes
  fine-tune is a proprietary model — the SKU **flips our application from the deprioritized
  category (open-weight host) into the prioritized one**. This is the single strongest
  pro-GO distribution fact found, and it has value even if the SKU itself earns little.
- Single-provider models are barely touched by routing mechanics: Auto Exacto thresholds need
  4+ providers; price-weighted load balancing needs competitors. Discovery is the model page,
  the rankings board, and whatever demand the author brings. OR's own docs: "If there are
  models or providers you are interested in that OpenRouter doesn't have, please tell us in
  our Discord" — listing demand-side pull exists but is informal.

### Other channels

- **Tool integrations (Cline/Kilo/aider class):** Kilo Code supports custom models for any
  provider via base-URL config (kilo.ai/docs/code-with-ai/agents/custom-models, fetched
  2026-08-02); Cline's provider list crossed 30 with community vendor modules
  (docs.cline.bot/api/models, 2026-06-25). Cline now sells **ClinePass** ($4.99-9.99/mo
  bundling 10 open-weight models from 6 labs — github.com/diegosouzapw/OmniRoute issue #5518,
  2026-06-30) — a curated partnership surface an unknown SKU does not walk into. BYO-endpoint
  is open to any user but generates no discovery. Morph proves the integration route works —
  for *task-specific* models (apply/edit) where the tool needs the SKU, not for "a better
  general agent model".
- **HF trending:** Ornith-1.0-35B-GGUF did 3.45M downloads/30d (see
  `research/model-demand-20260801/REPORT.md`) — the local-download audience is real but
  converts to zero paid serving revenue for the author, and Ornith's eval claims collapsed
  under independent harnesses (Q2 below). HF distribution markets weights, not endpoints.
- **Direct API:** everything in `research/or-provider-20260802/` §4 applies; highest margin,
  slowest, needs the same trust signals plus sales effort. Not a cold-start channel.

### Distribution verdict

Not literally zero-path — the provider-application synergy is real and the Nex/KAT precedents
prove a new 35B agentic listing *can* get traffic. But the evidenced capture band for a
well-executed launch by a *funded lab* is **2-16B tok/day ~ $300-600/day gross**, the "provider's
own fine-tune" category has zero precedent, and we have neither a lab brand nor a tool
integration. Distribution alone does not KILL, but it caps the payoff at a level that the
quality-bar and cost evidence then has to justify — and it does not (Q2, Q4).

---

## Q2 — Quality bar reality check

(Research thread run 2026-08-02; all scores fetched from live model cards/blogs that day.)

**Vanilla Qwen3.6-35B-A3B is already a strong agentic model — and its published numbers are
harness-inflated.** Official card (huggingface.co/Qwen/Qwen3.6-35B-A3B): SWE-bench Verified
**73.4**, Terminal-Bench 2.0 51.5, TAU3-Bench 67.2, LiveCodeBench v6 80.4, MMLU-Pro 85.2. Under
Kwaipilot's unified Claude-Code-style harness the same vanilla model scores **64.4** SWE-V — a
~10-pt scaffold gap Kwaipilot attributes to Qwen's harness and test-set tuning
(huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev, fetched 2026-08-02).

**The best verified same-backbone post-train gained +5 SWE-V / +9 TB2.1 — and needed far more
than trace-SFT.** KAT-Coder-V2.5-Dev (built ON Qwen3.6-35B-A3B): 127K-example SFT **plus 10
epochs of RL** in verifiable repo environments with hierarchical rewards and backbone-specific
penalties. Same-harness: 69.4 vs 64.4 SWE-V, 41.0 vs 32.0 TB2.1. Their own data shows a chunk
of the win is behavior repair (abnormal tool-call rate 9.34%->0.28%) — i.e. format, not
capability (HF card + marktechpost.com 2026-07-26).

**The loud counter-example failed reproduction.** Ornith-1.0-35B (DeepReinforce, June 2026,
Qwen3.5-35B-A3B backbone) claims SWE-V 75.6 / TB2.1 64.2; under Kwaipilot's independent unified
harness it collapses to **55.8 / 36.0 — below vanilla Qwen3.6** (KAT card, fetched 2026-08-02;
community: "benchmaxxed... fails miserably on existing codebases", r/unsloth 2026-06-28). The
sharpest live demonstration that 35B "SOTA agentic" claims are frequently scaffold-fit.

**Format vs capability is quantified in public work.** SWE-Star's scaling study: every model
size jumps to ~40% SWE-V within ~800 trajectories (learning tools/scaffold), then gains go
"super-exponentially" expensive; one 100K-trajectory run ~ 4,500 H100-hours
(logicstar.ai/blog/swe-star, ~Jan 2026). Harness-only changes move TB2 pass@1 by 7-15 pts
(arXiv 2605.23950, May 2026); SWE-bench Pro shows 22-pt swings from scaffold alone. R1-distills
(the canonical near-SOTA-teacher->32B precedent) produced huge *reasoning* jumps but are
explicitly "not optimized for function calling" — reasoning distills != agentic distills.

**K3-teacher distills don't exist yet.** Kimi K3 verified: **2.8T total / 104B active**, 1M
context, near-frontier agentic scores (TB2.1 88.3, BrowseComp 91.2, GPQA-D 93.5), weights
released 2026-07-27 (huggingface.co/moonshotai/Kimi-K3, fetched 2026-08-02) — roughly
Opus-class on agentic boards, so the distillation *ceiling* is genuinely high. But no
K3-distilled student with published scores exists (weights 6 days old); public K3 trace
corpora are 10^2-10^3 trajectories (e.g. greghavens 582 trajectories, CC-BY-4.0) — seed
material, 2-3 orders of magnitude short of a training corpus. We would be first, carrying
first-mover risk, not following a proven recipe.

**Honest eval battery to claim "better for agentic/coding than vanilla":** SWE-bench Verified
on TWO harnesses (ours + Claude Code/OpenHands class — otherwise the number is unfalsifiable
per the Ornith case), Terminal-Bench 2.1 avg-of-5 (what Qwen and Ornith both ran), TAU3 or
tau2-bench (needs a paid user-sim model), BFCL v4, LiveCodeBench v6, plus regression
(MMLU-Pro, GPQA, AIME) where any >1-2pp drop is a visible tax. Plus — from the prune research —
a large-n loop/termination-rate check (n>=2000 agentic prompts). Budget: low four figures USD
per candidate checkpoint, dominated by SWE-V rollouts and TB sandbox hours.

**Quality verdict: pure trace-SFT cannot honestly differentiate this backbone.** Vanilla starts
at 64-73 SWE-V — already past the point where public SFT curves flatten. The only verified path
to a real gap (+5/+9) required 127K curated examples plus sandboxed RL infrastructure. A
K3-teacher corpus raises the ceiling but is unproven, and the claim would have to survive the
exact cross-harness test that killed Ornith.

---

## Q3 — Prune-and-distill variant (the economically interesting one)

(Research thread 2026-08-02; in-repo methodology: `research/per-expert-quant/`,
`tools/recover_hy3_reap_mask.py`. Note the in-repo Hy3 evidence itself: the owner ruled
pruning OFF for Hy3 after it "showed bad results" — memory `hy3-plain-arm-fused-tiering`,
2026-07-26. Qwen3.6 evidence below is more favorable but the caution transfers.)

**Public prune results on this exact model exist.** Qwen3.6-35B-A3B = 256 experts/layer, top-8
+ 1 shared, 40 layers; expert bank ~ **93% of weights** (config.json fetched 2026-08-02;
matches crucible-labs' 34.66B->19.17B at 48% prune). Cerebras REAP (arXiv 2510.13999, ICLR
2026): on Qwen3-30B-A3B, 25% prune ~ code-lossless, 50% ~ 96% retention; on Qwen3-Coder-480B,
25% lossless (SWE-V 54.0->54.0), 50% ~ -2pp. Community checkpoints on Qwen3.6-35B-A3B itself:
REAP-20/25/30/48/50 all published; measured: 20% prune costs MMLU -2.6pp and wikitext ppl
7.85->10.06 (barozp, lm-eval); 48% prune holds HumanEval/BFCL but drops BigCodeBench -3.3pp
(crucible-labs; both HF cards fetched 2026-08-02). Qwen3.6's shared expert absorbs ~69% of
layer output norm, cushioning pruning. Critical REAP caveat: calibration data dominates at
50% — generic-C4 calibration collapses coding to near-zero vs the agentic/code mix (paper
Table A6), so any plan must calibrate on agentic traces.

**Recovery-by-SFT is precedented on this exact model at mild fractions.** barozp's REAP-20% +
reasoning LoRA recovered MMLU +1.05pp and pushed ARC-C *above* the unpruned base. Router-KD
(arXiv 2603.02217) recovers REAP-pruned Qwen3-30B with ~2 GPU-hours. **The hard ceiling is at
higher fractions:** 0xSero's GLM-5.2 REAP-34% + router-KD reached Aider-parity but its
loop/non-termination rate **doubled at n=2000** (7.2% vs 3.6%), 6x more KD data failed to close
it — expert knowledge is genuinely gone; small evals hide it. Full recovery at >=34% has only
been shown with pretraining-scale budgets (SlimQwen: 120-400B tokens; Minitron: 380B).

**Serving-cost delta — honest math.** REAP does NOT reduce active params (top-8 preserved over
the shrunken bank, router renormalized): **batch-1 tok/s is unchanged** when both fit in VRAM
(219 vs ~240 tok/s on the same GPU class, single-sided community numbers; no controlled A/B
published anywhere — a measurement gap we could own). Real wins: footprint (fp8: 35GB->27GB at
25%, ->19GB at 50%), KV/batch headroom, and the batched regime where each step touches most of
the bank — halving the bank roughly halves expert weight traffic at high concurrency, and
halves any spill working set. **Honest caveat for OUR envelope:** vanilla 35B-A3B already fits
the 2-4x5090 SKU rule (fp8 in 2 cards with KV room; Q4 in one), so pruning here buys
concurrency/KV headroom (more paying requests per box), not existence. The prune variant's
economics are real but incremental — unlike Hy3, where fit was the issue.

**Nobody serves a pruned MoE commercially.** Cerebras explicitly: prod API models "are not
pruned" (r/LocalLLaMA, 2025-10-24). Zero REAP/pruned listings in the OR catalog (fetched
2026-08-02). Being first is a differentiation story AND an unpriced trust risk (quant-label
policing by the community is aggressive; a pruned endpoint under a vanilla slug would be a
scandal — it would have to list as its own SKU, which returns us to Q1's cold-start problem).

**Feasibility verdict: REAP-25 + agentic-SFT-heal on Qwen3.6-35B-A3B is technically
well-precedented** (public checkpoints on the exact model, recovery demonstrated, our own
tooling transfers), cheap to try (~$2.5k clean-teacher LoRA path, Q4), and produces serving
value on owned boxes even if never listed publicly (more concurrent requests per box). 50%
is not honest territory without large-n agentic evals and real SFT budget.

---

## Q4 — Cost + licensing

### Licensing — teachers

- **Kimi K3 weights license** (huggingface.co/moonshotai/Kimi-K3 LICENSE, fetched 2026-08-02):
  MIT-style grant incl. fine-tune/derivatives. No explicit restriction on training on outputs
  (outputs treated as distinct from "Software"). Two triggers, both far above our scale:
  MaaS clause — separate agreement required if MaaS revenue >$20M over 12 months; attribution
  ("Kimi K3" display) only at >100M MAU or >$20M/month. A small provider serving a K3-distilled
  Qwen student is clear on the plain reading; the "outputs aren't derivatives" reading is
  untested ambiguity to diligence before betting a product line on it.
- **DeepSeek V4: plain MIT** (HF LICENSE fetched 2026-08-02) — zero distillation/attribution/
  revenue clauses. **Cleanest teacher, and ~15x cheaper (below).**
- **Student base Qwen3.6-35B-A3B: Apache-2.0** (HF repo, fetched 2026-08-02). Unencumbered.
- **Provider ToS for API-bought traces:** OpenRouter ToS (2026-07-27) flows model terms down,
  claims nothing itself; Fireworks (2026-07-10) and Together (2026-05-19) disclaim training on
  or claims over customer outputs. The model license is the binding constraint.

### The recommended pilot path: opencode -> OpenRouter -> pinned DeepSeek V4-Flash (verified 2026-08-02)

Owner's proposed setup, links verified with exact quotes; raw HTML receipts in this dir
(`openrouter-tos-20260802.html`, `deepseek-open-platform-tos-20260802.html`,
`deepinfra-tos-20260802.html`, `novita-tos-20260802.html`, `baidu-intl-ai-terms-20260802.html`).
opencode is local OSS software (no service link); the V4-Flash model license is MIT-class; the
two service links check out as follows.

**Link 1 — OpenRouter ToS (openrouter.ai/terms, "Last Updated: July 27, 2026"): clean.**
Section 6.1: "You retain copyright and any other proprietary rights that you may hold in the
Input. Your ownership rights in the Output are set forth in the Model Terms for each Model you
use." OR takes only an operational license to User Content "solely in connection with operating
and providing the Service" (broader logging license is opt-in only, §6.2). No restriction on
using outputs for training anywhere in the document. Two obligations flow to us: §5.1 — "You
are solely responsible for reviewing the Model Terms applicable to each Model" — and Model
Providers are third-party beneficiaries of §§5/6.1 (§20), i.e. the per-endpoint host terms are
enforceable against us. That makes the provider allowlist below load-bearing, not optional.

**Link 2 — per-provider terms for `deepseek/deepseek-v4-flash` (21 endpoints live, receipt
`dsv4-flash-endpoints-20260802.json`).** The four cheapest + first-party, checked:

| Provider | $/M in / out / cache-rd | Quant, ctx | Verdict (quote) |
|---|---|---|---|
| **DeepSeek first-party** | 0.140 / 0.280 / **0.0028** | unknown, 1M | **ALLOW — explicitly permissive.** Open Platform ToS §4.2 (cdn.deepseek.com, effective 2026-04-29): "We assign any rights, title, and interests—if any—in the Outputs of the Services to you; (3) You may apply the Inputs and Outputs of the Services to a wide range of use cases, including ... derivative product development, **training other models (such as model distillation)**, etc." The only first-party ToS found anywhere in this study that names distillation as a permitted use. Best cache-read price on the list (10x cheaper than anyone) and 99.996% uptime-30m. |
| **DeepInfra** | 0.090 / 0.180 / 0.018 | fp4, 1M | **ALLOW.** ToS: "inputs you provide to our API and outputs it generates are your private data. We will not store, sell or train using this data"; no output-use restriction. Generic non-compete covers competing *with DeepInfra's service*, not model training. Note quant=fp4 — acceptable for trace gen, worth knowing. |
| StreamLake (cheapest, 0.0868/0.1736) | fp8, 1M | — | **HOLD — unverifiable.** Kuaishou's cloud; its ToS pages (streamlake.ai/document/...) are JS-rendered and returned no text; no English terms located. Cheapest but unverified = excluded by the allowlist rule. |
| Baidu (0.0882/0.1764) | fp8, 1M | — | **DENY.** Baidu AI Cloud International agreement 3A.5: "You shall not ... directly or indirectly, use the AI service to develop or improve similar or competing products or services." Same clause family as the blocked subs. |
| **Fireworks** | 0.140 / 0.280 / 0.028 | unknown, 1M | **ALLOW** (per the licensing-thread verification of Fireworks ToS, 2026-07-10: customer owns outputs, no training restriction). |
| **Novita** | 0.140 / 0.280 / 0.028 | fp8, 1M | **ALLOW.** ToS (updated 2025-11-04) §9: "You retain copyright and any other proprietary rights that you may hold in the Input... we shall not log, store, or retain any User Content... or any Outputs"; no output-use restriction ("compete with us" clause covers their marketplace, not model training). |

**Pinned allowlist for OR request preferences** (`provider.only`, plus `allow_fallbacks:false`):
`["deepseek", "deepinfra", "fireworks", "novita"]` — order = preference. GMICloud (0.0938/0.1876,
fp8) had no reachable ToS text (JS-only site) — HOLD with StreamLake; both can be promoted later
if their terms verify. Everything else on the endpoint list stays off until read.

**Per-trace pilot cost** (assumptions: 25k-token final trajectory, ~30 turns, cumulative prompt
re-read ~8x transcript at 90% cache-hit, ~8k output tokens, 3x rejection-sampling overhead —
own arithmetic from the endpoint receipt): DeepSeek first-party **~$0.017 per kept trace**
(~$665 per 1B kept tokens — the cache-read price is the dominant term and DeepSeek's $0.0028/M
beats the field 10x); DeepInfra ~$0.019 (~$780/1B); Fireworks/Novita ~$0.030 (~$1,210/1B).
A 5K-trajectory SWE-smith-scale pilot ≈ **$85–150 total**; a 50K-trajectory KAT-scale corpus
≈ $850–1,500. At these prices the pilot is essentially free relative to eval costs — V4-Flash
prices came down so far that the earlier V4-Pro-based estimate (~$2k/1B) is now the *upper*
bound of the clean path, not the floor.

### Licensing — the owner's paid subs (both checked 2026-08-02, raw HTML receipts saved): BLOCKED

- **Kimi Code sub — prohibited.** Kimi Model-Use agreement
  (kimi.com/user/agreement/en/modelUse): "(i) you have no authority to use Kimi and the content
  generated by Kimi in any commercial manner; (ii) you may not use our Services to develop
  products or services that compete with us" and, in the prohibited-uses list: "By using Kimi
  to develop, train, or improve algorithms, models, etc., that are in direct or indirect
  competition with us". Also bans removing the deep-synthesis mark (trace cleaning trips it).
  This is the SERVICE ToS — separate from and stricter than the open-weights license.
- **Qwen coding plan — prohibited twice.** (1) Alibaba Cloud Product Terms Art. 4.48(d)(v)
  (Model Studio): you shall not "use Model Studio, AI models provided through Model Studio
  (including any Output of such AI models) to train or develop products or services that
  compete with Alibaba Cloud and/or its affiliates' products and services, unless expressly
  authorised". Use-agnostic; blankets all models served through the plan (incl. DeepSeek/GLM/
  Kimi/MiniMax) — no per-model carve-out, and training even a *Qwen* student for a commercial
  inference SKU is inside the plain reading. (2) The Coding Plan page itself: "This plan is for
  interactive use in programming tools... Do not use the plan's API key for automated scripts,
  application backends, or other non-interactive scenarios" — bulk trace gen is the named
  violation; quota (Pro $50/mo: 6,000 req/5h, 90,000/mo) is sized for interactive use anyway.

### Harvest-lane reality check (correcting the task's premise)

Data-gen inference rides the harvest lane at zero marginal cost **only if the teacher fits the
serving envelope**. It doesn't: K3 is 2.8T (community serving evidence: W4A16 on **TP16** —
16xB200/H200-class, $59-80/node-hr rented); even DeepSeek V4-Flash (284B, the smallest strong
agentic teacher) needs ~145GB at NVFP4 — over the 128GB ceiling of a 4x5090 box. **The teacher
phase is new spend on every path.** Only the student-side work (rejection-sampling scoring,
eval rollouts, drafter builds) rides the harvest lane.

### Cost bottom line (external numbers fetched 2026-08-02; arithmetic own, assumptions stated)

- **Corpus size:** public agentic recipes span 5K trajectories (SWE-smith -> 40.2 SWE-V from a
  near-zero base) to 127K examples + RL (KAT) to 800K samples (R1-distill). Defensible plan:
  **1-5B training tokens** of verified trajectories.
- **Teacher generation** (assumptions: 3x rejection-sampling overhead, ~35% assistant-token
  share, 8x input re-read at 90% cache-hit):
  - K3 via API ($3/M in, $15/M out uniform across 7 providers; Morph $2.90/$14 —
    receipt `kimi-k3-endpoints-20260802.json`): **~$30k per 1B kept tokens** ($20-50k range);
    5B -> $100-250k.
  - K3 self-hosted on rented TP16: $10-40k per 1B kept after prefill drag — no cheaper than
    API, much higher engineering risk.
  - **DeepSeek V4-Pro API ($0.435/$0.87 first-party): ~$2k per 1B kept; 5B ~ $10k.** The
    licensing is also the cleanest (MIT). GLM-5.2: $2-6k/1B.
  - Public K3 trace corpora: free but 10^2-10^3 trajectories — seed/eval only.
- **Gradient phase is noise:** 8xH100 ~ $16-24/hr (vast/RunPod/Lambda, fetched 2026-08-02);
  3B-active MoE full-FT at realistic throughput -> 1B tokens ~ $100-420, 5B ~ $460-2,100;
  LoRA ~half. **The teacher choice swings total cost ~15x; the training method barely 2x.**
- **Total:** LoRA x 1B with V4 teacher ~ **$2.5k**; full-FT x 5B with K3 teacher ~ **$100-250k**.
  Eval battery per candidate: low four figures (Q2). RL infrastructure (what KAT actually
  needed): unpriced here, but it is engineering-months, not dollars.

---

## Verdict

**Distribution (primary axis): the plain idea fails it.** A provider-created fine-tune with no
lab brand and no tool integration has zero precedent of capturing traffic; the evidenced
ceiling for *funded-lab* launches on this same backbone class is $300-600/day gross for the
whole model. Against that payoff: the quality evidence says differentiation requires
127K-example SFT + sandboxed RL (not trace-SFT), the honest gain measured on this backbone is
+5 SWE-V, the loudest bigger claim (Ornith) failed reproduction, and the clean-path cost is
$10-50k plus months of RL-grade engineering plus a marketing motion we don't have. The same
boxes serving vanilla Qwen3.6-35B-A3B tap a proven $13.6k/day pool tomorrow. **As proposed —
KILL.**

**What survives (GO-later), with triggers:**

1. **The provider-application asset is real and costs almost nothing to hold.** OR explicitly
   prioritizes providers with proprietary models; a credible fine-tune SKU in the application
   flips darklanes out of the deprioritized open-weight-host category. This argues for keeping
   the option alive, not for building the model now.
2. **Earliest sensible trigger:** darklanes is an approved OR provider, AND box #1 meters
   >= its coverage line on vanilla SKUs (per the operating model's metering rule), AND we have
   >=30 days of our own routed agentic traffic. That traffic is the one moat no competitor has —
   real production trajectories for calibration and SFT, replacing the teacher-corpus spend
   with data that is ours (subject to our published privacy policy permitting it — a listing
   design decision to make BEFORE the privacy policy ships, per the or-provider checklist).
3. **The variant to build then is prune-first: REAP-25 on Qwen3.6-35B-A3B + agentic SFT heal**
   (~$2.5k clean-teacher LoRA path + low-4-figure eval battery). It is technically
   well-precedented on this exact model, feeds the spill/serving lanes either way, and its
   payoff (KV/concurrency headroom per box) accrues to owned hardware even if the SKU is never
   listed. Plain SFT changes nothing economically; pruning does.
4. **Eval bar the artifact must clear before any listing:** beat vanilla on a unified
   third-party harness (not our own): SWE-bench Verified +3 minimum, Terminal-Bench 2.1 +5
   (avg-of-5), BFCL v4 >= vanilla, <=2pp regression on MMLU-Pro/GPQA/LiveCodeBench v6, and
   loop/non-termination rate at n>=2000 agentic prompts <= vanilla. Cross-harness or it didn't
   happen (the Ornith rule).
5. **Teacher policy if/when trace-gen happens:** the verified pilot path is
   **opencode -> OpenRouter -> DeepSeek V4-Flash pinned to the allowlist
   `["deepseek","deepinfra","fireworks","novita"]`** (~$0.017-0.03 per kept trace; DeepSeek
   first-party is both cheapest-effective and the only ToS that *names* distillation as
   permitted). V4-Pro API for harder traces (~$2k/1B kept) or K3 API under its weights license
   for ceiling experiments — **never the Kimi Code or Qwen coding-plan subs** (both
   ToS-blocked, receipts above), never Baidu-hosted endpoints (3A.5 non-compete), and no K3
   self-hosting fantasy on our own boxes (it is TP16 hardware).

---

## Source index (all fetched 2026-08-02 unless dated)

**In-repo:** `research/or-provider-20260802/REPORT.md`; `research/model-demand-20260801/REPORT.md`;
`research/per-expert-quant/` (arms.lock.json, hourish-eval-plan.md, hy3-layer103p5-release.md);
memories `darklanes-operating-model`, `hy3-plain-arm-fused-tiering`.

**OpenRouter (receipts in this dir):** /api/v1/models; /api/frontend/v1/rankings/models
(2026-07-28 -> 08-01 window; 08-01 is the full N=440 day); endpoints for qwen3.6-35b-a3b,
nex-n2-mini, kimi-k3, sao10k/thedrummer/KAT/Arcee/Hermes; docs/guides/community/for-providers;
providers/apply; docs/guides/overview/models.

**Quality bar:** huggingface.co/Qwen/Qwen3.6-35B-A3B; Kwaipilot/KAT-Coder-V2.5-Dev;
deepreinforce-ai/Ornith-1.0-35B + deep-reinforce.com/ornith_1_0.html; r/unsloth 1uly7d7
(2026-06-28); QwenLM/Qwen-AgentWorld; deepseek-ai/DeepSeek-R1-Distill-Qwen-32B;
SWE-bench/SWE-smith (arXiv 2504.21798); logicstar.ai/blog/swe-star; arXiv 2504.07164 (R2E-Gym);
openhands.dev blog (2025-03-31); moonshotai.github.io/Kimi-Dev; mistral.ai/news/devstral-2;
arXiv 2605.23950; particula.tech (2026-03-25); arXiv 2501.17161; moonshotai/Kimi-K3 card;
greghavens/kimi-k3-coding-and-debugging-traces; marktechpost.com (2026-07-26).

**Prune:** arXiv 2510.13999 + cerebras.ai/blog/reap (2025-10); cerebras/Qwen3-Coder-REAP-*
cards; vllm-project/llm-compressor REAP example; RangerX/barozp/atbender/crucible-labs
Qwen3.6-REAP cards; arXiv 2603.02217 (Router-KD); 0xSero/GLM-5.2-504B card; arXiv 2605.08738
(SlimQwen); NVIDIA Minitron blog (2024-10-17); r/LocalLLaMA 1o98f57, 1ok1tkh, 1qn0dtg,
1oefu29; pipenetwork/Kimi-K3-REAP73 card; unsloth.ai/docs/models/qwen3.6.

**Pilot-path verification (all fetched 2026-08-02, receipts in this dir):** openrouter.ai/terms
(2026-07-27); cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html
(effective 2026-04-29); deepinfra.com/terms; novita.ai/legal/terms-of-service (2025-11-04);
intl.cloud.baidu.com Baidu AI Cloud agreement (3A.5); openrouter.ai/api/v1/models/deepseek/
deepseek-v4-flash/endpoints; streamlake.ai + gmicloud.ai ToS unreachable (JS-only) -> HOLD.

**Licensing/cost:** moonshotai/Kimi-K3 LICENSE; moonshotai/Kimi-K2-Instruct LICENSE;
deepseek-ai/DeepSeek-V4-Pro LICENSE; Qwen/Qwen3.6-35B-A3B LICENSE; venturebeat.com K3-license
piece (2026-07-27); kimi.com/user/agreement/en/modelUse (receipt saved);
platform.kimi.ai/docs/agreement/modeluse (~2026-07-29); alibabacloud.com Product Terms Art.
4.48 + Coding Plan page (receipts saved); OpenRouter ToS (2026-07-27); Fireworks ToS
(2026-07-10); Together ToS (2026-05-19); festr2/kimi-k3-full-mxfp4-kld-reference (TP16
serving evidence); H100/B200 rental: vast.ai/RunPod/Lambda/Spheron (2026-06 -> 2026-08);
Cline/Kilo docs (2026-06); github.com/diegosouzapw/OmniRoute #5518 (2026-06-30);
buildmvpfast.com Nex-N2 analysis (2026-06-13).
