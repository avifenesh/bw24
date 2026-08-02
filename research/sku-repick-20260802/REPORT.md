# SKU re-pick after the Hy3 drop — best OpenRouter serving SKU inside 64-128 GB (2-4x RTX 5090)

Date: 2026-08-02. Lane: `lane/sku-repick`. No GPU used; analysis-first over committed data +
fresh price pulls (every fresh number dated, raw JSON receipts in `raw/`).

**Owner rule executed:** Hy3 is dropped (needs 150 GB+ resident; measured spill floor 2.49 tok/s,
`research/hy3-hopper-20260801/baseline.md`). Question: which OpenRouter-demanded model is the
best serving SKU inside 64-128 GB (2-4x 5090 @ 32 GB, PCIe PP-2/PP-4 receipts-backed
bit-identical; multi-replica for single-card models)? Owner amendments during the lane:
skip the 70B-dense tier (confirmed by data below), the ~300B tier is out of envelope, and give
full seats to **MiMo-V2.5**, **Step-3.7-Flash**, and **DeepSeek V4 Flash** — done, all three
evaluated below.

**Data sources**
- OR 4-day usage (2026-07-28 → 07-31): `research/model-demand-20260801/raw/or_rank_models.json`
  (pulled 2026-08-01). Rows are 4-day cumulative per model; `total_tool_calls`,
  `total_native_tokens_cached`, `total_native_tokens_reasoning`, and media counters are
  **unpopulated (all zero)** in this pull — cache/tool/multimodal shares below are therefore
  bracketed, not measured.
- OR catalog: `.../or_models.json` (336 models, 2026-08-01).
- Endpoint tables: **fresh pulls 2026-08-02** from `openrouter.ai/api/v1/models/{id}/endpoints`,
  17 receipts in `raw/or-endpoints-*.json` (provider, quant label, ctx, prices, cache price,
  tools, uptime_last_30m).
- Artifact sizes: HF API `?blobs=true`, 2026-08-02 (`raw/hf-gguf-sizes-20260802.json`) — real
  GGUF byte sizes, not bpw estimates.
- Engine receipts: `research/tune-data/current-board.json` (2026-08-02),
  `research/hw-buy-20260802/` (TCO method + qwen35 market receipt),
  `research/or-provider-20260802/` (OR mechanics), `research/m0-nccl-20260801/` +
  `research/m1-pp2-20260801/` (PP transport + bit-identical gates).
- Web (cited inline, fetched 2026-08-02): HF model cards, llama.cpp support state.

**Verdict in one line:** launch box #1 (2 cards) on the already-shipped
**Qwen3.6-35B-A3B lane**, and make **Step-3.7-Flash on 4x 5090 (PP-4)** the flagship listing
the moment the box reaches 4 cards — it is the only candidate that combines a top-5 demand pool
($89K/day at list prices), near-zero price competition (3 endpoints, all holding $0.20/$1.15),
an honest 4-card fit (95.3 GB IQ4_XS + 33 GB headroom), and zero new kernel classes.

---

## 1. The fit-class scan

Filter: OR catalog x 4-day rankings, open weights, text-servable, weights fit 64-128 GB at an
honest quant (Q4/IQ4-class or better; Q2/Q3 is not an honest serving quant for a flagship
endpoint — CLAUDE.md quant discipline). Real GGUF sizes (HF API, 2026-08-02):

| Candidate | Total/active | Honest quant | GB | Cards (32 GB) | GGUF |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 35B/3B | IQ4_XS | ~18 | **1**/replica | shipped in-repo |
| Gemma-4-26B-A4B | 25.2B/3.8B | QAT Q4_0 | ~15 | **1**/replica | shipped in-repo |
| Gemma-4-31B | 30.7B dense | QAT Q4_0 | ~17 | **1**/replica | shipped in-repo |
| Laguna-XS-2.1 | 33B/3B | Q4_K_M | 20.3 | **1**/replica | official poolside |
| Qwen3.5-122B-A10B | 122B/10B | UD-IQ4_XS | 60.2 | **2** (tight, 3.8 GB KV) | unsloth 113K dl/30d |
| gpt-oss-120b | 116.8B/5.1B | MXFP4 | 63.4 | **3** (2 cards = no KV room) | official ggml-org |
| Nemotron-3-Super | 120B/12B | UD-IQ4_XS | 64.5 | **3** | unsloth |
| **Step-3.7-Flash** | 198B/11B | UD-IQ4_XS | **95.3** | **4** (+32.7 GB headroom) | official stepfun + unsloth; official NVFP4 repo |
| Laguna-S-2.1 | 118B/8B | Q4_K_M (NVFP4 ~62 GB alt) | 96.0 | **4** (3 at NVFP4) | official poolside + DFlash draft GGUF |
| MiniMax-M2.7 | 230B/10B | UD-IQ4_XS | 108.4 | **4** (19.6 GB KV, admission-bounded) | unsloth |
| DeepSeek V4 Flash | 284B/13B | UD-IQ4_XS | 136.7 | **5** (owner's "little more") | unsloth 143K dl/30d |
| MiMo-V2.5 | 310B/15B | UD-IQ4_XS | 149.4 | **5 tight / 6** — out of envelope | unsloth (text-only) |
| Qwen3-235B-A22B-2507 | 235B/22B | IQ4_XS ~120 GB (est.) | ~120 | 4 marginal, no KV room | exists |

Out at the gate: **Ling-3.0-flash** (124B-A5.1B, 1.35T tok/4d — but its ONLY OR listing is
`:free` on Novita at $0/$0, receipt `raw/or-endpoints-inclusionai-ling-3p0-flash*-20260802.json`;
free-only everywhere else checked 2026-08-02 — a demand mirage, zero paid market to enter),
**mistral-nemo** ($0.019/$0.03 pricing — no revenue), **qwen3-next-80b** ($1.2K/day pool,
superseded class), **step-3.7's vision traffic and all OCR/vision models** (text engine),
**qwen3-235b-a22b** (marginal fit + $2.4K/day pool over 12 endpoints).

**Owner priors, checked against data:** 70B-dense tier CONFIRMED dead — llama-3.3-70b is
$822/day of floor-priced pool across 13 endpoints ($0.32 floor, fp8 DeepInfra), hermes-4-70b is
1 endpoint and 3B tok/4d. Noted and dropped. **GLM-4.7-Flash** (owner-rejected for local):
serving data agrees — $515/day pool, 4 endpoints at a $0.40 floor, 26.9B tok/4d. Dead either way.

## 2. Demand, competition, and the revenue pool (the money table)

4-day OR usage (2026-07-28→31) x fresh 2026-08-02 endpoint floors. "Pool" = daily traffic
priced at today's floor (in x floor-in + out x floor-out) — the honest market-size number.
in:out is measured per model.

| Model | 4d tok | 4d req | in:out | Endpoints (fresh) | Floor $/M in/out | Pool $/day | Pool per endpoint |
|---|---:|---:|---:|---:|---|---:|---:|
| MiMo-V2.5 | 8,094B | 116.9M | 117:1 | 7 (GMICloud fp8 degr. 91.6%) | 0.112/0.224 | $228.5K | $32.6K |
| V4-Flash (preview slug) | 7,561B | 705.3M | 14:1 | **21** | 0.087/0.174 | $175.3K | $8.3K |
| **Step-3.7-Flash** | 1,701B | 25.5M | 93.5:1 | **3 — all at identical held prices, 99.9-100% up** | 0.200/1.150 (cache 0.04) | **$89.3K** | **$29.8K** |
| Laguna-S-2.1 | 597B | 7.8M | 133:1 | 1 (Poolside first-party fp4) | 0.090/0.180 (cache 0.009) | $13.5K | $13.5K |
| gpt-oss-120b | 521B | 122.3M | 6:1 | 19 | 0.030/0.170 | $6.5K | $0.3K |
| Gemma-4-31B | 391B | 75.2M | 13:1 | 18 | 0.090/0.340 | $10.5K | $0.6K |
| Gemma-4-26B-A4B | 371B | 85.9M | 16:1 | 9 | 0.070/0.300 | $7.7K | $0.9K |
| Nemotron-3-Super | 358B | 10.2M | 31.5:1 | **3** (floor bf16 at 89.9% = Degraded) | 0.085/0.400 | $8.5K | $2.8K |
| Laguna-XS-2.1 | 190B | 3.4M | 124:1 | 1 (Poolside fp8) | 0.060/0.120 | $2.9K | $2.9K |
| MiniMax-M2.7 | 153B | 6.5M | 39:1 | 12 (Mara floor 100% up; GMICloud/Atlas 88-91%) | 0.240/0.960 | $9.8K | $0.8K |
| Qwen3.6-35B-A3B | 77B | 8.8M | 10.3:1 | 9 | 0.100/0.950 | $3.4K | $0.4K |
| Qwen3.5-122B-A10B | 16B | 1.3M | 13.3:1 | 5 | 0.260/**2.080** | $1.6K | $0.3K |
| GLM-4.7-Flash | 27B | 6.7M | 19:1 | 4 | 0.060/0.400 | $0.5K | $0.1K |
| Llama-3.3-70B | 28B | 17.0M | 13:1 | 13 | 0.100/0.320 | $0.8K | $0.1K |

Two structural reads:
1. **Step-3.7-Flash is the Hy3 wedge profile, upgraded.** Hy3 was 4.8T tok on 5 endpoints at a
   $0.49 floor; Step is 1.7T on **3** endpoints at a **held $1.15** — no price war (all three
   endpoints, including two third parties, sit at StepFun's exact list price; receipt
   `raw/or-endpoints-stepfun-step-3p7-flash-20260802.json`). Best paid demand-to-competition
   ratio on the whole board.
2. The agentic in:out ratios (93-133:1 for Step/Laguna/MiMo) mean these pages are
   **prefill+cache businesses**, not decode businesses — which reshapes both the revenue math
   (section 4) and the moat fit (section 5).

## 3. Engine fit per candidate (kernel classes, honest days-of-work)

Baseline (README supported table + current-board receipts): GQA, GDN hybrid (qwen3_5/3_6),
SWA+full interleave (gemma-4), qwen/gemma MoE arms incl. shared expert, K-quant/NVFP4/Q4_0/Q8_0,
MTP + EAGLE3 + own-gen drafters, GGUF+safetensors, PP-2 bit-identical (PP-4 = same seam, target
shape), serving surface shipped (tools, prompt-cache 25% billing, metering, early-429).

| Candidate | Kernel classes needed | Class | Honest cost to listing-grade |
|---|---|---|---|
| Qwen3.6-35B-A3B (+AgentWorld/Ornith post-trains) | none | **shipped, over deployment bar** (178.2 plain / 302 spec measured, 1.68-1.76x receipts) | 0 engine days; OR-checklist only |
| Gemma-4-26B-A4B / 31B | none | shipped, board rows | 0 |
| **Step-3.7-Flash** | none new: SWA-512+global 3:1 = gemma-4 seam; 288+1-shared top-8 MoE = existing arms; **official MTP head** (`num_speculative_tokens: 3` in the HF card) = existing spec seam; hidden 4096, 45L (3 dense + 42 MoE) | onboarding + PP-4 shakeout | **2.5-4 weeks**: tokenizer/template 2-4d, SWA-ratio + shared-expert mapping 2-5d, MTP integration + K-sweep 2-4d, PP-4 on the 95.3 GB artifact 3-5d, full gate battery + interleaved head-to-head 3-4d |
| MiniMax-M2.7 | none new: plain GQA full-attention + 256-expert top-8 MoE (62L); MiniMax family loader groundwork exists (M3 REAP50 "loads + generates" in-repo) | onboarding + PP-4 | 1.5-3 weeks (drafter = own-gen, unproven on this family: +3-5d or ship plain if >=1.1x) |
| Laguna-S-2.1 + XS-2.1 | small deltas: softplus router/output gating; SWA-512 1:3 on existing seam; llama.cpp support NOT upstreamed (PR #25165 open) — differentiation window, Bonsai-style | one bring-up, two SKUs | 2-4 weeks combined |
| gpt-oss-120b | **new**: MXFP4 dequant path + attention sinks (SWA exists) | new format work | 2-4 weeks (demand report priced it "moderate") |
| Nemotron-3-Super | **new class**: Mamba-2 scan + LatentMoE + hybrid state cache | biggest lift | 4-8 weeks honest |
| V4-Flash | MLA increments 3-6 (1-2 merged: parse/loader/CPU-forward), DSA lightning indexer (design only), MTP (have) | major, partially started | 4-8 weeks |
| Qwen3.5-122B-A10B | none — literally our arch family (qwen3_5_moe) | onboarding + PP-2 | ~1 week |
| MiMo-V2.5 | none new (SWA-128+full 5:1, 256-expert top-8 — gemma/qwen classes), text-only GGUF | onboarding + PP-5/6 | 2-3 weeks — but out of envelope |

## 4. Economics — `sku-tco.py` (method in file header; hw-buy capex/energy/resale method + explicit prefill/decode split)

Throughput basis: measured 178.2 tok/s (Qwen3.6-35B-A3B IQ4_XS, 896 GB/s laptop rig) → 40%
MoE bandwidth efficiency → per-candidate `D_card = 0.40 x 1790 / active_GB_per_tok` on the
1.79 TB/s desktop card; bracket ±50%; superseded at bring-up. PP-N saturated = N x D_card
(hop cost 7-11.5 µs receipts). Prefill = 10x decode (hw-buy assumption) with a 5x sensitivity
row — **at 90:1+ shapes prefill is the revenue engine and it is our least-receipted stage on
sm_120a: measure before pricing.** Wedge prices ~5-10% under 2026-08-02 floors. Used-5090
capex ($2.7K/card), 30% utilization, $0.18/kWh, 60% resale at 3y.

```
=== cache 0%, P=10xD | util=30% ===                    $/hr sat  3y rev$  3y net$
q3.6-35b-a3b 2 replicas [SUPPORTED]  (capex  $8.4K)       2.00    15,759   +7,706
q3.6-35b-a3b 4 replicas [SUPPORTED]  (capex $14.8K)       4.00    31,517  +18,713
step-3.7-flash PP-4                  (capex $14.8K)       3.20    25,232  +12,428
minimax-m2.7 PP-4                    (capex $14.8K)       3.67    28,917  +16,113
qwen3.5-122b-a10b PP-2               (capex  $8.4K)       2.17    17,130   +9,078
laguna-s-2.1 PP-4                                         1.76    13,890   +1,086
laguna-xs-2.1 2 replicas                                  1.10     8,647     +594
gemma-4-26b-a4b 2 replicas                                1.03     8,104      +51
v4-flash PP-5 [5 CARDS]              (capex $17.5K)       1.00     7,888   -6,792
nemotron-3-super PP-3                                     0.85     6,664   -4,264
gemma-4-31b 2 repl spec                                   0.58     4,547   -3,505
gpt-oss-120b PP-3                                         0.53     4,189   -6,739
mimo-v2.5 PP-5 [OUT OF ENVELOPE]                          1.52    11,968   -2,713

=== cache 70%, P=10xD ===  step 4.40/hr +21,880 | m2.7 4.34/hr +21,400 | q35x4 4.50/hr +22,706
=== cache 70%, P=5xD  ===  step 2.53/hr  +7,161 | m2.7 2.82/hr  +9,419 | q35x4 3.64/hr +15,924
```

(Full three-scenario output: run `python3 sku-tco.py`.)

What the table proves:
- **The 35B-A3B lane remains the best $/hr per card in every scenario** — the hw-buy pick
  survives this re-pick. It is also the only candidate already over the deployment bar.
- **Step-3.7-Flash and MiniMax-M2.7 are the only big-model candidates with materially positive
  3-year net** ($7-22K) inside the envelope. They are economically twins; the demand pool
  separates them (9x) — utilization capture at 30% needs 0.04% of Step's pool vs 0.3% of M2.7's.
- **V4-Flash is TCO-negative at its own floor in all scenarios** (-$6.8 to -$9.1K): 21
  undifferentiated providers ground the price to $0.174/M-out. Owner's "extremely cheap to
  serve" is TRUE (13B active) — but everyone else enjoys the same fact, and the floor already
  reflects it.
- gpt-oss-120b's 521B tok/4d monetizes at $0.53/hr — 19-endpoint commodity race; skip.
- Nemotron-3-Super: the only thin-competition page we must reject on engine cost — the biggest
  bring-up (Mamba-2+LatentMoE) for a $0.40-floor page that nets negative.

## 5. The moat lens

Our edges: tools-capable cheap endpoint (lane/serve-tools shipped 2026-08-02 — the exact gap
GMICloud left open on hy3), prompt-cache with 25% cache-read billing (shipped), spec decode
(180-302 tok/s receipts; MTP seam), exactness contract, early-429 admission discipline.

- **Step-3.7-Flash exploits every edge at once**: 93.5:1 agentic traffic makes prompt-cache the
  margin driver (cache 0%→70% lifts $/hr 37%); it ships an **official MTP head** so our
  strongest tech (spec decode) applies day one; all 3 incumbents support tools, so Auto Exacto
  eligibility is table stakes we already meet; and a 4th endpoint 5% under a held list price
  takes 100% of `:floor` plus outsized default-routing weight (inverse-square: ~1.22x vs the
  three at list).
- Laguna-S/XS is the purest *positioning* moat (sole incumbent, coding-agent tools traffic,
  official DFlash draft GGUF, OpenMDW, llama.cpp not upstreamed = "fastest Laguna on NVIDIA"
  story) — but 133:1 at a $0.162 held out-price monetizes weakly ($1.8-2.4/hr). It is the best
  *second* big listing, not the first.
- V4-Flash's moat angle is different and real but not a listing case: **internal workhorse**
  (13B-active MIT, 1M ctx — our own agent/research traffic becomes harvest-tenant #1 on idle
  capacity, worth roughly its API price in avoided spend, low-hundreds $/mo at our volumes) and
  the **speed wedge** (21 providers, uptime spread 87.5-100%, fp4/fp8 mix — a measurably
  fastest endpoint wins latency-aware routing). Both angles activate AFTER MLA 3-6 + DSA land;
  neither pays for 5 cards at a $0.174 floor.

## 6. The owner candidates, adjudicated

- **MiMo-V2.5** ("best next by users, not necessarily business-wise") — exactly right, now with
  receipts: biggest pool on the board ($228K/day) but 7 providers in an active floor war
  ($0.112/$0.224, GMICloud already degraded at 91.6% uptime), 117:1 shape, and it **fails the
  envelope honestly** (IQ4_XS 149.4 GB → 5 cards with 10.6 GB total headroom, or 6 cards;
  Q2/Q3 fits are not honest serving quants). At 5 cards it nets -$2.7K to +$1.6K — worse than
  Step on 4 cards in every scenario. Omnimodal → GGUF text-only forfeits an unmeasurable
  traffic share. Verdict: the 5-6-card-era candidate list exists (MiMo vs V4-Flash), decide
  then; not now.
- **Step-3.7-Flash** (owner: "the Hy3 wedge at a high output price") — VERIFIED on all six
  asks: (1) 198B total / **11B active**, SWA-512+global 3:1, GQA-family, 288+1 experts top-8,
  256K ctx, Apache-2.0, official MTP head — zero exotic classes; (2) **4 cards at IQ4_XS
  95.3 GB + 32.7 GB headroom** (SWA on ~3/4 of layers keeps global-KV small; official NVFP4
  ~100-110 GB alt also 4-card); (3) official stepfun GGUF + unsloth; llama.cpp via StepFun's
  fork/branch, upstream merge unverified → differentiation window, verify at bring-up;
  (4) 1.70T tok/4d, 25.5M req, 3 endpoints all at $0.20/$1.15/cache-$0.04, 99.9-100% uptime;
  (5) $89.3K/day pool; box economics above — best big-model net; (6) 2.5-4 weeks, no new
  kernels. Honest risks: China-centric demand skew (older demand report), unknown multimodal
  share (we serve text-only), the held price could break if a price war starts (a 4th cheap
  endpoint — us — may itself trigger it; model the floor at -20% and it still beats M2.7's
  pool-adjusted capture), and prefill-heavy revenue rides our least-receipted stage (measure
  PP-4 prefill before submitting the OR application).
- **DeepSeek V4 Flash** — owner's "158b" is not a real variant: both the preview and 0731 are
  **284B total / 13B active** (OR catalog + HF, 2026-08-02); ~137 GB IQ4_XS is likely the
  number remembered. 5 cards honest (Q3_K_XL 128.2 GB is a 4-card squeeze with zero KV room —
  no). Business shape as scored above: internal-use + speed-wedge are real, revenue at floor is
  not. Sequencing: it is the natural *engine-showcase* SKU once MLA 3-6 + DSA exist for their
  own research value — not the reason to build them.

## 7. Verdict — ranked top 3 (business), the pick, and the plan

**#1 — Qwen3.6-35B-A3B lane (launch NOW, box #1 as bought).** 2 used cards, 2 replicas.
Already over the deployment bar with the drafter regime; tools+cache+metering shipped; 9
providers at a defended $0.95 floor. Best per-card economics in every scenario. Listing
breadth for free: gemma-4-26b-a4b/31b (supported; `is_ready`-switchable onto the same cards
when their traffic pays better — 26b is ~breakeven, 31b negative: list, don't chase).

**#2 — Step-3.7-Flash PP-4 (THE re-pick; turns on at card 4).** The single SKU
recommendation for the 64-128 GB envelope. Coverage math on the grown box (4x 5090, 128 GB):
IQ4_XS 95.3 GB weights + ~33 GB KV/activations; saturated projection ~490 tok/s out or
~4.4K tok/s prefill-mix (bracket ±50%); revenue $3.20-4.40/hr saturated at 0-70% cache
(prefill-sensitivity floor $2.53/hr). At 10/30/60% utilization: ~$0.32-0.44 / $0.96-1.32 /
$1.92-2.64 per hr → roughly **$230-320 / $700-960 / $1,400-1,900 per month** against $14.8K
box capex and ~$12.8K 3-year TCO — 3-year net +$7.2K to +$21.9K. Capture needed at 30% util:
0.04% of the page's current pool.

**#3 — MiniMax-M2.7 PP-4 (the hedge; same box, 1.5-3 weeks).** Economic twin of Step with a
9x smaller pool but a friendlier 39:1 shape (leads in the prefill-pessimistic scenario), 12
endpoints with the floor drifting, standard kernel classes, loader groundwork in-repo. Build
it if Step's held price collapses or its bring-up hits a wall — or after Step, as the second
big page. (Laguna-S+XS slots behind it as the sole-incumbent positioning play; Qwen3.5-122B
is a ~1-week opportunistic add at a $2.08 floor on our own arch, thin pool.)

**Required work before the Step listing (honest):**
1. Engine bring-up, 2.5-4 weeks (section 3 breakdown): onboarding → SWA-3:1 + shared-expert
   mapping → MTP spec integration → PP-4 shakeout on the 95.3 GB artifact → full battery
   (kernel-check / run-gen argmax / run-spec K=1..8) + N=5 interleaved head-to-head vs the
   StepFun llama.cpp fork. Deployment bar: >=1.1x e2e.
2. **Measure PP-4 prefill throughput before pricing** — at 93.5:1 it IS the revenue; the 10x
   assumption above is the least-receipted number in this report.
3. OR-checklist remainder (shared across all SKUs, from `research/or-provider-20260802`):
   merge `lane/dl-metering`, OR-schema `/v1/models` (quant label: `int4` for IQ4_XS or `nvfp4`
   for the official NVFP4 — declared honestly), privacy/ToS pages, `capacity_tpm` from the
   load harness, then apply (backlog: open-weight hosts deprioritized — submit early).
4. Wedge price at application: **$0.19/M in, $1.09/M out, $0.045/M cache-read** (~5% under the
   held $0.20/$1.15/$0.04; cache-read kept above 20% of input to avoid margin inversion on
   90:1 traffic).

**Not picked, for the record:** gpt-oss-120b (19-endpoint commodity, $0.53/hr — revisit only
if MXFP4 lands for other reasons), Nemotron-3-Super (best thin-competition page but Mamba-2 +
LatentMoE bring-up for a negative-net floor), Ling-3.0-flash (free-only mirage), MiMo-V2.5 /
V4-Flash (5-6-card era, section 6), 70B tier and GLM-4.7-Flash (pools $0.5-0.8K/day — data
confirms both owner calls).

---

## Source index

Fresh pulls 2026-08-02 (receipts in `raw/`): OR endpoints API x17 models; HF API blob sizes
(ggml-org/gpt-oss-120b-GGUF, unsloth Nemotron-3-Super/MiniMax-M2.7/MiMo-V2.5/V4-Flash(+0731)/
Step-3.7-Flash/Qwen3.5-122B GGUF repos, poolside Laguna-S/XS-2.1-GGUF). Web fetched 2026-08-02:
huggingface.co/stepfun-ai/Step-3.7-Flash (Apache-2.0, MTP config, stepfun llama.cpp fork);
llama.cpp PR #25165 (Laguna, open); marktechpost.com 2026-05-29 + recipes.vllm.ai (Step-3.7
arch: 198B/11B, SWA-512+global 3:1, 288+1 experts, 3+42 layers); ggml-org discussion #20421 +
NVIDIA model card (Nemotron-3 LatentMoE/Mamba-2); unsloth docs (MiniMax-M2.7 230B-A10B,
62L, 8/256); openrouter.ai model pages (Ling free-only). Committed data: as listed in the
header. Engine receipts: repo paths cited inline.
