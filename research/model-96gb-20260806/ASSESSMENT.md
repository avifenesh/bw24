# Model re-assessment for the 96 GB card — what darklanes serves on one RTX PRO 6000

Date: 2026-08-06. Lane `lane/model-96gb` (from `restructure/public-split` @ 701af619).
CPU-only research — no GPU runs in this lane. Owner directive (2026-08-06, verbatim): *"we
planned q27b because of the 5090 so we need re assesment of the correct model for the 6000
capacity."* H100/sm_90a is excluded per the same directive ("h100 is out, non relevant
anymore") — no H100 numbers or SKUs are considered below.

Deployment frame: darklanes serves on RTX PRO 6000 Blackwell WS 96 GB, boxes-of-2 — serve on
card 1, lab on card 2 — so this is a **one-card** assessment. Owner priority axis: models
users actually want, never models-for-the-models. Standing owner rules that bind candidates:
prod quant = 8-bit (memory `prod-8bit-and-2x5090-rental-go`), deployment bar = trimmed
drafter gated before deploy, honest quant labels on any listing (or-provider REPORT §3).

Evidence base: OR 4-day rankings + catalog (`research/model-demand-20260801/raw/`, pulled
2026-08-01), fresh endpoint/config pulls 2026-08-06 (receipts in `raw/` here), sku-repick
receipts (`research/sku-repick-20260802/`, incl. real GGUF byte sizes), the PRO 6000 prod
board (`research/pro6000-prod-20260804/pro6000wk-runpod.jsonl`), the q27 phase-2 deep dive
(`research/q27-deepdive-p2-20260805/`), and the 8-bit format decision
(`research/8bit-decision-20260803/DECISION.md`).

---

## 0. The Qwen3.8 wildcard — RESOLVED (this changes the question)

Fetched 2026-08-06 (openlm.ai/qwen3.8; yottalabs.ai 3.8-Max post, updated post-launch;
Alibaba announcement 2026-08-03; unsloth on X 2026-08-03):

- Qwen3.8 open-weight variants announced: **Qwen3.8-Max (2.4T total, ~95B active, MoE, 1M
  ctx)** and **Qwen3.8-27B**. That is the whole announced open list. **No 72B, no mid-size
  variant announced.**
- Qwen3.8-Max at ~95B active is out of the one-card envelope by an order of magnitude
  (~1.2 TB at 4-bit). It is not a candidate on any darklanes hardware.
- Unsloth: "Qwen3.8-27B will run locally on 17GB RAM/VRAM setups" — consistent with a
  27B-dense-class artifact in the 3.6-27B shape (16 GB NVFP4 daily today).
- All Qwen open-weight models are Apache-2.0 (openlm.ai license section; 3.6/3.5 precedent).

**Consequence:** the "3.8 may ship larger variants we could serve" branch is closed. Day-one
3.8 leverage on a 96 GB card is exactly the 27B, riding the existing same-arch runbook
(`research/qwen38-prep-20260803/`). The re-assessment question collapses to: *what do we run
NEXT TO and INSTEAD OF q27 with the other ~80 GB.*

## 1. VRAM arithmetic per candidate class (weights + KV, shown)

memra KV default = q8_0 K + q5_1 V ≈ **0.906 B/elem** (58 B per 32-elem K+V block pair =
45.3% of BF16; `research/kv-compress-20260802/REPORT.md` §1.1). KV/token = 2 (K+V) ×
kv_heads × head_dim × n_full_attn_layers × 0.906 B. GDN linear-state is per-session-fixed
and negligible (~75-100 MB). Working overhead (activations, graphs, pools) budgeted 6 GB.

| Candidate | Weights (receipt) | Full-attn KV/token | 128k session | Fits 96 GB? | KV budget left → concurrent 128k sessions |
|---|---|---|---|---|---|
| **q27 NVFP4+MTP daily** | 15.7 GB (+1.2 drafter) | 64L/int4 → 16 full × 4 kvh × 256 hd = 29.0 KB | 3.7 GB | yes, trivially | ~73 GB → **~19** |
| **q27 Q8_0 (prod-8bit)** | 28.6 GB | same 29.0 KB | 3.7 GB | yes | ~61 GB → ~16 |
| **q27 Q8_0 + Q8RP mirror** | 53.2 GB resident class (measured 63.7 GB total at c=16 incl. KV) | same | 3.7 GB | yes — **the 96 GB-only lever** | ~37 GB → ~9 (or 40× 32k) |
| **Qwen3.5-122B-A10B UD-IQ4_XS** | 60.2 GB (HF blob receipt, sku-repick) | 48L/int4 → 12 full × 2 kvh × 256 hd = 10.9 KB | 1.4 GB | **yes** — the card's differentiator | ~30 GB → ~21 |
| Qwen3.5-122B-A10B Q8_0 | ~130 GB | — | — | **NO** | 8-bit form does not exist on one card |
| **Dense 70B-class Q8_0** (llama-3.3-70B shape) | 77.2 GB (72.7B × 1.0625) | 80 full × 8 kvh × 128 hd = 148 KB(!) | **19.4 GB** | weights yes, envelope no | ~13 GB → **0** at 128k; 2-3 at 32k |
| Laguna-S-2.1 NVFP4 | ~62 GB (sku-repick est.; Q4_K_M receipt 96.0 GB does NOT fit) | GQA + 1:3 SWA-512 → small | ~1-2 GB | yes (NVFP4 only) | ~28 GB → ~15-20 |
| Step-3.7-Flash IQ4_XS | 95.3 GB (receipt) | — | — | **NO** (zero KV room) | 128 GB-class SKU, not 96 |
| Hy3 295B-A21B NVFP4 | ~150 GB | — | — | **NO** resident | spill only: 5.13 tok/s served (docs/PERFORMANCE.md) |
| gpt-oss-120b MXFP4 | 63.4 GB (receipt) | sinks+SWA small | small | yes | fits, but see demand |
| **Multi-SKU: q27-Q8_0+Q8RP + q9-Q8_0** | 53.2 + 9.4 = 62.6 GB | q27 29 KB / q9 ~7 KB | — | yes | ~27 GB shared → q27 ~6× 128k + q9 ~20× 32k |

Notes on the table:
- The dense-70B row is the arithmetic kill: full attention on every one of 80 layers means
  **148 KB/token KV — 5.1× q27's and 13.6× the 122B's** — so an 8-bit 70B on 96 GB is
  weights-dominant and batch-starved: one 128k session doesn't fit, and the card that wins
  on batch throughput (Q8RP +57% came from c=16/32) can never reach batch. The class is
  hardware-shaped wrong for this card, before demand is even consulted.
- Q8RP receipt (prod pod, 2026-08-04, `pro6000wk-runpod.jsonl`): c=16 agg 486.2 vs 310 off
  (+57%), c=32 488.5, p50 6.61→4.21 s, 0 errors, 63.7 GB resident. "27B trunk+mirror+KV =
  63-72 GB — impossible on 24/32 GB, comfortable here."
- MEMRA_CTX floor sweep on the same pod: 2048/8192/32768 → 421.4/421.5/420.5 tok/s (flat) —
  **KV headroom is free at 96 GB**; 32k+ floors cost nothing.

## 2. Demand per candidate (OR 4-day window 2026-07-28→31 + fresh 2026-08-06 endpoints)

| Model | 4d tok | in:out | Endpoints | Floor $/M in/out | Pool $/day | Read |
|---|---:|---:|---:|---|---:|---|
| qwen3.6-27b | 27.1B | 12.3:1 | **9** (fresh 08-06: Chutes fp8 $0.30/$2.00 floor; Morph $0.289/$2.40; Alibaba $0.45/$2.70) | 0.30/**2.00** | ~$2.9K | modest pool, **high held out-price** — a decode-margin page, not a prefill-war page |
| qwen3.6-35b-a3b | 76.7B | 10.3:1 | 9 | 0.10/0.95 | $3.4K | supported #1 per-card economics (sku-repick TCO) |
| gemma-4-31b / 26b-a4b | 391B / 371B | 13:1 / 16:1 | 18 / 9 | 0.09/0.34 · 0.07/0.30 | $10.5K / $7.7K | big pools, commodity floors, $0.1-0.9K/day *per endpoint* — list, don't chase |
| **qwen3.5-122b-a10b** | 16.3B | 13.3:1 | **5** (fresh 08-06: SiliconFlow/Alibaba fp8 $0.26/**$2.08**, DeepInfra fp4 $0.29/$2.40, AtlasCloud, Novita bf16) | 0.26/**2.08** | $1.6K | thin pool but the **highest held out-price of any fit-on-96GB open model**, only 5 providers, price held since ≥08-02 |
| llama-3.3-70b | 28.4B | 13:1 | 13 | 0.10/0.32 | $0.8K | dead: 13 providers grinding a $0.32 floor |
| qwen2.5-72b | 3.5B | 5.7:1 | few | — | ~$0 | legacy; license "qwen/other" (fresh HF receipt), not Apache |
| laguna-s-2.1 | 597B | 133:1 | **1** (Poolside fp4 $0.09/$0.18, fresh 08-06) | 0.09/0.18 | $13.5K | biggest unserved-by-third-parties pool; prefill business at a weak out-price |
| gpt-oss-120b | 521B | 6:1 | 19 | 0.03/0.17 | $6.5K | $0.3K/day/endpoint commodity; MXFP4 = new format work |
| hy3 | 4.8T | — | 5 | 0.13/0.53 | big | does not fit; spill = 5 tok/s; not interactive-grade |

Agent-workload demand side (`research/model-demand-20260801/REPORT.md`): the most-pulled
local agent models of the summer are post-trains of the **same two arches memra already
serves** — Ornith-1.0-9B/35B (4.59M + 3.45M HF dl/30d, #2/#3 of all GGUF repos),
Qwen-AgentWorld-35B-A3B, KAT-Coder-V2.5. Coding-agent users (pi/aider/cursor-class) run
qwen-family mid-size models; nothing in that audience runs dense-70B anymore. The owner's
own daily driver is q27+drafter (memory: memra = owner default engine).

## 3. Candidate verdicts

### C1 — q27-class at 96 GB: **KEEP as primary SKU; the 5090→6000 move changes the config, not the model**

- **VRAM**: §1. The 24 GB constraint that forced NVFP4-only is gone; the 96 GB card runs
  the prod-8bit artifact (Q8_0 28.6 GB) **with** the Q8RP mirror (+57% at c=16/32 — the
  measured 96 GB-only lever) **and** 128k-ctx admission **and** spec ON, simultaneously.
- **License**: Apache-2.0 (Qwen open-weights; 3.8 same posture announced).
- **Drafter**: MTP head + own-gen trimmed drafter shipped and gated; PRO 6000 receipts:
  spec serve c=1 170.55 tok/s (K=3, N=5), bare 186.7; K-policy from p2: q8 K=4(B32)/K=6(B128),
  nv K=5; burst 128 + p-min 0.3 stack = +14%/+20% further (community pod, relative).
- **Revenue ceiling per card** (assumptions labeled): wedge at $0.27/$1.80 (10% under the
  fresh Chutes floor $0.30/$2.00). Prefill-decode shared-card model at the measured
  12.3:1 in:out shape: per out-token = 1/488 s decode (Q8RP c=32 receipt) + 12.3/4118 s
  prefill (pp512 receipt) = 5.04 ms → ~198 out tok/s + 2.4k in tok/s sustained ≈
  **$3.7/hr saturated gross**; with 70% prompt-cache hits (agentic traffic; cache-read
  billed 25% of in) ≈ **$4.1/hr**. At 30% utilization ≈ **$0.9-1.2/hr ≈ $650-900/mo per
  card**. For calibration: the or-provider hy3-floor estimate was $2-4/hr saturated on
  *rented* H100s against 5 incumbents at a $0.49 out-floor; q27 on owned silicon at a
  $2.00 out-floor is the better margin shape (fewer competitors, 4× the out-price,
  decode-heavier traffic that matches the spec-decode edge).
- **Engineering cost**: **zero** — supported since v0.1.0, gated on this exact rig class
  (kernel-check model-backed, argmax, run-spec, serve gates: `pro6000-prod-20260804/`).
  3.8-27B day-one runbook standing.
- **The honest caveat**: q27's OR pool is ~$2.9K/day — a listing is distribution + receipts
  + owner-dogfood infrastructure, not a standalone profit engine at one card. That was
  already true of every candidate at this fleet size (or-provider REPORT §3).

### C2 — Dense 70B-class 8-bit: **REJECT (arithmetic + demand + license + no drafter, each independently fatal)**

1. **Arithmetic** (§1): 148 KB/token KV → zero 128k sessions on 96 GB at Q8_0; the card's
   proven win (batch residency) is unreachable. 
2. **Demand**: the class is dead — llama-3.3-70b $0.8K/day across 13 endpoints,
   qwen2.5-72b 3.5B tok/4d. The sku-repick lane already confirmed this owner prior with
   data ("70B-dense tier CONFIRMED dead"). Nothing in the coding-agent audience runs 70B.
3. **No modern candidate exists**: the Qwen 3.5/3.6 generation has **no 70B dense** — the
   lineup jumps 27B dense → 35B-A3B → 122B-A10B MoE. The newest class members are
   llama-3.3-70B (Dec 2024, Llama license, no MTP head) and Qwen2.5-72B (Sep 2024,
   license "qwen"/other per fresh HF receipt — fails the Apache/MIT preference). Both are
   ~1.5-year-old models the market has moved past.
4. **Drafter path**: neither ships an MTP/NextN head; an own-gen donor-block drafter on a
   foreign arch is unproven work for a dead page. Bring-up 2-4 weeks (new tokenizer/
   template/gates) for a $0.8K/day pool at a $0.32 floor. No.

### C3 — Large MoE 100B+ on one card: **split verdict**

- **Hy3-class (the spill machinery): REJECT for interactive serving.** 295B-A21B NVFP4
  ≈ 150 GB > 96 GB resident; the measured spill serve floor is 5.13 tok/s — demo-grade,
  not endpoint-grade. The Hy3 study stays parked research. (Honest fit note the owner
  asked for: MoE-high-ctx multi-GPU is where vLLM-class engines dominate; one 96 GB card
  is not that fight.)
- **Qwen3.5-122B-A10B: ADOPT as the second SKU — this is what the 96 GB card is FOR.**
  The one model class the 24/32 GB era could never touch that this card serves resident:
  - Fits: UD-IQ4_XS 60.2 GB + ~30 GB KV → ~21 concurrent 128k sessions (§1). NVFP4
    house-quant (~61 GB est.) is the memra-native alternative.
  - **Zero new kernel classes**: `qwen3_5_moe` — literally the supported 35B-A3B arch at
    bigger shape (config receipt fetched 2026-08-06: GDN hybrid interval-4, 256 experts
    top-8 + shared expert, GQA 2 kvh × 256 hd). sku-repick priced bring-up at **~1 week**.
  - **MTP head confirmed**: `mtp_num_hidden_layers: 1` in the config → the standard
    drafter regime applies; deployment bar reachable by the existing recipe.
  - License: **Apache-2.0** (HF cardData receipt 2026-08-06).
  - Demand honesty: $1.6K/day pool is thin, but it is the **highest held out-price
    ($2.08/M) of any open model that fits this card**, 5 providers, no price war. At a
    $1.90 wedge and ~10B-active decode (bandwidth estimate ~130 tok/s c=1 class, batch
    aggregate several hundred), the saturated ceiling is $3-4/hr-class — comparable to
    q27 — and the *pair* (27B fast + 122B premium brain, one arch family, one drafter
    regime, one gate battery) is a coherent two-tier product page.
  - **Conflict to surface, not decide**: prod=8bit rule vs 122B-only-fits-at-4bit. Options:
    serve it declared `int4`/`nvfp4` (OR label enum supports both; quality-sensitive users
    filter — same posture as every fp4 incumbent: DeepInfra already serves this page fp4),
    or skip the SKU. **Owner call.**
- **Laguna-S-2.1 NVFP4 (~62 GB): WATCH, fast-follow.** Best big-SKU demand-to-competition
  on the board ($13.5K/day through a single first-party endpoint, fresh 08-06: still
  Poolside-only, fp4, $0.09/$0.18), OpenMDW, official DFlash draft. But 133:1 prefill-heavy
  at a $0.18 out-price monetizes weakly, and bring-up is 2-4 weeks (softplus gating,
  SWA-512 interleave). Slot after the 3.8 drop settles, if its single-provider status holds.
- **Step-3.7-Flash / gpt-oss-120b / MiniMax-M2.7**: out — Step is a 128 GB-class SKU
  (95.3 GB, zero KV room on 96), gpt-oss is a 19-endpoint commodity with new-format cost,
  M2.7 needs 108 GB.

### C4 — Multi-SKU on one card: **ADOPT (it's free) — q27 + q9 first, 122B changes the pairing**

- Arithmetic (§1): q27-Q8_0+Q8RP (53.2 GB) + q9-Q8_0 (9.4 GB) + shared ~27 GB KV — two
  price points on one card with full residency levers intact. q9 serves 281/211/187 tok/s
  spec on an 82-SM card; on 188 SM it is the "fast cheap tier" endpoint (qwen3.5-9b page:
  $0.10/$0.15, 32.6B tok/4d — more traffic than q27's page).
- Isolation story exists: tenant prompt-cache namespacing + per-replica VRAM budgets +
  admission caps (docs/SERVING.md machinery; supervisor runs one model per replica —
  running two replicas on one card is config, not engine work).
- If the 122B SKU is approved, the better pairing is **122B (60 GB) + q27-NVFP4 (17 GB)**
  ≈ 77 GB + ~13 GB KV: premium tier + daily tier on card 1, and q27-Q8_0 large-batch
  capacity lives on additional boxes as the fleet grows. (Q8RP-mirror q27 + 122B does NOT
  fit together: 53 + 60 > 96.)

## 4. Ranked recommendation

1. **q27-class stays the primary serve SKU — re-shaped for 96 GB, not replaced.** Q8_0
   prod artifact + Q8RP residency mirror (+57% measured), MEMRA_CTX floor 32k+ (measured
   free), 128k admission, spec ON with the p2 K-policy, drafter attached. Swap to
   Qwen3.8-27B day-one via the standing runbook when it lands (~this week). Auto-decidable:
   all receipts exist; nothing above is a default-flip on the engine (machine-specific
   serving config per flags doctrine).
2. **Add Qwen3.5-122B-A10B as the 96 GB-differentiating premium SKU** (~1 week bring-up,
   zero new kernels, Apache-2.0, MTP head confirmed) — gated on the owner's 4-bit-quant
   exception call, and on the standard deployment-bar battery once brought up. If Qwen ships
   a 3.8-gen successor in this class later, same slot, same runbook.
3. **Multi-SKU co-residency** — q27+q9 now (free, receipts exist); re-pair as 122B+q27-NVFP4
   if #2 is approved.
4. **Laguna-S-2.1** — watchlist fast-follow (recheck single-provider status + 3.8 landscape
   in 2-4 weeks).
5. **Dense 70B-class — rejected** (four independent kills, §C2).
6. **Hy3-class spill serving — rejected for interactive**; research lane stays parked.

## 5. Owner decisions vs auto-decidable

**Owner must decide:**
1. **The 122B quant exception** — prod=8bit rule vs the only fit being int4/NVFP4 (declared
   honestly on the listing). This is the gate on recommendation #2.
2. **Listing wedge prices** for the q27 page ($0.27/$1.80 proposed, 10% under the fresh
   floor) — pricing is a business call per the or-provider study.
3. **Laguna go/no-go** when the watch window closes.
4. (Standing, re-flagged) which 8-bit form ships 3.8 day-one — Q8_0 bridge vs FP8-ST — the
   open call from `8bit-decision-20260803`. Note: the Q8RP mirror lever is currently a
   GGUF/Q8_0 lever; an FP8-ST equivalent is unbuilt engineering, which weighs on the
   day-one side of that call for the 96 GB serving shape.

**Auto-decidable (no owner input needed):**
- q27 remains primary; 96 GB serving config re-shape (Q8RP on, ctx floor, K-policy) — all
  machine-specific env config under the flags doctrine, receipts in hand.
- q9 co-residency behind the same proxy (config only).
- 122B bring-up *preparation* (tokenizer/template diff, gate scripts, drafter recipe) can
  start regardless — the quant call only gates the public listing, not the engineering.

## Source index

Fresh (2026-08-06, receipts in `raw/` here): OR endpoints qwen3.6-27b + qwen3.5-122b-a10b +
laguna-s-2.1; HF config.json Qwen3.5-122B-A10B; HF cardData Qwen2.5-72B (license) — plus
web: openlm.ai/qwen3.8, yottalabs.ai Qwen3.8-Max post (launch specs 2.4T/~95B active),
unsloth X post (27B/17GB), latent.space AINews 08-04.
Committed: `research/model-demand-20260801/raw/or_rank_models.json` (+REPORT),
`research/sku-repick-20260802/` (GGUF sizes, endpoint floors, TCO, 70B kill),
`research/pro6000-prod-20260804/pro6000wk-runpod.jsonl` (Q8RP, ctx sweep, serve rows, gates),
`research/q27-deepdive-p2-20260805/` (K-policy, burst, p-min), `research/8bit-decision-20260803/DECISION.md`,
`research/kv-compress-20260802/REPORT.md` (KV bytes law), `research/or-provider-20260802/REPORT.md`
(listing mechanics, revenue realism), `research/qwen38-prep-20260803/{AUDIT,WATCH}.md`,
`docs/PERFORMANCE.md` (rigs, 27B serving board, Hy3 spill 5.13 tok/s).
