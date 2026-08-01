# Model demand research — 2026-08-01

Lane: `lane/model-demand`. All numbers pulled live on **2026-08-01**; raw evidence in `raw/`.
No training-knowledge claims: every model spec below was read from its HF `config.json` /
model card on fetch day, every demand number from a live API or page.

**Sources & windows**
- OpenRouter usage: `openrouter.ai/api/frontend/v1/rankings/models` — per-model daily tokens,
  window **2026-07-28 → 2026-07-31** (4 days), fetched 2026-08-01 (`raw/or_rank_models.json`).
- OpenRouter catalog/pricing/endpoints: `/api/v1/models` + `/api/v1/models/{id}/endpoints`,
  fetched 2026-08-01 (`raw/or_models.json`).
- HuggingFace: `api/models?sort=trendingScore` and `?filter=gguf&sort=downloads` — the
  `downloads` field is **30-day** downloads, fetched 2026-08-01 (`raw/hf_*.json`).
- Ollama library (all-time pulls + update recency), fetched 2026-08-01
  (`raw/ollama_library_popular_20260801.txt`).
- r/LocalLLaMA: old.reddit top/month + per-model search top/quarter, scraped 2026-08-01
  (`raw/reddit_localllama_top_month_20260801.html`).
- llama.cpp demand: GitHub issue/PR search `repo:ggml-org/llama.cpp created:>2026-05-15`,
  sorted by +1 reactions, fetched 2026-08-01.
- Architecture facts: each repo's `config.json` + README on huggingface.co, fetched 2026-08-01.

**memra baseline (what "rides existing kernels" means)** — from
`research/current-model-targets.md` + supported board: GQA full attention, Gated-DeltaNet
hybrid linear attention (qwen3_5/qwen3_5_moe), qwen/gemma MoE arms, K-quant + NVFP4 dequant,
MTP (NextN) + EAGLE3 spec decode, GGUF + safetensors loading, SLRU expert spill (research lane).
**Not shipped:** MLA (bring-up lane in flight), DSA sparse indexing, KDA, MSA, vision/audio
encoders, sub-2-bit quant formats.

---

## The mid-2026 landscape in one paragraph

The OpenRouter open-weights board (4-day tokens) is led by **xiaomi/MiMo-V2.5** (8.1T tok,
310B-A15B), **deepseek/DeepSeek-V4-Flash** (7.6T tok and 705M requests — #1 by request count
site-wide), **tencent/Hy3** (4.8T, 295B-A21B), **deepseek/V4-Pro** (3.6T), **z-ai/GLM-5.2**
(3.1T — the already-picked primary target, validated), then nemotron-3-ultra, minimax-m3,
step-3.7-flash, **moonshotai/Kimi-K3** (1.4T, 2.8T params) and inclusionai/ling-3.0-flash.
On the local side the mid-2026 story is: the supported Qwen3.5/3.6 + Gemma-4 families still
dominate GGUF downloads (unsloth Qwen/gemma repos at 0.4–1.4M/30d each, plus million-download
community finetunes like DavidAU and HauhauCS that ride the same arches), and the new demand
is (a) **coding/agentic post-trains of those same backbones** (Ornith-1.0, Qwen-AgentWorld,
KAT-Coder-V2.5), (b) **sub-2-bit rebuilds** (prism-ml Bonsai-27B, viral all July), and (c)
big-MoE-on-unified-memory (Hy3/V4-Flash/Laguna on 64–128GB boxes — see the 653-pt
r/LocalLLaMA thread "We need a 80-160B model urgently. The unified memory device market needs
more Models", 2026-06-17).

---

## Q1 — LOCAL (5090-class, 24–32GB): ranked shortlist

### 1. Ornith-1.0 family (deepreinforce-ai) — and with it Qwen-AgentWorld-35B-A3B + KAT-Coder-V2.5-Dev

| | |
|---|---|
| Params | 9B dense; 35B-A3B MoE (also 31B dense on Gemma-4, 397B MoE out of local scope) |
| Arch class | **Bit-for-bit the supported arches.** Ornith-9B `config.json` = `qwen3_5_text`, identical shape to Qwen3.5-9B (32L, 4096h, 16/4 heads, hd256, vocab 248320, 3:1 linear:full hybrid). Ornith-35B = `qwen3_5_moe_text`, identical to Qwen3.6-35B-A3B (40L, 2048h, 256 experts, moe_inter 512). Ornith-31B = Gemma-4-31B backbone (per model card). |
| GGUF / 32GB fit | Official GGUF repos; same quants as the Qwen twins memra already runs (35B-A3B Q4_K_M ≈ 17–21GB resident, 9B Q8_0 ≈ 9GB) |
| New engine work | **≈ zero kernels.** Model onboarding only: tokenizer/added-tokens + chat template + argmax/spec gates. MTP/EAGLE drafts: same seams as Qwen3.5/3.6. |
| Demand evidence | HF 30d: **Ornith-1.0-9B-GGUF 4.59M, Ornith-1.0-35B-GGUF 3.45M downloads** — #2 and #3 of ALL GGUF repos on the Hub (2026-08-01). Ollama `ornith` 323.3K pulls (added ~1 month). Reddit: release post 367 pts (06-25), "Ornith 35B is great so far" 98 pts (06-27). MIT license. Claims SOTA open agentic-coding (Terminal-Bench 2.1, SWE-Bench) vs Qwen3.5/3.6/Gemma4 at equal size. |
| Honest caveat | Download-to-likes ratio (4.6M dl / 591 likes) suggests part of the volume is automated pulls from their agent tooling; Reddit chatter is modest. Even discounted 10x it stays a top-10 GGUF family. This is "actually downloaded, quietly discussed" — the opposite of loud-on-Reddit. |

Same onboarding batch, same zero-kernel cost: **Qwen-AgentWorld-35B-A3B** (Qwen's own agentic
post-train, unsloth GGUF 586K dl/30d, created 06-24) and **KAT-Coder-V2.5-Dev** (Kwaipilot,
open-weights 07-23, 35B-A3B, config identical again; Reddit 07-27: "Kat Coder 2.5 is insane.
Especially considering I ran it at Q4_K_M", 195 pts). One arch, four in-demand coder/agent
models.

### 2. Bonsai-27B — 1-bit + Ternary (prism-ml)

| | |
|---|---|
| Params | 27B dense-equivalent, rebuilt on the **Qwen3.6-27B hybrid backbone** (~75% linear attention — the arch memra already runs as a daily driver) |
| Arch class | Supported hybrid + **new weight formats**: GGUF `Q1_0_g128` (1.125 bpw, ~3.9GB) and `Q2_0_g128` ternary (1.71 bpw, ~7.2GB), binary/ternary weights end-to-end incl. embeddings + LM head; 4-bit KV; DSpark drafter attached (1.34–1.37x claimed) |
| GGUF / 32GB fit | Trivially — 3.9/7.2GB. On a 5090 that means full 262K context resident, or Bonsai + a second model. |
| New engine work | **Medium, well-bounded:** Q1_0_g128/Q2_0_g128 dequant + GEMV/MMQ kernels (packed-consume, never expand to fp16), optional 4-bit KV path, DSpark draft integration (memra's spec seam exists). No new attention/MoE classes. |
| Demand evidence | HF 30d: **Bonsai-27B-gguf 2.51M, Ternary 716K downloads** (created 07-04). Reddit July: 788 pts "runs locally on an iPhone" (07-17), 613 pts WebGPU demo (07-14), 526 pts "first 27B-class model on a phone" (07-14); the 1.7B sibling did 1,140 pts in April. Card claims 95% of FP16 quality at 1.71 bpw across 15 thinking benchmarks. |
| Differentiation | Upstream llama.cpp support is immature: Q1_0 merged but multiple **open** eval bugs July 2026 ("Ternary Bonsai 27B don't run", "fails to load with its DSpark draft", Metal Q2 broken); the good kernels live in prism-ml's fork. A correct, fast sm_120a Q1/Q2_0 path is an ownable niche ("fastest Bonsai on NVIDIA"), and 1–2-bit kernels also feed the Hy3 quant lane. |
| Honest caveat | Bonsai's core audience is laptops/phones; a 5090 owner can afford Q4_K of a 35B. The 5090 pitch is context/coexistence, not necessity. Quality claims are vendor-published, not yet independently reproduced. |

### 3. gpt-oss-20b (OpenAI)

| | |
|---|---|
| Params | 20.9B MoE, 3.6B active (32 experts, top-4), native MXFP4 |
| Arch class | New-to-memra: MXFP4 expert format, attention sinks, alternating SWA/full — no linear-attention, standard GQA otherwise |
| GGUF / 32GB fit | ~12–13GB MXFP4 — easy |
| Demand evidence | Ollama all-time 11.3M pulls (top-17); unsloth GGUF still **551K dl/30d** a year after release (2025-08); Reddit release post 2,026 pts. Steady-state, not growing — no major refresh since 2025 (nothing found in 2026 searches). |
| New engine work | MXFP4 dequant + sinks + SWA interleave — moderate; MXFP4 also appears in Kimi-K3 native quant, so the format has reuse. |
| Verdict | Solid third: big install base, aging demand curve. Do it when MXFP4 is wanted for other reasons. |

### Not local, despite the noise

- **Kimi-K3** — the loudest model of July on r/LocalLLaMA (3,229-pt weights-release post 07-27,
  arena-win posts 2,047+ pts) and **498 Ollama pulls**, because 2.8T params. The perfect
  "loud on Reddit vs actually downloaded" cautionary row.
- **DeepSeek-V4-Flash locally** — real phenomenon ("32 tok/s on AMD Ryzen AI MAX+ 395", 388 pts
  07-28; antirez/deepseek-v4-gguf 664K dl/30d) but that's the 128GB-unified-memory segment,
  not 32GB. Relevant to memra only via the spill lane (below).
- **Hy3 / Laguna-S locally** — same segment ("Tencent-HY3 is the real deal on 128GB!" 289 pts
  07-10; "Hy3 1Bit 89-93 GB" 176 pts). See intersection.

---

## Q2 — SERVING (multi-H100, darklanes): beyond GLM-5.2

GLM-5.2 validation en passant: 3.06T tokens / 73M requests in the 4-day window, 33 endpoints,
$0.76/$2.39 per M — premium tier, crowded, already decided. Not re-litigated.

### 1. DeepSeek-V4-Flash-0731 — the demand monster on GLM-5.2's kernel classes

| | |
|---|---|
| Params / arch | 284B total / 13B active (43L, 256 routed experts top-6 + 1 shared); MLA-family compressed KV (1 KV head × 512d, q_lora 1024) + **DSA** (lightning-indexer `index_topk=512`, sliding 128) + **1 MTP layer**; FP8 native; 1M ctx; MIT |
| Demand | Preview slug: **7.56T tokens and 705M requests in 4 days** — #1 open model by requests on all of OpenRouter, #2 by tokens. 0731 refresh (released 2026-07-31): #2 HF trending in 24h, unsloth GGUF day-one, Ollama `deepseek-v4-flash` tag updated 6h before this fetch, 8 endpoints on day one (22 on the preview). Reddit 0731 post 1,035 pts. |
| Fleet economics | FP8 ≈ 284GB → 4×H100; NVFP4 ≈ ~145GB → **2×H100 replica**. Price point $0.14/$0.28 — commodity tier, wins go to tokens/$; 13B-active MoE + DSA + attached **DSpark draft module** is exactly a throughput-engine's game. |
| New engine work | MLA-class attention (bring-up lane already in flight for GLM-5.2), DSA indexer, MoE arm, MTP (have). **The marginal cost on top of GLM-5.2 is small — GLM-5.2 is `glm_moe_dsa` (MLA kv_lora 512 + DSA index_topk 2048 + MTP); one MLA+DSA investment covers both SKUs.** |
| GLM-5.2 overlap verdict | **Complementary, not overlapping.** Different tier: $0.14/$0.28 vs $0.76/$2.39 (≈5–8x), different buyers (bulk agents/apps vs premium reasoning). The 705M-request profile says high-QPS/small-request traffic, where per-replica capex (2 GPUs vs 8) decides margin. Risk: 22-provider price war; you compete on efficiency, not scarcity. |

### 2. Tencent Hy3 — best demand-to-competition ratio, zero new kernel classes

| | |
|---|---|
| Params / arch | 295B total / 21B active (80L, 192 experts top-8, moe_inter 1536); **plain GQA** (8 KV, hd128), 1 MTP layer; Apache-2.0; 262K ctx |
| Demand | **4.80T tokens / 60.6M requests in 4 days with only 5 OpenRouter endpoints** (GLM-5.2 has 33, V4-Flash 22) — the thinnest supply for top-5 demand on the board. $0.132/$0.528. GGUF locals: vcruz305/Hy3-GGUF 890K + AngelSlim 430K dl/30d. llama.cpp `hy_v3` + MTP merged 2026-07-07. |
| Fleet economics | FP8 ≈ 295GB → 4×H100; NVFP4 ≈ ~150GB → 2×H100 replica. |
| New engine work | **No new kernel class at all**: GQA + MoE arm + MTP + K-quant/NVFP4 — every box ticked by shipped memra code, and the repo already carries a dedicated Hy3 lane (spill paths, REAP mask recovery, five-arm quant study). Bring-up = finish what's started. |
| Risk | Tencent's own churn (Hy3.5 would reset the board); demand is 60% of V4-Flash's. |

**Wildcard third (watch, don't commit): poolside Laguna-S-2.1** — 118B-A8B, GQA + 1:3
full:SWA(512) + softplus router/output gating, OpenMDW license, official FP8/NVFP4/INT4/GGUF +
DFlash draft. **597B tokens in 4 days through a single first-party endpoint** — the largest
unserved-by-third-parties demand found; NVFP4 ≈ 59GB → **one-H100 replica**. New work is
small deltas (softplus gating, 512-SWA interleave) on existing classes. Unproven how much of
that traffic would route to a second provider; check again in 2–4 weeks (Ollama 32.6K pulls
in 4 days says interest is compounding).

Also measured and passed over: **MiMo-V2.5** (top tokens, 8.1T/4d, but 310B with 7 endpoints
already and Xiaomi first-party serving; GQA+MoE so feasible — reconsider if endpoints thin
out), **MiniMax-M3** (2.0T/4d, 9 endpoints, but natively multimodal + new MSA attention class),
**Step-3.7-Flash** (1.7T/4d, 3 endpoints, 3 MTP layers — interesting spec-decode showcase,
China-centric demand), **Ling-3.0-flash** (1.3T/4d but 0 current OR endpoints — data anomaly,
recheck), **Nemotron-3-Ultra-550B** (2.6T/4d, 4 endpoints, 550B-A55B capex-heavy).

---

## Q3 — the intersection (one bring-up, two markets)

**Hy3 is the intersection, with an honest asterisk on what "local" means.**
One bring-up (GQA+MoE+MTP, all shipped kernel classes, research lane already invested) yields:
(a) a darklanes serving SKU in the top-5 demand tier with only 5 competing providers, and (b)
the local >VRAM story — 890K+430K GGUF downloads/30d prove people already run Hy3 on their own
metal; on the 5090 it is exactly the repo's stated conditional win ("SLRU-MoE on >VRAM
model"): 295B-A21B with REAP-50 + mixed 2–4-bit tiers + SLRU spill is the flagship
demonstration of memra's spill pipeline, not a resident model. The mass local Hy3 market is
64–128GB unified-memory boxes; the 32GB 5090 gets the tech demo + the spill benchmark, not
mainstream UX. No candidate found serves both audiences resident-on-32GB *and* fleet-scale —
that model doesn't exist in mid-2026; the closest runner-up is **Laguna-S-2.1** (118B-A8B:
1-GPU serving replica + spill-friendly ~59GB NVFP4 for high-VRAM locals + the "we need
80–160B" segment).

---

## Do NOT support (with reasons)

| Model | Reason |
|---|---|
| GLM-4.7-Flash | Owner-rejected ("nobody wants to run it"). For the record: 1.4M Ollama pulls exist, decision stands. |
| Kimi-K3 (2.8T) | Fleet-infeasible (>1.4TB at 4-bit — beyond 8×H100) and locally irrelevant (498 Ollama pulls). Loudest model of July; unrunnable. KDA+AttnRes would be a whole new attention stack for zero servable product. |
| Kimi-K2.7-Code (1T-A32B) | 4-bit ≈ 500GB+; 16 endpoints already; capex per replica kills the margin story. |
| Inkling / Inkling-Small (Thinking Machines) | 975B-A41B / 276B-A12B, natively multimodal (image+audio encoders) — engine-scope explosion for a text engine; only 2–3 endpoints; custom acceptable-use license. Watch: unsloth GGUF + NVFP4 exist and llama.cpp arch PR is open (11 votes) — revisit if a text-only path emerges and demand holds. |
| MiniMax-M3 | Multimodal + new MSA sparse-attention class; 9 endpoints; demand real (2.0T/4d) but work/moat ratio poor. |
| Nemotron-3-Ultra-550B-A55B | 550B-A55B, 4 endpoints, NVIDIA serves it first-party; nothing a small fleet adds. |
| Solar-Open2-250B | Not on OpenRouter at fetch time; KR/JP-centric demand; NoPE linear-hybrid novel but unproven outside Korea. |
| OCR/vision wave (glm-ocr 6.2M pulls, baidu Unlimited-OCR, PaddleOCR-VL 698K dl/30d) | Real demand, wrong engine — vision input is out of memra's scope. Named here so the pull numbers don't tempt anyone. |
| DiffusionGemma | Block-diffusion decoding — architecturally alien to an AR engine (41 votes on llama.cpp issue notwithstanding). |
| Community finetunes (DavidAU, HauhauCS, Qwythos, ThinkingCap…) | 0.3–1.8M dl/30d each but they ARE the supported arches — support-by-construction once the base runs; never a bring-up target. |

**Watch list:** Qwen3.8 teased (2,733-pt r/LocalLLaMA hype post 07-19) — likely lands on the
qwen3_5 lineage memra already owns; DeepSeek-V4-Pro official release announced "soon" (0731
post); Ling-3.0-flash endpoint anomaly; Laguna-S third-party routing share.

---

## Recommendation

Do the near-free thing first and the compounding thing second: onboard the
**Ornith-1.0-9B/35B + Qwen-AgentWorld + KAT-Coder-V2.5** batch immediately — four of the most
downloaded new local models of the summer, all config-identical to the Qwen3.5/3.6 arches
already on the perf board, so the cost is tokenizer/template plumbing plus gates, and it
converts memra's existing kernel speed into visible "runs the models people actually pull"
breadth. Then spend the real engine budget on **Hy3** as the one bring-up that serves both
markets — top-5 OpenRouter demand (4.8T tok/4d) with only 5 competing providers, Apache-2.0,
zero new kernel classes, and the already-invested spill/REAP/quant lane as the local 5090
story — while treating **DeepSeek-V4-Flash-0731** as the fast-follow serving SKU once
GLM-5.2's MLA+DSA kernels land (shared classes make its marginal cost small, and its 705M
requests/4d is the single largest paid-demand pool a 2×H100-replica fleet can address).
**Bonsai-27B's Q1/Q2_0 kernels** are the local differentiation play (2.5M dl/30d, upstream
CUDA support visibly broken in July) and double as groundwork for Hy3's sub-2-bit tiers —
slot them after the Ornith batch, before or alongside V4-Flash by taste.
