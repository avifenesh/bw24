# NVIDIA free hosted inference (build.nvidia.com / integrate.api.nvidia.com) — ToS verdict for SFT trace generation

Date: 2026-08-02. Lane: `lane/nvidia-tos`. Extends the finetune-sku study
(`REPORT.md` in this directory). Question: does NVIDIA's free API catalog / NIM trial
API permit using model outputs (DeepSeek/Llama/Qwen-class teachers) to build an SFT
trace corpus for training our own model?

**Verdict up front: DENY.** The host-layer NVIDIA API Trial Terms of Service contains
both an Alibaba-4.48-style "any portion of Generated Content" no-transfer clause (§4.2)
and a Kimi/Baidu-style non-compete clause explicitly covering Generated Content (§4.12),
plus an evaluation-only scope (§1.2/1.4). The permissive model-layer license underneath
(MIT for DeepSeek V4) does not rescue it — same host-overrides-model pattern as the
Baidu DENY. Details and exact quotes below.

---

## Receipts added in this directory

| File | What |
|---|---|
| `nvidia-api-trial-tos-20260802.pdf` | NVIDIA API Trial Terms of Service, v. September 19, 2025 (9 pp, the governing doc) |
| `nvidia-open-model-agreement-20260802.html` | NVIDIA Open Model Agreement, v. March 9, 2026 (model-layer for DeepSeek V4 on catalog) |
| `nvidia-community-model-license-20260802.html` | NVIDIA Community Model License, v. April 15, 2025 (model-layer for Qwen-class; anti-distillation §2.1.2) |
| `nvidia-dsv4-flash-modelcard-20260802.md` | DeepSeek-V4-Flash catalog model card with GOVERNING TERMS stack |
| `nvidia-dsv4-pro-modelcard-20260802.md` | DeepSeek-V4-Pro catalog model card (same stack) |
| `nvidia-qwen3-coder-modelcard-20260802.md` | Qwen3-Coder-480B catalog card (Community Model License stack) |
| `nvidia-llama33-modelcard-20260802.md` | Llama 3.3 70B catalog card (Llama Community License stack) |
| `nvidia-nim-faq-20260802.html` | NIM General FAQ (production definition, free-tier scope) |
| `nvidia-forum-40rpm-20260802.html` | Dev-forum staff post acknowledging the 40 RPM free-tier limit, no increases |
| `nvidia-forum-trial-api-use-20260802.html` | Dev-forum staff answer on User/Generated Content data handling |

---

## Q1 — Which agreement governs the free tier?

The governing document is the **NVIDIA API Trial Terms of Service** ("Agreement"),
v. September 19, 2025:
<https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf>
(fetched 2026-08-02; receipt `nvidia-api-trial-tos-20260802.pdf`).

> "The Agreement, as updated from time to time, governs your use of a designated NVIDIA
> API service (an 'API Service') available from a catalog of offerings (the 'NVIDIA API
> Catalog')."

Every catalog model card on `docs.api.nvidia.com` / `build.nvidia.com` carries a
GOVERNING TERMS line pointing at this PDF. Example, DeepSeek-V4-Flash card (fetched
2026-08-02, receipt `nvidia-dsv4-flash-modelcard-20260802.md`):

> "**GOVERNING TERMS:** This trial service is governed by the NVIDIA API Trial Terms of
> Service. Use of this model is governed by the NVIDIA Open Model Agreement. Additional
> Information: MIT."

Not applicable: "AI Foundation Models Community License" (superseded naming; the current
model-layer docs are the Open Model Agreement / Community Model License, see Q3) and the
NVIDIA Cloud Services agreement (that governs paid DGX Cloud, not the free catalog).
Free access itself comes via NVIDIA Developer Program membership (NIM FAQ, receipt
`nvidia-nim-faq-20260802.html`): "Members of the NVIDIA Developer Program have free
access to NIM API endpoints for prototyping."

## Q2 — Output-use terms (the decisive clauses)

Three clauses in the Trial ToS kill trace-corpus use, independently of each other.

**(a) Evaluation-only scope — §1.2 Trial Access Rights:**

> "Subject to this Agreement, NVIDIA will provide you access to the API Service for
> limited trial purposes only and without use of the API Service or Generated Content
> in production."

and §1.4 Trial Terms and Credits:

> "Unless you purchase a Subscription from NVIDIA or a Service Provider (as applicable),
> you may only use the API Service for internal testing and evaluation purposes, not in
> production."

Bulk generation of a training corpus for a model we ship is not "internal testing and
evaluation"; the corpus and the trained weights are production artifacts of the
Generated Content.

**(b) No-transfer of Generated Content — §4.2** (the Alibaba Art-4.48 analog):

> "Except as indicated in the Section 1.2 ('Trial Access Rights') above, you may not
> copy, sell, rent, sublicense, transfer or distribute or make available to others any
> portion of the API Service or Generated Content."

An SFT corpus is copied/stored Generated Content, and weights distilled from it are
distributed derivatives — "any portion" language, same shape that produced the Alibaba
BLOCKED verdict.

**(c) Non-compete over Generated Content — §4.12** (the Kimi/Baidu analog):

> "You will not use (or allow others to use) the API Service including Generated Content
> to develop or improve products or services that compete with the API Service."

The API Service is hosted LLM inference. A darklanes-served fine-tuned model is a
product that competes with hosted LLM inference, built using Generated Content. This is
a direct anti-distillation/non-compete hit.

Ownership does not help: §6.3 says "as between you and NVIDIA or the relevant Service
Provider (as applicable), you own all Generated Content that may result from your use of
an API Service" — but ownership is granted "[u]nless otherwise stated in an accompanying
license" and remains subject to the §4 use restrictions. Owning the bytes doesn't license
the training use.

**Data flow the other direction — NVIDIA claims YOUR content.** §3.3:

> "NVIDIA will collect the following data, without identifying specific users, to operate
> and improve the API Services and other products and services: (i) session metrics ...
> (ii) error logs ... (iii) your feedback ... and (iv) User Content and Generated Content
> to improve NVIDIA products and services, including AI models."

So the free tier is a one-way valve: NVIDIA reserves the right to train its models on
your prompts and the generated outputs (§3.3(iv)), while §4.12 forbids you from training
on them. §2.3's "NVIDIA will not store or use User Content or Generated Content at the
end of each API Service session" is qualified by "unless expressly disclosed to you for
an API Service" — and §3.3(iv) is exactly such a disclosure. NVIDIA forum staff, asked
directly whether prompts/outputs are retained for training, only pointed back at
Sections 2 and 3 rather than denying it (receipt `nvidia-forum-trial-api-use-20260802.html`).
Do not send calibration prompts or anything lane-sensitive through this endpoint.

## Q3 — Per-model layering

Yes, NVIDIA layers like OpenRouter does, with an explicit flow-down clause. Trial ToS §9:

> "The API Service may come bundled with, or otherwise include or be distributed with,
> components with separate legal notices or terms ... Without limiting the foregoing,
> you are responsible for your compliance with third-party AI model licenses."

Each model card declares its stack. Measured 2026-08-02:

| Model on catalog | Layer 1 (host) | Layer 2 (model) | Layer 3 (upstream) |
|---|---|---|---|
| deepseek-ai/deepseek-v4-flash, -v4-pro | API Trial ToS | **NVIDIA Open Model Agreement** (v. 2026-03-09) | MIT |
| qwen/qwen3-coder-480b-a35b-instruct | API Trial ToS | **NVIDIA Community Model License** (v. 2025-04-15) | Apache 2.0 |
| meta/llama-3.3-70b-instruct | API Trial ToS | Llama 3.3 Community License | — |

Two model-layer gotchas worth recording:

- The **NVIDIA Open Model Agreement** is genuinely permissive: "NVIDIA does not claim
  ownership to any outputs generated using the Works or Derivative Works" and grants a
  "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license"
  incl. Derivative Works. If it were the only document, DeepSeek V4 via NVIDIA would be
  clean — the block comes entirely from the host-layer Trial ToS.
- The **NVIDIA Community Model License** (governing Qwen-class and most Nemotron on the
  catalog) has its own anti-distillation clause, §2.1.2: "[You may not] Use the NVIDIA
  Models, Derivative Models or any output or results of them to develop or improve any
  other AI models (excluding NVIDIA Models or Derivative Models), unless approved by
  NVIDIA in writing (email approval being sufficient)." And §1.4(c) defines Derivative
  Models to include "distillation methods that use intermediate data representations or
  methods based on the generation of synthetic data by the NVIDIA Models for training
  the other model." So Qwen-class via NVIDIA is double-blocked (host + model layer),
  while DeepSeek-class is blocked at the host layer only.

Pattern match to prior verdicts: this is the **Baidu-host shape** — a permissive model
license underneath a hostile host wrapper. Host terms govern the outputs produced by
that host's service; the verdict is set by the strictest layer.

## Q4 — Rate limits / quotas

NVIDIA publishes no per-token price and no guaranteed quota for the free tier.
What is documented:

- **~40 requests/minute baseline**, model- and traffic-dependent, shown per-account in
  the build.nvidia.com dashboard. NVIDIA staff (MarkusHoHo, Developer Forums, receipt
  `nvidia-forum-40rpm-20260802.html`): "I can assure you that the team is aware of the
  implications of the 40 rpm rate limit." Same thread, moderator boilerplate: the limit
  is "dependent on model, use-case and the amount of current overall traffic using the
  same access. There is no official way to circumvent this rate limit or to receive a
  rate limit increase on that same tier."
- Historically 1,000 trial credits (1 credit ≈ 1 request); NVIDIA has since moved the
  free tier to rate-limit-based rather than credit-based (forum threads, 2026).
- No free-tier increases, ever, via any channel; the sanctioned paths past the ceiling
  are self-hosting the downloadable NIM (free for dev/test on up to 16 GPUs, you pay the
  hardware) or NVIDIA AI Enterprise (from $4,500/GPU/yr, or ~$1/GPU/hr license on top of
  cloud instance cost; 90-day free eval).
- NIM FAQ production line (receipt `nvidia-nim-faq-20260802.html`): "Production use
  involves any use of NIM for purposes other than development, testing, research or
  evaluation such as conducting business transactions and any non-testing activity
  including activity serving real end-users."

Mechanical viability, stated honestly: 40 RPM ≈ 57,600 requests/day theoretical ceiling
per account if hammered 24/7 with backoff. A 10-50k kept-trace corpus at tens of
requests per multi-turn trace would take weeks but is not mechanically impossible — the
quota alone is NOT the blocker. The terms are. (And sustained 24/7 batch extraction is
itself outside the "internal testing and evaluation" scope of §1.4, so running it would
compound the violation, with §11.2 auto-termination as the mildest consequence.)

## Q5 — Verdict

**DENY** for SFT trace-corpus generation. The free build.nvidia.com / NIM trial API is
governed by the NVIDIA API Trial Terms of Service (v. 2025-09-19), which (a) scopes the
free tier to internal testing and evaluation only (§1.2, §1.4), (b) forbids copying or
making available to others "any portion of ... Generated Content" (§4.2), and (c)
forbids using "the API Service including Generated Content to develop or improve
products or services that compete with the API Service" (§4.12) — a fine-tuned model we
serve is exactly that. The permissive model-layer terms for DeepSeek V4 (NVIDIA Open
Model Agreement + MIT) do not override the host layer; for Qwen-class models the
NVIDIA Community Model License adds a second, explicit anti-distillation block
(§2.1.2 + §1.4(c)). Bonus hazard: §3.3(iv) lets NVIDIA use your prompts AND the
generated outputs to improve its own AI models — a one-way valve against us.

Slotting vs the verified pin: it does not slot. Free only beats the ~$0.017/kept-trace
DeepSeek first-party floor if terms AND quotas clear; here the terms fail on three
independent clauses, so the quota analysis is moot. The scoreboard stands:
**DeepSeek first-party (§4.2 distillation-permissive) remains the pin**, with
novita/deepinfra/fireworks via OpenRouter as the clean alternates. NVIDIA's free
catalog stays useful for what its terms actually permit: interactive model evaluation
and prompt prototyping before buying capacity elsewhere. Fine to use it to *pick* the
teacher; not to *milk* the teacher.

Calibration table update:

| Provider | Verdict | Decisive clause |
|---|---|---|
| Kimi Code sub | BLOCKED | anti-distillation |
| Alibaba coding plan | BLOCKED | Art 4.48 "any Output" + interactive-only |
| DeepSeek first-party | ALLOWS | §4.2 explicitly permits distillation |
| OpenRouter | CLEAN | passthrough, no output-use restriction |
| Baidu host | DENY | 3A.5 non-compete over permissive models |
| **NVIDIA free API catalog** | **DENY** | Trial ToS §4.2 + §4.12 + §1.2/1.4 eval-only |

Sources index (all fetched 2026-08-02):

- NVIDIA API Trial Terms of Service (v. 2025-09-19): <https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf>
- NVIDIA Open Model Agreement (v. 2026-03-09): <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/>
- NVIDIA Community Model License (v. 2025-04-15): <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/>
- DeepSeek-V4-Flash catalog card: <https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash>
- Qwen3-Coder-480B catalog card: <https://docs.api.nvidia.com/nim/reference/qwen-qwen3-coder-480b-a35b-instruct>
- NIM General FAQ (production definition, free-tier scope): <https://docs.api.nvidia.com/nim/docs/product>
- Dev forum, 40 RPM staff acknowledgment: <https://forums.developer.nvidia.com/t/api-rate-limit-increase-is-not-granted-by-requesting-it-here/368420>
- Dev forum, trial data-handling staff answer: <https://forums.developer.nvidia.com/t/clarification-on-trial-api-use/334275>
