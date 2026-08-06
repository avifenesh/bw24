# Two-card model assessment — what darklanes serves on 2x RTX PRO 6000 (192 GB)

Date: 2026-08-06. Lane `lane/model-192gb` (from `restructure/public-split` @ 4cbf5e39).
CPU-only research — no GPU runs in this lane. Extends `research/model-96gb-20260806/ASSESSMENT.md`
(the one-card verdicts; method and demand receipts reused, refreshed only where stale).

Owner question (2026-08-06, verbatim): *"what model is the most plausible on two 6000pro?"*

Frame: owned trajectory = boxes of 2x RTX PRO 6000 WS 96 GB. **Standing doctrine: serve on
card 1, lab (research/training) on card 2.** Every 2-card SKU in this file therefore carries a
line item the one-card assessment never had: the doctrine cost — what the owner gives up when
the lab card serves. Fit alone does not answer the question.

Evidence base: the 96 GB assessment + its fresh 08-06 receipts, the P2P verdict
(`research/p2p-5090-validation-20260803/NOTE.md` — PRO 6000 has native stock-driver P2P,
verified both directions), the PP receipts (`research/box-phase1-20260802/SUMMARY.md` — M1
PP-2 DONE, six gates bit-identical; `research/m2-pp8-20260802/RESULTS.md` — PP-N 28/28
bit-identical, deferred-pipelined 1.87x, and its standing quarantines), the bounce/boundary
arithmetic (`research/hw-growth-rethink-20260803/ASSESSMENT.md` §1 — ~0.3–2%/tick class),
2-WK-box thermal/stacking verdict (`research/pro6000-stacking-20260804/ASSESSMENT.md`),
Hy3 receipts (`docs/HY3-SPILL.md`, `research/hy3-hopper-20260801/`,
`research/hy3-spec-20260802/`, `research/hy3-accept-profile-20260802/`), sku-repick
(`research/sku-repick-20260802/REPORT.md`), PRO 6000 prod receipts
(`research/pro6000-prod-20260804/pro6000wk-runpod.jsonl`). Fresh pulls 2026-08-06 in `raw/`
here: HF GGUF blob sizes for Qwen3.5-122B (all quants), Hy3 HF license + config + safetensors
size, OR endpoints hy3 + step-3.7-flash.

---

## 0. What the second card actually adds (engine truth, receipts first)

- **Transport is NOT the problem on this pair.** PRO 6000 advertises P2P on the stock driver
  (P2P NOTE §5: `can_device_access_peer` True/True on 2x PRO 6000, driver 580.95; our own G7e
  experience). `pp.rs` already ships the `cudaMemcpyPeerAsync` boundary — the host-staged
  bounce arm (the 5090 gap, issue #67 class) is **not needed** here. Boundary payload is
  `n_embd` f32 only: 122B-class = 12 KB/token; even at 5090-bounce worst-case pricing that
  was 0.3–2%/tick (hw-growth §1.3); over native PCIe P2P on this pair it is noise.
- **PP-2 correctness is receipted, PP-2 *serving* is not built.** M1/M2 gates: bit-identical
  48-step logits across devices, serial and pipelined, N=2..8 (box-phase1, m2-pp8). But the
  pp door lives in the **eager decode loop only** — `crates/memra-engine/src/pp.rs` header:
  batch/dc/graph/spec loops are explicitly unwired (`warn_unwired_once`). No microbatch
  (M2's named next increment), no graph capture, no spec-across-the-split, and the
  deferred-pipelined arm (the 1.87x) is **not serving-default-cleared** (~0.5% cross-device
  flake, 1/~190 runs, root-cause open). PP-2 gates have also never run on a PRO 6000 pair
  (H100 NVSwitch receipts only) — a fresh on-target battery is mandatory before any listing.
- **Consequence for every PP-2 candidate below:** the honest engineering cost is not the
  transport (days-to-zero) — it is wiring PP through the serving loops (batched tick, graph,
  admission, drafter) + gates: **weeks-class**, and it is the same bill whichever PP-2 model
  is picked. Two *independent single-card SKUs* on the two cards cost **zero** new engine work
  (the supervisor already runs one model per replica; placement is `CUDA_VISIBLE_DEVICES`).

## 1. VRAM/KV arithmetic per candidate (weights + KV, shown; method = 96 GB assessment §1)

KV default q8_0/q5_1 ≈ 0.906 B/elem; working overhead 6 GB/card. Fresh byte receipts
(`raw/hf-gguf-unsloth-qwen3p5-122b-20260806.json`): 122B Q8_0 **129.9 GB**, Q6_K 101.0,
UD-Q6_K_XL 112.4, UD-IQ4_XS 60.2. Hy3 config receipt (`raw/hf-config-hy3-20260806.json`):
80 layers ALL full-attention GQA, 8 kv-heads × 128 head_dim.

| Candidate | Weights | KV/token | 128k session | Fits 192 GB? | KV left → concurrent 128k |
|---|---|---|---|---|---|
| **122B-A10B Q8_0, PP-2** | 129.9 GB (≈65/card) | 12 full-attn × 2 kvh × 256 hd = 10.9 KB | 1.46 GB (+~0.15 GB GDN state) | **yes** — the 8-bit form exists at 192 | ~50 GB → **~31** |
| 122B-A10B UD-Q6_K_XL, PP-2 | 112.4 GB | same | same | yes | ~68 GB → ~42 |
| **Hy3 295B-A21B NVFP4 resident, PP-2** | ~157 GB (295B × 4.25 bpw; BF16 source 597.6 GB receipt) | 80 full × 8 kvh × 128 hd = **145 KB(!)** | **19.5 GB** | weights yes | ~23 GB → **~1** (or ~4-5 × 32k) |
| Hy3 Layer103.5 overlay resident, PP-2 | ~104 GB logical (release receipt) | same 145 KB | 19.5 GB | yes | ~76 GB → ~3-4 |
| **122B IQ4 (card 1) + q27-Q8_0+Q8RP (card 2)** | 60.2 + 53.2 GB | 10.9 KB / 29.0 KB | 1.46 / 3.7 GB | yes — two independent cards | 122B ~21× 128k; q27 ~9× 128k |
| **Two q27 Q8_0+Q8RP instances** | 53.2 × 2 | 29.0 KB | 3.7 GB | yes | ~9 per card |
| Step-3.7-Flash IQ4_XS, PP-2 (adjacent candidate) | 95.3 GB | SWA-512 on ~3/4 layers → small global KV | few GB | yes, huge KV room | tens |
| Qwen3.8-Max 2.4T | ~1,275 GB at 4-bit (2.4T × 4.25 bpw) | — | — | **NO — out by 6.6x** | verified out of envelope |

Notes:
- The 128k×N KV profile is where the 122B and Hy3 diverge violently: the 122B's GDN hybrid
  (36 linear + 12 full-attn layers, config receipt) gives **10.9 KB/token**; Hy3 is
  full-attention on all 80 layers → **145 KB/token — 13.3x worse**. At 192 GB that is the
  dense-70B kill shape all over again: Hy3-NVFP4-resident holds ONE 128k session. The card
  pair that wins on batch (Q8RP +57% came from c=16/32) can never reach batch on Hy3.
- PP-2 split shape for the 122B: 24+24 layers puts exactly 6 full-attn layers per card
  (interval-4 pattern, verified from `layer_types`) — KV load balances naturally; boundary
  = 12 KB/token f32.
- Qwen3.8: the 96 GB assessment resolved the wildcard — announced open list is Max (2.4T,
  ~95B active) + 27B, **no mid-size**. Max is out at any quant (arithmetic above). Day-one
  3.8 leverage at 192 GB is exactly the same as at 96: the 27B swap via the standing runbook.
  If a 3.8-gen ~120B-class ships later, it inherits the 122B slot and runbook unchanged.

## 2. Candidate verdicts

### C1 — Qwen3.5-122B-A10B Q8_0 over PP-2: **the headline answer, gated — the quant exception dies in principle, not today**

- **Fit**: yes, comfortably — 129.9 GB + ~31 concurrent 128k sessions. This is the only way
  any 8-bit form of the 122B exists on owned silicon; the one-card assessment's conflict
  (prod=8bit vs only-fits-at-4bit) is resolved by hardware rather than by an exception.
- **Decode class (honest projection)**: A10B at Q8 ≈ 10.6 GB/token reads → single-card
  bandwidth class ~67 tok/s c=1 (0.40 × 1790 / 10.6, sku-repick method). Serial PP-2 does
  not add single-stream speed (each stage idles half the tick); the 1.87x deferred-pipelined
  prize exists on H100 receipts but is quarantined from serving. Batch aggregate is where the
  pair should pay — and the batched tick is exactly what is not PP-wired.
- **Spec/MTP across the split**: MTP head confirmed (`mtp_num_hidden_layers: 1`), head lives
  with the last stage, draft feedback re-enters stage 0 — the spec loop is not pp-wired;
  drafter-across-PP is new engineering inside the same weeks-class bill.
- **Demand/$**: unchanged from 96 GB assessment — $1.6K/day pool, 5 providers, highest held
  out-price of any open fit ($2.08/M, held since ≥08-02; fresh 08-06 receipt). At 8-bit we'd
  hold the quality corner of the page (incumbents: fp8 ×3, fp4 ×1, bf16 ×1): a Q8_0/FP8
  listing filters into the `quantizations:["fp8","bf16"]` segment that a 4-bit listing cedes.
- **Engineering cost**: PP-2 serve wiring (microbatch + batched tick + graph + admission +
  drafter across split) + on-target gate battery — **weeks-class**, the biggest item in this
  file. Until it lands, the 122B-at-192 story is eager-decode-only: not listable.
- **Doctrine cost: total.** Both cards serve one SKU; the lab card is gone entirely — worse,
  a PP-2 replica is one failure domain across both cards (a lab preemption isn't even
  possible without dropping the SKU).

### C2 — Hy3-class resident at 192 GB: **arithmetically revived, and REJECT again — the kill just moves from spill to KV**

- The 96 GB kill was the spill floor (2.48 tok/s measured, hy3-hopper baseline; 5.13 tok/s
  on the 24 GB local profile). At 192 GB the bank is resident — NVFP4 full bank ~157 GB fits,
  and the served Layer103.5 overlay (~104 GB logical) fits with room. Decode class projected
  ~60-100 tok/s (21B active) — endpoint-grade at c=1, a real revival of the interactive
  number.
- **But the KV table is fatal for the serving shape**: 145 KB/token → one 128k session on
  the full-NVFP4 fit, ~3-4 on the overlay fit. Hy3's OR page sells 262k context; we'd list
  the big-context model we can hold one big-context session of. Same hardware-shaped-wrong
  verdict as the dense-70B row in the 96 GB file, before demand is consulted.
- **Drafter path is weak by receipt**: nextn=1 head chains zero draft tokens (K=1 forever);
  acceptance 44-75% on real content → spec ceiling ≈ +8.5%..+48%, and the PP-2/resident
  regime has never run its own K=1 sweep (hy3-spec SUMMARY's own required gate).
- **Market (fresh 08-06 receipt, `raw/or-endpoints-hy3-20260806.json`)**: now **6** endpoints
  (Baidu joined since 08-02), floor eff. ~$0.118/$0.49 (GMICloud bf16, discounted), Tencent
  first-party fp8 at $0.132/$0.528. More incumbents than any candidate here, at a fifth of
  the 122B's out-price. License apache-2.0 (fresh HF receipt).
- **Engineering**: the overlay/SLRU stack has no PP-2 arm at all (safetensors-overlay
  sharding across two cards is unbuilt, on top of the same serve-wiring bill).
- Verdict: the research lane stays parked; 192 GB does not buy Hy3 a listing. If a resident
  Hy3-class experiment is ever wanted, it is a *lab-card research run*, which is exactly
  what the doctrine already permits without any of this.

### C3 — 122B IQ4 on card 1 + q27 on card 2 (or both on card 1): **the incumbent, and it survives — with a sharper split**

Two doctrine-distinct sub-shapes, both zero-new-engineering:

- **(3a) Doctrine-preserving (the 96 GB assessment's #2+#3, unchanged):** card 1 =
  122B-IQ4 (60.2 GB) + q27-NVFP4 daily (17 GB) + ~13 GB KV; card 2 = lab. Premium tier +
  daily tier on one card, lab intact. Cost: q27 gives up the Q8RP mirror (+57% at c=16/32)
  and the 122B serves at 4-bit (the quant-exception owner call stands).
- **(3b) Doctrine-spending:** card 1 = q27-Q8_0+Q8RP (53.2 GB, the full measured lever);
  card 2 = 122B-IQ4 (60.2 GB) + ~30 GB KV. Both SKUs get their best single-card config;
  replica granularity stays 1 card (a card can be pulled back to lab in minutes, unlike PP-2).
  Cost: the lab card, for ~$0.9-1.2/hr-class extra gross at 30% utilization (both pages are
  thin pools; receipts in the 96 GB file §3).

### C4 — Two q27 instances: **REJECT as a target config**

Pure capacity scaling of a page whose pool (~$2.9K/day) does not saturate one card at our
capture rates. Spends the lab card for redundancy nobody has asked for. Config-only if ever
needed (supervisor placement), so it needs no slot in the plan — it is what you do for an
afternoon traffic spike, not a model decision.

### C5 — The 3.8 drop: **adds nothing at 192 that it didn't add at 96**

Max is out by 6.6x at 4-bit (verified above). No mid-size announced. 3.8-27B swaps into
whatever q27 slot exists, day one, runbook standing. A later 3.8-gen 122B-class successor
inherits C1/C3's slot and this file's arithmetic.

### Adjacent candidate the 192 envelope newly opens (flagged, not ranked #1): Step-3.7-Flash PP-2

The sku-repick flagship (95.3 GB IQ4_XS — zero KV room on 96 GB, killed there) fits PP-2
with ~80 GB of KV/headroom; UD-Q6_K_XL-class (~162 GB est.) also fits. Fresh 08-06 receipt
(`raw/or-endpoints-step37flash-20260806.json`): **still 3 endpoints, all still holding
$0.20/$1.15, 99-100% up** — the held-price no-price-war structure survived four more days.
Pool $89.3K/day = **56x the 122B's**; official MTP head; SWA-keeps-KV-small = the right
batch shape for this pair. Costs: 2.5-4 week bring-up (new tokenizer/template/SWA-3:1
mapping) ON TOP of the same PP-2 serve-wiring bill, 93.5:1 prefill-heavy traffic rides our
least-receipted stage, and the full doctrine cost. It is the strongest *business* case for
ever spending both cards on one model — stronger than the 122B's — and it should be the
model re-evaluated the day the PP-2 serving bill is actually paid.

## 3. Ranked recommendation

1. **Qwen3.5-122B-A10B is the most plausible model on two PRO 6000 — but enter it through
   the doctrine-preserving config (C3a), not PP-2.** Card 1 = 122B-IQ4 + q27 daily, card 2 =
   lab. ~1 week bring-up, zero new kernels, Apache-2.0, MTP drafter path standard, highest
   held out-price of any open fit. This is the 96 GB assessment's answer, and 192 GB does
   not overturn it — it adds options behind gates.
2. **The Q8_0-over-PP-2 upgrade (C1) is the standing successor config, gated on the PP-2
   serving bill** (microbatch + serve-loop wiring + drafter-across-split + on-target gate
   battery, weeks-class) **and on the owner accepting the doctrine cost.** When both gates
   clear, the 122B quant exception dies and the listing moves to the fp8/bf16 quality
   segment of the page.
3. **C3b (one SKU per card)** — the intermediate doctrine-spend: zero engineering, both
   SKUs at their best single-card config, reversible in minutes. Take it only when metered
   demand on the C3a card actually saturates.
4. **Step-3.7-Flash PP-2** — the flagged fast-follow: re-price the moment the PP-2 serving
   bill is paid for any reason; its pool/competition structure (held $1.15 out, 3 endpoints)
   is the best big-SKU business case in this file.
5. **Hy3-class resident — REJECT for serving at 192** (KV-starved: 145 KB/token → ~1×128k
   session; K=1-only drafter; 6 incumbents at a $0.49-class floor). Research lane stays
   parked; any resident experiment is a lab-card run under the existing doctrine.
6. **Two q27 instances — not a plan** (config-only spike response).
7. **Qwen3.8-Max — out of envelope, verified** (~1.28 TB at 4-bit vs 192 GB).

## 4. The doctrine tradeoff, stated explicitly

The standing doctrine — serve on card 1, lab on card 2 — is what makes darklanes' operating
model work: the lab card is where the drafter corpus runs, the FP8-ST and NVFP4-strict lanes
gate, the 3.8 day-one bring-up rehearses, and the tuning campaign that moves the boards
lives; under the operating model, that research *is* the product, and every 96 GB assessment
receipt was produced by exactly that kind of capacity. Spending the lab card on serving buys,
at today's pools and 30% utilization, roughly one more $0.9-1.2/hr-class thin-pool endpoint
(C3b) or one premium 8-bit listing bound to a two-card failure domain (C1) — and it converts
the box from "one SKU + full-speed research" into "two SKUs + research happens on rentals or
not at all." PP-2 is strictly worse than one-SKU-per-card on this axis: a co-resident lab job
on a PP-2 stage isn't a slowdown, it's an SLA breach, and reclaiming the card means dropping
the SKU. The honest framing for the owner: the 2-card SKU question is not "what fits in
192 GB" but "is any second listing worth more than the lab card's research output" — and at
the current pool sizes ($1.6K/day for the 122B page, $2.9K for q27) the answer the receipts
support is **not yet**; the number that would flip it is Step-class ($89K/day pool) demand
actually captured, or a saturated C3a card.

## 5. Owner decisions vs auto-decidable

**Owner must decide:**
1. **The doctrine call itself** — whether any config that spends the lab card (C1, C3b,
   Step PP-2) is on the table at current pool sizes. This file recommends: not yet.
2. **The 122B quant exception** (unchanged from the 96 GB file) — it gates C3a's listing
   today. Note the new fact: the exception is now *temporary by construction* — the Q8_0
   PP-2 path (C1) retires it once the PP-2 serving bill is paid, which may make approving
   the interim 4-bit listing easier.
3. **Whether to schedule the PP-2 serving bill** (microbatch + serve-loop wiring + drafter
   split + PRO-pair gate battery) now vs after the 3.8 drop settles — it unlocks C1 and the
   Step re-price, and it is the same bill for both.
4. (Standing) Q8_0-bridge vs FP8-ST for 3.8 day-one — unchanged from `8bit-decision-20260803`.

**Auto-decidable (no owner input needed):**
- 122B bring-up preparation (tokenizer/template diff, gate scripts, drafter recipe) —
  already auto-decidable in the 96 GB file; unchanged.
- PP-2 gate-battery *scripts* for a PRO 6000 pair (port of run-m2-gates.sh; runs whenever
  a pair is available) — preparation, not a default flip.
- Hy3-at-192: no action (reject is data-driven; the research lane's parked state is the
  owner's existing call).

## Source index

Fresh (2026-08-06, receipts in `raw/` here): HF blob sizes unsloth/Qwen3.5-122B-A10B-GGUF
(Q8_0 129.9 GB et al.); HF tencent/Hy3 license (apache-2.0) + config.json (80L full-attn GQA
8 kvh × 128 hd) + safetensors total (597.6 GB); OR endpoints tencent/hy3 (6 endpoints,
Baidu new) + stepfun/step-3.7-flash (3 endpoints, $0.20/$1.15 held).
Committed: `research/model-96gb-20260806/` (the one-card verdicts + its raw/),
`research/p2p-5090-validation-20260803/NOTE.md`, `research/box-phase1-20260802/SUMMARY.md`,
`research/m2-pp8-20260802/RESULTS.md`, `research/hw-growth-rethink-20260803/ASSESSMENT.md`,
`research/pro6000-stacking-20260804/ASSESSMENT.md`, `research/sku-repick-20260802/REPORT.md`,
`research/or-provider-20260802/REPORT.md`, `research/hy3-hopper-20260801/baseline.md`,
`research/hy3-spec-20260802/SUMMARY.md`, `research/hy3-accept-profile-20260802/SUMMARY.md`,
`docs/HY3-SPILL.md`, `research/pro6000-prod-20260804/pro6000wk-runpod.jsonl`,
`research/kv-compress-20260802/REPORT.md`, `crates/memra-engine/src/pp.rs` (serve-wiring gap).
