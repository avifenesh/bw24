# RTX PRO 6000 Blackwell: Workstation vs Max-Q vs Server — the STACKING assessment

Date: 2026-08-04. All web sources fetched 2026-08-04 unless noted.
Owner hypothesis under test (verbatim): "workstation can get x1.4 performance because of w600
limit as compare to w300 on maxq, but problematic when starting to stack. two are fine, but if
scaling get to 4-6 i might need to replace it all to server/maxq edition to not blow the heat
on each other."

Scope: everything the rented-WK-pod measurement lane (perf-vs-watts at 300/450/600W on our
workloads) cannot answer — thermals-in-a-box, chassis reality, market evidence. This file does
NOT contain our own numbers; the parallel lane owns those.

---

## TL;DR verdict matrix

| Fleet size | Verdict | One-line evidence |
|---|---|---|
| **2 cards** | **WK (600W)** — buy it | NVIDIA positions WK for "one or two" class use; field reports of 2x WK stable (70°C top card); WK is *cheaper* than Max-Q street right now ($12.1-13.3k vs $15.4-16.4k eBay); decode gap vs Max-Q ≈ 0 anyway, prefill 1.3-1.5x free upside |
| **4 cards, one chassis** | **Max-Q** (air) or **WK derated + open-frame/spacing** or liquid | No integrator ships 4x WK air-cooled; even 4x *Max-Q* air needed a custom Exxact cooling solution (3x was the prior air limit); 4x600W ≈ 3.5-4kW wall = over one 230V/16A circuit |
| **4 cards, two boxes (2+2)** | **WK — no problem at all** | Each 2x WK box is the proven config; ~1.1kW serving load per box, one circuit each; this is the escape hatch that makes buy-WK-now safe |
| **6 cards** | **Max-Q or Server Ed.** (or liquid-converted WK) | 6x600W = 3.6kW GPUs alone → colo-only power; WK's 5.4"-tall body doesn't fit standard GPU-server chassis (Server/Max-Q are 4.4" standard height); Server Ed. needs ducted chassis airflow = colo chassis anyway |

**Bottom line on the owner's hypothesis: directionally correct, but the danger is overstated
and the 1.4x is workload-dependent.** Two WK are fine (confirmed). 4-6 in ONE chassis does
force Max-Q/Server/liquid (confirmed). But (a) scaling by *boxes of 2* keeps WK viable far
longer than the hypothesis assumes, (b) WK derates to a functional Max-Q at the same watts
with a *bigger* cooler, and (c) for bandwidth-bound decode the 1.4x is actually ~1.0x, so the
"WK premium" you'd be walking away from at the 4-6 stage is small for our serving mix.

---

## Q1. Cooler geometry + official multi-GPU language

**NVIDIA family page** (nvidia.com rtx-pro-6000-family, fetched 2026-08-04) — the official
positioning, verbatim:

- **Workstation Edition**: "Engineered for maximum performance in **single-GPU workstations**…
  Ideal for professionals prioritizing **single-GPU throughput on desktops**." Thermal:
  "Double flow-through". 600W. Form factor **5.4" (H) x 12" (L)** dual slot.
- **Max-Q**: "Optimized for **dense workstation configurations (up to four GPUs)**…" Thermal:
  "Active" (blower). 300W. Form factor **4.4" (H) x 10.5" (L)** dual slot.
- **Server Edition**: "Designed for **multi-GPU server deployments requiring passive cooling**."
  400-600W (configurable). Passive. 4.4" x 10.5".

So NVIDIA's own ceiling language exists: WK = single-GPU positioning, Max-Q = "up to four
GPUs" explicit. No official "WK not recommended above N" sentence found, but the single-GPU
positioning plus retailer amplification (Central Computer, 2026-01-22: "it is not recommended
to use the RTX Pro 6000 Blackwell Workstation Edition in multi-GPU configurations, since the
cards will simply blow hot air on each other") is the industry read. Central Computer
(2026-03-23) draws the line at: 1-2 GPUs → WK; 2-4 GPUs → Max-Q; racks → Server.

**Geometry mechanics** (Puget Systems, 2025-07-24, Max-Q vs WK article):

- WK flow-through (5090-FE-style): pulls air from below, exhausts **out the top of the card,
  into the case** — "if there are multiple GPUs in the system, then the lower card will push
  hot air into the intake of the upper card."
- Max-Q blower: intakes inside the case, **exhausts out the rear bracket** — "heat generated
  by one card won't (significantly) affect other cards."
- **Even Max-Q should not be stacked flush**: 96GB puts GDDR7 chips on the PCB backside,
  cooled only by the backplate, which needs airflow. Puget's recommendation: **one empty slot
  between cards** — for Max-Q. (Exxact's build guidance repeats the 1-slot spacing, cited in
  r/LocalLLM 2026-04-25.)
- Physical slot math this implies: a dual-slot card + 1 gap = 3 slots per card. Four cards
  need ~11 slots of board length — impossible on any ATX/WRX90 board (7 slots). 4-with-gaps
  means open frame + risers, or a chassis engineered around it. This binds *both* editions,
  but Max-Q survives flush-stacking better (blower) with strong forced intake; WK does not.
- WK's 5.4" height is over standard full-height (4.4"); it does not fit standard 4U GPU-server
  card cages. Server Ed. and Max-Q are standard height. **This, not watts, is the hard
  stacking wall for WK**: you cannot migrate WK cards into a proper GPU server chassis later.

## Q2. Field evidence: what actually ships and what actually runs

**Vendors:**

| Vendor | 4-6x PRO 6000 offering | Edition | Cooling | Source/date |
|---|---|---|---|---|
| Puget Systems | Multi-GPU workstation (debut SIGGRAPH 2025) | **Max-Q** | Air, 1-slot gaps | pugetsystems.com labs 2025-07-24; awn.com 2025-08 |
| Puget Systems | Dual 600W-class GPU builds | (5090/WK class) | Pushed to **rackmount** lineup, "installed in server rooms" | pugetsystems.com blog 2025-10-28 |
| Exxact | Valence 4x Max-Q workstation — **first validated air-cooled 4x Max-Q anywhere** | **Max-Q** | Air, custom cooling solution, 2500W PSU, TR PRO 9995WX | exxactcorp.com blog 2025-08-18, updated 2026-03-12 |
| Comino | Grando 4U: up to **8x PRO 6000 at full 600W TDP** (server) / up to 6 (rackable workstation), 6.5kW loop | WK-class silicon | **Liquid** (sells a PRO 6000 waterblock separately) | storagereview.com 2026-04-17; comino.com |
| BIZON | ZX5500 4-7x GPU incl. PRO 6000 | mixed | **Water-cooled** | bizon-tech.com (2026) |
| Dell | R7725 rack server hosts PRO 6000 but **caps it at 450W** | Server-class hosting | Chassis airflow | dell.com community + NVIDIA dev forums, 2025-11-05/06 |

Key vendor datapoints:
- **Exxact (2026-03-12): "Three NVIDIA RTX PRO 6000 Max-Q were considered the thermal limit
  for air-cooled configurations"** before their custom 4x solution; in a *stock* config they
  "observed four Max-Qs thermal throttling, undervolting, and leaving valuable performance on
  the table." Their validated 4x Max-Q peaks 85°C over a 3h OCCT stress at 24°C ambient.
  If 4x **300W blower** cards need a bespoke cooling solution, 4x **600W flow-through** in a
  tower is not a product anyone air-cools. Nobody ships it.
- **No system integrator found shipping 4x+ WK Edition air-cooled.** The 600W stacked path in
  the market is liquid (Comino, BIZON) or the Server Edition in ducted server chassis.

**Community (stacking reports):**

- r/LocalLLM "Upgraded to 2x RTX Pro 6000 [WK]" (2026-07-03): "These cards are designed to
  cool 600W, so if you've got good airflow **they run cooler at 300W than the Max Qs**! Top
  GPU tops at 70°C under sustained load." — 2x WK works; derated WK beats Max-Q thermally
  per-card (oversized cooler at half power).
- r/LocalLLaMA "Anyone running 4x RTX Pro 6000s stacked directly on top of each other?"
  (2025-12-28): owner of 2x WK: "I undervolt mine to 500W and a slight downclock… keeps them
  under 65°C… **but I have 18 fans now**." r/threadripper same thread: 4x RTX 6000 Ada
  precedent "thermals were manageable, though I was running at **250W each** for power
  envelope reasons." Flush-stacking full-power WK: no positive reports found.
- r/LocalLLaMA "8x RTX Pro 6000 server complete" (2025-12-13): 768GB via **4 Workstation + 4
  Max-Q mixed** on TR PRO 9955WX — mixed fleets are workable (open-frame/server build); same
  poster: "I also started with workstation cards and **didn't anticipate it to escalate**" —
  the exact buy-WK-then-outgrow path the owner is asking about, resolved by mixing rather
  than replacing.
- r/LocalLLaMA "Dual WK vs Max-Q, open frame" (2026-04-18): "I know I can power-limit the
  Workstation to 450W and still beat a 300W Max-Q. But I keep reading that people
  underestimate what the Workstation cards demand for airflow in a multi-GPU [build]" —
  the community's live debate at N=2-3 lands on WK + power cap on open frames.
- r/pcpartpickerbuilds (2026-04-18): "Once you're talking about 4x RTX 6000s in one server,
  then sure, Max-Q makes more sense."
- NVIDIA dev forums (2026-03-05): WK card with wrong temperature thresholds throttled to
  510MHz (1/6 perf) via SW power cap — thermal-management edge cases on this card are real
  and punishing when they trigger (driver bug, but shows the failure mode's cost).
- Mechanical note: a $10k WK card **snapped at the PCIe connector during transit** with the
  card installed (tomshardware, 2025) — 4-6 heavy WK cards in a box that later moves to a
  colo is a real risk; remove cards before transport.

**Independent 4x600W-host thermal evidence** (aurorainfra.ai, ~2026-07, 8x Max-Q vs 4x 600W WK
server comparison): during an all-GPU Wan2.2 soak, the 4x600W host's GPU 3 hit **92°C and
logged software thermal slowdown** (health-gate fail); one-hour BF16 soak peaked **90°C vs
69°C** on the Max-Q host. Even in a purpose-built multi-GPU server, 4x WK at full power rides
the thermal limit; 4x/8x Max-Q sits 20°C below it.

## Q3. The throttle math: power class per density, and where 1.4x collapses

Power classes (owner is on 230V circuits, ~3.5kW usable per 16A circuit; IL power per
hw-growth-rethink §: $0.116/kWh):

| Config | GPU nameplate | Realistic wall (serving load, +CPU/PSU losses) | Circuit class |
|---|---|---|---|
| 2x WK 600W | 1.2kW | ~1.1-1.5kW | one standard 230V circuit — fine (matches hw-buy §4.1: "one dedicated 240V circuit") |
| 4x WK 600W | 2.4kW | ~3.0-4.0kW peak (vrlatech 2026-07-01: "total system approaches 3,500-4,000W… multiple dedicated circuits") | **saturates/exceeds one 230V/16A circuit** → dedicated high-amp circuit or colo |
| 4x Max-Q / 4x WK@300W | 1.2kW | ~1.6-2.0kW | one dedicated circuit; single 2500W PSU (Exxact's validated build) |
| 6x WK 600W | 3.6kW | ~4.5-5kW | **no single residential/office circuit; colo-only** |
| 6x Max-Q | 1.8kW | ~2.3-2.6kW | one dedicated 230V circuit — still residential-possible |

Where the 1.4x headline collapses:

1. **By workload before by heat**: the 1.4x exists only in compute-bound work (see Q6).
   For bandwidth-bound decode it is ~1.0x from the first card — nothing to collapse.
2. **By thermals at density**: flush-stacked WK recirculates exhaust; the aurora 4x600W host
   shows 90-92°C and SW thermal slowdown at sustained load even in a server chassis. Sustained
   clocks degrade toward the thermal cap; Max-Q sustains its 300W indefinitely at ~69-85°C.
   No published side-by-side gives the exact N where stacked-WK *effective* < Max-Q *sustained*,
   but the market behavior brackets it: integrators sell WK at 1-2, Max-Q at 3-4, liquid/Server
   beyond — i.e. the industry's revealed answer is that WK's advantage is not bankable at ≥3
   flush cards on air.
3. **By the wall socket at 4+**: at 4x600W you must derate or re-wire; derating to 450W
   surrenders a third of the compute gap voluntarily (450W ≈ 95% of ResNet perf though —
   see Q4 — so this is the cheap fix, not the disaster).

## Q4. The de-rate option: is WK@300W == Max-Q?

**Electrically/compute: yes.** Same GB202 die, same 24,064 cores, same 96GB/1.79TB/s. WK is
software-limitable via `nvidia-smi -pl` down to 150W (naveen.ing, 2025-08-07). Measured WK
power curve (ResNet-50 train): 600W=100%, **450W=94.8%**, **300W=75.4%**; peak efficiency at
300W (3.22 img/s/W) — i.e. a WK at 300W behaves like a Max-Q, at 450W it still "beats a 300W
Max-Q" (community consensus + the 450W sweet spot: 85-95% perf across LLM inference loads).

**Thermally per-card: WK-derated is BETTER than Max-Q** — a 600W-sized dual-fan cooler at
300W runs cooler than the Max-Q's small blower at 300W (field report: 2x WK@300W "run cooler
than the Max Qs", top card 70°C sustained, r/LocalLLM 2026-07-03).

**But the cooler geometry does not derate.** The three things `nvidia-smi -pl` cannot fix:

1. **Exhaust direction**: still flow-through — 300W/card still dumps inboard, still preheats
   the neighbor. At 2-3 cards with gaps and case airflow: non-issue (heat load halved). At
   flush 4-6: still recirculates; the blower's out-the-bracket path is architecturally absent.
2. **Form factor**: 5.4" x 12" body vs 4.4" x 10.5". Derated or not, WK doesn't fit standard
   GPU-server card cages or dense chassis. This is the binding constraint at rack stage.
3. **Slot spacing**: backside VRAM still wants the 1-slot gap; board slot budgets cap
   gap-spaced cards at ~3 per ATX/WRX90 board either way.

**Verdict on Q4: WK-derated ≈ Max-Q for fleet sizes ≤3-4 in a roomy tower/open frame with
gaps (arguably better: bigger cooler, same watts, cheaper card). It is NOT equivalent at
rack/colo density — geometry, not watts, is the binding constraint there.**

## Q5. Resale asymmetry: buy 2 WK, swap at the 4-6 stage

- Class retention (hw-buy-20260802/REPORT.md §6, fetched sources Aug 2026): RTX 6000 Ada
  48GB launched ~$6.8k (2022), asks $5-7k used after ~3.5y = **75-100% nominal retention**.
  96GB workstation cards are the best-retaining class in the study.
- Current prices (hw-buy report + this sweep): WK new $12,099 (Newegg) / $13,250 (NVIDIA
  marketplace); WK used/refurb $9.5-11k. **Max-Q street $15.4-16.4k on eBay (scarcity
  premium)**; occasional flash deals exist (Microcenter $7,999, r/LocalLLM 2026-01-23, one-off).
  Server Ed. $19.5k.
- **The asymmetry currently runs in WK's favor**: WK is the highest-volume, most-liquid
  variant; Max-Q is scarce and premium-priced. Buying 2 WK now (~$24k) and selling them at
  the 6-card stage should recover ~80-95% each in shortage conditions (workstation-class
  curve + 96GB scarcity), a **round-trip haircut of roughly $1-5k on the pair** — less than
  the ~$6-8k premium you'd pay TODAY to buy Max-Q instead at street prices.
- No WK-vs-Max-Q split retention data exists yet (the SKUs are ~15 months old); the retention
  driver in the 6000 Ada precedent is the VRAM class, not the cooler. Honest unknown: a 2027
  supply unwind (hw-buy §hedge) would compress used prices on both variants; forced-sale
  timing risk is the real exposure, not the edition.
- Liquid-converting WK (Comino waterblock) preserves the cards but hurts resale (deshrouded
  cards sell at a discount and re-shrouding is labor); treat conversion as a keep-forever move.

## Q6. The 1.4x claim cross-checked

Same die, same core count, same memory system on both editions; the delta is boost clock
(WK ~2617MHz spec class vs Max-Q ~2100MHz class) sustained by the power limit.

| Workload class | WK/Max-Q gap | Source |
|---|---|---|
| Dense GEMM (BF16/FP16/FP8 isolated) | **1.40-1.54x** | aurorainfra.ai measured, ~2026-07 |
| Training / image / video gen (applications) | 1.25-1.35x | aurorainfra.ai |
| Content-creation suite | 1.05-1.16x (Max-Q "5-14% slower") | Puget 2025-07-24 |
| **LLM decode, short context (TP1, 1K)** | **~1.00x (+0.3%)** | aurorainfra.ai, Qwen3-4B vLLM |
| LLM decode, long context (TP1, 32K) | ~1.21x | aurorainfra.ai |
| DeepSeek dense TP2 decode C1-C64 | 1.07-1.13x | aurorainfra.ai |
| Memory bandwidth | **1.00x** (1,465 GB/s measured, both) | aurorainfra.ai |

**The owner's 1.4x is correct for compute-bound work (prefill, batch GEMM, training) and
wrong for bandwidth-bound decode, where the industry number is 1.0-1.2x.** For a serving mix
like ours (decode-dominated with prefill bursts), the blended e2e gap is plausibly ~1.1-1.25x,
not 1.4x. The parallel measurement lane owns the exact number for our engine; this is the
cross-check envelope — if the lane measures decode ≈1.0x and pp512 ≈1.3-1.5x at 600W vs 300W,
that matches the industry curve and nothing is anomalous.

Corollary: because decode doesn't pay for the 600W, capping WK to 450W (95% perf) or even
300W is nearly free for serving-heavy fleets — which is exactly what makes the de-rate path
credible.

---

## The decision tree: "buy 2 WK now, what happens at 6"

```
BUY 2x WK now (~$24k new / ~$20k refurb)
│  2 cards, one box, gaps + airflow: proven fine (70°C field report).
│  Full 600W available for prefill/finetune bursts; cap to 300-450W for serving.
│
├─ Fleet grows to 4:
│   ├─ PREFERRED: second box of 2x WK (2+2). Zero thermal problem, one 230V
│   │   circuit per box, PP-2 pairs per box match the hy3-SKU shape (2x96GB=192GB).
│   │   No replacement, no haircut. Cross-box serving = network, not P2P — but
│   │   memra's replica-granularity serving doesn't need 4-way P2P.
│   ├─ IF one-chassis-4 is required (colo rack economics, TP-4 across 384GB):
│   │   derate WK to 300-450W + open frame/gap spacing (community-proven), or
│   │   sell 2 WK (~85-95% recovery) → 4x Max-Q (Exxact-validated air path), or
│   │   waterblock the WKs (Comino block; keep-forever move).
│   └─ Mixed fleet also works: 4 WK + 4 Max-Q in one build exists in the field.
│
└─ Fleet grows to 6:
    ├─ Three boxes of 2x WK: still fine technically; 3 circuits, 3 chassis —
    │   ops overhead grows, colo consolidation starts to win.
    ├─ Colo consolidation (one chassis): WK is OUT on form factor (5.4" tall,
    │   flow-through) regardless of watts. Choose:
    │   ├─ 6x Max-Q in a big-airflow chassis (1.8kW, single PSU class), or
    │   ├─ Server Edition in a proper GPU server (passive, 400-600W cfg,
    │   │   ducted airflow — colo-native, 4.4" standard height), or
    │   └─ Comino-style liquid 4U (up to 6-8 cards at full 600W, ~$ premium).
    └─ WK exit cost at this stage: sell 2 (or 6) WK at 75-95% recovery =
        $1-3k haircut per pair at current market. NOT a fleet-replacement
        catastrophe — it's a normal upgrade cycle cost, and cheaper than
        paying the Max-Q scarcity premium (~$3-4k/card) today for headroom
        you may never stack into.
```

**Answer to the owner's question: buy-2-WK-now is NOT a dead-end.** The dead-end scenario
(forced full replace at 4-6) only triggers if all three of these hold simultaneously:
(1) the fleet must consolidate into ONE chassis, (2) that chassis must be air-cooled, and
(3) the used market has crashed. Against that: box-of-2 scaling defers (1) indefinitely on
owned premises; liquid and Server-swap options break (2); and the current 96GB shortage makes
(3) the only real risk, which is a timing risk, not an edition risk. Meanwhile WK today is
cheaper per card than Max-Q, faster at prefill/finetune, equal at decode, and derates into a
better-cooled Max-Q on command.

## Honest unknowns

- No published side-by-side of *sustained* stacked-WK-at-600W vs Max-Q in an identical
  tower chassis — the aurora comparison is cross-system (different chassis/CPU/NUMA); the
  exact N and chassis where WK-effective drops below Max-Q-sustained is bracketed by market
  behavior, not measured directly.
- Reddit thread comment bodies were partially inaccessible (Reddit 403s all fetch routes);
  the quotes above come from search-index snippets of the threads — thread titles/dates
  verified, full context not re-read. Repro: the six thread URLs in Sources.
- WK-vs-Max-Q split resale retention: no data exists yet (SKUs too young); class-level
  retention (75-100% at 3.5y) is the proxy.
- Boost clock figures (2617 vs ~2100MHz) are spec-class from press/DB sources, not verified
  against NVIDIA datasheet PDFs this sweep; the measured 1.40-1.54x GEMM ratio is the number
  that matters and is directly sourced.
- Server Edition in owner-premises air: not evaluated in depth — it is colo-chassis-only by
  design (passive), so it only enters at the colo stage.
- The Exxact "NVIDIA had designed the Max-Q to scale to 4x GPUs" claim aligns with NVIDIA's
  "up to four GPUs" language; whether NVIDIA supports >4 Max-Q in one workstation formally is
  not documented anywhere found (8x builds exist in server chassis in the field).

## Sources

| # | Source | Date | What it evidences |
|---|---|---|---|
| 1 | nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/ | fetched 2026-08-04 | Official positioning: WK single-GPU, Max-Q "up to four GPUs", Server passive multi-GPU; form factors, thermal types |
| 2 | pugetsystems.com/labs/articles/nvidia-rtx-pro-6000-blackwell-max-q-vs-workstation-for-content-creation/ | 2025-07-24 | Flow-through recirculation mechanics; 1-slot-gap recommendation; backside VRAM cooling; Max-Q 5-14% slower suite-wide; 3x Max-Q test system |
| 3 | pugetsystems.com/blog/2025/10/28/our-approach-to-dual-geforce-rtx-5090-workstations/ | 2025-10-28 | Puget pushes 600W-class dual-GPU to rackmount/server-room |
| 4 | exxactcorp.com/blog/news/exxact-validates-4x-nvidia-rtx-pro-6000-blackwell-max-q-in-a-workstation | 2025-08-18, upd. 2026-03-12 | 3x Max-Q = prior air-cooled limit; 4x Max-Q throttles stock; first validated air 4x Max-Q (85°C/3h OCCT, 2500W PSU) |
| 5 | aurorainfra.ai/blog/rtx-pro-6000-blackwell-max-q-vs-600w-benchmarks | ~2026-07 | Measured 600W/300W gaps: GEMM 1.40-1.54x, apps 1.25-1.35x, decode ~1.0x short-ctx; bandwidth equal; 4x600W host 90-92°C + SW thermal slowdown vs 69°C Max-Q; 8x300W > 4x600W by 44-57% at equal 2.4kW |
| 6 | naveen.ing/writings/benchmarking-rtx-pro-6000/ | 2025-08-07 | WK power curve: 450W=94.8%, 300W=75.4% (ResNet); peak efficiency at 300W; -pl configurable to 150W; LLM tables 450W=85-95% |
| 7 | storagereview.com/review/comino-grando-rtx-pro-6000-review-768gb-of-vram-in-a-liquid-cooled-4u-chassis | 2026-04-17 | Liquid path: 8x full-TDP PRO 6000 in 4U, 6.5kW loop; up to 6 in rackable workstation |
| 8 | vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/ | 2026-07-01 | 4x600W system = 3.5-4kW wall, multiple dedicated circuits; blower prevents recirculation; Server Ed. SW-configurable 400-600W |
| 9 | centralcomputer.com blog (lineup guide + all-Blackwell-GPUs) | 2026-03-23 / 2026-01-22 | Retailer guidance: 1-2 GPUs WK, 2-4 Max-Q; "not recommended… multi-GPU… blow hot air on each other" |
| 10 | reddit.com/r/LocalLLM/comments/1um8p2b (2x WK upgrade) | 2026-07-03 | 2x WK@300W cooler than Max-Q, top card 70°C sustained |
| 11 | reddit.com/r/LocalLLaMA/comments/1pxvp4t + r/threadripper/1pxvpbi (4x stacked?) | 2025-12-28 | Flush-stack reality: undervolt 500W + 18 fans for 2x; 4x 6000 Ada precedent at 250W/card |
| 12 | reddit.com/r/LocalLLaMA/comments/1plwgun (8x server complete) | 2025-12-13 | Mixed 4 WK + 4 Max-Q 768GB build exists; "didn't anticipate it to escalate" |
| 13 | reddit.com/r/LocalLLaMA/comments/1sov4el + r/pcpartpickerbuilds/1sovrno (dual WK vs Max-Q) | 2026-04-18 | Community N=2-3 verdict: WK + power cap on open frame; "4x in one server → Max-Q" |
| 14 | forums.developer.nvidia.com (510MHz throttle bug; R7725 450W cap) | 2026-03-05 / 2025-11-05 | WK thermal-cap failure mode; servers cap WK at 450W |
| 15 | dell.com community R7725 thread | 2025-11-06 | Server host power-caps the 600W card; cites 300W peak-efficiency finding |
| 16 | tomshardware.com ($10k WK snapped in transit) | 2025 | Mechanical transport risk for heavy WK cards |
| 17 | research/hw-buy-20260802/REPORT.md (internal) | 2026-08-02 | Prices: WK $12.1-13.3k new / $9.5-11k used; Max-Q street $15.4-16.4k; Server $19.5k; RTX 6000 Ada 75-100% retention @3.5y |
| 18 | thundercompute.com/blog/nvidia-rtx-pro-6000-pricing | 2026-07-31 (~4d before fetch) | Current pricing; Max-Q blower positioning |
| 19 | acecloud.ai/blog/multi-gpu-rtx-pro-6000-blackwell-workstation-build/ | 2026-02-27 | Build math: lanes, PSU classes, NVIDIA "up to 384GB in four GPU configuration" Max-Q language |
