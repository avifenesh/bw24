# What GPU to BUY (August 2026) — small self-funded serving+research fleet, box-by-box

Date: 2026-08-02. Question: what is the best GPU to **buy** in August 2026 for a small
self-funded serving+research fleet, purchased incrementally, one box at a time?

All market prices in this report were fetched from the live web on 2026-08-02 (source + date
inline on every number; no training-knowledge prices). All performance numbers are this repo's
own measured receipts where they exist; anything projected is labeled a projection and its
method is stated.

## 0. The owner doctrine this report serves

> "buying the best hardware per price and buying every time the earning crossed the line is
> much more safe, and at the end you can always sell and get half of the money back... h100 is
> the best to rent, is it the best to buy? development is cheaper and faster and we can add new
> hardware support if we need, buying the wrong hardware because we already developed is wrong."

Operating model: GPUs pay for themselves via marketplace serving (OpenAI-compatible endpoints —
`tencent/hy3` 295B-A21B at NVFP4 ~150 GB as the flagship revenue product, plus the 9-35B class);
our own inference-shaped research rides the spare capacity. memra is deepest-optimized on
sm_120a and solid on sm_90a — and per the doctrine the pick must NOT be biased by that. The
sm_90a lane is the receipt that arch support is addable fast (`ARCHITECTURE-H100.md`: bring-up
to beating vLLM on every board row in weeks, merged 2026-07-30). The sunk-development warning
cuts both ways: don't pick sm_120a because we love it, don't pick H100 because we just ported
to it.

## 1. Our receipts (what we actually measured)

### 1.1 Serving throughput

- **H100 80GB (sm_90a), single-GPU board** (`research/tune-data/current-board.json`,
  h100_board 2026-08-01, N=5 medians, interleaved vs vLLM 0.26): e2e single-request tok/s —
  Qwen3.5-9B **204**, Gemma-4 12B **146**, Qwen3.6-27B **96**, Gemma-4 31B **75**,
  Qwen3.6-35B-A3B MoE **226**, Gemma-4 26B MoE **196**. Every row >=1.02x vLLM.
- **H100 multi-user serving** (`research/darklane-serving-20260801/REPORT.md`): 9B-Q8_0
  replica saturates at c=8 ~ **308 tok/s aggregate**; pair-packed ~ **493 tok/s per GPU**;
  6 replicas / 3 GPUs = **1480 tok/s** direct, **~1380 tok/s** through the admission proxy,
  byte-exact outputs, chaos-tested. Key finding: the small-model serving tick is
  **latency/scheduler-bound, not bandwidth-bound** (weight streaming ~12% of the tick) —
  faster HBM does NOT linearly buy small-model serving throughput.
- **RTX 5090 *laptop* (sm_120a, 24 GB, 896 GB/s)** (`current-board.json`, 2026-08-02):
  Qwen3.5-9B NVFP4 **135.7** plain / **281** spec short-code; 35B-A3B MoE **178.2** plain /
  **302** spec. Dense NVFP4 plain decode runs at **~85% of peak memory bandwidth**
  (135.7 tok/s x ~5.6 GB ~ 762 GB/s of 896) — the desktop 5090 and RTX PRO 6000
  (both 1.79 TB/s GDDR7) are ~2.0x the bandwidth of this rig, so bandwidth-bound decode
  scales ~2x from these rows. MoE decode is far less efficient: the 35B-A3B row implies
  ~38% of peak on sm_120a and <=27% on the H100 e2e row — MoE expert gather does not
  saturate HBM.

### 1.2 hy3 (the flagship revenue model) on our engines

- 1x H100 80GB **spill floor** (bank does not fit): **2.48 tok/s** decode, N=3
  (`research/hy3-hopper-20260801/baseline.md`). MTP spec is a net slowdown at the spill
  floor at every K (`research/hy3-spec-20260802/SUMMARY.md`).
- 24 GB sm_120a spill profile: **5.13 tok/s** m=1 (`docs/HY3-SPILL.md`).
- **PP-2 resident is the serving shape** (2 GPUs holding the full bank kill the
  1.9-3.9 GB/token staging term). PP-2 resident hy3 throughput is **not yet measured**
  (the 8xH100 spike is queued); everything below that depends on it is labeled a projection.

### 1.3 PP over PCIe — why no-NVLink boxes are viable for our shapes

- PP-2 exactness gates: **BIT-IDENTICAL logits** across 2-stage splits on Qwen3.5-9B and
  Gemma-4 12B (`research/m1-pp2-20260801/`, `research/m1-inc2-20260801/`, incl.
  double-buffered overlap arms).
- Transport receipts (`research/m0-nccl-20260801/`, 8xH100 box): PP send/recv at decode-size
  messages (4-64 KB) costs **7-11.5 us per hop** (NCCL ~10 us, peer ~7 us, graph-captured
  NCCL ~7 us). A decode activation hop is `hidden x 2B` ~ 12-16 KB for hy3-class models —
  the hop is **latency-floor-bound, not link-bandwidth-bound**, and PCIe 5.0 p2p latency is
  the same order of magnitude. Against a 5-25 ms decode tick that is <0.5% overhead.
  Prefill hops (2048 tok x ~12 KB ~ 25 MB) cost ~0.5-0.6 ms over PCIe 5.0 x16 (~45 GB/s
  effective) — negligible against multi-second prefills. **NVLink is not load-bearing for
  PP-2/PP-4 pipeline serving**; it matters for tensor-parallel all-reduce shapes we don't
  use, and for very deep pipelines (PP-6+) only via more hops of the same tiny cost.

### 1.4 Revenue model (dl-metering + or-provider receipts)

From `research/or-provider-20260802/REPORT.md` (all sources fetched 2026-08-02):

- hy3 OpenRouter effective floor: **$0.1185/M in, $0.4909/M out** (GMICloud, bf16, no
  tools). Our wedge price: **$0.105/M in, $0.44/M out, $0.026/M cache-read** — cheapest
  endpoint *with* tools, captures `:floor` + agentic traffic.
- Gross per M output tokens at wedge prices, 10:1 in:out agentic mix: **$1.49** (uncached
  input) ... **$0.70** (fully cached input). Blended figure used below: **$1.10/M-out**.
- or-provider's replica estimate: **$2-4/hr gross per saturated PP-2 H100 replica**; a 2-4
  replica fleet at floor prices is distribution + utilization, not a standalone profit
  engine.
- Small-model lane (live OpenRouter pricing fetched 2026-08-02, receipt
  `or-models-pricing-20260802.json` in this dir): qwen3.5-9b $0.10/$0.15 per M,
  gemma-4-31b $0.10/$0.34, qwen3.6-35b-a3b $0.14/**$1.00**, qwen3.6-27b $0.30/$2.00.
  At our measured 493 tok/s/GPU (9B), a *saturated* H100 earns only ~$0.27/hr on 9B
  output — the 35B-A3B class ($1.00/M out at 226+ tok/s e2e) is the small-model revenue
  sweet spot, not the 9B class. Endpoint-level check (live API, receipts
  `or-endpoints-*.json` in this dir, 2026-08-02): qwen3.6-35b-a3b has **9 providers,
  floor $0.10/M in / $0.95/M out** (DeepInfra fp8) — a real, defended market; gemma-4-31b
  has 18 endpoints at a $0.34/M-out floor (crowded, cheap); 70B-dense class clears at only
  ~$0.40/M out (llama-3.3-70b). The money SKU below hy3 is the 30-40B MoE class. Metering is billing-grade (dl-metering: worker-truth usage
  on all shapes incl. SSE final chunk, reconciliation EXACT 13/13).

## 3. Projection method for cards we haven't measured

Stated once, used everywhere:

1. **hy3 resident serving (the revenue product) scales with aggregate memory bandwidth.**
   Active bytes/token at NVFP4 ~ 21e9 x 4.5 bits ~ **12 GB/token**; decode is
   bandwidth-bound once resident (receipts 1.1: dense 85% of peak, MoE 13-38%). Calibration
   point: **PP-2 H100 SXM (6.7 TB/s aggregate) ~ 300 tok/s saturated output** — the
   or-provider revenue-model assumption, consistent with 12 GB/token at ~27% effective
   bandwidth plus batching amortization. Projection: `tok/s = 44.8 x aggregate_TB/s`,
   bracket +/-50%, superseded the day the PP-2 spike measures it.
2. **Small-model serving does NOT scale with bandwidth** (receipts 1.1: latency-bound
   tick). H100-class and sm_120a-class GPUs are assumed to serve the 9-35B lane at roughly
   the measured per-GPU rates (~490 tok/s 9B-class pair-packed; 35B-A3B 226 e2e
   single-stream), independent of HBM tier.
3. **Non-CUDA hardware pays a backend tax** (CLAUDE.md accelerator doctrine): explicitly
   gated backend, golden-output gate, no default flips. sm_90a (shared CUDA source) took
   ~2-4 weeks bring-up-to-beating-vLLM; a ROCm/HIP port shares no compiled kernel —
   months, and every kernel-family gate (kernel-check, run-gen argmax, run-spec K=1..8)
   re-proven. That cost is priced into the AMD verdict, per the task instruction.

## 2. August-2026 market prices (all fetched 2026-08-02; source date inline)

Price-tape note: the single best used-market source found is CCIR's price tape
(ccir.io/hardware), built from eBay Seller Hub *sold*-listing captures (3,564 sold listings,
$28.9M volume, Jul 2023-Jul 2026) plus live dealer asks; data-as-of 2026-07-22. eBay item
pages 403-block automated fetches, so CCIR is the executed-price record used here.
Everywhere below: **negotiate from executed medians, not dealer asks** — the tape shows a
2.0x ask-vs-executed gap on H100.

### 2.1 Hopper (post-B300 state)

| Item | Price | Type | Source | Date |
|---|---|---|---|---|
| H100 SXM5 80GB module | **$11,500 executed median** (n=13, trailing 90d); $22,500 ask median | used, sold record | ccir.io/hardware | 2026-07-22 |
| H100 80GB refurb card | $18,000-28,000 ask (PCSP, 1-5 yr warranty); $15-28k (orangehardwares, "under $10k deserves scrutiny") | used/refurb ask | pcserverandparts.com used-GPU guide; orangehardwares.com | mid-2026; ~2026-07-26 |
| H100 PCIe 80GB | ask median $41,495 with **0 executed sales in 90d** — stale dealer posture; realistic used $18-28k | ask | ccir.io; pcserverandparts.com | 2026-07-22 |
| **H100 NVL 94GB (3.9 TB/s, drop-in PCIe form)** | **$26,250 executed median** (n=6); $37,089 ask | used, sold record | ccir.io/hardware | 2026-07-22 |
| HGX H100 8-GPU server new | $250-320k (~$285k typical, ~$36k/GPU deployed) | new OEM | mercatus-ai.com h100-server-price | verified 2026-07-14 |
| HGX H100 8-GPU used | listings exist (UNIXSurplus, eBay 168103028289), price quote-gated; module-derived floor 8x$11.5k + platform | used ask | ebay.com | 2026-06-24 |
| H200 141GB SXM module | **$24,500 executed median** (n=6 — thin); $35,898 ask | used | ccir.io/hardware | 2026-07-22 |
| H200 NVL PCIe new | $28-34k OEM reseller; ServerSupply $38,999 | new | thundercompute.com; serversupply.com | Aug 2026; ~2026-07-19 |
| HGX H200 8-GPU new | $320-420k (~$370k, ~$46k/GPU); refurb H200 systems "barely exist in 2026" | new | mercatus-ai.com h200-server-price | 2026-07-08 |
| 4x H200 SXM board | ~$175k (~$43.75k/GPU) — flagged 4 months stale | new | intuitionlabs.ai pricing guide | 2026-04-14 |
| Rental anchor | H100 $2.10 (SF Compute) - $2.21 (Vast) - $3.60 (index median); H200 $4.08-4.39 | on-demand | sfcompute.com/prices; thundercompute.com; gpuprice.fyi | fetched 2026-08-02 |

Market direction: the H100 crash **already happened** — ~$40k (2023-24 peak) to $12-18k by
April 2026 (axis-intelligence.com, ~1 mo old); Blackwell-refresh supply from
hyperscalers/labs is flowing into secondary channels at 40-70% below new (PCSP, mid-2026);
Vera Rubin (production May 2026) expected to cut another 10-20% off Hopper secondaries later
this year (orangehardwares, ~2026-07-26). Meanwhile H100 **rentals firmed ~40%**
(SemiAnalysis 1-yr rental index $1.70 -> $2.35/hr Oct 2025 -> Mar 2026; newsletter of
2026-04-06) — inference demand is absorbing Hopper. Buy-side implication: used H100 at
executed prices pays back vs its own rental price in ~4-10 months of 24/7 — ownership
economics on used Hopper are unusually good right now; the catch is SXM modules need an HGX
platform (that is exactly why they clear at half the NVL price).

### 2.2 Blackwell

| Item | Price | Type | Source | Date |
|---|---|---|---|---|
| B200 192GB single | $30-40k list (8-GPU volume) / **$45-50k street quote**; allocation-gated, 36-52 wk lead; **no used market exists** | new quote | mercatus-ai.com b200-server-price; tech-insider.org | 2026-07-16; re-checked 2026-07-31 |
| HGX B200 8-GPU | $400-500k (~$450k, ~$56k/GPU deployed) | new | mercatus-ai.com | 2026-07-16 |
| B300 single / DGX B300 | ~$53k; $300-350k+ (8-GPU), shipping since Jan 2026 | new | tech-insider.org (citing Spheron 2026-07-05) | 2026-07-05 |
| **RTX PRO 6000 Blackwell 96GB Workstation** | **$13,250 NVIDIA marketplace** (raised +55% from $8,565 launch, Jun 2026); Newegg ~$12,099 in stock; lowest US tracked $10,691; **used/refurb $9,500-11,000** | new + used | thundercompute.com rtx-pro-6000-pricing (Aug 2026); wccftech (~2026-07-01); gpuprix.com (~Jul 2026) | fetched 2026-08-02 |
| RTX PRO 6000 Max-Q (300W) | marketplace $13,250; eBay street avg $15.4-16.4k (scarcity premium) | new | gpupoet.com | Jul-Aug 2026 |
| RTX PRO 6000 Server Ed. | $19,500 (96GB) / $13,995 (94GB variant) ServerSupply | new | serversupply.com | 2026-06-16 |
| RTX PRO 5000 Blackwell 48GB | $6,694 (Cloud Ninjas, in stock; launch list was $4,200 Dec 2025) | new | cloudninjas.com | fetched 2026-08-02 |
| **RTX 5090 32GB** | new street: **$4,099 lowest in-stock** (Tom's live tracker; MSRP $1,999 is paper — memory-shortage price hikes); **used eBay sold comps median ~EUR 2,517 (~$2.7k)**, typical EUR 1,919-2,709, 48 completed sales | new + used | tomshardware.com live tracker (2026-08-02); pcprice.watch (Jun-2026 data) | 2026-08-02 |
| 8x 5090 rackable | Bizon ZX9000 4U liquid-cooled base "from $36,142" (3-7 day ship, GPUs configurable); Comino Grando pre-order (last public price Feb 2025 — stale); gray-market 2-slot blower 5090s exist ($5,999 datapoint Aug 2025 — stale) | new | bizon-tech.com/bizon-zx9000 (fetched 2026-08-02) | 2026-08-02 |
| Rental anchor | B200 $2.80 (Vultr) - $5.89 (RunPod) - $6.87 avg; RTX PRO 6000 $1.99-2.19; RTX 5090 $0.21 (Vast) - $0.99 (RunPod) | on-demand | getdeploying.com (2026-08-02); vast.ai guide (2026-07-20); madebyagents.com (2026-08-02) | 2026-08-02 |

Market direction: **no Blackwell oversupply — the opposite.** B300 ramp did not cut B200
(cloud median +8% YoY; $45-50k street premium holds; hyperscalers absorb allocation;
normalization is a 2027 story). On the sm_120 side the **GDDR7/memory shortage is the
driver**: NVIDIA cut consumer Blackwell output 30-40% in H1 2026 and raised RTX PRO 6000
MSRP to $13,250 (+55%) in June; 5090 street is ~2x paper MSRP. The exploitable inversion:
**used 5090s (~$2.7k sold comps) trade at ~65% below new-in-stock ask, and refurb RTX PRO
6000s ($9.5-11k) undercut NVIDIA's own marketplace price.** For B200-class compute, renting
($3-7/hr) beats owning ($450k + allocation fight) at single-box scale, full stop.

### 2.3 Everything else (used mid-range, AMD, Intel, ASICs, gray market)

| Item | Price | Small-buyer availability | Source | Date |
|---|---|---|---|---|
| L40S 48GB used | **$6,795 executed median** (n=10) / $9,199 ask (n=63); ~$7-9k refurb in tested servers | easy (eBay, PCSP w/ warranty) | ccir.io/hardware; pcserverandparts.com | 2026-07-22; ~Jul 2026 |
| RTX 6000 Ada 48GB used | ~$5,000-7,000 ask | easy | pcserverandparts.com | mid-2026 |
| RTX A6000 48GB (Ampere) | $3,650 executed / $5,200 ask; very liquid (1,906 sold in 3y) | easy | ccir.io/hardware | 2026-07-22 |
| A100 80GB PCIe used | $12,604 lowest-3-ask avg (**+42% since Jan 2026** — DRAM shortage re-inflation); ask median $20,650 | easy but pricey | gpupoet.com; ccir.io | fetched 2026-08-02 |
| A100 80GB SXM4 used | **$5,529 executed** (22.2% of launch) — needs HGX board | easy, cheap, platform problem | ccir.io/hardware | 2026-07-22 |
| V100 32GB used | $669 executed (3.6% of launch after 8.3 yr) | commodity | ccir.io/hardware | 2026-07-22 |
| AMD MI300X 192GB | **not sold standalone** (Dihuni states verbatim, 2026-07-01); analyst street $11-15k/GPU *inside* quote-gated 8-GPU systems (Dell XE9680/SMC/GIGABYTE) | **effectively unbuyable box-by-box** | dihuni.com; alibaba buying guide | 2026-07-01 |
| AMD MI325X 256GB / MI355X 288GB | **zero published purchase prices anywhere**; OEM quote-only, 8-GPU pods; rentals MI325X $3.18/hr avg, MI355X $2.59-8.60/hr | not a retail channel | getdeploying.com (2026-08-02); spheron.network (2026-07-08) | 2026-08-02 |
| Intel Gaudi 3 | $15,650 list (2024 figure still quoted — stale); IBM Cloud only CSP; discontinuation signaled 2026-27; Falcon Shores cancelled, Crescent Island (160GB LPDDR5X inference PCIe) samples late 2026 | dead end for small buyers | introl.com (2026-04-21); eenewseurope (2026-07-30) | 2026-08-02 |
| Tenstorrent Blackhole p150 | $1,399 (32GB GDDR6, web store); QuietBox 4x $9,999-11,999; Galaxy 32-chip $70-110k (GA Apr 2026); open stack (TT-Metal + vLLM fork), but multi-card LLM serving is still DIY | **the only ASIC actually buyable retail** | tenstorrent.com; theregister.com (2025-11-27); hpcwire (2026-05-01) | 2026-08-02 |
| Groq / Cerebras / SambaNova | Groq -> NVIDIA $20B licensing/acqui-hire (Jan 2026); Cerebras ~$2-3M/system sales-only; SambaNova reportedly being acquired by Intel | cloud/API only | fortune.com (2026-01-05) | 2026-08-02 |
| Qualcomm Cloud AI 100 Ultra 128GB | no posted retail price; gray resellers; Cirrascale rents 8x @ $3,759/mo | marginal | cirrascale.com/pricing | fetched 2026-08-02 |
| RTX 4090 48GB China mod | $4,000+ typical, $5,200 posted ask (Jawa) — **risen** with the memory shortage; no warranty, hacked vBIOS; no NVFP4 (sm_89) | gray | jawa.gg; alibaba Q&A | ~mid-Jul 2026 |


## 4. Candidate-by-candidate

Every box below is sized to serve hy3 NVFP4 (~150 GB artifact + KV headroom) as the flagship
product, with the 9-35B lane as filler. "sat tok/s" = projected saturated hy3 output using
the section-3 method (44.8 tok/s per TB/s aggregate, +/-50%); "$/hr sat" at the blended
$1.10/M-out. Full math: `tco-model.py` in this dir.

### 4.0 2x RTX 5090 (entry) -> grow to 4-6 cards — the owner's candidate, tested hard
**~$8.4k used / ~$11.2k new** (2x $2.7k sold-comps or $4.1k street + $3k AM5 host; +$1-1.5k
buys a Threadripper/EPYC host with 6-7 x16 slots — the growth chassis).

**The SKU-hardware coupling, explicit:** 64 GB total CANNOT serve hy3 (~150 GB NVFP4) —
spill serving is 5.13 tok/s on a 24 GB sm_120 card (docs/HY3-SPILL.md), not
marketplace-grade. What it CAN serve is exactly where our receipts are deepest: the
9-35B class resident per card (q35 IQ4_XS 178.2 plain / 302 spec MEASURED on the 896 GB/s
laptop rig; AgentWorld deployed at 1.68-1.76x vs llama.cpp, Ornith-35B plain e2e over 1.0x), projected
~300 tok/s/card plain on the 1.79 TB/s desktop part (2.0x bandwidth, MoE efficiency held
constant, bracket 180-500 — the one unmeasured step, flagged). PP-2 across the pair is
ALSO receipts-backed (bit-identical logits, peer-copy transport, section 1.3) and opens
the 50-64 GB tier: 70B-dense Q4 (~40-42 GB + KV) fits — feasible, but that market clears
at ~$0.40/M out (llama-3.3-70b, live 2026-08-02), so it is optionality, not the plan.
The 35B-A3B lane is the revenue product: floor $0.95/M out across 9 providers (receipts
above). So choosing this box = choosing the 30-40B MoE SKU first and deferring the hy3
listing until the box grows to 5-6 cards (160-192 GB) or a second box exists.

**Economics (SKU-B table, section 7):** $0.47/Mtok at 30% util — 2.5x cheaper per token
than the best RTX PRO configuration and 6x cheaper than 1x H100 NVL on the same lane;
saturated gross ~$2.59/hr/box at the blended $1.20/M-out. It is the only sub-$10k entry
that is TCO-positive at 30% utilization.

**Power/placement:** 1.4 kW nameplate (2 cards + host) — any home/office circuit
(NEC 80%: 120V/15A = 1.44 kW continuous; a 240V line is comfortable); ~$500-1,050/yr at
$0.12-0.25/kWh at the 30%-util duty profile. Zero colo dependency = zero $/kW fees while
small, and the EULA's "datacenter" word never enters the building (section 5). Growth
crosses residential limits around card 4 (2.5-3 kW serving) — plan the colo move (or a
240V/30A circuit) at that point, which is also where the consumer-card asterisk begins.

**Risks:** used consumer cards, no warranty (Puget 2025: RTX 50 FE 0.25% fleet failure —
best-in-class, 2026-02-02); GeForce EULA sections 2.7/2.8 once the box moves to a colo;
32 GB granularity means no single-card model above ~30 GB without PP; desktop-5090 serving
numbers are projected from the laptop board until first-box bring-up measures them (that
measurement is free: it happens on the box we buy).

### 4.1 2x RTX PRO 6000 Blackwell 96GB (Workstation/Max-Q) — $24-28.2k/box
192 GB total, PP-2 over PCIe 5.0 (receipts 1.3), 42 GB KV headroom — the roomiest 2-card
hy3 box. 3.58 TB/s aggregate -> ~160 tok/s sat (~$0.63/hr). sm_120-class: zero porting, our
deepest kernels, native NVFP4; MIG on Max-Q for research partitioning; one 96 GB card also
serves the whole 9-35B lane resident (35B-A3B 18 GB x multiple replicas + drafts). Power:
1.45 kW nameplate box; ~1.1 kW at serving load — one dedicated 240 V circuit; Max-Q pair
(0.85 kW nameplate) is trivially residential (NEC 80% rule: a 120V/15A circuit carries
1.44 kW continuous). Electricity at $0.12-0.25/kWh: ~$950-2,000/yr (0.30-util profile).
Colo committed-kW $150-250/kW/mo (QuoteColo 2026-05-23) -> $220-360/mo if hosted.
Pro driver — no GeForce datacenter clause (section 5). Warranty: NVIDIA 3-yr repair/replace
(nvidia.com RTX PRO warranty page, fetched 2026-08-02); PNY sells a free 3->5-yr extension
program (news.pny.eu, 2026-04-16). Resale: RTX 6000 Ada (launch ~$6.8k, 2022) still asks
$5-7k used after ~3.5 yr = ~75-100% nominal retention — workstation 48-96 GB cards are the
best-retaining class on the tape, currently amplified by the GDDR7 shortage. No NVLink —
irrelevant for PP-2 (receipts 1.3).

### 4.2 6x RTX 5090 32GB used (~$2.7k sold comps) — ~$23.2k/box
192 GB as PP-6, 10.74 TB/s aggregate -> ~481 tok/s sat (~$1.90/hr) — **3x the throughput of
the RTX PRO box for the same money; the raw tokens-per-capex-dollar winner** (32 GB slices
also serve the whole 9-35B lane per card at 2x our laptop-board rates). Costs: (a) GeForce
EULA — section 5: contractual risk, zero known enforcement at this scale, but it is the one
candidate with a legal asterisk; (b) 3.45 kW GPU nameplate (~2.7 kW serving) — at the edge
of residential (dedicated 240V/20A+), $0.12-0.25/kWh -> $2,800-5,900/yr; colo ~$550-925/mo
and some facilities' AUPs mirror the EULA; (c) used consumer cards: no warranty, though
Puget's 2025 fleet data has RTX 50 FE at 0.25% failure — the most reliable GPUs they shipped
(pugetsystems.com, 2026-02-02); (d) PP-6 = 5 inter-stage hops ~ still <0.1% of a decode tick
(receipts 1.3), but 6x the fan/VRM failure surface and a physically awkward build (3-slot
coolers; 2-slot blowers are gray-market; Bizon-class liquid 4U starts $36k before GPUs).
Resale: used-bought at $2.7k, worst plausible 3-yr floor ~$1.5k = ~55%+ retention aided by
the memory shortage.

### 4.3 2x H100 NVL 94GB used ($26.25k executed) — ~$57.5k/box
188 GB PP-2, NVLink-bridged pair (not load-bearing for us), 7.8 TB/s -> ~349 tok/s
(~$1.38/hr). The only *practical* small-buyer H100: SXM modules at $11.5k executed need an
HGX platform; NVL drops into a 4U. sm_90a: zero porting (lane already merged). Power 0.8 kW
nameplate. Twice the capex quantum of 4.1/4.2 for ~2x RTX PRO throughput -> nearly identical
$/Mtok, worse box-by-box granularity. Used enterprise resellers give 1-5 yr warranty (PCSP,
mid-2026). Resale: H100 arc = $30-40k (2023-24) -> $11.5k executed (2026-07-22), ~29-38%
of new after ~3 yr; buying at the crashed price resets that curve in our favor (~45%
assumed at 3 yr, A100-arc-informed).

### 4.4 Used 8x H100 SXM server (~$120k est: 8x$11.5k modules + platform, quote-gated)
640 GB, NVLink, 26.8 TB/s -> ~1,200 tok/s (~$4.75/hr): 4 hy3 PP-2 replicas — the best
**clean/legal $/Mtok at scale** ($2.57 at 30% util) and the direct continuation of our
Mumbai/rented-fleet receipts. But: one purchase = 4-5 small boxes (violates the incremental
doctrine's risk sizing), 5.9-6.5 kW = colo-only ($900-1,600/mo committed-kW), platform
price is quote-gated (the $120k is an estimate, flagged), and Meta's fleet data says H100
HBM3 is the dominant failure mode at scale (419 interruptions/54 days on 16,384 GPUs, half
GPU/HBM-caused — tomshardware.com, 2024-07-27). The box to buy at fleet stage, not first.

### 4.5 2x H200 NVL new ($28-39k) — ~$67k/box
282 GB PP-2, 9.6 TB/s -> ~430 tok/s (~$1.70/hr). Huge KV headroom (262k-ctx hy3 without
squeeze) and the same sm_90a zero-port. But new-unit pricing at ~2.4x the RTX PRO box for
~2.7x throughput = no per-token advantage, bigger quantum, and refurb H200 "barely exists"
(mercatus-ai, 2026-07-08). The card rental markets price at $4+/hr — better rented.

### 4.6 B200 / B300 — not a small-buyer channel in Aug 2026
$45-50k street single-GPU quotes, allocation-gated, 36-52 wk leads, no used market
(tech-insider.org, 2026-07-31). 8x HGX ~$450k models to $2.73/Mtok — worse than the used
8x H100 server despite 2.4x bandwidth, because capex per TB/s is still 1.6x higher and
there's no half-back resale precedent on an allocation-gated part. Rent B200 ($2.80-6.87/hr)
for any sm_100 experiments; do not buy in 2026.

### 4.7 4x L40S used ($6.8k exec) / 4x RTX 6000 Ada used (~$6k) — ~$29-32k/box
192 GB PP-4 but only 3.46-3.84 TB/s aggregate -> 155-172 tok/s — RTX-PRO-box throughput at
higher capex, power (1.2-1.4 kW), slot count, and an sm_89 port (weeks, gated backend, no
NVFP4 tensor path). Dominated by 4.1 on every axis. Skip. (The high used ask on these is
exactly the workstation-card value retention that makes 4.1's resale case.)

### 4.8 AMD MI325X/MI355X — unbuyable at our scale + backend tax
No published purchase prices exist anywhere; MI300X "not sold standalone" (Dihuni,
2026-07-01); OEM quote-only 8-GPU pods. Even if a pod were quotable, the CLAUDE.md
secondary-backend doctrine prices in: gated backend, golden-output gate before any scored
evidence, and a ROCm/HIP port that shares zero compiled kernels with our CUDA tree — months
of bring-up vs the weeks sm_90a took, all to reach silicon we can already rent at $2.59-3.18/hr
to evaluate. Not an August-2026 buy.

### 4.9 Intel / ASICs / gray mods
Gaudi 3: discontinuation signaled, successor (Crescent Island) samples late 2026 — dead end.
Tenstorrent Blackhole p150 ($1,399, 32 GB GDDR6, open TT-Metal stack) is the only
retail-buyable ASIC — genuinely interesting as a $1.4k research toy, but 512 GB/s-class
GDDR6 and a DIY multi-card LLM story cannot carry the hy3 product; same golden-output-gate
tax as AMD. RTX 4090 48GB China mods ($4-5.2k, rising): no warranty, hacked vBIOS, sm_89,
export-control gray — the $/GB is no longer even good vs a used 5090. All skipped.

### 4.10 Used 8x A100 SXM4 server (~$65k est) — the value dark horse, rejected
640 GB HBM2e, 16.3 TB/s -> ~731 tok/s, $2.89/Mtok. But sm_80 is a *third* arch lane (weeks,
even CUDA-shared), no FP8/FP4 units (prefill compute-poor), 2020 silicon entering its
V100-decay phase (V100: 3.6% of launch after 8 yr — ccir.io tape 2026-07-22), and the
DRAM-shortage re-inflation (+42% on A100 PCIe asks since Jan — gpupoet, 2026-08-02) means
buying INTO a spike. The per-token math is fine; the 3-year exit is not.

## 5. The legal angle: consumer cards in a serving fleet

Current GeForce driver license (v. February 25, 2025 — nvidia.com/en-us/drivers/geforce-license/,
fetched 2026-08-02), verbatim:

> **2.7** "Except as expressly granted in this Agreement, you may not sell, rent, sublicense,
> distribute or transfer the SOFTWARE or **provide commercial hosting services with the
> SOFTWARE**."
> **2.8** "You agree that GeForce or Titan SOFTWARE: (i) is licensed for use only on GeForce
> or Titan hardware products you own, and (ii) **is not licensed for datacenter deployment**."

What this actually is: a restriction in the **driver software license**, not on the silicon,
and scoped to "GeForce or Titan SOFTWARE". Three load-bearing facts:

1. **The old blockchain exception is gone.** The 2017-2018 text read "not licensed for
   datacenter deployment, except that blockchain processing in a datacenter is permitted"
   (TechPowerUp 2017-12-26; DCD coverage). The current version drops the exception and adds
   the "products you own" clause — the license got *stricter*, not looser.
2. **"Datacenter" is undefined** in the document, and NVIDIA's only public gloss remains the
   2018 CNBC statement era. A colo rack plainly risks the label; a workstation in an office
   or home does not. Selling *tokens* from our own server (memra) is arguably not "commercial
   hosting services with the SOFTWARE" (we host our software, not customers' workloads) —
   but 2.7+2.8 together give NVIDIA a colorable termination claim for any 5090 colo fleet.
3. **Enforcement history is commercial pressure, not litigation.** No known lawsuit against
   any provider since the clause appeared in Dec 2017; the mechanism was sales pressure on
   large Japanese/Western clouds in 2018 (DCD/Digital Trends coverage). Meanwhile Vast.ai,
   RunPod community tier, and SaladCloud have openly run thousands of consumer 4090/5090s
   for years. Practical exposure for a 1-3 box provider: driver-license termination risk and
   zero enterprise support — not damages (the license caps NVIDIA's own liability at $5, and
   remedies against a licensee are termination-shaped).

**RTX PRO 6000 Blackwell is outside all of this**: it is not "GeForce or Titan hardware",
runs the professional/enterprise driver branch, and carries NVIDIA's RTX PRO 3-year
warranty. Same sm_120 family, no licensing asterisk — that legitimacy is exactly what the
$12.1k-vs-$4.1k premium over a 5090 buys, alongside 96 GB and ECC.

## 6. Resale curves and reliability (the "sell and get half back" audit)

Depreciation arcs from the executed-price tape (ccir.io, data through 2026-07-22) and dated
reporting:

| GPU | New (year) | Used executed now | Nominal retention | Age |
|---|---|---|---|---|
| V100 32GB | $18,625 basis (2018) | $669 | 3.6% | ~8 yr |
| A100 80GB SXM | $24,875 basis (2021-22) | $5,529 | 22.2% | ~4.5 yr |
| A100 80GB PCIe | ~$15-17k (2021-22) | $12.6k lowest-3 asks (+42% since Jan 2026) | ~75-80% ask | ~4.5 yr |
| H100 80GB SXM | $30-40k street (2023-24) | $11,500 | ~29-38% | ~3 yr |
| RTX 6000 Ada 48GB | $6,800 (2022) | $5-7k ask | ~75-100% | ~3.5 yr |

Reading: **"half back after 3 years" is NOT what new-datacenter-GPU buyers got** (H100:
~1/3; A100 SXM: ~2/9 over 4.5 yr). It IS approximately what a *used* buyer gets (bought
post-crash, the remaining decay is much flatter — an H100 bought at $11.5k today landing at
the A100's current $5.5k in 2029 = 48%), and it is *conservative* for workstation-class
cards (RTX 6000 Ada at 75-100% after 3.5 yr). Two 2026 forces prop up used prices: the
DRAM/GDDR7 shortage (A100 asks +42% YTD; 5090 at 2x paper MSRP) and firmed inference rental
demand (SemiAnalysis H100 rental index +40% Oct 2025 -> Mar 2026). One force cuts against:
Vera Rubin ramp knocking 10-20% off Hopper secondaries later in 2026 (orangehardwares,
~2026-07-26). The doctrine's "half back" is a fair planning number for used-bought and
workstation gear; use 1/3 for anything bought new at datacenter list.

Context on the industry fight over exactly this number: hyperscalers depreciate GPUs over
5-6 years (CoreWeave: 6 since 2023 — cnbc.com 2025-11-14); Burry's counter is 2-3 years of
real economic life and ~$176B of understated depreciation 2026-28 (deepquarry 2025-12-07).
Our buy-used posture sidesteps the debate: someone else already ate the steep half of the
curve.

Reliability/warranty:

- Puget Systems 2025 fleet data (published 2026-02-02): RTX 50-series FE **0.25% failure**
  — their most reliable GPU class; ASUS RTX 50 0.4%. Consumer Blackwell is not a reliability
  risk in workstation duty.
- At datacenter scale, H100 HBM3 is the failure driver: Meta's Llama-3 run logged **419
  unexpected interruptions in 54 days on 16,384 H100s (~1 per 3 hrs), ~half GPU/HBM-caused**
  (tomshardware.com 2024-07-27). Per-GPU that is ~0.9%/yr-class interruption — fine at our
  scale, but used SXM boards carry HBM history we can't audit.
- Warranty norms: used-enterprise resellers 1-5 yr (PCSP); eBay module buys effectively
  30-90 days; RTX PRO new = NVIDIA 3 yr (+PNY 5-yr program); used consumer = none.

## 7. Verdict — ranked 3-year TCO per million tokens served

Two tables, because the SKU and the hardware constrain each other (`tco-model.py`, both
runs; energy $0.18/kWh mid, PUE 1.1, resale per section 6).

### 7a. The hy3 SKU (~150 GB NVFP4; revenue $1.10/M-out blended, bracket 0.70-1.49)

| rank | box | capex | sat tok/s | 3y TCO | **$/Mtok @30%** | @60%,$0.12 |
|---|---|---:|---:|---:|---:|---:|
| 1 | 6x RTX 5090 used | $23.2k | 481 | $19.6k | **1.43** | 0.72 |
| 2 | 6x RTX 5090 new | $31.6k | 481 | $26.6k | **1.95** | 0.98 |
| 3 | 8x H100 SXM used server | ~$120k | 1,200 | $87.7k | **2.57** | 1.30 |
| 4 | 8x B200 HGX new | $450k | 2,866 | $222.4k | **2.73** | 1.37 |
| 5 | 8x A100 SXM used server | ~$65k | 731 | $59.9k | **2.89** | 1.45 |
| 6 | 2x H200 NVL new | $67k | 430 | $39.0k | **3.19** | 1.59 |
| 7 | 2x RTX PRO 6000 refurb | $24k | 160 | $16.0k | **3.51** | 1.73 |
| 8 | 2x H100 NVL used | $57.5k | 349 | $36.3k | **3.66** | 1.82 |
| 9 | 2x RTX PRO 6000 new | $28.2k | 160 | $17.9k | **3.92** | 1.94 |
| 10 | 2x RTX PRO 6000 Max-Q | $30.5k | 160 | $18.1k | **3.97** | 1.95 |
| 11 | 4x RTX 6000 Ada used | $29k | 172 | $21.2k | **4.34** | 2.15 |
| 12 | 4x L40S used | $32.2k | 155 | $23.2k | **5.28** | 2.62 |

Sobriety: at the hy3 floor only the 5090 boxes get TCO below the $1.10/M-out revenue line,
and only above ~40% utilization. hy3 at floor prices is distribution and public perf
receipts (the or-provider conclusion), not the margin engine.

### 7b. The 9-35B SKU (what a 2-4 card box sells; revenue $1.20/M-out blended on the
qwen3.6-35b-a3b market — floor $0.95/M out, 9 providers, live 2026-08-02; bracket 0.95-1.45)

| rank | box | capex | sat tok/s | $/hr sat | 3y TCO | **$/Mtok @30%** | 3y net @30% |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **2x 5090 used** | **$8.4k** | 600 | 2.59 | $8.1k | **0.47** | **+$12.4k** |
| 1= | 4x 5090 used | $14.8k | 1,200 | 5.18 | $12.8k | **0.38** | +$28.1k |
| 1= | 6x 5090 used | $23.2k | 1,800 | 7.78 | $19.6k | **0.38** | +$41.8k |
| 4 | 2x 5090 new | $11.2k | 600 | 2.59 | $10.4k | **0.61** | +$10.0k |
| 5 | 1x RTX PRO 6000 refurb | $12.5k | 300 | 1.30 | $9.1k | **1.07** | +$1.1k |
| 6 | 1x RTX PRO 6000 new | $14.6k | 300 | 1.30 | $10.1k | **1.18** | +$0.1k |
| 7 | 1x H100 NVL used | $30.3k | 250 | 1.08 | $20.3k | **2.86** | **-$11.8k** |

The couplings this table proves with numbers:

- **1x RTX PRO 6000 does NOT beat 2x 5090** on the small SKU: same 1.79 TB/s per card, so
  the pair has 2x the aggregate bandwidth for 58-67% of the money — $0.47 vs $1.07-1.18
  per Mtok. What the single PRO 6000 buys instead is *VRAM aggregation without PP* (one
  96 GB card holds 50-90 GB models whole), the pro driver, and the 3-yr warranty — a
  research-and-legitimacy premium, not a serving-economics win.
- **H100 is the wrong hardware for the small SKU**: measured 226 tok/s e2e (35B-A3B) on
  silicon that costs $26k+ used — its HBM advantage doesn't convert on latency-bound MoE
  serving (receipts 1.1), so it loses to a $2.7k consumer card per token by ~6x.
- **The hy3 SKU requires >=160 GB**: 5-6x 5090, 2x RTX PRO 6000, or H100/H200 pairs. The
  buy pick and the SKU pick are one decision.

### First-box recommendation

**Box #1: 2x RTX 5090 (used, ~$2.7k executed comps each) in a 6-7-slot Threadripper/EPYC
growth chassis — ~$9.5-10k all-in.** The owner's reading is confirmed by the receipts —
with one deliberate consequence attached: this choice launches on the 30-40B MoE SKU
($0.95/M-out market, our strongest measured class) and defers the hy3 listing until the
same chassis reaches 5-6 cards. The math that carries it: $0.47/Mtok vs $1.20 revenue =
the only sub-$10k TCO-positive entry at 30% utilization; +$12.4k projected 3-yr net; 1.4 kW
residential (zero colo/$-per-kW while proving demand); per-card growth quantum ($2.7k) is
the purest expression of buy-when-earnings-cross; and the growth path ends at the #1 row
of BOTH tables (6x 5090 = 1,800 tok/s on 35B-class AND 481 tok/s hy3-capable at 192 GB).
Spending +$1-1.5k on the big chassis day one is what makes the path card-by-card instead
of box-by-box.

Sunk-development symmetry check: this is NOT "we love sm_120a". The same tables rank the
sm_120a RTX PRO 6000 pair *below* used Hopper on the hy3 SKU, and rank H100 dead last on
the small SKU — the arch we just spent weeks porting to. The 5090 wins on executed street
price per TB/s ($2.3k/TB/s used — 3.4x better than RTX PRO, 4-6x better than Hopper NVL),
a market fact any engine would face; our measured 1.68-1.76x-vs-llama receipts on this
class are real revenue math on top, counted per the owner's instruction, not the reason.

**The legal-clean sibling**: 2x RTX PRO 6000 (~$26k) remains the right FIRST box only if
the hy3 OpenRouter listing must exist from day one (it is the cheapest clean hy3-capable
box) or if the fleet must sit in a colo immediately (no GeForce EULA asterisk — section 5).
That is a go-to-market call, not a hardware call: it costs 3x the entry capex and ~2.5x
the $/Mtok on the lane that actually mints margin.

**H100: rent, don't buy — still.** Explicitly: the answer differs from H100. Used H100
only becomes rational at fleet stage as the 8x SXM server (~$120k, $2.57/Mtok on hy3) —
after roughly four small boxes of metered demand justify its quantum — and H100 rental
($2.10-3.60/hr, firmed +40% in 6 months) remains the burst-research tool meanwhile. The
owner's "h100 is the best to rent, is it the best to buy?" — the tape answers: best to
rent, not first to buy.

### The earnings-crossed-the-line threshold (dl-metering trigger)

Rule (doctrine-priced: resale returns ~half, so at-risk capital = half the price):
**add the next unit when cumulative metered gross margin (revenue minus power/colo) since
the last purchase >= 0.5 x next unit's price.**

- Card #3 (used 5090, ~$2.7k): trigger at **~$1.35k net**. Box #1 at 30% util nets
  ~$0.6/hr (~$440/mo) -> fires in ~3 months; at 10% util ~9-10 months. The meter decides.
- Cards #4-6: same ~$1.35k step each; card #5-6 unlocks the hy3 listing (160-192 GB).
- Alternative box #2 (2x RTX PRO 6000, ~$26k) if the colo/hy3-legitimacy route opens:
  trigger at ~$13k cumulative net.
- Fleet stage (8x H100 SXM used, ~$120k): trigger at ~$60k cumulative net — by
  construction only after the small fleet has proven ~2 years of real demand.

### Biggest uncertainties (ranked)

1. **Desktop-5090 35B-class serving rate is projected** (300 tok/s/card from the laptop
   board's measured 178.2 at exactly half the bandwidth) — measured the day box #1 boots;
   the buy decision survives the low bracket (180/card still = $0.78/Mtok, TCO-positive).
2. **Demand capture at 30% utilization** — the 35B-A3B market is real (9 providers) but
   OpenRouter's application backlog deprioritizes open-weight hosts; utilization below
   ~12% makes even box #1 TCO-negative. This is the true risk, and the per-card quantum
   is the hedge.
3. **The hy3 PP-2 calibration (300 tok/s per 6.7 TB/s, +/-50%)** — moves table 7a
   proportionally; rank order is capex-driven and stable; superseded by the queued spike.
4. **Executed-vs-ask discipline on used gear** (2x spreads on the tape) and **memory-
   shortage persistence** (props both our resale assumptions and our capex; a 2027 unwind
   cuts both symmetrically).

## 8. What a budget buys under the doctrine

- **~$10k**: Box #1 — 2x used 5090 in a 6-7-slot growth chassis. Lists the 30-40B MoE
  lane, runs all sm_120a research, sits on a home circuit, and starts the meter.
- **~$25k**: Box #1 grown to 6 cards as earnings cross (per-card triggers) — at which
  point it holds 192 GB, serves hy3 PP-6, and tops BOTH $/Mtok tables. (Same total as
  buying the 6x mule day one — the difference is that the meter, not hope, funded cards
  3-6.)
- **~$50k**: The grown box + either a second mule or the 2x RTX PRO 6000 pair (if the
  colo/legal-clean hy3 endpoint is the next unlock).
- **$120k+**: Still not day-one H100. That money is the grown box + a clean hy3 box +
  18 months of colo and power — with three resale exits and the option to buy the used
  8x H100 server later at what Vera Rubin does to its price (10-20% further down,
  orangehardwares ~2026-07-26).

## Source index (all fetched 2026-08-02; page dates inline above)

ccir.io/hardware (executed-price tape, 2026-07-22) - pcserverandparts.com - orangehardwares.com -
mercatus-ai.com (H100/H200/B200 server guides, 2026-07) - thundercompute.com (H200 NVL, RTX PRO
pricing, Aug 2026) - serversupply.com - intuitionlabs.ai - tomshardware.com (5090 live tracker
2026-08-02; Llama-3 failures 2024-07-27; Puget reliability 2026-02-01) - wccftech - gpuprix.com -
gpupoet.com - bizon-tech.com - getdeploying.com - vast.ai - sfcompute.com - gpuprice.fyi -
madebyagents.com - axis-intelligence.com - SemiAnalysis rental index (2026-04-06) - dihuni.com
(2026-07-01) - spheron.network (2026-07-08) - introl.com (2026-04-21) - eenewseurope (2026-07-30) -
tenstorrent.com - theregister.com (2025-11-27) - hpcwire (2026-05-01) - fortune.com (2026-01-05) -
cirrascale.com - jawa.gg - nvidia.com/en-us/drivers/geforce-license/ (v. 2025-02-25, fetched
2026-08-02) - nvidia.com RTX PRO warranty - news.pny.eu (2026-04-16) - techpowerup.com
(2017-12-26) - datacenterdynamics.com - digitaltrends.com (2018-01-08) - pugetsystems.com
(2026-02-02) - quotecolo.com (2026-05-22/23) - wacolo.com (2026-05-29) - brightlio.com -
cologpu.com (2026-04-12) - cnbc.com (2025-11-14) - deepquarry.substack.com (2025-12-07) -
openrouter.ai/api/v1/models + per-model endpoints APIs (JSON receipts in this dir) - plus this
repo's own receipts cited inline in section 1.
