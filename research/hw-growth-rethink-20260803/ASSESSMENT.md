# Hardware growth-strategy rethink — does no-P2P-on-5090 break the growth path? (2026-08-03)

Owner stake (verbatim): *"if p2p not available the plans to grow with time are non relevant
with this hardware and we need to re-think it. otherwise we are stuck with small model no
matter how many hardware we buy, or we get hit in performance. we will need to check the DGX
or 6000."*

Inputs: `research/p2p-5090-validation-20260803/NOTE.md` (P2P verdict), `research/hw-buy-20260802/REPORT.md`
(pricing sweep, cited as hw-buy), `research/dev-platform-20260802/REPORT.md` (card matrix),
`research/vast2x5090-bringup-20260803/SUMMARY.md` (validated 2x5090 numbers),
`research/m2-pp8-20260802/RESULTS.md` (M2 PP-N receipts), `research/m0-nccl-20260801/`
(transport latencies), `crates/memra-engine/src/pp.rs` (the N-stage transport). Fresh web
checks only where hw-buy was stale/missing (DGX Spark/Station, Rubin, HGX asks — sources
dated inline). Everything computed here is arithmetic on repo receipts unless labeled a
projection.

---

## 0. Verdict up front

**The P2P scare dissolves for the serving shapes we actually run.** The load-bearing number
(§1): a host-staged D2H→H2D bounce at the measured 21–25 GB/s effective adds **0.06–0.6% to a
serial PP-2 decode tick and ~1–2% to a deferred-pipelined tick** — the 1.87x deferred-readback
win measured on the 8xH100 box survives essentially intact (~1.83–1.85x projected) over a
bounce transport. We are **not** stuck with small models on 5090 fleets, and we do **not**
need the driver patch to grow. Where the missing P2P genuinely bites is tensor-parallel
all-reduce shapes (~27% of a decode tick — a shape memra deliberately does not serve),
deep pipelines (N=8: 9–14%), and nothing else at our scale.

The growth path that wins the matrix (§4) is **the hybrid (e), entered exactly as hw-buy
already recommended**: 5090 fleet for the ≤32 GB SKUs now, host-staged PP-2 opens the
33–90 GB tier on the same cards, and the first **RTX PRO 6000 Blackwell** (single card, then
pair) enters when a >64 GB-per-replica SKU has metered demand — same sm_120a, zero engine
work, native P2P, no EULA asterisk. DGX Spark is disqualified on bandwidth (273 GB/s), DGX
Station GB300 on price-per-servable-GB + a new arch lane, and Rubin is roadmap-only in the
buyable class (§3.d2) — its near-term effect on us is *cheaper used Hopper later*, which
favors, not delays, buying incrementally now.

**First buy trigger: unchanged from hw-buy** — Box #1 = 2x used RTX 5090 (~$2.7k executed
comps each) in a 6–7-slot growth chassis, ~$9.5–10k all-in; next unit when cumulative metered
gross margin since last purchase ≥ 0.5 × next unit's price.

One real work item falls out (§5): **the host-staged PP boundary transport does not exist in
pp.rs yet** — the M2 guard *refuses* when `cuDeviceCanAccessPeer=0` (exactly what the vast
2x5090 box hit). Building the pinned-bounce arm is the engine cost of path (a), and it is
days-class, not weeks-class.

---

## 1. The PP-2 bounce arithmetic (the load-bearing section)

### 1.1 What actually crosses a PP boundary

Engine truth (`pp.rs` header, M1/M2 contract): **the hidden state `[n_embd]` f32 is the ONLY
tensor that crosses a boundary.** Per token, per boundary, at bs=1 decode:

| model class | n_embd (assumed range) | bytes/hop (f32) |
|---|---|---|
| 9B class (qwen3.5-9b, verified `hidden=4096`) | 4096 | 16 KB |
| 27B class | ~5120 | 20 KB |
| hy3-class (295B-A21B) | ~6–8K | 24–32 KB |

(The task's "5–12 KB/token" figure is the fp16 spelling; pp.rs ships f32 today, so use
16–32 KB — same order, same conclusion. fp16 boundary compression is a free 2x if it ever
matters; it won't, see below.)

Prefill chunks: `chunk_tokens × n_embd × 4B` — pp512 at n_embd 5120 = **10.5 MB**; a 2048-tok
hy3-class chunk = **67 MB**.

### 1.2 Cost per hop: P2P vs host-staged bounce

- **Direct P2P / NCCL (measured, M0 on 8xH100 NVSwitch):** 7–11.5 µs per hop at 4–64 KB
  messages — latency-floor-bound, not bandwidth-bound (`research/m0-nccl-20260801/`,
  hw-buy §1.3).
- **Host-staged bounce (5090 pair, stock driver):** effective bandwidth 21–25 GB/s measured
  (tinygrad issues #35/#44 quoted in the P2P NOTE; same as NCCL SHM). At 16–32 KB the
  bandwidth term is **0.8–1.6 µs** — negligible; the cost is the latency floor of two staged
  copies (D2H pinned + H2D) plus ordering. **Projection: 30–60 µs one-way per hop**, bracketed
  deliberately wide until measured (NCCL SHM small-message latency sits ~20–30 µs; two
  cudaMemcpyAsync launches ~10–20 µs). Every number below uses the 60 µs worst case.
- Prefill chunk over the bounce: 10.5 MB / 21 GB/s = **0.50 ms**; hy3-class 67 MB = **3.2 ms**.
  (Patched P2P at ~45–50 GB/s: 0.23 ms / 1.5 ms.)

### 1.3 The hit, as a share of what the token already costs

Anchors: q27 Q8_0 on desktop 5090 = 53.63 tok/s plain decode (18.6 ms/token) and pp512 =
4151 tok/s (123 ms wall) — measured, vast bring-up 2026-08-03. Pipelined tick from M2:
q9 N=2 deferred = 346.9 tok/s (2.88 ms/token).

| shape | transport cost | tick | share |
|---|---|---|---|
| PP-2 **serial** decode, 27B class, bounce @60 µs | 60 µs | 18.6 ms | **0.32%** |
| PP-2 serial decode, same, direct P2P @11.5 µs | 11.5 µs | 18.6 ms | 0.06% |
| PP-2 **pipelined (deferred)** decode, q9-class, bounce @60 µs | 60 µs | 2.88 ms | **2.1%** |
| PP-2 pipelined, q27-class (projected 1.87x → ~100 tok/s) | 60 µs | 10.0 ms | **0.6%** |
| PP-2 **prefill**, pp512 chunk 10.5 MB @21 GB/s | 0.50 ms | 123 ms | **0.41%** |
| hy3-class prefill chunk 67 MB @21 GB/s | 3.2 ms | multi-second | <0.3% |
| N=8 pipelined, small model, 7 hops @40–60 µs | 280–420 µs | 3.0 ms | **9–14%** |
| **TP-2** (contrast): 2 all-reduce/layer × 62 layers @40 µs | 5.0 ms | 18.6 ms | **27%** |
| batched fleet hop, bs=32 (640 KB) @21 GB/s + latency | ~56 µs | tick also grows ∝bs | ~1% |

**Reading:**

1. **The bounce is invisible at PP-2 for every serving shape we ship** — decode serial
   0.3%, decode pipelined 0.6–2.1% (worst on the smallest model, where PP-2 is least needed
   anyway), prefill 0.4%. The measured 1.87x deferred-readback win, whose transport share on
   NVLink was ~0.3%, degrades to ~**1.83–1.85x** over the bounce. The P2P question is worth
   ~2% at most on the shapes that earn money.
2. **Where it DOES bite** — honestly:
   - **TP / all-reduce-heavy shapes**: 124 small messages per token → 27% of the tick at
     bounce latencies. This is why vLLM disables custom all-reduce on 5090s. memra does not
     use TP; if a future shape demands it, that is the day the pro-card/patched-P2P
     decision reopens — flagged as the standing trigger.
   - **Deep pipelines (N≥8) on small models**: 9–14%. But N=8 was already the *worst* M2
     row on NVLink (1.79x vs 1.87–1.88x at N=2/4) — the win saturates at N=2; deep pipelines
     are not the plan.
   - **KV-migration / weight-reshard events** (bulk GB-scale moves): 21–25 GB/s vs ~50 GB/s
     patched halves those; they are rare, off-tick operations.
3. Spec-decode verify hops (K+1 tokens ≈ 180 KB at K=8) ride the same arithmetic: ~9 µs
   bandwidth + latency floor ≈ still <1% of a spec tick.

**Conclusion of the section: P2P is not load-bearing for PP-2/PP-4 pipeline serving on 5090
fleets. The owner's "stuck with small model" scenario does not materialize — the growth
constraint is VRAM aggregation granularity (32 GB/card) and slot/power, not link bandwidth.**

### 1.4 The engine gap this exposes (real, bounded)

`pp.rs` today implements only `cudaMemcpyPeerAsync` for cross-device boundaries and the M2
guard **refuses loudly** when `cuDeviceCanAccessPeer=0` — exactly what the vast 2x5090 box
hit ("refusing a silently-staged path"). The P2P NOTE's bottom line stands: **build the
host-staged (pinned D2H→H2D double-buffered) boundary transport as the product default for
GeForce pairs.** Scope: one new transport arm inside the existing evented slot machinery
(ev_tx/ev_rx contract unchanged), plus the full bit-identity gate battery re-run (ppn-gate
serial + pipelined, kernel-check, run-gen, run-spec) on a 2x5090 host. Days-class agent-time;
it is the cost of path (a) and it is *also* required for paths (b)/(e) as the fallback arm,
so it is on the critical path of every 5090-fleet future — schedule it regardless of the
buy decision.

---

## 2. Scoring dimensions (per the operating model)

$/GB-of-servable-model (capex ÷ VRAM that holds paying SKUs), $/tok at fleet level (hw-buy
tables 7a/7b carry over), PP/scaling viability (§1), power/hosting reality (owner in IL —
business electricity **ILS 0.353/kWh ≈ $0.116/kWh**, globalpetrolprices.com Dec-2025, +1.5%
Q1-2026 per Channel-12 — i.e. *below* the $0.18 mid assumption hw-buy used; IL power favors
self-hosting), resale/liquidity, engine-work cost (stated as a line item, per the doctrine:
hardware pick not biased by sunk development, but new-lane cost is real agent-time).

Raw $/GB and bandwidth-per-dollar (prices: hw-buy 2026-08-02 unless newly cited):

| unit | price | GB | $/GB | TB/s per $k |
|---|---:|---:|---:|---:|
| RTX 5090 used (sold comps) | $2,700 | 32 | **84** | **0.66** |
| RTX 5090 new street | $4,099 | 32 | 128 | 0.44 |
| RTX PRO 6000 refurb | $10,250 | 96 | 107 | 0.17 |
| RTX PRO 6000 new (Newegg, re-checked 2026-08-01, Thunder Compute) | $12,099 | 96 | 126 | 0.15 |
| DGX Spark GB10 (MSRP raised to **$4,699**, OC3D 2026-02-27) | $4,699 | 128 | 37 | **0.06** |
| DGX Station GB300 (OEM $97–123k, pi3g 2026-06-17) | ~$97,000 | 288 HBM3e | 337 | 0.08 |
| H100 NVL 94GB used (executed) | $26,250 | 94 | 279 | 0.15 |
| 8x H100 SXM used server (est) | ~$120,000 | 640 | 188 | 0.22 |
| A100 80GB SXM used (needs HGX board) | $5,529 | 80 | 69 | 0.37 |

The 5090 remains the outlier on both axes simultaneously. Note the trap in raw $/GB: DGX
Spark "wins" it at $37/GB while being unable to serve (0.273 TB/s feeds ~15 tok/s-class
27B decode — decode is bandwidth-bound, receipts hw-buy §1.1). $/GB only counts **servable**
GB: GB × the bandwidth to move it.

---

## 3. Candidate paths

### (a) 5090 fleet + host-staged PP — VIABLE; the default growth spine

§1 is the whole case: bounce hit 0.3% serial / ≤2% pipelined / 0.4% prefill at PP-2. Fleet
mode (independent replicas) needs no inter-GPU traffic at all and is already validated at
~1.8x aggregate on the vast 2x5090 box (97.5 tok/s agg, 0 errors, flat c=8→32). Host-staged
PP-2 opens 64 GB (2 cards), PP-3 96 GB — the 80–150B NVFP4 class (45–85 GB artifacts) fits
on 2–3 cards per replica. Works on **any** host including rented/vast boxes (no BIOS/kernel
control needed). Engine cost: the §1.4 bounce transport (days). Risks unchanged from hw-buy
§4.0 (used consumer, EULA asterisk at colo scale, 32 GB granularity). Power: IL $0.116/kWh
makes the 6-card endgame ~$1.9–3.5k/yr at serving duty — cheaper than hw-buy's US-mid figures.

### (b) 5090 fleet + patched P2P (aikitoria fork) on owned boxes — OPTIONAL fast path, not a plan

Same arithmetic with 7–11.5 µs hops and ~50 GB/s bulk: buys back the ~0.25–1.5 percentage
points the bounce costs, plus 2x on rare bulk moves. Against that: driver-pin per CUDA
upgrade, one-volunteer-lineage rebase risk (though actively tracked 570→610 through 2026),
`iommu=pt`/ACS-off = single-tenant only, ReBAR BIOS requirement, and **the exactness battery
re-runs under every patched driver bump** (kernel-check + run-gen argmax + run-spec + ppn
bit-identity) — that recurring gate cost is the real price, and it buys ~1–2% on PP-2.
Verdict: keep as the NOTE says — flag-gated fast path on owned appliance boxes, promoted
only after gates pass under the patched driver; **never product-load-bearing**. Do not spend
it until a shape shows up where §1 says it matters (TP, N≥8, migration-heavy).

### (c) RTX PRO 6000 Blackwell multi-card — the big-SKU node; enters on SKU demand, not now

$12,099 new / $9.5–11k refurb (re-verified current, Thunder Compute + Newegg 2026-08-01/02
— hw-buy's numbers hold). 96 GB/card, native stock-driver P2P (verified both directions,
L1T Dec-2025 + our own G7e), same sm_120a fatbins byte-for-byte (proven on G7e July
bring-up), MIG, pro driver (no EULA asterisk), 3-yr warranty, best-in-class resale
(RTX 6000 Ada: 75–100% retention at 3.5 yr).

**When does it beat stacking 5090s?** Never on raw $/GB ($107 vs $84) or bandwidth-per-dollar
(0.17 vs 0.66) — and §1 removed the P2P argument for it at PP-2 scale. It wins on a
different axis: **single-card residency for the 45–90 GB SKU class** (80B NVFP4 ≈ 45 GB,
120B ≈ 68 GB, 150B ≈ 85 GB — one card, no PP at all, full KV headroom, replica granularity
of 1 card instead of 2–3), plus power/slots (192 GB = 1.2 kW in 2 slots vs 3.45 kW in 6),
plus colo legitimacy. **Does the 3.8–27B daily need it? No** — 27B Q8_0 is 28.6 GB, resident
on a 32 GB 5090, measured 2026-08-03. The PRO enters when a >64 GB-per-replica SKU has
metered demand or the fleet moves to a colo.

### (d) DGX-class — owner named it; checked, and it splits three ways

- **DGX Spark (GB10, 128 GB unified, $4,699** after the Feb-2026 +$700 memory-shortage raise,
  OC3D 2026-02-27; widely in stock — Amazon/Best Buy/Newegg): **disqualified for serving.**
  273 GB/s LPDDR5X (NVIDIA docs; Tom's Hardware review 2026-01-27) is ~15% of one 5090 —
  bandwidth-bound decode lands ~7x slower per replica; 48 SMs. Arch is **sm_121**
  (kubesimplify GB10 teardown 2026-06) — our sm_120a SASS does not load on CC 12.1; ARM
  (Grace) host = new build lane on top. Weeks-class engine work for a box that can't serve.
  At most a $4.7k big-memory *dev toy* — and G7e rentals already cover that need better.
- **DGX Station (GB300, 288 GB HBM3e @ ~8 TB/s + 496 GB LPDDR5X @ 396 GB/s, ConnectX-8):
  real but wrong price class.** Shipping via OEMs since ~June 2026 at **$97–123k**, 4–13 wk
  lead (pi3g.com 2026-06-17; aiHola pre-order ~$97k 2026-03-17). Superb single-node hy3 box
  (bank fully resident in HBM), but: $337/HBM-GB (vs $84–188 for the alternatives), one
  purchase = the whole doctrine's risk budget for ~2 years, **and a new arch lane** — GB300
  is Blackwell *Ultra* (sm_103-family, tcgen05-class kernels ≈ a second H100-scale porting
  effort) plus the Grace-ARM host lane. Two new lanes for one box. Fleet-stage candidate at
  best, and even there the used 8x H100 SXM server (~$120k, 640 GB, sm_90a lane already
  merged, NVLink native) dominates it on $/servable-GB and engine cost.
- **Used H100/A100 NVLink (HGX) systems**: the real DGX-flavored option. Fresh datapoint:
  eBay shows an "HGX H100 8-GPU barebones system bundle" pre-owned at **$15k bid / $25.1k
  BIN** (ebay.com, seen 2026-08-03 — "barebones" almost certainly means baseboard w/o GPU
  modules; flagged, quote-gated) against $11.5k/module executed — the hw-buy ~$120k
  all-in estimate stands. A100-SXM at $5.5k/module is cheap VRAM ($69/GB) but sm_80 = third
  arch lane + no FP8/FP4 + V100-decay phase: still rejected (hw-buy §4.10). The 8x H100
  box remains the *fleet-stage* buy — see the Rubin note below for why waiting on it is free.

### (d2) Rubin generation (owner addition) — roadmap-only in the buyable class; verified state

1. **Does a Rubin-based Spark/Station successor exist?** **No — roadmap-only.** NVIDIA's
   Computex-2026 roadmap puts **Vera Rubin Spark-class (LPDDR6) at 2027–2028** and Rosa
   Feynman at 2029–2030 (Tom's Hardware 2026-06-01; VideoCardz 2026-06-01: "Rubin in 2027").
   No price, no specs beyond the memory generation, nothing orderable.
2. **Datacenter Rubin** (the "massive improvements" the owner heard, real): R100/VR200 —
   288 GB HBM4, up to 22 TB/s (~2.75x GB300's 8 TB/s), NVLink 6, 3rd-gen Transformer
   Engine; VR200 NVL72 claimed 3.3x GB300-NVL72 inference (tech-insider GTC-2026 analysis
   2026-06-04; Thunder Compute 2026-07-31; cudocompute VR200 page). **Sampling Q4 2026,
   volume Q1 2027, DGX Rubin rack $3.5–4M** (~$50k+/GPU), NVL72 racks at 190–230 kW —
   hyperscaler-channel only, not a small-buyer product in any 2026–2027 window.
3. **Engine cost, honestly:** Rubin is a new SM generation — new compute capability (exact
   CC unpublished; CUDA-13 arch tables today top out at sm_110 = Thor and sm_121 = GB10,
   Arnon Shimoni table upd. 2026-04-13), new tensor-core ISA on top of a Grace/Vera ARM
   host for any DGX-shaped unit: **two new lanes**, H100-porting-scale or worse, unpriceable
   until hardware and nvcc support exist. Not a 6–12-month engine plan.
4. **Wait-for-Rubin vs buy-Blackwell-now:** for the Spark/Station class the wait is
   **2027–2028 against specs that don't exist** — waiting means 12–18 months of zero fleet
   revenue to maybe save on a box class we've disqualified anyway (Spark) or can't justify
   (Station). For our actual path the Rubin ramp is a *tailwind*: hw-buy already cites
   orangehardwares (~2026-07-26) expecting Vera Rubin volume to cut another 10–20% off
   Hopper secondaries in late 2026 — i.e. the fleet-stage used-HGX-H100 buy gets **cheaper**
   while the earnings meter runs. Incremental buying + Rubin-driven Hopper deflation is
   strictly better than waiting: capital at risk stays per-card-sized, resale on used-bought
   gear holds (~half back), and the big-quantum purchase lands post-deflation. **Verdict:
   row noted, wait rejected; revisit when a Rubin Spark/Station has a price and a CC.**

### (e) Hybrid — 5090 fleet for ≤32 GB SKUs + one big-VRAM node for large SKUs — the winner

Growth = replicate what earns, per the operating model. Concretely: 5090 cards carry the
9–35B lane (deepest receipts, $0.38–0.47/Mtok — hw-buy 7b), host-staged PP-2/PP-3 on the
same cards carries the 45–90 GB class *until* its demand is proven, then the first PRO 6000
(single card, then pair at 192 GB for hy3-class) takes that lane over with single-card
residency, KV headroom, and colo legitimacy. The fleet-stage jump (≥$60k trigger) buys the
used 8x H100 SXM server at post-Rubin-deflation prices. Every stage is sm_120a or sm_90a —
**zero new engine lanes across the whole path.**

---

## 4. Decision matrix

Scores 1–5 (5 best), per dimension. $/tok carries hw-buy tables 7a/7b; PP viability is §1;
engine cost is the line item (5 = zero new work).

| path | $/GB servable | $/tok (small SKU) | $/tok (hy3 SKU) | PP/scaling | power/hosting (IL) | resale | engine cost | legal/ops | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| (a) 5090 fleet + host-staged PP | 5 ($84/GB) | 5 ($0.38–0.47) | 4 ($1.43 @6x) | 4 (PP-2/3 clean; N≥8/TP out) | 4 (residential→240V; colo EULA asterisk) | 4 (~55%+ used-bought) | 4 (bounce arm, days) | 3 | **33** |
| (b) (a) + patched P2P on owned boxes | 5 | 5 | 4 | 4.5 (+~1–2% back; TP still unwise) | 4 | 4 | 2 (driver treadmill + per-bump gate battery) | 2 (unsupported fork) | 30.5 |
| (c) RTX PRO 6000 2-card | 3 ($107–126/GB) | 2 ($1.07–1.18) | 3 ($3.51) | 5 (native P2P, single-card 96 GB) | 5 (1.2 kW, pro driver, colo-clean) | 5 (75–100% class) | 5 (zero) | 5 | 33* |
| (d) DGX Spark | 1 (GB w/o bandwidth) | 1 | 1 | 1 (273 GB/s) | 5 | 3 | 1 (sm_121 + ARM) | 4 | 17 |
| (d) DGX Station GB300 | 2 ($337/HBM-GB) | 1 | 4 (resident hy3) | 4 | 3 (1.5 kW, fine) | 3 (no track record) | 1 (sm_103 + ARM, two lanes) | 5 | 23 |
| (d) used 8x H100 SXM | 3 ($188/GB) | 1 (wrong silicon for small) | 5 ($2.57, best clean) | 5 (NVLink) | 2 (6 kW colo-only) | 3 (~45–48% post-crash) | 5 (sm_90a merged) | 4 | 28 |
| (d2) wait-for-Rubin | — | — | — | — | — | — | 1 (two unpriceable lanes) | — | rejected (roadmap-only; §3.d2) |
| **(e) hybrid: (a) now → (c) on big-SKU demand → 8xH100 at fleet stage** | 5→3 staged | 5 | 4→5 staged | 5 | 4→5 | 4–5 | 4 (bounce arm only) | 4 | **35** |

\* (c) ties (a) on points but loses as the *first* box: 3x the entry capex for 2.5x worse
$/Mtok on the lane that mints margin now (hw-buy 7b) — it is the right *second-stage* node,
which is exactly what (e) encodes.

### Matrix winner per model-size class

| SKU class | artifact size | winner | why |
|---|---|---|---|
| 3.8–27B daily (incl. q27 Q8_0 28.6 GB) | ≤30 GB | **5090 fleet replicas** | resident/card, no inter-GPU traffic at all; measured 53.6 tok/s + 1.8x pair aggregate |
| 30–64 GB (35B dense hi-quant, 70B Q4) | 33–64 GB | **5090 pair, host-staged PP-2** | bounce 0.3–0.6% of tick (§1); $84/GB |
| 80–150B NVFP4 (45–85 GB) | 45–90 GB | **RTX PRO 6000 single card** (5090 PP-2/3 as the bridge until demand proven) | single-card residency, KV headroom, replica granularity 1; $107/GB buys no-PP ops |
| hy3-class (~150 GB+) | 150–190 GB | **PRO 6000 pair (192 GB)** now; **used 8x H100 SXM** at fleet stage | 42 GB KV headroom / best clean $/Mtok at scale, sm_90a merged |

---

## 5. Engine-work line items (agent-time is money; not hidden)

| item | trigger | scope | class |
|---|---|---|---|
| **Host-staged PP boundary transport** (pinned D2H→H2D double-buffer arm in pp.rs; guard downgrade from refuse→bounce) | now — critical path of (a)/(e); also the fallback for (b) | transport arm + ppn-gate bit-identity + full battery on a 2x5090 host | **days** |
| Bounce-latency measurement (replaces the 30–60 µs projection in §1.2) | with the above | one bench on the first owned/rented pair | hours |
| Patched-P2P qualification (aikitoria) | only if TP/N≥8/migration shape appears | gate battery per driver bump, recurring | days + recurring tax — deferred |
| PRO 6000 bring-up | first PRO purchase | zero (sm_120a proven on G7e; g7e-rtx6000.jsonl exists) | ~0 |
| 8x H100 bring-up | fleet stage | zero (sm_90a merged; darklanes receipts) | ~0 |
| DGX Spark (sm_121 + ARM) / Station (sm_103 + ARM) / Rubin (unknown CC + ARM) | not on the path | new arch lane(s), H100-scale each | weeks–months — avoided by this plan |

## 6. Staged buy plan (earnings-gated, per the doctrine)

Trigger rule (unchanged): **buy the next unit when cumulative metered gross margin (revenue
minus power/colo) since the last purchase ≥ 0.5 × next unit's price.** IL power at $0.116/kWh
business rate improves every net figure vs hw-buy's $0.18 mid.

| stage | buy | price | unlocks | trigger |
|---|---|---:|---|---|
| **1 (now)** | 2x used 5090 + 6–7-slot TR/EPYC chassis | ~$9.5–10k | 9–35B fleet SKUs; PP-2 dev on owned silicon; the bounce-transport measurement | first purchase — the meter starts here |
| 2 | cards #3–4 (used 5090, $2.7k each) | ~$5.4k | more replicas; PP-2 64 GB tier live (70B Q4, 80B NVFP4) | ~$1.35k net per card |
| 3 | card #5–6 OR first RTX PRO 6000 (refurb $9.5–11k) | ~$2.7–11k | 6x5090: 192 GB + hy3 PP-6; PRO: 45–90 GB SKU single-card + colo-clean node | 5090: $1.35k/card; PRO: ~$5k net **and** metered demand for a >64 GB SKU |
| 4 | second PRO 6000 → 192 GB pair | ~$10–12k | hy3-class resident PP-2 on pro silicon, native P2P, colo | ~$5–6k net + hy3 listing decision |
| 5 (fleet) | used 8x H100 SXM server | ~$120k, falling (Rubin deflation, 10–20% expected late 2026) | 4x hy3 PP-2 replicas, best clean $/Mtok | ~$60k cumulative net — by construction ~2 yrs of proven demand; price the tape again at trigger time |

Path-dependence note: stage 3's fork (more 5090s vs first PRO) is decided by the meter —
if >64 GB SKU demand hasn't shown, cheap 5090 GB keeps winning; the PRO never needs to be
bought on faith.

## 7. Biggest uncertainties (ranked)

1. **The 30–60 µs bounce-latency bracket is a projection** — measured on the first owned
   pair (hours, free with box #1). The conclusion survives the bracket: even 100 µs is 0.5%
   of a q27 serial tick and 3.5% of the worst-case pipelined tick.
2. **Deferred-readback pipelined arm is not serving-default-cleared** (~0.5% cross-device
   flake open, M2 RESULTS) — the 1.87x is an engine prize pending root-cause, independent
   of transport choice; serial PP-2 (bit-identical, free at N=2) carries until then.
3. Desktop-5090 35B-class serving rate: now partially de-risked (q27 Q8_0 measured 2026-08-03);
   the 35B-A3B-class rate on desktop silicon is still projected.
4. Demand capture at 30% utilization (hw-buy's #2) — unchanged, still the true business risk,
   and the per-card quantum is still the hedge.
5. "Barebones" HGX pricing ambiguity and Rubin-deflation timing — both only matter at
   stage 5, ~2 years out, re-priced then.

## Source index (fresh fetches 2026-08-03; all else cited to hw-buy/dev-platform/NOTE inline)

- DGX Spark price raise to $4,699: overclock3d.net 2026-02-27. Specs/stock: docs.nvidia.com
  DGX Spark hardware guide; tomshardware.com GB10 review 2026-01-27 (273 GB/s); Best
  Buy/Amazon/Newegg listings. sm_121: blog.kubesimplify.com GB10 teardown (2026-06-05).
- DGX Station GB300: pi3g.com 2026-06-17 ($97–123k, 4–13 wk lead, 748 GB = 496 LPDDR5X @
  396 GB/s + 252–288 HBM3e); aihola.com 2026-03-17 (~$97k OEM pre-orders); servethehome.com
  2026-03-30.
- Rubin: tomshardware.com + videocardz.com Computex-2026 roadmap (2026-06-01): Vera Rubin
  Spark 2027–2028 (LPDDR6), Rosa Feynman 2029–2030; tech-insider.org GTC-2026 Rubin analysis
  (2026-06-04): sampling Q4 2026, volume Q1 2027, DGX Rubin rack $3.5–4M; thundercompute.com
  Rubin architecture (2026-07-31): R100 288 GB HBM4, 22 TB/s; cudocompute.com VR200 page;
  moduledge.com (190–230 kW/rack).
- RTX PRO 6000 current: thundercompute.com pricing 2026-08-01 ($13,250 marketplace / $12,099
  Newegg / PNY ~$11,360); newegg.com listing live.
- Used HGX: ebay.com search 2026-08-03 (barebones bundle $15k bid/$25.1k BIN — flagged);
  mercatus-ai.com h100-server-price (~$285k new typical).
- IL electricity: globalpetrolprices.com Israel Dec-2025 (business ILS 0.353/kWh ≈ $0.116);
  sadanews/timesofisrael Q1-2026 +1.5%.
- CUDA arch tables: arnon.dk (upd. 2026-04-13) — sm_110 Thor, sm_121 GB10, sm_120a RTX 50.
