# Product-vision assessment — gate for the Aug-2 8×H100 window

Date: 2026-08-01. Owner rule being applied: *a box is rented (and its hours spent) only when
the product vision is strong, delivered, proved, and assessed.* The Aug-2 block
(cr-02251913b8f9ea0a6, $996.67, 2026-08-02T11:30Z → 2026-08-03T11:30Z, non-cancellable) was
purchased before this rule landed; this document is the assessment that gates its mission.

## The vision (one paragraph)

darklanes is a managed serving product: multi-GPU fleets running an exactness-gated engine
(memra) on a small set of **high-demand** OSS models, sold on reliability (chaos-tested
admission/breaker/supervisor stack) and competitive $/tok on owned or rented H100 capacity.
memra's public per-model board is the credibility engine; darklanes is the revenue surface.

## What is PROVEN (receipts in-repo)

| claim | evidence |
|---|---|
| Fleet serving stack works under load and failure | 1,477 tok/s managed on 3×H100, +7% on v0.60 core, 18/18 greedy-hash identical, chaos kill→recovery in 9s losing only in-flight (research/fleet-v060-20260801/) |
| Engine beats the field reference per model | 7/7 e2e wins vs vLLM 0.26 on the H100 board, decode 7/7 (round 50, README board) |
| Multi-GPU foundation is real | M0 comms floor measured (peerAsync 2.8× NCCL at PP sizes, PP ~0.3-0.5% of tick); M1 PP-2 seam bit-identical on two engine paths (research/m1-pp2-20260801/) |
| Serving demand exists and is measurable | live OpenRouter 4-day usage: Hy3 4.8T tok on 5 endpoints; V4-Flash 705M requests (#1 open model); GLM-5.2 premium tier $0.76/$2.39 (research/model-demand-20260801/) |

## What is NOT yet proven

- **GLM-5.2 serving** (the prior primary pick): MLA is at reference-math + loader-code stage.
  Real bring-up is a multi-week track. Premium-tier economics remain attractive, but the
  engine is not ready — betting the window on it would be spending $1k on an unready idea.
- **A servable multi-GPU SKU**: no model has yet ticked across ≥2 GPUs in memra
  (M1 transport lands in lane/m1-inc2; PP8/microbatch is unbuilt).
- **Demand durability**: OpenRouter windows are 4 days; V4-Flash sits in a 22-provider
  price war. Demand-to-competition, not raw demand, is the metric that matters for a
  small fleet — which is exactly why Hy3 (4.8T tok, five endpoints) ranks first.

## The redirect the demand data forces

GLM-5.2-as-first-SKU was **engineering-led** (its attention is the interesting problem).
The demand evidence says the first servable SKU should be **Hy3**: top-5 demand, the
thinnest competition on the board, **zero new kernel classes** (GQA+MoE+MTP — all
board-proven in memra), our spill/quant lane already invested, and a local-audience story
(890K+430K GGUF downloads/30d). 295B-A21B at NVFP4 ≈ 2×H100 per replica → PP-2 (exactly
M1) makes an 8×H100 box a 4-replica Hy3 fleet. GLM-5.2 stays the premium roadmap item,
gated on MLA maturity; V4-Flash is the fast-follow sharing GLM's kernel classes.

## Aug-2 mission (only proven/ready items ride)

1. **Hy3 PP-2 serving spike** (product centerpiece): prove one Hy3 replica ticks exactly
   across 2 GPUs (M1 transport + existing Hy3 arms), then measure a 4-replica fleet point.
   Exit artifact: exactness gates + a tok/s + $/Mtok table vs the 5 incumbent endpoints.
2. **M1 multi-device finish** (gates scripted in lane/m1-inc2) — the enabler for (1).
3. **Fleet cap re-sweep** 8/12/16 on the v0.61 core (in-window denominators only).
4. **M2 PP8 development hours** with whatever remains — microbatch scaffolding.
5. **GLM-5.2 weights pull + tensor audit ONLY as low-priority fill** (~1-2h of NVMe and
   CPU time keeps the premium option alive without betting GPU hours on it).

## Verdict

The serving stack, the engine board, and the demand data are strong and receipted; the
gap is the multi-GPU SKU, and the window is scoped to close exactly that gap on the
highest demand-to-competition model that needs zero new kernels. Vision: **assessed —
strong, with the SKU order corrected by demand evidence.** Future rentals: this document
class gets written and owner-reviewed BEFORE purchase.
