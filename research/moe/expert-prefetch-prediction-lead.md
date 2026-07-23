# Lead: prediction-guided expert prefetch (unworked, candidate lever for the 4.6 → 10 tok/s climb)

Status: CLOSED NEGATIVE (2026-07-23) — built and measured through three A/B arms; see
`expert-prefetch-prediction-pilot.md` for the receipts. The router-signal window is too short
to move MB-scale weights and the long-lead window carries no signal (lead-time/precision
scissors), on top of a measured concurrent-DMA compute tax. Machinery retained env-off.

## The idea

We already event-fork the copy stream for next-expert prefetch during matmul; the missing piece is
knowing *which* experts to warm before their layer's router runs. External result worth porting:
predict upcoming fetch addresses from an intermediate layer's signal and issue the fetches early,
overlapping them with the remaining layers' compute.

Source: TF-Engram (arXiv 2607.07388, July 2026), measured on our hardware class (RTX 5090 desktop
32 GB, i9-14900K, 64 GB DRAM, NVMe). Their "early-exit guided predictive prefetching" uses an
intermediate layer's output distribution to predict likely memory keys before decoding completes:

- baseline 481.2 tok/s; SSD-backed memory without prefetch 439.2; with top-64 predictive prefetch 460.3
- external-memory latency overhead cut from 9.57% to 3.20%

Their fetches are KB-scale phrase-memory rows, not experts, but the overlap structure is the same.

Related prior art establishing that expert choice is predictable from earlier-layer signal:
Pre-gated MoE (route layer k+1 with layer k's gate), SiDA-MoE (offline predictor for expert
activation), EdgeMoE. Check these before designing the predictor; the cheap options are "reuse
previous layer's router logits" and "tiny probe on layer-k hidden state".

## Regime caveat — do not port blindly

TF-Engram's fetches are KB-scale and latency-bound; Hy3 expert fetches are MB-scale and
bandwidth-bound against the ~7 GB/s NVMe wall. A misprediction there wastes latency; here it wastes
the scarce resource itself and can evict useful cache. Any port must be precision-weighted:

- confidence-gated, top-1/top-2 prefetch — not their top-64 spray
- speculative loads enter the SLRU as lowest-priority insertions; a mispredicted expert must never
  displace a hot resident entry
- account prefetch traffic separately in spill benchmarks so a "win" isn't just cache warming from
  doubled reads

## Cheapest first measurement (no GPU time beyond runs already happening)

Log per-layer router decisions during existing Hy3 eval runs, then measure predictability offline:
given layer-k router logits (or hidden state), what fraction of layer k+1..k+m expert selections are
predicted in the top-1/top-2? If hit rate is high (prior art says it should be), wire the
confidence-gated prefetch into the existing copy-stream fork. If it is low on Hy3 specifically,
record the negative result here and drop the lead.
