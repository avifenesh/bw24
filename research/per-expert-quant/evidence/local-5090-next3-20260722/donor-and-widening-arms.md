# 2026-07-22 evening arms: residency donors + E-core widening

Same protocol as the day's receipts (N=32 post-freeze windows, guards, argmax MATCH on every
scored run); most arms restore the freeze profile, so runs cost ~6 min. Single runs unless
noted — directional screens, not board rows.

| arm | tok/s | frozen blocks / complete experts | verdict |
|---|---:|---|---|
| anchor (winner config + profile) | 4.60 | 5285 / 1719 (restored) | band reference |
| KV-fp8, old profile | 4.85 | 5285 / 1719 (restored) | attention-side signal, N=1 |
| KV-fp8, fresh rewarm | 4.29 | **5275 / 1672** | REJECT as-wired: freed KV never reaches the expert budget (`VRAM_FRAC` samples free VRAM before the KV allocation change matters) and the rewarmed residency set is worse |
| KV-fp8, restore of that profile | 4.08 | 5275 / 1672 | confirms the worse set, not variance |
| threads=16 (P+E OMP team) | 3.49 | — | REJECT: compute 2.8 → 5.2 s |
| threads=20 | 1.67 | — | REJECT: compute 14.6 s — static-loop barriers straggle on Skymont cores; E-cores stay io-only |
| `VRAM_FRAC` 0.90→0.92, fresh rewarm | 4.74 | **5403 / 1757** (+118 / +38, no OOM) | mechanism CONFIRMED, magnitude sub-noise (+2.2% blocks ≈ +1.5% predicted) |
| frac92, restore | 4.60 | 5403 / 1757 | in band |

Conclusions carried forward:

1. The residency-donor mechanism works through the budget knob and scales per the Phase-0
   curve; a measurable arm needs ~0.94 (≈ +1 GB ≈ +380 blocks ≈ +4-5% predicted). OOM risk
   rises — run with the failure-capture guard.
2. KV-fp8 must not be re-tested as a donor until the expert-budget computation happens after
   (or is made aware of) the KV allocation change; as an attention-side lever it has one
   positive N=1 run and needs pairs.
3. Naive OMP team widening across heterogeneous cores is dead — any future E-core compute
   needs asymmetric partitioning (explicit work split), not shared worksharing loops.
4. Freeze profiles pin residency by design: donor experiments must rewarm, never restore —
   this session's method note for every future HBM-budget arm.
