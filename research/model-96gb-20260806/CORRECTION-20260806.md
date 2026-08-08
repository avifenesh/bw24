# CORRECTION (2026-08-06): OpenRouter pool $/day figures were trailing-window totals, not per-day

The demand-validation lane (darklanes repo, `exp/122b-earning-case.md`, commit dfbc0c4) found
the pool arithmetic in this assessment and `../model-192gb-20260806/ASSESSMENT.md` reads the
OpenRouter frontend rankings API rows as per-day when they are **trailing-window totals**
(verified by construction: `week/day = 7.14x`, `month/day = 30.73x` on the same model).

Corrected headline figures:

| page | published | real |
|---|---|---|
| qwen3.5-122b-a10b | $1.6K/day | **~$700/day**, of which ~71.4% Alibaba-captive → **~$200/day third-party addressable** |
| step-3.7-flash | $89.3K/day | **~$41.5K/day** |

Ratios between pages survive (Step vs 122B stays ~59x), so the *relative* rankings in these
assessments stand. Absolute $/day arguments do not — re-derive before using any pool number
from these files. The 122B listing verdict flipped to **NO-EARN as a revenue SKU**
($23–55/day gross at measured third-party capture; 11x opportunity cost vs q27's $2,224/day
third-party pool; page −27.5% in 5 days, generational decline). The 122B remains onboarded
(gqa=16 coverage + the fa_v4 overflow find made bring-up worthwhile); the *listing* decision
is a darklanes product call — see the darklanes repo for the full case and receipts.
