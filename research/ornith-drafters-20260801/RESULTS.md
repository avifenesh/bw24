# Ornith/KAT own-gen trimmed drafters — results (2026-08-01)

All GPU runs on the rig under `flock /tmp/gpu5090.lock`, single session (same window);
e2e ratios are same-invocation interleaved (plain generate then spec in one process),
N=3 medians at the serving K. Acceptance under greedy decoding is deterministic per
(prompt, K); single-run tok/s cells in the K-sweep tables are labeled by that fact.
Raw logs: `gates/<model>/`. Corpora: `corpus/`. Artifact shas: `manifests/`.

## Corpora (own-gen, 254-prompt canonical pack, ngen 512, greedy, chat template ON)

| target | tokens | distinct ids | top-32768 coverage | ranks sha (gguf) |
|---|---|---|---|---|
| Ornith-9B | 127,390 | 10,639 | 100.00% | d40d7f4f |
| Ornith-35B | 128,617 | 11,367 | 100.00% | a423317a |
| KAT-Coder | 119,273 | 12,198 | 100.00% | e2aed7f6 |

Same corpus class as the supported builds (q9 109k / q35-daily 108k / gemma 130k).

## Gate 1 — run-spec K=1..8 self-consistency (spec ≡ plain, acceptance > 0)

**PASS 8/8 for all three drafters** (`gates/<model>/gate-k1-8.log`).

## Gate 2 — acceptance tables K=2..4 (ngen 256, board prompt classes)

See `acceptance-ornith9b.md`, `acceptance-ornith35b.md`, `acceptance-katcoder.md`.

## Gate 3 — e2e spec vs plain at the serving K (x3 medians) + verdicts

| model @K | p1-code-short | p2-code-medium | p3-agentic-long | verdict |
|---|---|---|---|---|
| Ornith-9B @K=3 | **2.16x** (61.1%) | **1.77x** (47.0%) | **1.70x** (47.8%) | **ADOPT** — sweeps every class |
| Ornith-35B @K=2 | **1.38x** (65.9%) | **1.09x** (63.8%) | **1.05x** (63.8%) | **ADOPT** — wins every class |
| KAT-Coder @K=2 | 1.09x (82.5%) | 0.91x (61.7%) | 0.85x (55.4%) | **NO ADOPT** (e2e law) |

KAT K=1 probe (`gates/katcoder/probe-k1-*.log`): p2 1.00x (wash), p3 0.95x — no K
rescues p2/p3. KAT carries the BEST acceptance of the batch (donor block transfers
cleanly to the coder post-train) but its plain decode is the anomaly (~104 tok/s vs the
supported q35's ~170 on the same arch class — the onboarding lane saw the same gap), so
draft rounds cost more than they save outside code-short. Artifact + ranks stay on /data
with manifests (instant re-verdict if KAT's decode gets fixed — that slowness is its own
future lane).

Reference bars (supported family, same protocol): q9 @K=3 70.3%/1.91x, 54.6%/1.59x,
49.5%/1.42x; q35 @K=2 80.6%/1.56x, 65.3%/1.35x, 63.3%/1.27x. Donor drift costs the
Ornith pair 7-15 acceptance pts on code-short and ~0-2 pts on p2/p3; per-model e2e
ratios are not cross-artifact comparable (different plain-decode denominators — the
Ornith-9B Q8_0 ratio EXCEEDS the reference because its plain decode is slower).

Ornith-35B runs its experts on the SLRU spill cache on this 24.5GiB card (21.2GB
Q4_K_M): its spec rounds pay expert fetches, compressing the ratio vs the resident
reference. The lane also fixed the gen-graph door crashing this configuration
(commit b5a08d23, receipts in `corpus/ornith35b-owngen.log`).

## Serving

```
MEMRA_MTP_DRAFT=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/draft-ornith9b-owntrim-nvfp4head-q4blk.gguf   # + MEMRA_SPEC_K=3
MEMRA_MTP_DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf # + MEMRA_SPEC_K=2
```
