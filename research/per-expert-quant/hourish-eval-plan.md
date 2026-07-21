# One-hour directional capability panel

This panel answers one narrow question: does a smaller Hy3 expert policy retain enough capability
to justify a deeper evaluation? It is a directional screen, not a final benchmark claim.

## Compared arms

All NVIDIA arms use the same pinned Hy3 source, bw24 server, tokenizer, prompts, and non-expert
payload. Logical size is the shared 24,999,514,624-byte body plus the expert overlay.

| Arm | Policy | Logical bytes | Logical GiB |
|---|---|---:|---:|
| `plain_quant` | all experts NVFP4 | 186,035,622,400 | 173.259 |
| `plain_reap_quant` | public REAP50 expert selection, retained experts NVFP4 | 105,517,568,512 | 98.271 |
| `mix_quant` | hottest 25% NVFP4, middle 50% Q3, coldest 25% Q2; no pruning | 150,249,820,672 | 139.931 |
| `mix_quant_prune25` | 25% NVFP4, 25% Q3, 25% Q2, 25% pruned | 119,496,397,312 | 111.290 |
| `traffic_mix_quant` | 22% traffic Q8, to 50% NVFP4, to 85% Q2, last 15% traffic pruned | 117,096,698,368 | 109.055 |

The exact public `pipenetwork/Hy3-REAP50-MLX-4bit` checkpoint is an optional external reference,
not one of the first NVIDIA arms. It is MLX affine-4-bit and requires an Apple Silicon host with at
least 128 GB unified memory. The NVIDIA control above isolates the public REAP50 expert selection
using the same source weights and NVFP4 runtime as the other bw24 arms.

`mlx-reap50-reference.lock.json` pins the public checkpoint revision, the Hy3-capable MLX runtime
revision, and the same panel and lm-eval identities. Run that reference with greedy decoding,
one request at a time, and no draft model. Compare its paired quality outcomes to the NVIDIA arms,
but keep it outside the artifact-size Pareto calculation because its format and runtime differ.
`run_hourish_mlx_reference.sh` applies that lock to an already-running pinned MLX server and refuses
an artifact manifest, runtime receipt, or draft-model declaration that differs from the lock.
On macOS, run it with Homebrew Bash 4+ and GNU coreutils (`gtimeout` is detected automatically).
After both runs are complete, `summarize_hourish_external_reference.py` independently validates
the MLX artifact/runtime receipts and produces a paired quality comparison against `plain_quant`.
It records MLX size for context but marks it as excluded from the NVIDIA artifact-size Pareto.

## Frozen question budget

`hourish-panel.lock.json` records exact dataset revisions, lm-eval commit, document indices,
document IDs, strata, and hashes of every document, rendered prompt, and target. The selector is
deterministic and excludes the held-out programming calibration documents.

| Domain | Public task | Questions | Generation ceiling | Plain-quant projection |
|---|---|---:|---:|---:|
| Programming | HumanEval instruct | 14 | 512 tokens | ~29.5 min |
| Math | MATH-500, balanced across subject and level | 32 | 256 tokens | ~15 min |
| History | MMLU-Pro history, source-stratified | 5 | 256 tokens | ~10.8 min |
| Other knowledge | MMLU-Pro other, source-stratified | 5 | 256 tokens | ~9.7 min |
| **Total** | | **56** | | **~65 min** |

Model loading adds about three minutes for `plain_quant`; smaller arms may load or answer faster.
There is no one-hour wall-clock truncation. Every arm must finish every frozen question even when
it takes longer than the projection; the runner has only a 12-hour per-shard failure failsafe.

## Runtime contract

- Greedy decoding, temperature 0, one request at a time.
- No MTP, speculative decoding, prompt-response reuse, or candidate-specific generation setting.
- Same frozen server binary and lm-eval checkout for all NVIDIA arms.
- Each arm starts from a fresh server and exact one-model health validation.
- Model generation and scoring are separate for code. Generation uses lm-eval prediction-only;
  generated code is later executed in a resource-limited, network-disabled container.
- MATH generation and scoring are also separate. The API server did not enforce the task's
  `Problem:` stop marker, so the raw harness aggregate is excluded. A network-disabled container
  scores only the first non-empty answer line (or an `answer:` clause on that same line), ignores
  all later corrections and leaked prompts, and verifies equivalence with pinned Math-Verify
  0.9.0 plus narrow MATH-style literal normalization. No answer is regenerated or repaired.
- Results are accepted only when all 56 expected sample records exist and document, prompt, target,
  generation-config, runtime, artifact, and server hashes match the lock and the other arms.
- Preserve wall time, output-token counts, spill counters, server logs, and immutable receipts, but
  quality—not speed—is the screening decision.

## Scoring and precommitted decision

- Programming: pass@1 over the 14 sandboxed code questions.
- Math: strict first-answer equivalence over 32 questions, from the locked scorer receipt rather
  than the invalid raw harness aggregate.
- History and other: exact choice match over five questions each.
- Report each domain independently, unweighted total correct out of 56, and a domain-balanced macro
  in which programming, math, history, and other each contribute 25%.
- Report paired wins/losses/ties against `plain_quant`, exact sign tests, and paired bootstrap
  intervals. With this small panel these quantify direction; they do not prove equivalence.

An arm is **aligned enough to keep pushing** when all are true:

1. its total correct is no more than three questions behind the best arm;
2. its domain-balanced macro is no more than 7.5 percentage points behind the best arm; and
3. it does not score zero in a domain where the best arm scores at least two questions correctly.

Among aligned arms, the smallest logical model is the efficient finalist. Any arm that beats the
best larger arm remains on the Pareto frontier even if another aligned arm is smaller. Only those
survivors advance to a larger trusted evaluation.

## Calibration evidence

Plain-quant programming calibration used held-out indices 0 and 1 from HumanEval and MBPP, with the
same greedy 512-token ceiling. The HumanEval pair completed in 253 seconds, projecting 14 frozen
HumanEval questions to roughly 29.5 minutes plus harness setup. Both HumanEval answers passed the
isolated scorer. MBPP was excluded before any candidate run because the harness filter removed a
leading `def` and the model emitted replacement characters at code line endings; the screen does
not normalize or repair model output. No generated code was executed during calibration. The
calibration output is preserved under
`/data/results/per-expert-quant/hourish-calibration/plain_quant/` on the G7e host.

## Expanded matched triage panel

`expanded-capability-panel.lock.json` is the next-stage directional panel for parallel multi-GPU
triage. It keeps the same four domains, pinned lm-eval commit, dataset revisions, generation
ceilings, greedy decoding, and one-request-per-arm contract, but freezes 115 untouched questions:

| Domain | Public task | Questions |
|---|---|---:|
| Programming | HumanEval instruct | 32 |
| Math | MATH-500, balanced across subject and level | 56 |
| History | MMLU-Pro history, source-stratified | 10 |
| Other knowledge | MMLU-Pro other, source-stratified | 17 |
| **Total** | | **115** |

The seed is `bw24-expanded-panel-v1-20260712`; the frozen lock SHA-256 is
`33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b`. Every index in the
56-question hourish lock is explicitly recorded as excluded, as are HumanEval calibration indices
0 and 1. The lock records selected document IDs plus document, rendered-prompt, and target hashes.

Rebuild it only from the pinned harness after `prepare_harness.py` has applied `suite.lock.json`:

    HF_HOME=/data/cache/huggingface \
      /data/cache/lm-eval-97a5e2c710e2/.venv/bin/python \
      research/per-expert-quant/build_hourish_panel.py \
      --profile expanded \
      --output /tmp/expanded-capability-panel.lock.json

Then validate the rebuilt lock before comparing it byte-for-byte with the committed lock:

    python3 research/per-expert-quant/validate_capability_panel.py \
      /tmp/expanded-capability-panel.lock.json --print-sha
    cmp /tmp/expanded-capability-panel.lock.json \
      research/per-expert-quant/expanded-capability-panel.lock.json

Run each arm through the existing restartable launcher by selecting the new frozen lock; no prior
hourish result directory is edited or reused. Parallel arms may use distinct loopback ports through
`ADDR`, but one arm must keep the same port for all of its shards:

    PANEL_LOCK=research/per-expert-quant/expanded-capability-panel.lock.json \
      ADDR=127.0.0.1:8081 ARM=... ARTIFACT=... RUN_ID=... SERVER_BIN=... \
      research/per-expert-quant/run_hourish_one_arm.sh

The launcher, code scorer, math scorer, and summarizer derive task counts and panel SHA from that
validated lock. On Linux, harness checkout, preparation, and environment setup take one cache-wide
`flock`; revision injection is atomic and becomes a read-only no-op once prepared, so parallel
launchers can share the pinned evaluator. The supported macOS/MLX path remains single-run when
`flock` is unavailable. Container scorers take one host-wide lock and run with the minimum
Docker CPU share plus a one-CPU quota. Each completed HumanEval or MATH shard starts its scorer in
the background under that lock while the same server advances to later generation shards. The arm
waits for those scorer receipts before declaring generation complete; its final scoring pass remains
an idempotent completeness check. Expanded scorer receipts bind the count and SHA explicitly. The
summarizer accepts the new server binary only when every arm and shard has the same hash; pass
`--server-sha256` to pin that hash explicitly. It records and validates each loopback endpoint but
does not treat the port as model identity across parallel arms. It also recomputes document,
task-rendered prompt, target, and lm-eval runtime fingerprints from every logged sample before
scoring. The legacy hourish lock, its fixed endpoint and server requirements, and its existing
receipts remain valid.

    python3 research/per-expert-quant/summarize_hourish_results.py \
      --out-root /data/results/per-expert-quant/expanded-v1 \
      --run-id "$RUN_ID" \
      --arms "$ARMS" \
      --baseline plain_quant \
      --panel-lock research/per-expert-quant/expanded-capability-panel.lock.json \
      --server-sha256 "$SERVER_SHA256" \
      --output /data/results/per-expert-quant/expanded-v1/summary.json
