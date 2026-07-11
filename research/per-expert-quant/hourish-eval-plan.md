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
| `mix_quant` | 25% NVFP4, 50% Q3/Q2, 12% pruned by measured use | 150,249,820,672 | 139.931 |
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
- Results are accepted only when all 56 expected sample records exist and document, prompt, target,
  generation-config, runtime, artifact, and server hashes match the lock and the other arms.
- Preserve wall time, output-token counts, spill counters, server logs, and immutable receipts, but
  quality—not speed—is the screening decision.

## Scoring and precommitted decision

- Programming: pass@1 over the 14 sandboxed code questions.
- Math: exact match over 32 questions.
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
