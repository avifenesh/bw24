# Local 5090 MTP trim and native batching decision

Target: NVIDIA GeForce RTX 5090 Laptop GPU, driver 595.71.05, 24,463 MiB HBM.
Runtime base: merged native bw24 Hy3 serving at `40f02fb`. These are single observations in the
active-desktop thermal regime, not medians and not performance-board numbers.

## Frozen test profile

- Full Hy3 Layer-103.5 runtime from the verified dual-NVMe view.
- 45-token chat-templated prompt, N=32 for the paired screen.
- 13.82 GiB HBM expert budget, 20 GiB native CPU expert cache, eight P-core compute and I/O
  workers, direct I/O, and 128 discarded K=3 tokens before residency freeze.
- Full-vocabulary MTP artifact for the control. The trim arm used the existing 32,768-token MTP
  artifact and changed no other intended knob.

## 32k MTP vocabulary trim: reject

| Arm | HBM blocks | Plain N=32 | K=1 N=32 | Acceptance | Exactness |
|---|---:|---:|---:|---:|---|
| full vocabulary | 5,080 | 3.72 tok/s | 3.14 tok/s | 55.0% | PASS |
| 32k trim | 5,037 | 2.95 tok/s | 2.59 tok/s | 60.0% | PASS |

The trim recovered only 43 HBM blocks while losing 20.7% plain throughput and 17.5% K=1
throughput. The higher single-run acceptance did not offset the verification slowdown. The local
default therefore remains the full-vocabulary MTP head.

Raw logs: `fullvocab-k1-chat-ngen32-run1.log` and `trim32k-k1-chat-ngen32-run1.log`.

## Native speculative verification batching: reject

The experimental implementation was entirely bw24-native. It added a versioned multi-token C ABI,
grouped identical CPU experts across frozen speculative verification tokens, loaded/prepared each
unique expert once, kept token/expert accumulation order exact, and overlapped the one CPU batch
with resident GPU work. It did not compile, link, or load llama.cpp or ggml. The packed-row oracle
passed all supported formats and required byte-for-byte equality between one batched call and the
corresponding repeated token calls. The tested companion SHA-256 was
`f927cf1056ff2ac1e1f3f62f155ba23e7669125f4a07fbeeca7977f4df761834`.

The matched N=32 screen was flat:

| Path | Plain | K=1 | K=1 acceptance | CPU read volume | CPU compute |
|---|---:|---:|---:|---:|---:|
| tokenwise control | 3.72 tok/s | 3.14 tok/s | 55.0% | 405.721 GB | 57.407 s |
| native batch | 3.71 tok/s | 3.15 tok/s | 55.0% | 405.721 GB | 56.463 s |

The candidate then passed self-consistency at every K from 1 through 8. Across that gate it reduced
38,962 CPU-routed items to 27,538 unique expert groups (29.3% reuse), but the larger working sets
did not become an end-to-end win. K=6 through K=8 remained 1.60, 1.43, and 1.25 tok/s. The gate's
CPU profile reported 160.951 s compute and 817.986 GB reads; the prior tokenwise full-depth receipt
reported 117.794 s and 819.035 GB. That cross-run profile comparison is directional because the
short output and frozen residency differed, while the matched N=32 pair above is the decision
measurement.

`native-batch-v3-k1to8-chat-ngen8-run1.log` is an unscored harness interruption: the foreground
runner received exit 143 before model load produced output. Run 2 used a durable user service,
completed with status 0, and is the scored full-depth gate.

Raw logs: `native-batch-v3-k1-chat-ngen32-run1.log`,
`native-batch-v3-k1to8-chat-ngen8-run2.log`, and `cpu-native-batch-v3-check-final.log`.

## Decision

Do not bump the companion ABI, add a batching flag, or change the local default. The implementation
was removed under the winners-only rule. Cross-request serving batches remain a separate aggregate
throughput feature; this result rejects only grouping one request's speculative verification rows.
The next local lever must reduce the per-token CPU/storage wall itself rather than depend on sparse
router overlap that did not materialize here.
