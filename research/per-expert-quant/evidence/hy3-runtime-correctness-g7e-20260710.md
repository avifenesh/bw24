# Hy3 runtime-correctness gate on G7e (2026-07-10)

Machine: AWS G7e, RTX PRO 6000 Blackwell Server Edition 96 GB. Branch:
`feat/per-expert-quant`; corrected runtime commit: `38a5b08` (norm fix: `0a4b5b9`).

## Failure and root cause

The first public-eval attempt was invalid: greedy text was incoherent and all resulting scores were
therefore discarded. Tokenizer round-trip, expert NVFP4 bytes, routing bias, FAST/MMVQ, and the
stage-A path were checked before a source-reference layer oracle localized the first catastrophic
divergence to dense layer 0.

`resolve_ggml()` had applied Qwen 3.5's `NormPlusOne` transform to every architecture for which
`Arch::is_hybrid()` was true. Hy3 uses `HybridModel` as a dense-attention MoE runtime, but its
official RMSNorm multiplies by the raw checkpoint weight. On the live checkpoint, the erroneous
transform changed layer-0 input-norm mean from about `0.00407` to `1.00407` and post-attention-norm
mean from about `0.02141` to `1.02141`. Commit `0a4b5b9` scopes Qwen MTP/SSM/+1 transforms to
`Qwen35 | Qwen35Moe`; the independent MiniMax Gemma-norm arm is unchanged and Hy3 norms are plain.

## Official-reference gate after the fix

The diagnostic compares the real eager T=1 serving path against a shard-selective implementation
of Transformers commit `d610229d0f0d80c7927694f164e3dd362750ca19`. Tokens `120000 120044`
exercise the chat prefix at positions 0 and 1.

| Stage | token 120000 cosine | token 120044 cosine | token 120000 RMSE | token 120044 RMSE |
|---|---:|---:|---:|---:|
| embedding | 0.999936 | 0.999986 | 0.000941 | 0.0000328 |
| attention output | 0.999946 | 0.999955 | 0.000500 | 0.000384 |
| after-attention residual | 0.999957 | 0.999953 | 0.001064 | 0.000397 |
| dense MLP output | 0.999763 | 0.999967 | 0.000106 | 0.000663 |
| layer-0 residual | 0.999961 | 0.999979 | 0.001026 | 0.000739 |

Durable evidence:

- `/data/results/per-expert-quant/diagnostics/hy3-layer0-fixed-20260710T174500Z/bw24.jsonl`
  SHA-256 `8be862e01c49edaf8144e519cd11d299eba2e958ebfc319159c80c8195b515b1`;
- sibling `reference-bf16.jsonl`
  SHA-256 `132c33baf9d30332c70a41759eb62baef31c5f18245a74873e7acc5caa7b89af`.

The remaining small deltas are expected runtime quantization effects: large BF16 matrices are
re-encoded to Q8_0, activations use Q8_1, and KV is quantized.

## Full-generation smoke

With `FAST=0`, no prewarm/prefetch, and the uniform NVFP4 full-bank artifact, the prompt
`The capital of France is` produced 16 greedy tokens:

```text
 Paris.
The capital of Germany is Berlin.
The capital of Italy is
```

Response SHA-256: `ae6502f68430a537b05a452e9a2bcead0cb102b96ca6cba916bfec14f8e151d4`.
This is a coherence/correctness smoke, not a speed result (`97.805 s` in the deliberately slow
configuration).

## Explicit-I/O parity gate

Before the norm fix, `BW24_SPILL_IO=pread` and mmap produced byte-identical two-token diagnostic
JSONL (both SHA-256
`0416929bcfd62096340f4fb1838062c382aa864d8224e328759b38e8639aef7e`). The positioned-read run
performed 2,352 reads / 8,323,596,288 bytes with zero read errors, short reads, mmap fallbacks, or
buffer waits. This proves byte and H2D parity; blocking demand pread is not the throughput endpoint.

## Invalidated routing evidence

The following traces were captured before both the routing-bias mapping fix and the norm fix and
must never select tiers or support results:

- `/data/runs/hy3-calibration.trace`, SHA-256
  `f75c71be8bb77d3ab62e02b85e28c4bea5ce30c9abc438857937dfbeec949e2f`;
- `/data/runs/hy3-reap50-calibration.trace`, SHA-256
  `a2ac0a8c17e1a2e6d20fe787df3dc0c7fbea31176b7713cbc29cd365c5a535cc`.

Fresh commit-tagged traces must be captured from the unchanged frozen request corpus before
usage-ranked plans or quality evaluation are valid.
