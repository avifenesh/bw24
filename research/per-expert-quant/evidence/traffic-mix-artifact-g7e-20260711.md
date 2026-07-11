# Traffic-shaped mixed artifact — G7e, 2026-07-11

This is the promoted follow-up built from the corrected non-REAP routing trace. Selection used no
public-evaluation data. Experts are ranked independently inside each of the 79 MoE layers by router
selection count, with ascending expert ID as the deterministic tie-break.

| Per-layer rank | Encoding | Experts/layer | Layer-expert slots | Projections | Expert bytes | Cumulative routing traffic |
|---|---|---:|---:|---:|---:|---:|
| 1–16 | Q8_0 | 16 | 1,264 | 3,792 | 25,348,276,224 | 22.3201% |
| 17–53 | NVFP4 | 37 | 2,923 | 8,769 | 31,032,999,936 | 49.8495% |
| 54–126 | Q2_K | 73 | 5,767 | 17,301 | 35,715,907,584 | 84.9830% |
| 127–192 | pruned | 66 | 5,214 | 0 | 0 | 100.0000% |

The expert overlay is **92,097,183,744 bytes (85.772 GiB)**. Adding the same
24,999,514,624-byte non-expert payload used by every arm gives **117,096,698,368 logical bytes
(109.055 GiB)**. This is 37.06% smaller than `plain_quant` and 2.01% smaller than
`mix_quant_prune25`. Logical size describes the shared body plus sparse expert overlay; a later
standalone GGUF export can differ because of container metadata and common-tensor encoding.

## Provenance and staging identity

- Repacker implementation: `e0480548119b413c478c0a5dbaf3f0fc7029966b`.
- Calibration trace:
  `/data/runs/hy3-calibration-normfix-66394bf-worker-d8-full.trace`.
- Trace SHA-256: `01a05fe0f928bc04963cba581a3cc09afcce7e737c9bac5ce5ed7345056b6adf`.
- Matched routing events: 15,168 layer records, each covering 163,409 prompt tokens.
- Source artifact:
  `/data/artifacts/per-expert-quant-traffic-e048054/traffic-mix-quant`.
- Scratch artifact:
  `/scratch/artifacts-per-expert-quant-traffic-e048054/traffic-mix-quant`.
- Both directories are exactly 92,110,819,729 bytes.
- Both manifests have SHA-256
  `c7780aeb9da28b57a5def5207d3cc26ce243542ef12c3b7bd3c4f2bd59b8e587`.

The N=1 directional screen scored 2/7 and therefore promoted this arm to the clean matched N=50
comparison. That directional result is only a gate; it is not evidence that quality aligns with
the uniform baseline.
