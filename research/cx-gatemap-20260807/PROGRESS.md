# cx-gatemap progress — 2026-08-07

Branch: `lane/cx-gatemap`

Base: `80f47796`

Gate-map commit: `6af888d1`

## Finding

`amargin`, `amarginc`, `e4b`, and `kat` were valid `models.tsv` probes with pinned behavior, but
no `map.tsv` row could select them. Explicit `--probes` runs worked; change-scoped dispatch could
never reach them.

All required model artifacts fit the 5090 gate regime, so none needs an explicit-only size exemption.
The fix keeps `DEFAULT = g12,q9,q35,gwstress` and assigns narrow homes instead of charging every
unmatched change for four extra model boots.

## Dispositions

| Probe | Automatic home | Rationale |
|---|---|---|
| `amargin` | `decode.rs`, `forward.rs`, `hybrid_forward.rs`, `argmax_margin_probe.rs`, and `tools/argmax-margin-gate.sh` | The gate compares tokenwise `decode_step` with batched `forward_last`; these are the implementation surfaces it actually executes. It is not mapped to `spec_sample.cu` or the spec-control files because it would not exercise those changes; `accept` remains the served-spec acceptance/long-text gate. |
| `amarginc` | Same rows as `amargin` | The fault-injected wide-margin row is the comparator's teeth. Pairing it with the real arm also prevents wrapper/probe changes from silently weakening the parser. |
| `e4b` | Gemma attention TUs, `kernels.cu`, `decode.rs`/`hybrid_forward.rs`, `hybrid.rs`/`model.rs`/`lib.rs`, and `memra-gguf` | E4B shares Q4_0 kernels with other Gemma arms but uniquely exercises per-layer embeddings, KV sharing, short-window MQA, and E4B-only glue fusions. It is not blanket-added to every Q4_0 row already covered by `g12`/`g26`. |
| `kat` | `qmatvec.cu`, `mmq_iq_experts.cu`, `decode.rs`/`hybrid_forward.rs`, `hybrid.rs`/`model.rs`/`lib.rs`, and `memra-gguf` | KAT shares qwen35moe geometry with `q35`, but it is the only registered arm with non-expert IQ4_XS trunk weights. It uniquely reaches `qmatvec_iq4_XS_dp4a` and the dense IQ4_XS MMQ arm. Shared router, hybrid, and spill rows retain their cheaper representative probes. |

## Verification

- `git diff --check`: PASS.
- TSV structure: every non-comment `map.tsv` row has 4 fields; every `models.tsv` row has 6.
- Map references: every probe named by columns 3/4 exists in `models.tsv`.
- Registry coverage: 30 registered IDs, 30 dispatched IDs, empty set difference both ways.
- Exact planner block from `fast-gate.sh:71-147`: synthetic changes selected the expected new
  probes for `qmatvec.cu`, `mmq_iq_experts.cu`, `flash_attn.cu`, `kernels.cu`, `decode.rs`,
  `forward.rs`, `model.rs`, `lib.rs`, `memra-gguf/src/config.rs`,
  `src/bin/argmax_margin_probe.rs`, and `tools/argmax-margin-gate.sh`.
- Tier-0 command (`MEMRA_GATE_LOGDIR=/tmp/fast-gate-cx-gatemap-tier0-final-20260807`):
  `tools/fast-gate/fast-gate.sh --tier 0` GREEN. The real diff contains only the two fast-gate
  TSVs, so the plan correctly built the workspace and selected no kernel-check or GPU probe.
