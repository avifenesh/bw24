# H100 chunk-16 inheritance receipt (2026-08-02, Mumbai H100, v0.62 tree)

The docs-v062 sweep flagged: MEMRA_Q8RP defaults ON under memra_hopper_mma, so the
9B fleet model auto-selects the EXACT-16 chunk tier on Hopper with only 5090 receipts.
Closed same-night on the board box (build 4m00s, gate under flock /tmp/gpu-h100.lock):

    ./target/release/decode-batch-gate ~/models/Qwen3.5-9B-Q8_0.gguf --batch 16
    gate2 (B=16 vs isolated batched-B=1, bit-checked, 32 steps): PASS
    gate3 (device sampling: greedy==host-argmax + sampled B=16 vs isolated + lean-logits identity): PASS
    ALL GREEN: decode_step_batch exactness battery

gate1-config on H100: green within the battery (its calibration rig). The exact-16
tier is now bit-receipted on BOTH arches; the Aug-2 fleet cap re-sweep measures its
throughput there (policy receipts, this file, is the exactness half).
