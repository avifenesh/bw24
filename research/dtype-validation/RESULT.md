# Dtype validation vs ggml ground truth (PASSED 2026-06-26)

The 5 new dtypes (NVFP4/Q5_K/Q3_K/IQ3_S/IQ4_XS) were added to dequant.rs + qmatvec.cu and validated
against ggml's OWN dequantize_row_<type> (tools/ggml_dequant_ref.cpp links libggml), on REAL tensors
from the daily GGUFs (9B-NVFP4, 35B-MoE). Harness: tools/run_dequant_validation.sh.

| dtype | tensor | Stage-A qmatvec vs ggml | Stage-B dp4a vs ggml |
|---|---|---|---|
| NVFP4 | 9B blk.0.ffn_gate | rel 2.6e-6 | 5.3e-3 |
| Q5_K  | 9B blk.0.attn_gate | 4.9e-7 | 9.6e-4 |
| IQ3_S | 35B blk.0.ffn_gate_exps | 8.9e-8 | (Stage-A only) |
| IQ4_XS| 35B blk.0.ffn_down_exps | 3.4e-8 | 1.2e-4 |
| Q3_K  | 35B blk.40.ffn_gate_exps | 1.4e-7 | 6.4e-4 |

ALL GREEN. Discipline-debt closed: dtypes proven correct vs an INDEPENDENT ggml reference, not just
internal consistency. dp4a int8-activation-quant error ~1e-3 (expected, matches llama.cpp MMQ).
