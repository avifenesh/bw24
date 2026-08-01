# Round 50 — per-model MEMRA_MOE_F16G gate: gemma-MoE (gelu) class defaults OFF (2026-08-01)

Rig: Mumbai H100 SXM (darklanes-bench, 13.203.16.133), single-occupancy board box.
Tree: rsync'd main-lane tree + the two-file fix below; release build 2026-08-01 ~10:17Z.

## The regression (why)

Round 49 promoted `MEMRA_MOE_F16G=1` (grouped f16 expert prefill) as the Hopper default for
all MoE classes on the strength of the q35 board win (+53%) and a +6-15% g26 probe verdict.
The probe verdict did not survive the board workload (the stale-verdict law): on board-2048
real-text prefill, g26 (gemma-4-26B a4b, gelu MoE) REGRESSED -8.3%.

Arbitration A/B (`g26-f16g-ab.log`), N=5 interleaved pairs, same session, on-box:

| arm | prefill tok/s (5 runs) | median |
|---|---|---|
| def (f16g on)  | 8923.6, 10362.6, 10380.3, 10423.7, 11729.2 | **10380.3** (wild spread 8.9k-11.7k) |
| off            | 11317.4, 11340.7, 11319.0, 11311.7, 11314.4 | **11317.4** (±0.13%) |

## The fix

Per-model gate `moe_f16g_gemma_on()` (`crates/memra-engine/src/lib.rs`): true only when
`MEMRA_MOE_F16G` is explicitly set to a non-`0` value. The gemma gelu-MoE dispatch site
(`crates/memra-engine/src/hybrid_forward.rs`, ~line 4673) now consults it — naked runs skip
the grouped f16 lane on the gemma class. The silu/qwen admission site (~line 3186,
`moe_f16g_on()`) is untouched: q35 keeps the round-49 Hopper default. `MEMRA_MOE_F16G=0`
still kills everywhere. Branch source is byte-identical to the tree the Mumbai binaries
were built from (diff-verified both files).

## Gate verification (each row: run-gen, board-2048.txt prompt, MEMRA_NGEN=32)

| model | env | prefill tok/s | argmax gate | N | log |
|---|---|---|---|---|---|
| g26 | naked | 11322.4 | prefill=decode=236772 MATCH (maxdiff 1.848e0) | 1 | r50-gate-g26-naked.log |
| g26 | MEMRA_MOE_F16G=1 | 10244.3, 9828.0 | MATCH both (maxdiff 1.666e0) | 2 | r50-gate-g26-f16g1.log |
| q35 | naked | 8454.6 | prefill=decode=485 MATCH (maxdiff 8.304e-1) | 1 | r50-gate-q35-naked.log |
| q35 | MEMRA_MOE_F16G=0 | 5497.6 | MATCH (maxdiff 8.402e-1) | 1 | r50-gate-q35-f16g0.log |

Verdicts: gemma door gated OFF naked (11.3k = the off arm), explicit =1 still opens it
(10.2k/9.8k = the f16g arm's signature and spread), q35 silu default untouched (naked 8.4k
vs 5.5k at =0). Single runs labeled as such; the off arm's ±0.13% A/B band makes N=1
sufficient for the door-state check (the two arms are far outside each other's bands).

## Battery

- q35 `tools/validate-h100.sh --quick`: **ALL GATES GREEN** (`r50-battery-q35.log`) —
  kernel-check, decode-batch config B=8 + strict, decode-dc, graph-decode, graph-session.
- g26 `--quick` (`r50-battery-g26.log`): kernel-check green (`r50-kernel-check.log`:
  "ALL GREEN: kernels match CPU reference."), decode-dc PASS, run-gen argmax MATCH (above).
  decode-batch(config+strict) panics `decode_step_batch v1 covers the hybrid non-gemma4
  trunk only` and both graph gates error `"slotted tail: dense ffn only"`
  (`r50-g26-graph-decode.log`, `r50-g26-graph-session.log`) — the ledger-documented
  PRE-EXISTING gemma-MoE coverage gap (Round 46 first-time-gating note: lockstep decode
  rejects gemma4 BY DESIGN; the gemma graph door on MoE was never Hopper-gated). Not a
  round-50 regression; the fix touches only the prefill f16g dispatch.

## The new naked g26 board cell (tools/h100-vllm-board.sh, p2048/g512, N=5 medians both arms)

Run 2026-08-01T10:53:26Z, same-session block, GPU idle at start
(`r50-board-g26.log`; raw per-run logs `g26-memra.log`, `g26-vllm.log`, `g26-vllm.json`;
rows appended to `h100board-vllm-20260731-realtext.jsonl` by the harness):

| arm | artifact | prefill tok/s | decode tok/s | e2e tok/s (2560/(2048/pp+512/dec)) |
|---|---|---|---|---|
| memra | gemma-4-26B_q4_0-it.gguf | **11337.1** | **210.30** | **978.9** |
| vllm | RedHatAI/gemma-4-26b-a4b-it-FP8-dynamic | 43964.4 | 194.73 | 956.7 |

**e2e ratio memra/vllm = 1.023x** (at the regressed def median 10380.3 the same cell would
have been 972.7 = 1.017x). Prefill is back at the off-arm band (11337.1 vs the A/B off
median 11317.4); decode 210.30 is the cell's best to date (prior rows: 180.87 -> 182.09 ->
204.57).

## Receipts

All raw logs in this directory; board jsonl snapshot included. Cross-run/cross-day
comparisons remain clock-drift-invalid per the lane laws — the A/B pairs and the board
cell are each same-session interleaved/blocked measurements.
