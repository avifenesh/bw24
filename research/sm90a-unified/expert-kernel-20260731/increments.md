# mmq_iq_experts kernel-rate arc — round 45 (2026-07-31, H100 darklanes-bench)

Baseline: g26 pp1736 (depth-prompt-1736-ids, MEMRA_NGEN=4, interleaved pairs, N=3 each):
ncu at baseline: occupancy 12.5% (Block Limit Registers=1), SM 13-20%, DRAM 1.5-3%,
long_scoreboard = 66% of warp-stall samples (123.5k/187k) — global-latency underfill.

| increment | g26 pp1736 (pairs) | delta | verdict |
|---|---|---|---|
| baseline (v0.55.0 kernel) | 5036/5039/5032 vs — | — | — |
| launch_bounds minblocks=2 | 5036 vs 4798 | -4.7% | REFUTED (reg spill > occupancy) |
| MMQ_X=64 tile | 5041 vs 4582 | -9.1% | REFUTED (j-reuse > occupancy) |
| inc1: Y gather 16B cp.async | 5041 vs 6775 | +34.4% | SHIPPED |
| inc2: W kb-slice cp.async staging ring | 6776 vs 9577 | +41.3% | SHIPPED |
| inc3: Y half ping-pong (wait_group 1) | 9561 vs 10117 | +5.8% | SHIPPED |

Cumulative: 5041 -> 10117 tok/s = 2.01x g26 prefill. Kernel duration (ncu, nc form):
3.85ms -> 2.00ms at inc2. Stalls after inc2: long_scoreboard 123.5k -> 9.3k;
`wait` (fixed-latency dep chains) now dominant (23k) at 12.5% occupancy.

Numerics: BYTE-IDENTICAL data movement (same tile contents, same mma order) — g26
argmax MATCH with logit maxdiff constant (5.527e0) across every increment; q35 argmax
MATCH; kernel-check ALL GREEN (incl. NC26 + RAGK pins).

q35 prefill: +0.85% only (5334 -> 5379 board-2048) — q35's prime wall is not this
kernel's Q4_0 path share.

nsys prime share: 240 mmq_iq_experts calls ~= 95% of the g26 timed prime (the 52k-call
kernels in the trace are the argmax gate's per-token decode-verify pass, not the prime).

NEXT RUNGS (measured): `wait` dep-chain stalls at 2 warps/scheduler — occupancy via
register diet (accumulator tiling) or deeper ILP; W-ring depth 2+ with group-counted
waits; IQ3_S staging (110B rows need 2B tail handling).

## inc4 + the q35 form (post-v0.56.0 addendum)

| increment | cell | result |
|---|---|---|
| inc4: skip clamped-column gathers | q35 5447->5461, g26 10130->10169 | +0.3% both — kept (free, correct), not a win |

q35 ncu: TWO kernel forms. Down (16,256): SM 59.9%, short_scoreboard-dominant — near this
structure's ceiling. Gate/up (4,252): 3.11ms, long_scoreboard — groups average ~65 pairs
so every CTA runs ONE half-empty 128-token tile over 8 k-blocks; the whole shape is a tiny
per-expert GEMM (512 out x 65 tok x 2048 k). 64-token tile REFUTED ON PAPER: avg 65 > 64
means half the groups take 2 passes = 2x W dequant for the same mma. The fix class for
this shape is expert-BATCHED GEMM (CUTLASS grouped int8 / the vLLM shape) — a separate
arc; prime is ~15% of the q35 e2e wall, so the e2e leverage is bounded (~2-3%).
