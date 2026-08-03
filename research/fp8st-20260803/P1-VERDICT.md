# P1 — cuBLASLt block-scaled FP8 on sm_120: NOT SUPPORTED (2026-08-03)

Probe: `probe/fp8_lt_blk_probe.cu`, heuristic-only mode (host-side
`cublasLtMatmulAlgoGetHeuristic` query — deterministic API answer, no GEMM launched, no perf
claim). Raw log: `p1-blk-heuristic-5090laptop.log` (single run; support-matrix queries are
deterministic, N>1 adds nothing). Rig: RTX 5090 Laptop (sm_120), CUDA 13.1, cublasLt 130201,
run under `flock /tmp/gpu5090.lock`. Shape: q_gate 12288x5120, m = 512/2048/4096,
D = R_32F and R_16BF both probed.

| combo | A scale (weight) | B scale (act) | status | nh | verdict |
|---|---|---|---|---|---|
| 1 control | SCALAR_32F | SCALAR_32F | 0 | 4 | SUPPORTED (the shipped MEMRA_PP_FP8 config) |
| 2 | BLK128x128_32F | SCALAR_32F | 7 (INVALID_VALUE) | 0 | NOT_SUPPORTED |
| 3 DeepSeek recipe | BLK128x128_32F | VEC128_32F | 15 (NOT_SUPPORTED) | 0 | NOT_SUPPORTED |
| 4 | BLK128x128_32F | OUTER_VEC_32F | 15 | 0 | NOT_SUPPORTED |
| 5 | VEC128_32F | VEC128_32F | 15 | 0 | NOT_SUPPORTED |

Identical status at every m and both D dtypes — this is an arch/library gate, not a shape or
epilogue miss. Consistent with the July g7e finding (OUTER_VEC status=7 nh=0 on sm120,
g7e-rtx6000.jsonl:39): sm_120's cuBLASLt FP8 exposes ONLY per-tensor scalar scales in this
CUDA release. The CUDA 12.9+ block-scaled FP8 modes are documented for sm_90; they do not
reach sm_120 in cublasLt 130201.

## Consequence for the FP8-ST serve path (the GEMM architecture decision)

cuBLASLt cannot consume Qwen-official block-128 scales directly on the target arch. Two viable
consumers for the resident `Fp8BlockScales` grid, to be priced in the P1 follow-up:

1. **Scale-fold pre-pass**: dequant-requant the e4m3 weight ONCE at load against its block
   grid to a per-tensor-scalar e4m3 operand (scale = grid max; tiles far below max lose
   mantissa — accuracy must be gated, this is a real re-quant unlike the per-tensor case),
   then ride the SUPPORTED scalar path at the probed 668-779 TF class. Zero kernel work.
2. **Own kernel**: extend the `MEMRA_MMQ_F8F4` MMA tile (381-TF class, FLAGS.md:131) with
   per-128-block scale dequant — k-blocks of 128 align with the MMQ k-tile; the grid indexes
   `(o>>7)*cols + (e>>7)` off the resident layout as documented on `Fp8BlockScales`.
   ~half the Lt ceiling, fully deterministic, no accuracy compromise.

The Q8_0 re-encode floor (B1a) stays the correctness baseline for both: it already dequants
block-128 exactly, host-side, per-32-finer grid.

Caveat: verified on the laptop 5090 (sm_120). The desktop 2x5090 (sm_120a) shares the arch and
library; re-confirm there when the box frees up — one `./fp8_lt_blk_probe` run, heuristic-only.
