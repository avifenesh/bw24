# ARM B' — device-side block-128 FP8 -> Q8_0 dequant pass

Lane `lane/fp8-gemm-arm`, 2026-08-03. Rig: RTX 5090 Laptop (sm_120a), nvcc 13.1.

Kernel: `crates/memra-engine/cu/fp8_blk_dequant.cu`.
Wrapper: `Engine::fp8_blk_dequant_q8_0` (`crates/memra-engine/src/fp8_ffi.rs`).
Wiring: `model.rs::load_from_source`, gated `MEMRA_FP8_BLK_GPU=1` (default OFF).
Gate: `kernel-check` arm `fp8-blk-gpu`.

ARM A (already merged) folds the 128x128 scale grid into ONE per-tensor scale — lossy where
block dynamic range varies. ARM B' keeps every block's own scale and lands on Q8_0's finer
per-32 grid, so precision is class-equal-or-better than the existing CPU path, at the cost of
one load-time device pass. No new GEMM code: the output slab rides the existing Q8_0 MMQ/MMVQ
path unchanged.

## Gate 1 — synthetic byte-parity (kernel-check `fp8-blk-gpu`)

GPU output vs the host reference (per-block dequant `fp8_e4m3_to_f32(code)*grid[(o>>7)*cols +
(e>>7)]`, then `nvfp4_repack::f32_to_q8_0`). All 256 e4m3 codes cycled through the input; grid
spans ~2^-4..2^5 across blocks so the per-32 amax / f16-`d` path is exercised in several binades.

| shape | Q8_0 bytes | mismatching bytes | verdict |
|---|---|---|---|
| 256 x 512 | 139264 | 0 | OK |
| 136 x 160 | 23120 | 0 | OK (ragged out=136 row tail + ragged in=160 segment tail) |
| 8 x 32 | 272 | 0 | OK |

`ALL GREEN: kernels match CPU reference.`

Three exactness decisions the parity gate forced (all in the kernel's header comment):

1. e4m3 decode is the host's closed-form bit math, **not** `__nv_cvt_fp8_to_halfraw`. The
   intrinsic returns NaN for magnitude `0x7F`; `nvfp4_repack::fp8_e4m3_to_f32` returns 0.0
   (modelopt convention). Using the intrinsic poisons every block containing that code.
2. `rintf` (round-to-nearest-**even**), not `roundf`. Rust's `round_ties_even()` is RNE;
   `roundf` is ties-away-from-zero and disagrees on exact .5 products.
3. `id = 1/d` from the f32 `d`, not the f16-rounded `d` — same as the host.

## Gate 2 — real-checkpoint argmax

No block-128 FP8 checkpoint exists on this rig. Scan of `/data/ai-ml/hf-models` (2026-08-03):
the only FP8-bearing local checkpoint is `nvidia-qwen36-27b-nvfp4`, which ships per-tensor
`weight_scale`, not the `weight_scale_inv` grid — 0 local dirs carry a block grid.

So the gate input is a genuine block-128 FP8 checkpoint built from the local Qwen3-1.7B BF16 ST
dir by `make_blk128_fp8_ckpt.py` (in this directory): 196 2-D Linear weights -> `F8_E4M3` codes +
BF16 `weight_scale_inv` of shape `[ceil(out/128), ceil(in/128)]`, per-block `s = amax/448`
(bf16-rounded before encode so the loader's decode reproduces the exact `s`). Dynamic range
genuinely varies block to block — the property ARM A's global fold destroys. Written to
`/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth` (2.65 GB, 507 tensors).

`run-safetensors`, CPU-path load vs `MEMRA_FP8_BLK_GPU=1`, 4 prompts:

| prompt tokens | argmax | logit |
|---|---|---|
| 9419 11 1814 0 | 220 | 18.8825 |
| 1 2 3 4 | 5 | 22.5813 |
| 151643 9707 11 1879 30 33464 | 6832 | 19.3140 |
| 40 264 3766 315 | 279 | 21.1706 |

`diff argmax-cpu.log argmax-gpu.log` is **clean** — MATCH, and stronger than argmax-only: the
full top-5 logit values are bit-identical on every prompt. Expected, since the pass is
byte-parity with the host re-encode: same bytes in, same bytes out, same kernels after.

## Load-time measurement

Interleaved CPU/GPU pairs, same box, same session (no cross-run or cross-day comparison), whole
run-safetensors process wall on the 2.65 GB synthetic 1.7B block-128 checkpoint. N=3, warm page
cache (`loadtime.log`).

| iter | CPU re-encode wall (s) | GPU pass wall (s) |
|---|---|---|
| 1 | 35.652 | 9.218 |
| 2 | 35.751 | 9.218 |
| 3 | 35.149 | 9.318 |

Median: **35.652 s -> 9.218 s = 3.87x** faster load; **26.4 s** of host dequant+re-encode
removed on 2.65 GB of FP8 weights (~10 s/GB). This is the whole-process wall, so the ratio is a
floor for the pass itself — engine init, tokenizer, and the forward are in both numbers.

The 27B ST checkpoint the mission asked about is not present locally (`qwen36-27b-hf-min` is
config/tokenizer only, no shards), so the synthetic-checkpoint timing above is what this rig can
measure. Scaling the per-GB rate: a full 27B block-128 FP8 checkpoint (~27 GB of FP8 Linear
weights) would shed roughly 4-5 minutes of load wall.

## Files

- `make_blk128_fp8_ckpt.py` — builds the block-128 FP8 gate checkpoint from a BF16 ST dir.
- `argmax-cpu.log` / `argmax-gpu.log` — the 4-prompt argmax gate, raw, both paths.
- `loadtime.log` — the 3 interleaved timing pairs, raw.
- `rungen-cpu.log` / `rungen-gpu.log` — `run-gen` on this dense arch panics
  ("not a hybrid arch"); the argmax gate ran through `run-safetensors`, which is the correct
  harness for a dense ST checkpoint. Kept so the dead end is on the record.
