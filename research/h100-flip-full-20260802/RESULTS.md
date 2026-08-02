# h100-flip-full: mode 2 at full direct coverage + deep tail vs cublas mode 1 — FLIP, +52.6% (2026-08-02)

Lane `lane/h100-flip-full` (from `restructure/public-split` e42cc8e1 — the tree with
lane/iq-direct-loaders AND lane/sk-tail-form merged). Rig: darklanes-bench H100 80GB HBM3
(13.203.16.133), tree rsync'd to `~/memra`, `MEMRA_CUDA_ARCH=90a` nvcc 13.1 release build
(4m00s clean). Every GPU phase under `flock /tmp/gpu-h100.lock`; GPU idle at session start
(0 MiB, zero compute apps). Model: `~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (q35), prompt
`research/e2e/prompts/board-2048.txt`.

## The question

Round 54 (lane/h100-sk-direct) answered NO-FLIP: cublas 8547.1 / sk+direct 8112.1 (94.9%),
and priced the gap as coverage — direct loaders then covered only Q4_K/Q6_K = 5.2% of q35's
expert-bank bytes. Since then the tree gained (a) IQ4_XS/IQ3_S direct tiles
(lane/iq-direct-loaders — coverage now ~100% of q35's bank, worth +50.5% on the 5090
mode-2 arm) and (b) the 32x64x64 3-stage deep tail (lane/sk-tail-form, sm_80-portable).
Re-asked with both: does `MEMRA_MOE_F16G=2` flip past the cublas mode-1 Hopper default?

## Gates first

kernel-check sm_90a **ALL GREEN rc=0, 240 OK, 0 FAIL** (`kc-flipfull.log`) before any
measurement:

- `f16g-kq-direct`: iq4_xs + iq3_s synthetic AND **real q35 weights via ~/models**
  (blk.0.ffn_gate_exps IQ3_S in=2048 out=512; blk.0.ffn_down_exps IQ4_XS in=512 out=2048),
  every visitor form (hybrid / all-128 / all-32-deep-tail / all-32-legacy-tail) vs the
  workspace path: **maxdiff=0.00e0 byte-identical, 16/16**, on top of the q4_K/q6_K synth
  arms (8/8).
- `f16g-sk` deep-tail + legacy-tail arms vs grid-scan 0.00e0 incl the in_f=480 %64-fallback;
  explicit `f16g-sk-tail deep vs legacy 0.00e0`.
- KC-SKIPs are the known absent-model class only (Ornith/KAT/NVFP4 ggufs not on this box —
  same set as round 54).

## Three-arm probe — interleaved x5 process rounds, round-robin, one lock hold, same hour

`run-gen MEMRA_NGEN=32`, board-2048 prime (`probe-flipfull.log`). argmax **MATCH 30/30**
(prefill 485 == decode 485 AND batched-prime 485/485, every run, zero MISMATCH).

| arm | runs (tok/s) | median |
|---|---|---|
| cublas (mode-1, the round-49 Hopper naked default) | 8635.5 / 8092.4 / 8643.7 / 8523.9 / 8626.5 | 8626.5 |
| sk full form (`MEMRA_MOE_F16G=2` — direct + deep tail default-on, cross=32) | 13164.3 / 13152.2 / 13163.6 / 13180.7 / 13132.8 | **13163.6** |
| sk round-51 form (`MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0 MEMRA_F16G_TAIL=0`, cross=32) | 8073.4 / 8073.5 / 8071.5 / 8068.7 / 8074.5 | 8073.4 |

**VERDICT: FLIP.** The full mode-2 form = **152.6% of cublas** (zero overlap: min skfull
13132.8 >> max cublas 8643.7). The reference arm reproduces round 54's sk-ws cell to 0.01%
(8073.4 vs 8074.2, cross-session) — the whole +63.0% move over it is the direct loaders +
deep tail, exactly the coverage residual h100-sk-direct priced. The workspace pass H100
round 51 kept paying (and its h2f-adjacent traffic) was the entire gap AND then some: HBM3
did not make it cheap, it made removing it a 4.5 tok/us swing.

## Cross re-sweep on the winning arm — the stale-verdict law, third instance

Sweep-grade (1 process per value, `MEMRA_PP_ONLY` median of 5 in-process reps + 1 warmup,
sequential, one lock hold; NOT the claim number):

| `MEMRA_F16G_SK_CROSS` | pp2048 med (tok/s) |
|---|---|
| 16 | 12868.3 |
| 32 | 13192.4 |
| 64 | **13224.7** |

**cross=64 wins on H100 with the full form** — the 32 verdict (swept round 51, re-confirmed
round 54 on the kq-direct-only form) went stale the moment B tiles stopped riding the
workspace. The unset default (64) is now the swept winner on BOTH rigs: no per-arch cross
value, naked = full speed.

## The flip (shipped in this lane)

- `moe_f16g_mode()` Err arm: `cfg!(memra_hopper_mma) { 1 }` -> `{ 2 }` — Hopper naked
  default is now the single-kernel sk visitor with direct-from-quant loaders + deep tail
  (crates/memra-engine/src/lib.rs, doc block updated with the re-verdict).
- The gemma (gelu) site is UNAFFECTED by construction: `moe_f16g_gemma_on()` reads
  `MEMRA_MOE_F16G` directly and stays closed when unset (`Err => false`) — verified at the
  dispatch site (hybrid_forward.rs `moe_f16g_gemma_on()` guard); the flip only changes the
  `moe_f16g_mode()` Err arm, which that door never consults.
- sm_120a untouched: the Err arm's non-hopper branch stays 3 (AUTO-KQUANT), and
  `memra_hopper_mma` is compile-gated off in the naked sm_120a build.
- `moe_f16g_sk_params()` doc updated with the H100 re-sweep (64 confirmed both rigs);
  FLAGS.md §5 rows (`MEMRA_MOE_F16G`, `MEMRA_F16G_SK`, `MEMRA_F16G_SK_CROSS`,
  `MEMRA_F16G_DIRECT`, `MEMRA_F16G_TAIL`) + the §7 table carry the round-55 verdict.

## Battery (post-flip binary)

`tools/validate-h100.sh <q35> --quick` on the flipped tree: **VALIDATE-H100: ALL GATES
GREEN** (`vh100-quick-flipfull.log`) — policy tests, kernel-check, decode-batch config B=8,
decode-batch strict (both pin F16G=0 in-binary — immune to the flip by design), decode-dc,
graph-decode, graph-session. Per-run argmax on every probe/cell run above.

run-spec on-box: ATTEMPTED, structurally unavailable (`spec-flipfull.log`) — the q35
UD-IQ4_XS artifact carries no embedded NextN head (`nextn=0`, clean run-spec error) and
the own-trim drafter gguf is not on this box. The spec gate for the mode-2 numeric class
is carried by the 5090 receipts (research/iq-direct-loaders-20260802: q35 `MEMRA_MOE_F16G=2`
owntrim draft **K=1..8 PASS x8** — the exact class this flip promotes to Hopper naked).
Decode/verify are untouched by the flip either way (the t>=16 f16g floor keeps them on
dp4a). If the drafter ever stages onto darklanes-bench, run it there as a merge nicety.

## Board cell — q35 p2048/g512 N=5 both arms (the row)

`tools/h100-vllm-board.sh q35` appended to
`research/tune-data/h100board-vllm-20260731-realtext.jsonl` (ts 2026-08-02T07:08:30Z;
`cell-flipfull.log`, `q35-memra.log`, `q35-vllm.log/.json`); e2e = 512/(2048/pp +
512/dec); naked memra = the flipped mode-2 default, N=5 medians both arms, same-session
interleaved pair, argmax + batched-prime MATCH all 5 memra runs (0 MISMATCH).

| arm | decode tok/s | prefill tok/s | e2e tok/s |
|---|---|---|---|
| memra (flipped: mode 2 direct+tail, naked) | 242.60 | 13257.9 | **226.1** |
| vLLM 0.26 (Qwen/Qwen3.6-35B-A3B-FP8) | 225.35 | 18221.3 | 214.7 |

**ROW MOVES: 217 vs 215 = 1.01x -> 226 vs 215 = 1.05x.** Decode flat (242.92 -> 242.60);
the whole move is prefill 8136.2 -> 13257.9 (+63%). vLLM still primes faster (18.2k vs
13.3k) — the remaining prefill gap is the next rung, but the e2e row is no longer decided
by it.

## Files

`run-probe3-flipfull.sh`, `run-cross-sweep.sh`; raw logs `kc-flipfull.log`,
`probe-flipfull.log`, `sweep-cross{16,32,64}.log`, `vh100-quick-flipfull.log`,
`cell-flipfull.log`, `q35-memra.log`, `q35-vllm.log`, `q35-vllm.json`; `receipts.jsonl`.
