# GEMM Pre-Decode + Barrier-Cut Plan (capability: `gemm_predecode`)

## 0. State of play — what's already done, what the profiler actually says

The batched tensor-core GEMM tile **already exists and ships behind `BW24_GEMM`**:
`qmatvec_gemm_q8_0/q4_K/q5_K/q6_K/nvfp4` in `cu/qmatvec_gemm.cu`, dispatched at
`src/lib.rs:510` (`if m >= GEMM_M_THRESHOLD && self.gemm_supports(w)` →
`self.qmatvec_gemm(w, &aq, &ad, m)`, `lib.rs:512`), `GEMM_M_THRESHOLD = 16`
(`lib.rs:509`). It got prefill pp512 from **143 → 1272** by killing the structural
512× weight re-read (commit `8d1c0b7`). It is **not** at llama.cpp's 6240.

The first instinct — "we're occupancy-bound, redesign the tile (smem swizzle,
bigger BN, free accumulator regs)" — was **tried and DISPROVEN**. The smem-pad
swizzle is already documented as a measured no-op in the kernel itself:

> "A 16B-aligned pad to 48 reduces ldmatrix bank conflicts (33M→~10M) but the
> extra 8KB/CTA smem drops kernel2 occupancy 4→3 blocks, exactly cancelling the
> gain (pp512 flat). At this BN=128 / 64-reg-accumulator tile the kernel is
> occupancy-bound, not conflict-bound, so the pad is a no-op here"
> — `cu/qmatvec_gemm.cu:48-51`

That comment's own conclusion ("occupancy-bound") is what the **ncu run
overturned**. ncu on the prefill GEMM says:

- **barrier stall = 3.55** cycles/issue (the `__syncthreads()` per K-step, `cu/qmatvec_gemm.cu:283`)
- **long-scoreboard stall = 2.61** cycles/issue (global memory dependency landing in a math op)
- **"No Eligible" = 87.6%** of cycles have zero issuable warp
- the stall profile is **flat ~15% across occupancy** — adding CTAs/SM does nothing

Flat-across-occupancy + dominated by barrier + long-scoreboard = the kernel is
**memory-latency + barrier bound, NOT occupancy bound.** More warps cannot hide
the stall because the stall is a *dependency chain* every warp sits inside, not a
shortage of warps. The two named culprits are concrete and both in the K-loop:

1. **Weight decode is on the mma critical path.** `issue_load_stage` ALU-decodes
   the weight 32-block (`decode_block<QT>`, `cu/qmatvec_gemm.cu:234`) from a
   *global* read of `W + o*row_bytes` (`cu/qmatvec_gemm.cu:233`). The decode is
   synchronous into staged smem; only the **activation** half of the stage is
   `cp.async` (`cu/qmatvec_gemm.cu:247-248`). So the global weight read → ALU
   unpack → smem store → `ldmatrix` (`cu/qmatvec_gemm.cu:296`) → `mma`
   (`cu/qmatvec_gemm.cu:304`) chain is *serial*, and the global load at the head
   of it is exactly the long-scoreboard 2.61.

2. **One barrier per K-step.** `cu/qmatvec_gemm.cu:283` runs a `__syncthreads()`
   every 32-K iteration, guarding both stage visibility and WAR for the prefetch.
   With decode synchronous in front of it, every warp arrives late and waits =
   barrier 3.55.

This plan attacks **only those two**, in the order ncu ranks them. It does **not**
touch BN/accumulator/swizzle (proven flat).

---

## 1. Reference: how llama.cpp avoids both stalls

- **Pre-decode, not inline-decode.** `load_tiles_q4_K`
  (`llama.cpp/ggml/src/ggml-cuda/mmq.cuh:2093-2199`) unpacks all Q4_K nibbles to
  int8 into `x_qs[...]` (`mmq.cuh:2124-2125`) **once per tile**, called at
  `mul_mat_q_process_tile` (`mmq.cuh:3486`) **outside** the `vec_dot` loop. By
  the time `vec_dot` runs (`mmq.cuh:3499`) it reads `const int * x_qs = x`
  (`mmq.cuh:3015`) with **zero decode** in the hot loop. bw24 instead pays the
  decode *inside* every K-step.
- (llama.cpp's own barrier count is actually *worse* — 4 `__syncthreads`/K-step
  at `mmq.cuh:3497,3501,3513,3517` — but its loads are not behind an ALU decode,
  so its barriers don't gate a dependency chain. Our barrier is expensive
  *because* decode stalls in front of it; fixing decode shrinks the barrier
  stall too. We still cut barriers in step B for the residual.)

---

## 2. Fix A (PRIMARY, ~1.4×): pre-decode weights off the mma critical path

**Idea:** the weight 32-block is read+decoded once per (out-row, K-step) per CTA
*tile column*, but the decode output is identical for every token and every
CTA along the token axis. Move the global read **and** the ALU unpack out of the
K-loop dependency chain so the `mma` no longer waits on either.

Two implementation tiers, ship A1 first, measure, only escalate to A2 if A1
under-delivers:

### A1 — split-stage decode-ahead inside the existing NSTAGE ring (in-kernel, low risk)

Today `issue_load_stage` (`cu/qmatvec_gemm.cu:228-264`) does, per stage, in one
pass: global-read W → decode → store `sW` (synchronous), *and* `cp.async` the
activation. Split the weight half into two phases pipelined one stage apart:

1. **Phase fetch (stage s):** `cp.async` the *raw quant bytes* of the weight
   32-block from global `W + o*row_bytes` into a new
   `__shared__ uint8_t sWraw[NSTAGE][BM][bytes_per_block]` — no ALU, no
   dependency. Commit into the existing group alongside the activation copies
   (`cu/qmatvec_gemm.cu:270,290`).
2. **Phase decode (stage s, after its group lands):** ALU-unpack `sWraw[s]` →
   `sW[s]` (the existing int8 tile) *while the next stage's `cp.async` is in
   flight*. The decode now reads from **smem (already resident)**, not global —
   this removes the long-scoreboard read (2.61) from the chain. The ALU work
   itself overlaps the next stage's DRAM latency exactly as the activation copy
   already does.

This reuses the entire ring/commit/wait machinery (`cu/qmatvec_gemm.cu:267-290`);
the only new state is `sWraw` and a one-stage shift of the decode call. The
`bytes_per_block` is dtype-specific (Q8_0 34, Q4_K 144/8-superblock-shared, Q6_K
210, NVFP4 36) — the `cp.async` granularity must cover the superblock the block
indexes into; for the superblock quants stage on the **superblock** boundary
(`g & 7 == 0`) so 8 K-steps share one fetched superblock (this also cuts global
traffic 8×). smem cost: Q4_K adds 144 B/row × BM=64 × NSTAGE=3 ≈ 27 KB/CTA — this
is the one place we must re-check occupancy (it may force NSTAGE=2 for the raw
buffer only; keep the int8 `sW` ring at 3).

### A2 — layer-once pre-decode to a resident int8 staging buffer (host-side, if A1 short)

If A1's in-kernel overlap still leaves decode partly exposed (likely for Q6_K /
NVFP4 with their two-sub-scale 2× mma at `cu/qmatvec_gemm.cu:551-552`), run a
**separate lightweight decode kernel once per weight matrix per prefill** that
writes the full `int8[out_f][in_f]` (+ packed per-32 scales/bias) to a VRAM
staging buffer, mirroring llama's `load_tiles` being amortized across the whole
tile. Then `qmatvec_gemm` reads int8 directly via `ldmatrix` — **zero decode in
the K-loop, ever.** Trade: int8 staging is ~1:1 vs Q8_0 but ~4× vs Q4_K weight
bytes; only viable if the matrix fits VRAM headroom, so gate it per-matrix on
free VRAM and fall back to A1 otherwise. This is the same idea as the rejected
"resident int8" but scoped to *prefill staging only* (freed after), not a
permanent format change — so it does **not** sacrifice the 24 GB VRAM win that
keeps weights quantized for decode.

**Expected from Fix A: ~1.4×** (1272 → ~1780). Removing the global weight read
from the chain directly retires the 2.61 long-scoreboard; the barrier 3.55
shrinks partially as a side effect (warps stop arriving late behind decode) but
is finished off by Fix B.

---

## 3. Fix B (SECONDARY, ~1.2×): cut per-K-step barriers

The single `__syncthreads()` at `cu/qmatvec_gemm.cu:283` fires every 32-K. With
Fix A decoupling decode, the barrier is now only guarding (i) stage-`cur`
visibility and (ii) WAR for the post-barrier prefetch into `nxt`. Reduce its
*frequency*, not just its cost:

### B1 — multi-block-per-barrier (process 2–4 K-blocks between syncs)
Restructure the K-loop (`cu/qmatvec_gemm.cu:275-316`) to consume `KB_PER_SYNC`
(=2, then 4) already-landed stages per barrier: one `cp.async.wait_group` +
`__syncthreads()` covers a batch of `KB_PER_SYNC` mma K-steps (deepen NSTAGE to
`KB_PER_SYNC + 2` so that many stages are in flight). This is the standard
software-pipeline depth increase: barriers/token drop by `KB_PER_SYNC×`, which
directly attacks the 3.55. The WAR argument at `cu/qmatvec_gemm.cu:285-289`
generalizes — prefetch the batch of `nxt` stages after the batch barrier.

### B2 — async-group WAR instead of full barrier (if B1 still barrier-heavy)
The WAR half of the barrier (don't overwrite a stage still being read) can be
enforced by `cp.async.wait_group` depth alone rather than a full
`__syncthreads()` when the producer and consumer of a stage are the *same* warp
lanes (they are: `ld_A_s8`/`ld_B_s8` read the stage the same warp's
`issue_load_stage` wrote). Keep one `__syncthreads()` only where cross-warp smem
visibility genuinely requires it. Lower confidence — validate with the bit-equiv
gate before trusting it.

**Expected from Fix B: ~1.2×** on top of Fix A (~1780 → ~2100).

---

## 4. Honest projection

| stage | pp512 | source of number |
|---|---|---|
| pre-GEMM (per-token matvec) | 143 | commit `8d1c0b7` |
| current GEMM tile (shipped) | 1272 | measured baseline |
| + Fix A (pre-decode) ~1.4× | ~1780 | latency model: retire long-scoreboard 2.61 |
| + Fix B (barrier cut) ~1.2× | **~2100** | retire most of barrier 3.55 |
| llama.cpp target | 6240 | reference |

**This plan reaches ~2100, NOT 6240.** Be explicit: the ~1.4× and ~1.2× are not
multiplicative magic — they are the two ncu-named stalls (2.61 + 3.55 of a
~15%-flat profile) and retiring them fully is bounded by Amdahl on the rest of
the issue profile. ~2100 is still **~3× short of llama.cpp**, and that residual
gap is **not** addressable by latency/barrier work — it is **compute ceiling**:
plain int8 m16n8k32 ≈ 219 TFLOP/s on sm_120, while llama's heavily-tuned MMQ
sits closer to its int8 peak and bw24's tile does not. **Full parity (and
beyond) requires Stage-C block-scaled FP8** (381 TFLOP/s, the Blackwell-only
`BLACKWELL_MMA_AVAILABLE` path) — a separate capability, not in scope here. Do
not claim this plan closes the llama gap; claim it retires the two profiled
stalls and lands ~2100.

---

## 5. GATE (non-negotiable, ordered — identical bar to the shipped kernel)

1. **Bit-equivalence (kernel_check).** `qmatvec_gemm_raw` vs the dp4a reference
   must hold `rel < 1e-3` — the exact gate already in
   `src/bin/kernel_check.rs:401` (Q8_0/Q4_K/Q6_K), `:426` (Q5_K), `:443` (NVFP4),
   over T ∈ {16,64,128,512}. Pre-decode reorders *when* the decode runs, not the
   *bytes* it produces, so this must stay exact; any drift = the split mangled a
   superblock boundary → revert.
2. **End-to-end argmax (the real gate).** With `BW24_GEMM=1`, prefill argmax MUST
   stay **268** (qwen3 dense, `run_dense`), **271** (qwen35 hybrid, `run_hybrid`),
   **1178** (35B-A3B MoE). Any mismatch = revert.
3. **Perf gate.** `pp512 tok/s` MUST rise above the 1272 baseline. Milestone:
   Fix A alone should clear ~1700; Fix A+B ~2100. If a step does not move pp512,
   it failed its premise (re-profile with ncu, do not stack the next change blind).

---

## 6. Files to touch (concrete)

- **Edit `crates/bw24-engine/cu/qmatvec_gemm.cu`**
  - `issue_load_stage` (`:228-264`, and kernel2's at `:484-519`) — split into
    fetch (`cp.async` raw weight bytes → new `sWraw`) + decode (smem→`sW`),
    shifted one stage (Fix A1).
  - smem block (`:212-217`, `:472-476`) — add `sWraw[NSTAGE][BM][bytes/block]`;
    re-check occupancy, possibly NSTAGE=2 for the raw ring only.
  - K-loop (`:275-316`, `:528-561`) — `KB_PER_SYNC` batching + deeper NSTAGE,
    single batch-barrier (Fix B1); generalize the WAR prefetch (`:285-290`).
  - decode_block specializations (`:184-186`) — accept an smem `const uint8_t*`
    source instead of global `W + o*row_bytes` (Fix A1 reads from `sWraw`).
- **(Fix A2 only, if needed) Edit `crates/bw24-engine/src/lib.rs`** near
  `qmatvec_gemm` (`:686`) — add a per-matrix VRAM-gated pre-decode-to-int8-staging
  launch; reuse existing fatbin loader (`GEMM_FATBIN_PATH`, `lib.rs:24`).
- **Unchanged:** `gemm_supports` (`lib.rs:671`), dispatch (`lib.rs:510`),
  `GEMM_M_THRESHOLD` (`lib.rs:509`), the dp4a decode/Stage-A paths, all call
  sites — the kernel signature and the activation flow do not change.
- **Verify, do NOT loosen:** `kernel_check.rs:401/426/443` (`rel < 1e-3`).
