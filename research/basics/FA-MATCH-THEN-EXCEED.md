# FA — Match Then Exceed (bw24 sm_120a)

The bw24 FlashAttention plan. Binding framing (USER DIRECTIVE): **"In FA we want MORE than what
llama has, but AT LEAST what they have."** Two parts:

1. **FLOOR (match)** — reconstruct llama's sm_120 FA structure 1:1 in `mma.sync` (no wgmma/tcgen05;
   the laptop lacks both). Same silicon, same algorithm, same occupancy. NEVER land below llama.
2. **CEILING (exceed)** — add the bw-specific edges llama's *generic* FA does NOT do on sm_120.

This document is **read-only reasoning + a port plan**. No code was edited here (a GEMM build
workflow is editing the crate in parallel). It consolidates and supersedes the narrower
`FA34-PLAN.md` (port ranking) and `FA-FLOOR-PLAN.md` (floor measurement), and folds in the three
adversarial verdicts (runs-on-sm120a / matches-then-beats-llama / argmax-safe) — including the two
corrections those verdicts forced.

**Machine/model**: RTX 5090 Laptop, **sm_120 / CC 12.0 = `GGML_CUDA_CC_BLACKWELL`**, `build.rs:17`
targets `compute_120a,code=sm_120a`. Model = 9B-NVFP4 GGUF, **head_dim D=256** (not 128), n_head=16,
n_head_kv=4 → **gqa_ratio=4**.

**Verified head-to-head (this machine, WARM full-power — discard llama-bench cold run):** bw24 vs
llama 9B-NVFP4 — decode tg128 83 vs 122.5 (**0.68×**), prefill pp512 882 vs 5450 (**0.16×**).
Attention is part of both gaps; the **prefill catastrophe is the dominant one**, and it is an
*occupancy/launch* catastrophe, not a KV-bandwidth one (see §1).

---

## 1. The llama FA FLOOR — measured structure to match

Traced through the real `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:457-478`) and confirmed against
source + warm ncu on this box (GPU confirmed warm at 2272/3090 MHz, 172 W during the measurement —
NOT the idle-clock trap).

**Which FA llama PICKS on sm_120:**

- **Prefill (pp512, n_q=512): MMA-f16** — `flash_attn_ext_f16<256,256,16,4,0,0>`.
  `turing_mma_available` → true → `BEST_FATTN_KERNEL_MMA_F16`. **NOT wmma** (wmma is Volta-only
  here). Plus a stream-K `flash_attn_stream_k_fixup` combine pass.
- **Decode (tg, n_q=1): VEC** — `flash_attn_ext_vec<256,1,1,1,0>`.

**The MMA prefill config sm_120 actually instantiates** (ampere table `fattn-mma-f16.cuh:72`):

| Property | llama value | meaning |
|---|---|---|
| warps/CTA | **4** (128 threads) | `nthreads=128` |
| CTAs/SM | **2** | register- AND smem-limited (by design) |
| `nbatch_fa` | **32** | KV tile depth = our **BK** |
| `nstages` | **2** | cp.async double-buffer |
| `Q_in_reg` | **true** | **Q in REGISTERS, not smem** ← load-bearing, see §3 |
| `ncols2` | **4** | GQA: 4 query heads share staged K/V |
| Q-cols/CTA | **64** | tile = `BLOCK_Q` |
| O accumulation | **registers** (C-fragments) | not smem RMW |

**MEASURED floor (ncu, this machine, pp512 full-attn layers):**

- **llama prefill FA: ~15% warps_active** (cap 16.67%), **~52% SM SoL, ~167–171 µs/call**, 230
  regs/thread, **~35 KB dynamic smem**, occupancy capped at **2 CTAs/SM** by both regs and smem.
- **15% is by design, not a bug.** This is a register-heavy, high-ILP, compute-bound kernel at
  D=256. Chasing >16.67% by *shedding registers* trades away the ILP that makes the D=256 mma fast
  and **LOSES**. The floor is **15% warps @ ~50% SoL @ 2 CTAs/SM** — match it, do not "beat" it by
  dropping registers.
- llama decode vec: ~8.3% warps, ~12.6% mem throughput at short ctx — expected; **bw24's shipped
  split-K decode (`fa_decode_vec_q`) is already structurally ahead.** The gap is **prefill FA, not
  decode FA. Do not re-touch decode.**

**bw24 current state — the catastrophe (ncu, pp512):** `fa_prefill_f32` = **2.08% warps active,
9.26 ms/call, 58 regs/thread, grid (32,16,1) block (32,1,1) = 1 WARP/CTA**. The `sO[16][256]` f32
smem RMW per KV block is the dominant overhead; `__syncwarp` staging serializes. x8 full-attn layers
≈ 13% of prefill. One warp/CTA wastes 97% of the SM.

**Critical honesty (verdict-confirmed):** bw24 prefill is **STILL 0.16× even WITH `fa_prefill_q`
(the quant-KV kernel) already in the path.** So the quant-KV edge does **not** close the prefill gap
today — the gap is the 1-warp occupancy catastrophe (compute/launch-bound), not KV bandwidth. This
is why the FLOOR (occupancy) must land *before* any EXCEED edge can show.

**Structure map — llama config ↔ bw24 port phase (1:1):**

| llama property | bw24 phase |
|---|---|
| `Q_in_reg=true` (Q in registers) | **P0a** (Q-in-reg) ← the fix the original MATCH plan omitted |
| O in C-fragment registers | **P0b** (register-O) |
| `ncols2=4` (GQA reuse) | **P1** (grid.y=n_head_kv, inner gq loop) |
| `nthreads=128` (4 warps) | **P2** (multi-warp, BLOCK_Q 16→64) |
| `nstages=2` cp.async | **P4** (2-stage K double-buffer) |

---

## 2. EXCEED — the bw-specific ceiling (verified-real only)

llama's MMA prefill **hard-requires f16 K/V** (`fattn.cu:558` forces `need_f16_K/V=true`); its
quant-KV is **vec/decode-only (n_q ≤ 2)**. That is the structural seam bw24 exploits.

**Governing physics (decides which edge is real where):** prefill at D=256 is **compute-bound**
(~768 FLOP/byte); decode is **bandwidth-bound** (~1 FLOP/byte). An edge that saves KV *bytes* is
decisive in decode + long-context prefill, a **wash on short-ctx pp512**. An edge that speeds the
*mma itself* is decisive on compute-bound pp512, useless on decode.

### Edge 1 — Quantized-KV PREFILL (q8_0 K / q5_1 V) — REAL, already owned, the anchor

`fa_prefill_q` (`flash_attn.cu:486`) already inline-dequants q8_0 K (34 B/32 elem) + q5_1 V
(24 B/32 elem) during staging (lines 526-533, via `dq_q8_0_elem`/`dq_q5_1_elem`). Per 32 elem bw24
reads **58 B vs llama's 128 B (f16) = 0.45×** the KV HBM/L2 traffic.

- **pp512 (compute-bound): ~1.0× (a wash).** The byte saving is latency-hideable behind the mma; the
  dequant ALU on the staging path is the cost. **Honest: this edge does not show at pp512.**
- **Long ctx (≥4K) + MTP partial-accept replay (commit abef37d):** KV tile streaming becomes the
  co-bottleneck; at 0.45× bytes the load pipe drains 2.2× faster and bw24 goes **below llama's f16
  line. Projected 1.3–1.8× on the FA portion, scaling with ctx.**
- **The honest catch:** cp.async (P4) cannot dtype-convert. So the quant path **cp.asyncs the quant
  bytes** (58 B/tile) and dequants smem→smem (or at the ld point) — a *different* cp.async structure
  from the f16 floor (which cp.asyncs ready-to-mma bf16). Edge 1's cp.async stage is a separate
  kernel from the floor's, not a one-line dtype swap.
- **argmax-risk: LOW (verdict: bit-safe on values).** `dq_q8_0_elem`/`dq_q5_1_elem` do **exact** f32
  reconstruction and are the SAME functions already shipped + validated in `fa_decode_vec_q`. Moving
  them behind cp.async changes *when* bytes load, not their values.

### Edge 3 — Block-scale FP8 (mxf8) mma for QK — REAL primitive, payoff OVERSTATED, the prize but unproven

The only edge that attacks the **compute-bound pp512 catastrophe** (the 0.16× gap), because it
speeds the mma rather than the KV bytes. Keep K/Q in **e4m3** and run QK on sm_120's FP8 tensor path
instead of bf16.

- **Primitive is device-proven on THIS box** (verdict runs-on-sm120a): `probe/lowbit_peak.cu:16`
  benchmarks `mma.sync.aligned.m16n8k32...kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`
  and `probe/tc_peak.cu:25` proves plain `m16n8k32 f32.e4m3.e4m3.f32` assembles. The m16n8k32
  mxf8f6f4 block-scale path with `scale_vec::1X` (1X = the FP8 variant; 2X/4X are the mxf4 variants
  in `qmatvec_gemm.cu:893`) exists for the QK/PV shape. No tcgen05. Confirmed.
- **THE CORRECTION (verdict): the throughput math is wrong on this silicon.** The plan family
  claimed "FP8 = ~2× f16 / 381 TFLOP". The repo's OWN measured reference
  (`tc_peak.cu:34`/`lowbit_peak.cu:34`) prints **FP16=117, plain-FP8-f32acc=219 = 1.87× — and only
  for plain FP8**. The adversarial corpus (`research/raw-workflow-output.json:360`) states for this
  exact GB203: **"FP32-accumulate throttle ⇒ FP8 mma.sync == BF16 rate ... do NOT promise
  compute/ITL wins from FP8 attention math, expect FP8 to possibly LOSE on compute-bound prefill."**
  **FA REQUIRES f32 accumulate (softmax), so the throttle bites precisely the pp512 regime Edge 3
  targets.** The genuine full-rate FP8 lever is the **block-scaled mxf8f6f4 path** (which the plan
  named but conflated with plain-FP8 throughput); the corpus notes block-scaled MXFP8 attention
  **"does not yet exist for sm120" (build-it, not lift)**.
- **Scope: QK-only FP8 first.** Softmax-in-FP8 corrupts: P (probabilities) feeding PV in e4m3 has 3
  mantissa bits → the online-softmax `l_i` sum drifts. Keep PV's P operand bf16. sm_120's FP8 mma
  needs both operands same dtype, so PV-FP8 has an unsolved precision problem. QK is ~half the
  prefill FLOPs at D=256.
- **argmax-risk: HIGH (verdict: NOT bit-safe).** FP8 mma changes dot-product numerics + accumulation
  precision. Gate: rel < **1e-2** (looser — FP8 mma will not hit 1e-3) + mandatory end-to-end argmax
  + long-ctx `l_i` drift test over 4K keys.
- **Mandatory pre-commit gate:** re-ground the TFLOP claim with a **local mma microbench
  (`lowbit_peak.cu` already exists for exactly this)** BEFORE committing kernel work. If
  f32-accumulate block-scaled mxf8 does not beat the bf16 rate on this box, Edge 3 does not ship.
- **Projected gain (UNPROVEN, gated):** QK-only FP8 ≈ 1.2–1.4× on the FA portion *if* the microbench
  confirms a rate win; treat as unproven until then.

### Edge 2 — e4m3 KV cache — MARGINAL standalone, only Edge 3's feedstock

Per 32 elem: f16=64 B, q8_0=34 B, **e4m3=32 B**. But Edge 1's existing pair (q8_0 K 34 B + q5_1 V
24 B = **58 B**) **beats e4m3-both (64 B) on bytes** — q5_1 V at 24 B already beats e4m3 V at 32 B.
So e4m3 is **NOT a bandwidth win over Edge 1**. Its only value: (a) cheaper dequant (one hardware
convert vs nibble-unpack), (b) it is the **only format that feeds a block-scale FP8 mma directly
(Edge 3)**. **argmax-risk: MEDIUM** (3 mantissa bits, ~±448 range, no per-block adaptive scale →
outlier keys saturate; needs the long-ctx argmax gate). **Verdict: build only as a step toward
Edge 3, never standalone.**

### Edge 4 — Fuse online-softmax rescale with dequant — DROPPED

After P0 (register-O) the rescale is already a register FMA over C-fragments, and P3 skips it ~90%
of the time. V-dequant (staging, once/tile, shared across 4 GQA heads) and O-rescale (registers,
once/tile) operate on different data at different times — "fusing" removes no pass. The genuinely
useful sub-part (V stays quant in smem, dequant at the ld point) is **subsumed by Edge 3** (FP8 V
feeding an FP8 mma = no dequant at all). **Do not build standalone.**

### Edge 5 — FA3/FA4 SCHEDULING ALGORITHMS (hand-ported, the real "more than llama" — ADDED 2026-06-27)

User directive: FA3/FA4 can't run literally (no wgmma/tcgen05) but their **algorithms are hardware-
independent and hand-codable** with mma.sync + cp.async + warp roles. The instruction makes them
faster; the SCHEDULING is what wins, and llama's generic sm_120 MMA-f16 FA does NOT do it. This is
the layer the floor (occupancy) + Edges 1/3 (KV format) MISS — they make bw24 EQUAL llama's
structure + cheaper KV; the FA3 schedule is what pushes the compute/latency profile PAST it.

- **5a — Softmax–GEMM overlap (FA3 pingpong), the headline FA3 win.** Overlap the online-softmax
  (exp2 + max/sum reduce + O-rescale) of KV-tile `j` with the QK^T mma of tile `j+1`, so the
  softmax's transcendental+reduce latency hides behind tensor issue instead of serializing after
  PV. Hand-portable: 2-stage register pipeline over the KV loop — issue QK(j+1) mma, then while it
  retires compute softmax(j) on the prior scores in registers, then PV(j). Pure scheduling, ZERO new
  instructions, ZERO numeric change (same ops, reordered) → **argmax bit-safe**. llama's MMA-f16
  prefill at ~52% SoL still serializes softmax after the PV mma per tile; overlapping it is a direct
  structural beat on the compute-bound pp512 regime. **Projected: the largest "exceed" lever after
  the floor — gate on ncu issue-slot utilization (softmax stall cycles should drop) + per-call time.**
- **5b — Warp-specialization (producer/consumer), FA3/FA4.** Split the CTA's warps: producer warp(s)
  run cp.async (and dequant for Edge1/3) to fill the K/V smem ring; consumer warps run the mma +
  softmax. Async handoff via smem barriers (mbarrier/`__syncthreads` phase). The role-split is the
  algorithm — TMA (sm_120 HAS cp.async.bulk) is just a faster producer, optional. Higher complexity
  (named-barrier choreography, hand-rolled without CUTLASS); evaluate AFTER 5a since 5a captures most
  of the overlap win with far less risk. **argmax bit-safe (reorder only).**
- **5c — FA4 finer pipeline / exp approximation.** Deeper KV-tile pipeline depth + attention-specific
  exp2 polynomial. Marginal after 5a; defer.

**Build order for Edge 5: 5a FIRST (after the floor lands — it needs register-O/multi-warp in place
to have registers for the 2-stage pipeline), it is the bit-safe high-value scheduling beat; 5b only
if ncu shows the producer cp.async still exposed after 5a; 5c last.** Edge 5 STACKS on Edges 1/3
(scheduling is orthogonal to KV format). This is the layer that makes "more than llama" true on
COMPUTE, not just KV bandwidth (Edge 1).

---

## 3. The two corrections the verdicts forced (read before §4)

**Correction A — Q-in-reg is the real occupancy blocker, NOT the f32 `sS`** (verdict
matches-then-beats-llama). The original MATCH plan claimed "drop f32 sS → ~50-52 KB → 2 CTAs/SM."
**That is arithmetically false.** The f32 `sS` is only ~4 KB; removing it from the 68.5 KB
register-O config leaves sQ(32 KB)+sK(16)+sV(16)+sP(4) ≈ **68 KB = 1 CTA/SM = ~8% warps = HALF the
floor.** The actual blocker is **`sQ` (32 KB)**. llama hits its ~35 KB dynamic smem precisely
because **`Q_in_reg=true` means Q lives in REGISTERS** — only K/V/P rotate through smem
(sK+sV+sP ≈ 36 KB ≈ llama's 35 KB). **To reach 2 CTAs/SM the port MUST move Q out of smem into
registers (true Q_in_reg).** This is phase **P0a** below and is the load-bearing fix; without it the
edit set lands at 1 CTA/SM and **misses the floor.**

**Correction B — Edge 3's FP8 compute payoff is unproven on f32-accumulate FA** (verdict
runs-on-sm120a). See Edge 3 above: gate behind the `lowbit_peak.cu` microbench. The "more than
llama" story survives **today** only on the **bandwidth edge (Edge 1)**; the **compute edge
(Edge 3)** is real-but-unbuilt and must re-ground its TFLOP claim first.

**Budget math (binding, HEAD_DIM=256, 99 KB opt-in smem cap, 2 CTAs/SM ⇒ ≤49.5 KB/CTA):**

1. **BK must drop 64 → 32** (= llama `nbatch_fa=32`). BK=64 register-O = 104.5 KB > cap.
2. **Register-O at D=256 = 128 f32/lane for O alone** ((256/8)×4) + softmax/frag/addr ≈ ~160–230
   regs/thread = llama's 230 @ 2 CTAs/SM (register-limited, exactly llama's limiter). Do NOT fight
   regs down to chase >16.67% — it loses.
3. **Q-in-reg (P0a) is mandatory** to drop the sQ 32 KB → smem ≈ 36 KB (sK+sV+sP) ≈ llama's 35 KB.

| Config | smem | CTAs/SM | warps_active |
|---|---|---|---|
| **Current** 1-warp BK=64 smem-O | 94.1 KB | 1 | 2.08% |
| P2-only BLOCK_Q=64 BK=32 smem-O | 140.5 KB | **0 (won't launch)** | — |
| P2+P0b register-O, **Q still in smem** | ~68 KB | **1** | ~8% (HALF floor — the trap) |
| **P2+P0a+P0b (Q-in-reg too)** | **~36 KB** | **2** | **~15% (floor)** |
| **+P4** 2-stage K (+16 KB) | ~52 KB | 1–2 (measure) | ~15% latency-hidden |

P4's +16 KB pushes ~36 → ~52 KB → may cost the 2nd CTA. llama keeps BOTH 2 CTAs and 2-stage by
staging K/V in a single rotating buffer (not separate doubled sK/sV); mirror that layout. **Measure
both.**

---

## 4. Port order — ranked by (occupancy-gain × feasibility × argmax-risk)

P0a+P0b+P2 **compile together** (P2-alone with smem-O = 140.5 KB won't launch; register-O+Q-in-reg
share the lane-map refactor). The table separates them for **risk attribution**.

| Rank | Phase | warps_active | Reaches floor? | Why this order |
|---|---|---|---|---|
| **1** | **P2 multi-warp** (4 warps, BLOCK_Q 16→64, **BK 64→32**) | 2.08% → (fused) | partial | Biggest occupancy jump; math byte-identical (only thread-fanout + `__syncwarp`→`__syncthreads`). Compiles with P0. |
| **2** | **P0a Q-in-reg** (sQ smem → registers) | — | **enables 2 CTAs** | **The fix the old plan omitted.** Drops smem 68→36 KB. Without it the floor is missed. |
| **3** | **P0b register-O** (delete sO/f32 sS, O in C-fragments) | fused → **~15%** | **yes (with P0a+P2)** | Removes the dominant smem-O RMW; with P0a unlocks 2 CTAs/SM. **Highest silent-corruption risk** (lane map) — gate hard. |
| **4** | **P1 GQA K/V reuse** (grid.y=n_head_kv, inner gq=0..3) | ~15% flat — **time ↓ 30-40%** | **floor met on time** | = llama ncols2=4. Cuts KV traffic 4×, closes per-call time to ~170µs. Bit-safe. |
| **5** | **P4 cp.async 2-stage K** | ~15% flat — time ↓ long ctx | exceeds | = llama nstages=2. Hides K-load latency. High pipeline risk; gate behind 1-4. |
| **6** | **P3 conditional rescale** (tau=8, `__any_sync`) | flat — small | polish | Compute-bound; small win. NOT bit-safe (intra-prefill numeric change). |
| **7** | **P5 XOR swizzle** | flat — ~86→94% ldmatrix SoL | polish | Pure address math; lowest risk; LAST so a swizzle bug can't mask P0-P4 validation. |

**Decode D0 split-K is shipped and ahead of llama — do not re-touch decode.**

---

## 5. Exact edit sites (all paths absolute)

Kernel: `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu`
Launchers: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs`
Validation: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/bin/kernel_check.rs` (oracle
`sdpa_naive_f32`, `lib.rs:1009`).

### P2 — multi-warp tile (`fa_prefill_f32` 310-477 + twin `fa_prefill_q` 486-625)

- **Line 58/61:** keep `#define M_ROWS 16` (per-warp rows); add `#define N_WARPS 4`,
  `#define BLOCK_Q 64`; **`#define BK 64` → `#define BK 32`**.
- **Line 318:** add `const int warp = threadIdx.y;` (block `(32,4,1)`); each warp owns rows
  `[warp*16 .. warp*16+16)`. `q_base = q_tile*BLOCK_Q + warp*M_ROWS`.
- **Line 316:** `q_tile = blockIdx.x` strides 64 rows → launcher grid.x = `(t+63)/64`.
- **K/V staging (336-341, 357-363):** block-cooperative over 128 threads
  (`for i = warp*32+lane; i < BK*HEAD_DIM; i += 128`).
- **Barriers 343,364,393,428,436,466:** `__syncwarp()` → **`__syncthreads()`**. Softmax block
  (396-427) per-warp `if(lane<16)` on that warp's `sS/sP/sM/sL` sub-slice.

### P0a — Q-in-reg (fused with P2/P0b; THE occupancy fix)

- **Delete `sQ` from smem.** Each warp loads its 16 Q-rows × 256 cols into **A-fragment registers
  once** (Q is reused across all KV tiles — load once, hold). Use `ATile::get_i/get_j` (lines 70-84)
  for the lane map. This is what drops smem 68 → 36 KB and unlocks the 2nd CTA.
- Q-staging loop (current per-warp stage) → replaced by a one-time `ldmatrix`/load into the
  persistent A-fragments before the KV loop.

### P0b — register-O (fused; highest SILENT-corruption risk)

- **Delete `sO` (330) and f32 `sS` (331).** O lives in per-warp C-fragments:
  `CTile O_acc[HEAD_DIM/8]` = 32 CTiles = 128 f32/lane.
- **PV accumulate (442-465):** mma directly into `O_acc[d0/8 + lo/hi]`; stop spilling to `sO`. (The
  C-fragment is already f32 — register-O is **lossless**, same operands/order, bit-safe on numerics.
  The risk is the lane map, not precision.)
- **O rescale (430-435):** register loop `for(c) for(l) O_acc[c].x[l] *= alpha;` — `alpha` broadcast
  via `__shfl_sync` (replace the `sS[r*BK+0]=alpha` smem broadcast at 426).
- **Row softmax:** do it on the C-fragments via `__shfl_xor` reduce within the 4 lanes holding a
  row's col-pairs (removes the f32 sS). Note (Correction A): this saves only ~4 KB — it is *not* the
  occupancy lever, P0a is. Keep it because it's the natural register-O form, not for the budget.
- **Epilogue (470-476):** write `O_acc` → global with deferred `1/sL[r]` normalize, using
  `CTile::get_i(l)/get_j(l)` (72-73) **verbatim**.
- ⚠ **Single highest-risk detail (silent, no NaN):** the C-fragment→register index map
  (`r0=lane/4`, `c0=(lane%4)*2`). A wrong map silently corrupts P@V. **Mitigation:** the
  `get_i/get_j` accessors are already validated by `qkpv_test` — reuse them; gate with
  compute-sanitizer + `qkpv_test` FIRST, then rel + argmax.

### lib.rs launchers — `fa_prefill` (1041-1061) + `fa_prefill_view` (1067-1090)

```
grid_dim:  ((t + 63)/64, n_head_kv, 1),   // grid.x 64-row tiles (P2); grid.y n_head_kv (P1)
block_dim: (32, 4, 1),                     // 4 warps (P2)
shmem:     <recompute for BLOCK_Q=64, BK=32, Q-in-reg, register-O ⇒ target ≤49.5 KB / ~36 KB>
```

### P1 — GQA K/V reuse

- **grid.y `n_head` → `n_head_kv` (=4).** `kv_head = blockIdx.y; head = kv_head*gqa + gq` inside a
  new `for(int gq=0; gq<gqa_ratio; ++gq)` wrapping the FA-2 KV loop. Stage `sK/sV` **once per KV
  tile** (outside gq); re-load Q-fragments per gq. **Loop gq outer** (K/V resident in smem across
  4 heads) — holding 4× O_acc = 512 f32/lane is too many. Bit-safe (data reuse only; GQA index math
  already correct, line 321).

### P4 — cp.async 2-stage K (gate behind 1-4)

- `cp.async.cg.shared.global` helpers; two sK buffers (single rotating buffer to keep 2 CTAs — mirror
  llama). At KV-loop top: cp.async next K, `commit_group`, compute QK on current, `wait_group<1>`.
- **f16 floor:** cp.asyncs ready-to-mma bf16. **Edge 1 quant path (different kernel):** cp.async the
  **quant bytes**, dequant smem→smem (the dequant 529-532 happens after the byte-load). Do NOT
  attempt 3-stage (blows 99 KB; FA-3 found it slower).

### P3 / P5 — polish (after floor proven)

- **P3:** `delta = max(m_i,rmax)-m_i; bool need = __any_sync(0xffffffff, delta>8.0f);` skip O-rescale
  when `!need`. **The `__any_sync` warp-uniformity is load-bearing for CORRECTNESS, not just perf** —
  if lanes diverge on `need`, different lanes exp2 against different m and silently corrupt P. Keep
  it. NOT bit-safe (tau=0 vs tau=8 differ ~1e-5 due to exp2 rounding + bf16-P quantization order).
- **P5:** wrap smem indices in `swz<32>()` symmetrically on store (162,184-185) and ld_A/ld_A_trans
  load (379-380,449-450). Pure address; rel unchanged.

---

## 6. Per-phase gates + revert criteria

Validate after each phase against `sdpa_naive_f32` (the in-tree oracle), the llama cross-check, and
the **end-to-end argmax on the 9B GGUF** (token IDs must match the pre-change reference, the
268/220/271 reference from prior tasks). Then ncu for monotonic occupancy with no SoL regression.

**Two distinct failure modes (verdict argmax-safe correction — do not conflate):**
- **BIT-SAFE on numerics** (gate = exact argmax + rel < 1e-3 as a *tripwire*, expect exact):
  **P0a, P0b, P1, Edge 1.** Their real risk is **SILENT LANE-MAP CORRUPTION** (no NaN), caught by
  **compute-sanitizer + `qkpv_test`**, NOT by a rel threshold.
- **NEEDS TOLERANCE GATE** (argmax may shift, rel drift expected): **P3** (~1e-5, tau=0 vs 8),
  **Edge 2 e4m3** + **Edge 3 FP8-mma** (rel < 1e-2 + mandatory end-to-end argmax + long-ctx `l_i`
  drift test).

| Phase | Correctness gate | Risk type | Revert criterion |
|---|---|---|---|
| P2 | rel < 1e-3 vs naive; argmax unchanged | low (math identical) | argmax shifts |
| **P0a** | rel < 1e-3 **AND** sanitizer clean; `qkpv_test` first | **silent corruption** (A-frag map) | sanitizer error or argmax shift |
| **P0b** | rel < 1e-3 **AND** sanitizer clean; `qkpv_test` first | **silent corruption** (C-frag map) | sanitizer error or argmax shift |
| P1 | trap `n_head_kv ∈ {1, n_head}` both agree; rel < 1e-3 | low (data reuse) | argmax shifts |
| P4 | rel < 1e-3; sanitizer race-clean | high (pipeline) | race or argmax shift; **OR no time win** (revert if 2-stage costs the 2nd CTA without a net long-ctx speedup) |
| P3 | tau=0 vs tau=8 match < 1e-5; argmax unchanged | medium (numerics) + `__any_sync` uniformity | argmax shifts or uniformity broken |
| P5 | rel unchanged (pure address) | low | rel changes at all |
| **Edge 3** | **`lowbit_peak.cu` microbench shows mxf8-f32acc > bf16 rate FIRST**; then rel < 1e-2 + argmax + `l_i` drift | high (FP8 numerics) | **microbench shows no rate win ⇒ DO NOT BUILD** |

**Gate definition (ship a phase only if all hold):** (a) `kernel_check` rel < threshold vs naive
oracle, (b) end-to-end argmax on 9B GGUF matches pre-change token IDs, (c) ncu shows monotonic
warps_active improvement with no SoL regression.

---

## 7. Projected occupancy trajectory + acceptance

- **P2 + P0a + P0b** (4 warps, Q-in-reg, register-O ⇒ 2nd CTA fits): 2.08% → **~15%** ← **floor
  structurally met** (4 warps × 2 CTAs = llama's 8 warps/SM). *(Without P0a: stalls at ~8% / 1 CTA —
  the trap.)*
- **+P1** (GQA reuse): warps flat ~15%, **per-call time → ~170 µs ±20%**, SM SoL → ~50% ← **floor
  met on time.**
- **+P4** (cp.async): ~15% warps, time drops further on long ctx; may cost the 2nd CTA — measure.
- **+P3/P5:** SoL polish ~50% → mid-50s%.

**Acceptance (ncu, pp512, x8 full-attn layers):** warps_active ≥ ~15%, SM SoL ≥ ~50%, per-call
within ~1.2× of llama's ~170 µs. **P2+P0a+P0b+P1 is the minimal set that hits it.** Do NOT chase
>16.67% by shedding registers — 2 CTAs/SM is the proven point on this silicon.

---

## 8. Build order (full sequence) + honest match-vs-beat ledger

**Order:** floor first (P2+P0a+P0b together → P1 → P4) → then EXCEED (Edge 1 cp.async-on-quant-bytes
restructure → validate long-ctx/MTP win) → then Edge 3 (`lowbit_peak.cu` microbench gate → QK-only
FP8, new `qkpv_fp8_test` modeled on `qkpv_test`). Edge 2 rides in with Edge 3. Edge 4 dropped.
P3/P5 polish last.

| Item | Real? | Beats llama on | Regime | Gain vs floor | Risk | Owned today? |
|---|---|---|---|---|---|---|
| **FLOOR** (P2+P0a+P0b+P1) | — | matches | pp512 occupancy | parity (~15%/~50%/~170µs) | med (lane maps) | no — port |
| **Edge 1** quant-KV prefill | **REAL** | bandwidth | long-ctx + MTP replay | **1.3–1.8×** long; ~1.0× pp512 | LOW | **yes (arithmetic shipped)** |
| **Edge 3** FP8 mma QK | REAL (primitive) | **compute** | **pp512 (dominant gap)** | 1.2–1.4× **IF microbench confirms** | **HIGH** | no — needs mxf8 path + `qkpv_fp8_test` |
| Edge 2 e4m3 KV | marginal | bandwidth (ALU, not bytes) | Edge 3 feedstock only | ~1.0× vs Edge 1 | MED | no |
| Edge 4 fuse rescale+dequant | DROPPED | nothing | — | <1.05× | — | — |

**The whole truth (no overstatement):**
- The **FLOOR is reachable** but ONLY with the **Q-in-reg fix (P0a)** the original MATCH plan
  omitted; the stated edit set without it lands at ~8% warps / 1 CTA/SM = **half the floor**.
- **Edge 1 wins the long-context / MTP-replay bandwidth regime — real today, owned, the anchor.** It
  is a **wash on the compute-bound pp512 gap** (the worst gap), and quant-KV is *already in the path*
  yet prefill is still 0.16× — confirming the gap is occupancy, not bandwidth.
- **Edge 3 is the only edge that can exceed llama on the dominant pp512 compute gap** — but its FP8
  TFLOP payoff is **OVERSTATED on f32-accumulate FA** (the repo's own bench: plain-FP8 = 1.87× but
  FA needs f32-accumulate which throttles FP8 toward the BF16 rate). It is **real-but-unbuilt,
  high-risk, and gated behind a `lowbit_peak.cu` microbench** that must show a block-scaled
  mxf8-f32acc rate win before any kernel work.
- **No single edge beats llama everywhere.** The "more than llama" story is two-pronged: Edge 1
  (bandwidth, real today) + Edge 3 (compute, real-but-unproven). Anyone claiming one edge wins
  everywhere is overstating — the compute/bandwidth regime split is the whole story.

## Files (absolute)
- Kernel + dequant seam: `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` —
  `fa_prefill_f32` (310-477), `fa_prefill_q` (486-625), `dq_q8_0_elem`/`dq_q5_1_elem` (133-155),
  `append_quantize_kv_q8_0_q5_1` (186-238, gains e4m3 variant for Edge 2), `mma_bf16` (107-112,
  Edge 3 replaces with FP8 `mma.sync`), validated lane maps `ATile/BTile/CTile::get_i/get_j` (70-84).
- Launchers: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — `fa_prefill`
  (1041-1061), `fa_prefill_view` (1067-1090, Edge 1 entry), `fa_decode` (1095-1150,
  `fa_decode_vec_q` ships the decode-side quant edge).
- Validation: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/bin/kernel_check.rs`
  (oracle `sdpa_naive_f32`, lib.rs:1009); Edge 3 needs new `qkpv_fp8_test` on the `qkpv_test` model.
- FP8 microbench gate: `probe/lowbit_peak.cu` (mxf8f6f4 block-scale), `probe/tc_peak.cu` (plain FP8).
- Plans this consolidates: `/home/avifenesh/projects/bw24/research/basics/FA-FLOOR-PLAN.md`,
  `/home/avifenesh/projects/bw24/research/basics/FA34-PLAN.md`.
