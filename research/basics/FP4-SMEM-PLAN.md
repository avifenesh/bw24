# FP4 GEMM smem-read-volume redesign — 5.5% → 40%+ of 762 TFLOP FP4 peak

## Why this lever exists (the structural beat-llama claim)

The FP4 path (`qmatvec_gemm_mxf4_kernel`, `cu/qmatvec_gemm.cu:933`) is the ONE prefill lever
llama.cpp cannot match on consumer Ada: it feeds RAW e2m1 weight nibbles + RAW UE4M3 micro-scales
straight to `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale` (`cu/qmatvec_gemm.cu:889-898`), a
762 TFLOP/s primitive. llama has no FP4 tensor-core path on sm_120 — its prefill tops out on the
int8/dp4a/bf16 ladder (≤219 TFLOP/s). If bw24 reaches even 40% of FP4 peak (~305 TFLOP/s) it is
~1.4x the int8 *peak* and structurally beats llama prefill regardless of how well-tuned llama is.

**But we are at 5.5%** (~42 TFLOP/s). ncu says why: **MIO 42% / barrier 31% / 78% mem SoL.**
This is NOT compute-bound and NOT cp.async-latency-bound (the kernel comment at lines 923-927
already established deepening the ring regresses). It is **smem-read-volume + pre-barrier
imbalance bound.** This plan attacks both, in gated passes, with ncu after each.

---

## 0. Where the smem read volume actually comes from (the metric)

Per K-step, the mma loop (`cu/qmatvec_gemm.cu:1087-1103`) issues **scalar u32 smem loads**:

- **A-fragment**: 4 scalar loads/lane (`afrag[0..3]`, lines 1089-1092) + 1 scale (`sa`, line 1093).
- **B-fragment**: inside the `BN/8 = 16` n-tile loop, **2 scalar loads/lane** (`bfrag[0..1]`,
  lines 1099-1100) + 1 scale (`sb`, line 1101), ×16 n-tiles = **32 B-loads/lane/K-step**.

So per lane per K-step: 4 (A) + 32 (B) = **36 scalar u32 smem loads** + scales. Across 128 threads
that is the MIO queue depth that pins the kernel at 42% MIO stall. **Each of those scalar loads is
an independent MIO transaction.** ldmatrix collapses a 16-int8 (A) or 8-int8 (B) operand assembly
that would be 4 / 2 scalar loads into **ONE** warp-cooperative matrix load instruction — this is
exactly the lever the int8 GEMM already pulls (`ld_A_s8` line 100, `ld_B_s8` line 109).

The FP4 fragment layout is already device-verified (`cu/qmatvec_gemm.cu:881-887`), and it is
**structurally identical** to the int8 m16n8k32 layout the existing ldmatrix helpers serve — the
only differences are operand width (16 vs 4 int8/lane for A) and that mxf4 packs two int8's worth
of nibbles per byte. The repacked `sWq[s][r][0..7]` (8 u32 = 32 bytes = 16 rows' worth indexed by
`r`) and `sAq[s][n][0..7]` are ALREADY 16B-aligned (`__align__(16)`, lines 954, 956) and laid out
row-contiguous — i.e. **already in the shape ldmatrix wants**, no relayout of the repack needed.

---

## PASS 1 — ldmatrix.x4.b16 for the A-operand (weight nibbles)

### The exact m16n8k64 A-operand lane mapping

From the device-verified layout note (`cu/qmatvec_gemm.cu:882-883`):
- A-frag lane L: `reg0 = row(L/4), K[(L%4)*8 .. +7]`; `reg1 = row(L/4+8)`, same K-group;
  `reg2 = row(L/4), K[+32..]`; `reg3 = row(L/4+8), K[+32..]`. Nibble n → K base+n.

In the current scalar form that is exactly lines 1089-1092:
```
afrag[0] = sWq[cur][warp*WARP_M + r0    ][kg];      // r0 = lane/4, kg = lane%4
afrag[1] = sWq[cur][warp*WARP_M + r0 + 8][kg];
afrag[2] = sWq[cur][warp*WARP_M + r0    ][kg + 4];
afrag[3] = sWq[cur][warp*WARP_M + r0 + 8][kg + 4];
```
The four regs are `sWq[row][kg]`, `sWq[row+8][kg]`, `sWq[row][kg+4]`, `sWq[row+8][kg+4]`. This is a
**16x16 b16 tile** in `sWq` (16 rows × 8 u32 = 16 rows × 16 b16 units), read in the canonical
`m8n8.x4.b16` interleave. The ldmatrix per-lane address is identical in *form* to `ld_A_s8`
(line 101) — `base + (lane%16)*stride_b16 + (lane/16)*4` in b16 units — but the FP4 reg
**ordering** (reg1 = row+8, not the int8 reg2=row+8) means we must verify the variant maps regs
to the slots the mma opcode expects. The two candidates:

1. `ldmatrix.sync.aligned.m8n8.x4.b16 {r0,r1,r2,r3}, [addr]` with per-lane
   `addr = &sWq[cur][warp*WARP_M + (lane%16)] [(lane/16)*4]` (b16 stride = 16). This yields
   `r0=row K[0..3 u32], r1=row K[4..7], r2=row+8 K[0..3], r3=row+8 K[4..7]` for an 8x8x4 tile.
   That is reg order (row,kg),(row,kg+4),(row+8,kg),(row+8,kg+4) — **NOT** the afrag order above.
2. **Therefore** after the single ldmatrix, remap to mma order with a 0-cost register swap:
   `a[0]=t[0]; a[1]=t[2]; a[2]=t[1]; a[3]=t[3];` (pure register renaming, the compiler folds it
   into the mma operand list — zero instructions). Bit-identical to the scalar afrag.

**New helper** (mirror `ld_A_s8` at line 100, widen to the 16-row FP4 A-tile):
```cuda
__device__ __forceinline__ void ld_A_mxf4(unsigned (&a)[4], const unsigned* sWq_row0) {
    // sWq_row0 = &sWq[cur][warp*WARP_M][0]; 16 rows x 8 u32 contiguous, 16B-aligned.
    const unsigned* xs = sWq_row0 + (threadIdx.x % 16) * 8 + (threadIdx.x / 16) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    unsigned t[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(t[0]),"=r"(t[1]),"=r"(t[2]),"=r"(t[3]) : "r"(addr));
    a[0]=t[0]; a[1]=t[2]; a[2]=t[1]; a[3]=t[3];   // remap to mma A-reg order (verify on device)
}
```
Replaces 4 scalar loads (lines 1089-1092) with **1 ldmatrix**. The `sa` scale (line 1093) stays a
scalar load (it is 1/lane, not on the hot multiplier — leave it).

**Per-pass gain (honest):** A-loads drop 4→1 per lane/K-step. A was 4 of the 36 scalar loads
(B dominates at 32). So this alone cuts ~3 of 36 = ~8% of the smem transaction count. Expect
**MIO 42% → ~38-39%, pp512 +3-6%.** Small — because B is the bulk. Pass 1 is the *correctness
beachhead*: it proves the FP4-ldmatrix reg remap on device cheaply before touching the 16-deep B
loop. **Do not expect a big pp512 jump here; the gate is "doesn't regress + rel unchanged."**

### GATE Pass 1
`cargo run --bin kernel_check` → FP4 informational rel UNCHANGED (≤ baseline; gate band rel<6e-2,
`kernel_check.rs:625`); end-to-end **argmax 268/220 stable** (the AUTHORITATIVE FP4 gate,
`kernel_check.rs:466-469, 631`); pp512 ≥ baseline (rises or flat). ncu after: MIO % recorded.

---

## PASS 2 — ldmatrix.x2.b16 for the B-operand (activation nibbles) — THE BIG ONE

This is where the volume is. The B loop runs `BN/8 = 16` times/K-step (line 1096), each iteration
doing 2 scalar loads (lines 1099-1100). That is **32 of the 36 scalar loads.** Collapse each
n-tile's 2 loads into 1 ldmatrix → halves the dominant cost, AND because ldmatrix is one warp
transaction the MIO queue depth drops far more than 2x in practice (fewer in-flight requests).

### The exact m16n8k64 B-operand lane mapping

From `cu/qmatvec_gemm.cu:884`: B-frag lane L: `col(L/4); reg0=K[(L%4)*8], reg1=K[+32]`. Per n-tile
the token is `tok = nt*8 + lane/4` (line 1097), and the 2 regs are `sAq[tok][kg]`, `sAq[tok][kg+4]`
(`kg = lane%4`). `sAq[s][n][0..7]` is 8 u32 = 16 b16 units, row-contiguous, 16B-aligned (line 956).

For ONE n-tile (8 tokens), the 8 lanes that own those tokens form an **8x8 b16 sub-tile** →
`ldmatrix.sync.aligned.m8n8.x2.b16` is the exact int8 `ld_B_s8` pattern (line 109), widened:
per-lane `addr = &sAq[cur][nt*8 + (lane%8)][((lane/8)%2)*4]`. But the FP4 B-frag uses `tok=lane/4`
(4-lane groups own a token) whereas the m8n8 ldmatrix maps `lane%8` → row. So the n-tile's 8 tokens
map to ldmatrix rows by `lane%8`, and the resulting 2 regs land in the (reg0=K[0..3], reg1=K[4..7])
order — remap to mma B-order `bfrag[0]=t[0]; bfrag[1]=t[1]` (verify; may be identity).

**New helper:**
```cuda
__device__ __forceinline__ void ld_B_mxf4(unsigned (&b)[2], const unsigned* sAq_ntile0) {
    // sAq_ntile0 = &sAq[cur][nt*8][0]; 8 tokens x 8 u32 contiguous, 16B-aligned.
    const unsigned* xs = sAq_ntile0 + (threadIdx.x % 8) * 8 + ((threadIdx.x / 8) % 2) * 4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
        : "=r"(b[0]),"=r"(b[1]) : "r"(addr));
}
```
Loop becomes (lines 1095-1103): `ld_B_mxf4(bfrag, &sAq[cur][nt*8][0]); mma_mxf4(...)`. Scale `sb`
(line 1101) stays scalar (1/4 lanes).

**Per-pass gain (honest):** B-loads 32→16 transactions, and ldmatrix coalescing typically cuts MIO
queue pressure more than the raw 2x. **MIO 42% → ~26-30%** is the realistic band (the int8 GEMM saw
its scalar→ldmatrix B move land in this range). Combined with Pass 1, smem transaction count
≈ 36 → ~17 per lane/K-step (≈53% cut). **pp512 expect +25-40%** (this is the load-bearing pass).
This does NOT yet reach 40% of peak — barrier 31% and bank conflicts now dominate (Passes 3-4).

### GATE Pass 2 — same triplet. The B remap is the risk; argmax 268/220 MUST hold exactly.

---

## PASS 3 — smem swizzle for conflict-free ldmatrix

ldmatrix at stride-32B (`sWq`/`sAq` rows are 8 u32 = 32B) can hit bank conflicts: 8 lanes reading
8 consecutive 32B rows all land their first 16B in the same bank set. The repo already analyzed the
naive fix (pad to 48B) and **rejected it** — the +8KB/CTA drops occupancy 4→3 and exactly cancels
the gain (`cu/qmatvec_gemm.cu:56-60`). So **do NOT pad. Use XOR-swizzle at the existing 32B stride**
(the comment at line 60 explicitly defers to "XOR-swizzle at stride 32" once occupancy is freed —
Passes 1-2 free MIO headroom, making this the right time).

**Swizzle:** permute the u32-group index within a row by XOR with a row-derived value so the 8 rows
an ldmatrix touches scatter across all 8 bank sets. Apply the SAME permutation at repack-store
(lines 1029-1030, the `int4` stores) and at the ldmatrix address calc (Pass 1/2 helpers):
- store: `sWq[s][r][ (gi ^ ((r & 1) << 2)) ]` style — or, since the int4 stores write u32[0..3] and
  u32[4..7] as two 128b chunks, swizzle at the **128b-chunk granularity**: chunk index
  `c ^ ((r >> 0) & 1)` keeps the two 16B halves conflict-free across odd/even rows.
- ldmatrix addr: XOR the `(lane/16)*4` (A) / `((lane/8)%2)*4` (B) column offset with the same
  row-derived bits so the read matches the swizzled store. **Store and load swizzle MUST be the
  identical permutation** or the operands corrupt — this is the correctness knife-edge of Pass 3.

Keep stride 32B (legal for both `cp.async.cg`-16 and `ldmatrix.b16`, per line 56). Zero smem-size
change → **no occupancy loss** (the whole reason the pad was rejected does not apply to XOR-swizzle).

**Per-pass gain (honest):** conflict-free ldmatrix removes the residual MIO replays. Realistic
**MIO ~26-30% → ~18-22%.** Marginal pp512 **+5-12%** on top of Pass 2. The big win was Pass 2;
this is the polish that lets the tensor cores actually see their operands without serialization.

### GATE Pass 3 — argmax 268/220 exact (swizzle bug = silent operand corruption → argmax drift,
which is precisely what this gate catches). rel unchanged. pp512 rises.

---

## PASS 4 — fix the pre-barrier work imbalance (barrier 31%)

The 31% barrier stall is the `__syncthreads` at `cu/qmatvec_gemm.cu:1078`. Before it, `repack`
(lines 1005-1038) and the fetches run. **The imbalance:** `repack` and `fetch_*` stride over
`BM=64` rows / `BN=128` tokens with `tid < NWARP*WARP_SZ = 128` threads (`for r=tid; r<BM; r+=128`,
line 1007). With BM=64 and 128 threads, **only 64 of 128 threads do repack work; the other 64 spin
idle into the barrier** — but they still issued the loop's bounds check + the `o<out_f` predicate.
More importantly the repack inner nibble-gather (lines 1018-1026: 8 iters × 8-deep shift/OR chain)
is a long ALU dependency chain on the 64 active threads while the idle 64 wait — the barrier can't
release until the slow 64 finish, so the fast/idle half's latency is wasted.

Two stacked fixes:

**4a — balance the repack across all 128 threads by splitting the 8-u32-group work, not the rows.**
Currently 1 thread does all 8 groups of 1 row. Instead, map (row, group-half) to threads:
`tid → row = tid % BM, half = tid / BM` (BM=64 → half ∈ {0,1}), each thread repacks 4 of the 8
u32 groups and does ONE `int4` store. Now all 128 threads are active, each does half the ALU, and
the two `int4` stores (lines 1029-1030) are split across the row's two threads. **Pre-barrier ALU
critical path halves** → barrier release no longer gated by a 64-thread serial tail.

**4b — collapse the nibble-gather dependency chain (lines 1022-1024)** with `__byte_perm`. The
inner `for n=0..7: w |= ((q[n]>>hinib)&0xF) << (4*n)` is an 8-deep dependent shift/mask/OR chain
(~16+ cycle latency). Two `__byte_perm` calls extract 4 low/high nibbles each in parallel
(the `gtable16` pattern already in the file, ~line 597). Cuts the per-group repack latency ~8x,
shortening the longest pre-barrier path further.

**Per-pass gain (honest):** barrier **31% → ~14-18%** (4a removes the half-idle imbalance; 4b
shortens the tail). Because repack is mostly off the mma chain (it's hidden under cp.async
wait_group, lines 1077-1083), the barrier reduction folds into pipeline utilization rather than
raw latency — realistic pp512 **+8-15%.**

### GATE Pass 4 — repack is bit-identical (4a/4b change WHO does the gather and HOW, not the bytes
produced) → rel must be EXACTLY unchanged; argmax 268/220 exact; pp512 rises.

---

## Honest projection — does this reach 40% (305 TFLOP) → pp512 ~3.5x llama?

Stack the realistic (not optimistic) per-pass stall reductions, starting from MIO 42% / barrier 31%:

| Pass | What | MIO | Barrier | useful≈100−(MIO+barrier+other) | TFLOP (762×useful) | of peak |
|------|------|-----|---------|-------------------------------|--------------------|---------|
| base | scalar loads | 42% | 31% | ~27% (5.5% measured → see note) | ~42 (measured) | **5.5%** |
| P1 | ldmatrix A | 39% | 31% | ~30% | ~50-65 | ~7-9% |
| P2 | ldmatrix B | 28% | 31% | ~41% | ~110-150 | **~15-20%** |
| P3 | swizzle | 20% | 31% | ~49% | ~150-190 | ~20-25% |
| P4 | barrier fix | 20% | 16% | ~64% | ~200-260 | **~28-34%** |

**The honest ceiling: ~28-34% of peak (~215-260 TFLOP), NOT a clean 40%.** Note the gap between
the "useful%" column and the 5.5% *measured*: stall % and achieved TFLOP are not linearly related
(issue-slot waste, tail effects, the s32→f32 epilogue at lines 1106-1116, and BN=128's 64-reg
accumulator pressure all sit between "no stall" and "peak FLOP"). The stall-accounting projections
in the findings that read off "762 × (1−stall) = 327-457 TFLOP" are **over-optimistic** — they
assume zero issue-slot waste and a linear stall→FLOP map, which the 5.5%-at-27%-useful baseline
already disproves (if linear, base would be ~205 TFLOP, not 42).

**To actually clear 40%** you need a fifth lever the four passes don't supply: **the BN=128
accumulator/occupancy redesign** (BN=64 halves `sAq` 8.2KB→4.1KB and the 64-reg `facc[16][4]`
accumulator → more CTAs/SM → real latency hiding). The findings flag this (BN=64, occupancy 4→5).
That is a *tile redesign*, higher-risk, and prior tile-redesign experiments on the int8 kernel
reverted on occupancy (lines 58-60, 926-927) — so it is a **Pass 5 candidate, gated and measured**,
not a given. Realistic outcome of Passes 1-4: **~3.5-4.5x current FP4 pp512** (5.5%→~25-30% of
peak), which at ~200-230 TFLOP is already **~1.0-1.1x int8 peak (219)** — i.e. it crosses the
beat-llama threshold (llama has no FP4 path) even WITHOUT hitting 40%. Pushing 40%+ → pp512 ~3.5x
*llama* (not 3.5x current) is **plausible only with Pass 5**, and should be claimed only after the
BN=64 tile is measured, not projected.

**Bottom line:** Passes 1-4 are the grind that takes FP4 from 5.5% to ~25-30% of peak and crosses
beat-llama; the clean 40%/pp512-3.5x-llama claim is honestly a Pass-5 (tile redesign) outcome and
must be earned on the bench, not asserted.

---

## Execution discipline (this is a GRIND)

- One pass per commit. **ncu after EVERY pass** — record MIO%, barrier%, mem SoL, achieved TFLOP.
- GATE every pass, no exceptions: FP4 `kernel_check` rel UNCHANGED + **argmax 268/220 exact** +
  **pp512 rises** (or flat with a recorded stall-% win that sets up the next pass). A pass that
  drifts argmax is reverted, not patched forward.
- Build/run: the kernel is `qmatvec_gemm_nvfp4_fp4` (`cu/qmatvec_gemm.cu:1119`), launched via
  `Engine::qmatvec_gemm_nvfp4_fp4` (`src/lib.rs:343`). Validate with `cargo run --bin kernel_check`
  (FP4 informational rel at `kernel_check.rs:466-469`; authoritative argmax at `:631`).
- Order is fixed: P1 (cheap correctness beachhead for the FP4 ldmatrix reg remap) → P2 (the big
  B-loop win) → P3 (swizzle, only safe once P1/P2 prove the addressing) → P4 (barrier). P5 (BN=64
  tile redesign) is conditional and only if 40% is a hard requirement.
