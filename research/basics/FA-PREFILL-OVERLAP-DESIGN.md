# FA-Prefill Warp-Specialized Softmax–GEMM Overlap — Design Sketch (sm_120a)

**Status: READ-ONLY research + design. No production kernel edited.** This is the producer/consumer
warp-specialization design for the bw24 `fa_prefill` kernel (the long-context prefill-attention lever),
plus the honest measured case for *when* it is worth building.

Machine: RTX 5090 Laptop, sm_120 / CC 12.0, 82 SMs, 100 KB opt-in smem/SM, ~847 GB/s.
Model: 9B-NVFP4 GGUF (qwen35 hybrid), head_dim **D=256**, n_head=16, n_head_kv=4, gqa_ratio=4,
**8 full-attention layers** (full-attn every 4th layer of 33; the rest are Gated-DeltaNet linear-attn).

---

## 0. TL;DR — the honest leverage verdict (READ FIRST)

Two prior overlap layers ALREADY landed (commit `474f7f1`):
1. The **floor port** (4 warps/CTA, Q-in-reg A-fragments, register-O C-fragments, BK=32, BLOCK_Q=64).
2. **Edge 5a as "register-resident softmax"** — the default kernel `fa_prefill_f32_pp` keeps the QK
   scores in C-fragment registers and runs the *entire* online-softmax (mask + 4-lane `__shfl_xor`
   max/sum + exp2 + alpha) in registers, eliminating the `sSw` score smem round-trip. This is a *partial*
   realization of FA3 overlap (it removes the smem-dependency serialization), **but it is NOT** the
   cross-tile producer/consumer overlap this doc designs: softmax(j) still runs *serially between*
   QK(j) and PV(j) inside one tile; nothing of tile j+1 is in flight while softmax(j) computes.

The remaining, unbuilt win is the **two-warp-group producer/consumer pipeline** (§4): softmax(j) on
CUDA cores overlapping QK(j+1) MMA on tensor cores, with a named-barrier handshake.

**MEASURED leverage of building it (this box, clock-locked 1860 MHz):**

| seq (pp) | fa_prefill/layer | fa_prefill ×8 layers | total prefill | **fa share of prefill** |
|---|---|---|---|---|
| 512  | 248 µs  | 1.98 ms   | 278 ms  | **0.7 %** |
| 2048 | 1846 µs | 14.8 ms   | 1051 ms | **1.4 %** |
| 4096 | 6225 µs | 49.8 ms   | 2203 ms | **2.3 %** |
| 8192 | 22280 µs| 178 ms    | 4771 ms | **3.7 %** |

fa-share grows only ~**seq^0.59** (log-log fit). Projected crossover: **fa hits 10 % at ~48K tokens,
25 % at ~227K, 50 % at ~731K.** bw24 prefill is dominated by the MoE/FFN/lm-head GEMMs end-to-end;
attention's O(T²) growth is real but the constant is tiny and the GEMM denominator is huge.

**Therefore (no overstatement): the producer/consumer overlap is a 5–20 % speedup of a kernel that is
0.7 %→3.7 % of prefill at 512→8192 — i.e. a ≤0.1 %→≤0.7 % whole-prefill win in the practical range,
and does NOT become a double-digit prefill lever below ~48K context.** It is worth building only
(a) for very-long-context (≥32K) workloads, or (b) as the algorithmic groundwork that *also* serves the
quant-KV (`fa_prefill_q`) path where the producer's cp.async+dequant is the genuinely co-bottleneck at
long ctx (Edge 1). The design below is correct and HW-portable; its *priority* is low until a ≥32K
scenario is on the board. This matches the FA-MATCH-THEN-EXCEED §1 framing that prefill is GEMM-bound.

---

## 1. Current `fa_prefill_f32_pp` structure (the consumer-only baseline)

Per CTA: **4 warps** (block `(32,4,1)`), each warp owns 16 query rows of the 64-row BLOCK_Q tile.
grid `(ceil(T/64), n_head, 1)`. Per warp, per KV tile of BK=32 keys, the **serial** dependency chain is:

```
[stage K,V tile -> smem]  (128-thread cooperative copy, __syncthreads)        <- producer work, INLINE
   |                                                                              (no overlap with compute)
[GEMM0  QK^T]  16 mma.sync.m16n8k16  -> scores in 4 C-frag registers Sc[4]
   |    (tensor pipe)
[SOFTMAX]  scale+causal mask; row_max4 (__shfl_xor); exp2f x32; row_sum4;        <- CUDA-core + transcendental
   |       alpha; m_i/l_i recurrence  (ALL in registers, no sSw round-trip)         TENSOR PIPE IDLE HERE
[write P -> sPw smem]  (bf16, mandatory for PV's A-operand ldmatrix layout) __syncwarp
   |
[rescale register-O by alpha]  32 CTiles x FMA
   |
[GEMM1  P@V]  ld_A(sPw) + ld_A_trans(sV) + 32 mma.sync  -> accumulate O_acc[32]
   |    (tensor pipe)                                                            __syncthreads (tile end)
```

**Budget (measured / from launcher):**
- smem (persistent, all 4 warps): `2*(2*BK*D + BLOCK_Q*BK) + 4*(BLOCK_Q*BK + 2*BLOCK_Q)` bytes
  = `2*(2*32*256 + 64*32) + 4*(64*32 + 128)` = **2*(16384+2048) + 4*(2176)** = 36864 + 8704 = **45.6 KB/CTA**.
- registers: ~255/thread (commit). O_acc alone = 32 CTiles × 4 f32 = 128 f32/lane.
- **occupancy: ncu shows flat 8.33 % warps_active = 1 CTA/SM** (4 warps of 48-warp max). The commit
  claimed 12.2 % / 2 CTAs; the isolated-kernel ncu here measures **1 CTA/SM** — the 255-reg + 45.6 KB
  smem envelope caps it at 1 resident CTA. (At 256 regs × 4 warps × 32 lanes = 32768 regs/CTA vs the
  64K reg file = 2 CTAs by regs alone, but combined with smem + the wave-quant at small grid it lands
  at 1.) This is the single biggest reason the producer/consumer split is *harder* here than on a
  data-center GPU: there is no spare occupancy to hide a producer warp-group behind other CTAs, so the
  overlap must come from WITHIN the CTA's 4 warps.

**Already-overlapped? Partially.** The smem score round-trip is gone (Edge 5a). What remains serialized:
the softmax transcendental + reduce sits on the critical path between the two MMAs of the *same* tile,
and the K/V staging (`__syncthreads`-gated cooperative copy) is fully inline before QK.

---

## 2. ncu — the overlap headroom (clock-locked 1860 MHz, GPU idle, isolated launches)

`fa_prefill_f32_pp`, one launch per seq length, full SM grid:

| metric (per-issue-active ratio unless %) | T=512 | T=2048 | T=4096 | T=8192 | meaning |
|---|---|---|---|---|---|
| `smsp__issue_active` % | 7.84 | 8.72 | 9.03 | 9.28 | **SM issues an inst only ~8–9 % of active cycles → ~91 % idle** |
| `sm__throughput` % (SoL) | 1.47 | 1.68 | 3.36 | 6.95 | overall SM util — very low, rising with grid fill |
| `warps_active` % | 8.33 | 8.33 | 8.33 | 8.33 | **flat 1 CTA/SM** |
| stall **long_scoreboard** | 2.37 | 6.14 | 4.87 | 5.20 | global/L2 mem dep (K/V loads) — **dominant stall** |
| stall **short_scoreboard** | 1.35 | 1.84 | 1.96 | 2.16 | smem/MIO dep (sPw P-write→PV ld_A, sV ld) |
| stall **wait** (fixed-latency) | 1.35 | 1.58 | 1.50 | 1.49 | exec deps incl. **exp2f transcendental** = the softmax math |
| stall **barrier** (`__syncthreads`) | 5.88 | 0.12 | 1.02 | 0.13 | high only at T=512 (few tiles → sync-bound) |
| stall **mio_throttle** | 0.08 | 0.05 | 0.03 | 0.04 | negligible |

**Reading the headroom for the overlap:**
- The MMA (tensor) pipe is idle the vast majority of the time (`issue_active` ~8–9 %). The kernel is
  **latency/stall-bound, not throughput-bound** — exactly the regime warp-specialization fixes (memory).
- The stalls that the producer/consumer overlap can hide are **long_scoreboard (K/V global loads,
  ~5 inst)** + **short_scoreboard (sPw/sV smem deps, ~2 inst)** + **wait (exp2 softmax, ~1.5 inst)**.
  Of these, the *softmax-specific* part the FA3 pingpong targets (the `wait` + part of short_scoreboard
  = exp2 + sPv smem dep) is ~**2.5–3.5 inst of the ~9–13 inst total stall budget = roughly 25–35 % of
  the stall cycles**. The bigger chunk (long_scoreboard, the K/V load latency) is what the *producer
  cp.async warp-group* hides — which is why 5b (warp-spec producer) attacks the larger share than 5a's
  softmax pingpong alone.
- **Net overlap headroom inside fa_prefill: the kernel spends ~90 % of issue cycles stalled; an ideal
  producer/consumer pipeline that fully hid the K/V-load + softmax latency behind MMA issue would lift
  issue_active toward the ~15–16 % that the floor's 2-CTA target implies, i.e. up to ~1.7–2× the
  per-launch kernel speed in the limit.** Realistically (single CTA/SM, hand-rolled barriers, residual
  tail) expect **1.3–1.6× on the fa_prefill kernel**.

**Apply the §0 share to get the whole-prefill win:** 1.4× on a kernel that is 3.7 % of prefill @8192 =
**~1.0 % whole-prefill speedup at 8192**, ~0.3 % @2048, ~0.2 % @512. Double-digit whole-prefill only
materializes past ~48K context (where fa-share ≥10 %).

---

## 3. Edge-5 plan summary (from FA-MATCH-THEN-EXCEED.md §2, Edge 5)

The match-then-exceed plan (commit `7259972`) frames FA3/FA4 as **hardware-independent scheduling
algorithms** hand-portable with `mma.sync` + `cp.async` + warp roles (no wgmma/tcgen05/TMA-mbarrier on
this laptop). Edge 5 has three sub-parts:

- **5a — Softmax–GEMM overlap (FA3 pingpong):** overlap online-softmax of tile j with QK^T of tile j+1
  so softmax transcendental+reduce latency hides behind tensor issue. Pure reorder → argmax bit-safe.
  *Build order: FIRST.* **Status: a resource-constrained form already shipped** as the register-resident
  softmax in `fa_prefill_f32_pp` (kills the score smem round-trip). The *cross-tile* pingpong is §4 here.
- **5b — Warp-specialization (producer/consumer):** split the CTA's warps — producer warp(s) run
  cp.async (+ dequant for Edge 1/3) to fill the K/V smem ring; consumer warps run MMA + softmax. Async
  handoff via smem barriers. Higher complexity (hand-rolled named-barrier choreography). *Evaluate AFTER
  5a.* argmax bit-safe (reorder only). **This is the unbuilt design in §4.**
- **5c — FA4 finer pipeline / exp approximation:** deeper KV-tile pipeline + attention-specific exp2
  polynomial. Marginal after 5a; defer.

Decision (per task directive, recorded): producer/consumer warp-specialization, **NOT** persistent-kernel
cooperative — fa_prefill is stall/latency-bound (softmax/loads stall MMA), which warp-spec fixes; the
GEMM's warp-spec attempt regressed because the GEMM is occupancy-bound (different bottleneck).

---

## 4. DESIGN — two-warp-group producer/consumer pipeline for fa_prefill (sm_120a)

### 4.1 Goal and the constraint that shapes everything

Overlap, *across* the KV-tile loop, the latency work (cp.async K/V load of tile j+1 + the softmax
transcendental/reduce of tile j) with the tensor-issue work (QK/PV MMA), inside a SINGLE CTA, because
there is no spare occupancy (1 CTA/SM). The CTA's 4 warps are re-roled into two **warp groups**.

**The hard constraint:** D=256 register-O is 128 f32/lane already; we CANNOT add a second buffer of O.
So the consumer warp group keeps the *full* register-O + online-softmax state; the producer warp group
holds almost no live state (just addressing). This dictates an asymmetric split.

### 4.2 Warp-group split (4 warps total, block stays `(32,4,1)`)

Two viable splits — measure both, lead with **Split A**:

- **Split A — 3 consumer + 1 producer (recommended).** Warps 0–2 = consumers, each owning a
  **21–22-row** sub-tile of a BLOCK_Q raised to 64 (keep 16 rows/warp → BLOCK_Q=48 for 3 consumers, OR
  keep BLOCK_Q=64 with consumer warps owning 64/3 rows — prefer BLOCK_Q=48, 16 rows × 3 warps, to reuse
  the validated 16-row CTile lane maps verbatim). Warp 3 = producer: issues `cp.async.cg` for the next
  K and V tiles into the smem ring and does the Edge-1 dequant (smem→smem) when on the quant path.
  Consumers never touch global K/V — they only `ldmatrix` from the ring. The producer is 1 warp of 4 =
  25 % of the threads doing pure load — acceptable because loads are latency- not throughput-bound here.
- **Split B — 2 consumer + 2 producer.** Only if ncu after Split A still shows long_scoreboard exposed
  (it shouldn't — 1 producer warp can keep cp.async in flight for BK=32×D=256 bf16 = 16 KB/tile easily).

The MMA itself (QK and PV) is run by the **consumer** warps (they hold Q-frags and O-frags). The FA3
"pingpong" of softmax-vs-MMA is then *within* the consumer group across tiles (§4.5), while the producer
group runs orthogonally. This is the key structural difference from the shipped `_pp`: the producer
removes the inline `__syncthreads`-gated K/V staging from the consumers' critical path entirely.

### 4.3 smem double-buffer (the K/V tile ring) and layout

Stage = 2 (double buffer; 3-stage blows the 100 KB cap and FA-3 found it slower at D=256). One rotating
buffer pair, mirroring llama's single-rotating-buffer trick to keep smem within the 1–2 CTA envelope:

```
ring[2] each:  sK_s[BK][D] bf16  +  sV_s[BK][D] bf16    = 2 * (32*256*2)        = 32 KB / stage
shared:        sP[BLOCK_Q][BK] bf16 (PV A-operand)       = 48*32*2              = 3.0 KB
               sM[BLOCK_Q] f32, sL[BLOCK_Q] f32 (only as alpha-broadcast slots) = 0.4 KB
   total smem = 2*32 KB (ring) + 3.4 KB = ~67.4 KB / CTA   ->  caps at 1 CTA/SM (already the case)
```

If 67 KB is too tight against regs for even 1 CTA, fall back to **single-stage K + double-stage V** or
BK=16 (halves ring to 16 KB/stage). The producer fills `ring[(j+1)&1]` while consumers read `ring[j&1]`.

cp.async path: `cp.async.cg.shared.global` of the 16 KB K-tile + 16 KB V-tile (f32→bf16 conversion is
NOT possible in cp.async, so on the f32 path the producer cp.asyncs the f32 source and the consumers'
`__float2bfloat16` happens at the ld point — OR the producer does a smem→smem convert; on the quant path
the producer cp.asyncs the 58 B/32-elem quant bytes and dequants smem→smem, the Edge-1 structure).

```
producer warp (warp 3):
    for j in tiles:
        cp.async.cg ring[(j+1)&1].sK_s <- K[k0_{j+1} ...]      // 16 KB
        cp.async.cg ring[(j+1)&1].sV_s <- V[k0_{j+1} ...]      // 16 KB
        cp.async.commit_group();
        bar.arrive(BAR_PROD_DONE, 128);   // signal: tile j+1 issued
        cp.async.wait_group<1>();         // keep at most 1 group in flight (double-buffer)
        bar.sync(BAR_TILE_READY, 128);    // rendezvous: tile j+1 landed, consumers may advance
```

### 4.4 Named-barrier handshake (bar.sync / bar.arrive, NO mbarrier)

sm_120a has no `mbarrier` arrive/wait usable here without TMA; use **named barriers** (`bar.sync`/
`bar.arrive` with explicit barrier id and a thread count). Allocate two barrier ids (PTX supports 16):

- `BAR_TILE_READY` (id 1, count = 128 = all 4 warps): producer signals "ring slot for tile j is filled
  and cp.async has retired"; consumers wait on it before `ldmatrix` of tile j. Implemented as a full-CTA
  `bar.sync 1, 128` AFTER the producer's `cp.async.wait_group`.
- `BAR_CONS_DONE` (id 2, count = 96 = 3 consumer warps, OR full 128 and producer no-ops): consumers
  signal "done reading ring slot j" so the producer may overwrite it with tile j+2. Producer waits on
  this before issuing the cp.async that reuses that slot.

```
   producer:  fill[j+1]; commit; bar.arrive(BAR_TILE_READY) ; wait_group<1>
              ... wait BAR_CONS_DONE(slot to reuse) before fill[j+2]
   consumer:  bar.sync(BAR_TILE_READY)            // tile j present
              QK(j) mma ; softmax(j) regs ; write sP ; PV(j) mma -> O_acc
              bar.arrive(BAR_CONS_DONE)           // slot j free
```

PTX: `bar.sync 1, 128;` and `bar.arrive 2, 96;` (asm volatile). The count MUST equal the number of
threads that hit that barrier or the CTA hangs — this is the single highest-risk detail (a wrong count
or a divergent consumer that early-outs on the causal break and skips its `bar.arrive` deadlocks). The
causal early-out (`if (k0 > q_pos_max) break;`) must be made **CTA-uniform** (all warps break together)
or the producer must arrive on behalf of broken-out consumers.

### 4.5 The FA3 softmax-MMA pingpong (within the consumer group, across tiles)

With the producer removing K/V-load latency, the consumer group additionally overlaps softmax(j) with
QK(j+1) — a 2-stage register pipeline over the scores:

```
consumer steady state (per tile j), tensor pipe never idle:
    QK(j+1) mma issued  --->  retires into Sc_next[4] C-frags     (tensor pipe busy)
       while it retires:  softmax(j) on Sc_cur[4]                 (CUDA cores: exp2, __shfl_xor)
                          write P(j) -> sP ; rescale O_acc by alpha(j)
    PV(j) mma  (P(j) @ V(j)) -> O_acc                              (tensor pipe busy)
    swap Sc_cur <-> Sc_next
```

This needs **two score-CTile sets in registers** (`Sc_cur[4]`, `Sc_next[4]` = 8 CTiles = 32 f32/lane
extra). With O_acc at 128 f32/lane + Q-frags (16 ATiles = 64 u32/lane) the lane budget is tight; if regs
spill past the 1-CTA point, drop the pingpong to producer-only overlap (5b alone) and keep softmax
serial — the producer already captures the larger (long_scoreboard) share per §2. **Build 5b first, add
the 5a pingpong only if ncu shows the `wait`+short_scoreboard softmax stall still exposed after 5b.**

### 4.6 Register-resident online-softmax state across the pipeline

The running stats stay exactly as in `fa_prefill_f32_pp` — **register-resident, per consumer lane**:
`m_lo, m_hi, l_lo, l_hi` (the two rows a lane owns via the CTile map `r_lo=lane/4, r_hi=r_lo+8`),
and `O_acc[32]` C-frags. The producer warp holds NONE of this. The pingpong only adds the second score
set; m/l/O persist across tiles in registers untouched by the producer. exp2/LOG2E recurrence and the
4-lane `__shfl_xor` reductions (`row_max4`/`row_sum4`) are reused verbatim → **bit-identical numerics →
argmax-safe** (pure reorder, same as the shipped 5a).

### 4.7 Correctness gates (same regime as the floor)
- Reorder-only ⇒ **bit-safe**; gate = exact argmax on 9B + rel < 1e-3 vs `sdpa_naive_f32` oracle
  (tripwire — expect exact), AND **compute-sanitizer racecheck clean** (the new risk is the smem ring
  race + the named-barrier counts, NOT numerics).
- Revert criteria: any racecheck hit, any CTA hang (wrong barrier count / non-uniform causal break), or
  **no per-call time win** (if 1 CTA/SM has no MMA throughput to overlap into, the producer split can
  LOSE by stealing a consumer warp — the GEMM warp-spec regression cautionary tale).

---

## 5. Realistic gain estimate (long ctx)

| Where | fa_prefill kernel speedup | fa share | whole-prefill speedup |
|---|---|---|---|
| pp512  | 1.3–1.6× (kernel) | 0.7 % | ~0.2 % |
| pp2048 | 1.3–1.6× | 1.4 % | ~0.3–0.5 % |
| pp4096 | 1.3–1.6× | 2.3 % | ~0.5–0.8 % |
| pp8192 | 1.3–1.6× | 3.7 % | ~0.9–1.4 % |
| ~32–48K (projected) | 1.3–1.6× | ~8–10 % | **~3–5 %** |
| ≥128K (projected) | 1.4–1.7× (more tiles to pipeline) | ≥20 % | **≥7–12 %** |

The overlap is a **long-context (≥32K) lever**, not a pp512/2048/4096/8192 lever. In the practical
8K range it is a sub-1.5 % whole-prefill win. Build it when a ≥32K-context scenario is on the board, or
fold the producer half into the `fa_prefill_q` quant path (Edge 1) where the producer's cp.async+dequant
of the smaller (0.45× bytes) KV is the genuinely co-bottlenecked work at long ctx.

---

## 6. Files
- Kernel (read-only here): `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` —
  `fa_prefill_f32_pp` (555-743, the default, register-resident softmax / partial 5a),
  `fa_prefill_f32` (342-519, `BW24_FA_FLOOR` floor), `fa_prefill_q` (752-934, quant twin, 5a folded in).
- Launchers: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — `fa_prefill` (1380),
  `fa_prefill_view` (1414). Forward call site: `hybrid_forward.rs:150`.
- Validation: `fa_sanitize` bin (oracle `cpu_sdpa`), `kernel_check` (`sdpa_naive_f32`).
- Source plan: `research/basics/FA-MATCH-THEN-EXCEED.md` (Edge 5).
