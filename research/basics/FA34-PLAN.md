# FA3/FA4 sm_120 Improvements — Build Plan

The kernel and design doc are verified. Now I'll synthesize the concrete port plan. The current `flash_attn.cu` (read above) is the **correctness-first** baseline — single warp per q-tile, O in smem, synchronous staging, no cp.async, no conditional rescale. The design doc (`fa_v1_kernel.md`) is the **target** tuned kernel. The plan bridges the two with prioritized, concrete edits.

# Hand-Port Plan: FA3/FA4 Improvements → bw24 sm_120 FA2 Kernels

## Ground truth: where the kernel is *today* vs the target

The shipping kernel at `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` is the **correctness-first** assembly described in its own header (`flash_attn.cu:47-50`): one warp per 16-row q-tile, O accumulator in **shared memory** (`flash_attn.cu:128,153,255-258`), **synchronous** f32→bf16 staging (`flash_attn.cu:180-186`), **unconditional** per-block O rescale (`flash_attn.cu:255-258`), and a scalar per-key decode (`flash_attn.cu:360-394`). The design doc `research/fa/fa_v1_kernel.md:39-417` is the *tuned* target (O in registers, 4 warps, cp.async, conditional rescale). This plan ports the design-doc wins into the shipping kernel, **prioritized by measured impact and split prefill vs decode**.

Two things are already done and must NOT be re-touched: **exp2 fast-exp** is live in both paths (`flash_attn.cu:234,238,388-389,420`) and **GQA index math** is correct (`flash_attn.cu:144`). The current GQA is *correct* but *not bandwidth-reusing* — that's a P1 item below.

---

## Priority ranking (impact-ordered)

| # | Change | Path | Bottleneck it attacks | Expected win | Risk |
|---|--------|------|----------------------|--------------|------|
| **P0** | O accumulator: smem → registers | prefill | Removes per-block smem read-modify-write of `16×256` f32 (`flash_attn.cu:255-258,285-286`) | Large (this is the dominant overhead in the current kernel) | Med (lane-map) |
| **P1** | GQA K/V reuse: `grid.y = n_head_kv`, inner `gq` loop | prefill | KV bandwidth (847 GB/s co-bottleneck) — load K/V **once** for 4 query heads | ~4× KV-load reduction | Low |
| **P2** | Multi-warp tile (4 warps, `BLOCK_Q=64`) | prefill | SM occupancy — 1 warp/CTA wastes 75% of a 4-warp-capable SM | Large (occupancy) | Med |
| **P3** | FA-4 conditional rescale (`tau=8`, `__any_sync`) | prefill | Removes ~10× of O-rescale vector FMAs | Med (compute-bound prefill) | Med (numerics) |
| **P4** | cp.async 2-stage K double-buffer | prefill | Hides K-load latency behind QK mma (only on `FA_KV_FP16`) | Med (long ctx) | High (pipeline) |
| **P5** | Smem XOR swizzle on K/V/Q ldmatrix | prefill | ldmatrix 8-way bank conflicts (~86%→94% SoL) | Small-Med | Low |
| **D0** | Split-K decode (`grid.z = split_k`) + combine | decode | SM fill — 1 block/head leaves ~66 of 82 SMs idle at long ctx | Large (decode latency) | Low |
| **D1** | Inline KV dequant in decode dot/gather | decode | **KV memory bandwidth** (decode is BW-bound) | Large — **SHARED EDIT, see §KV-quant** | Med |

**Prefill is compute-bound** (`fa_v1_kernel.md` topic-2 finding: ~768 FLOPs/byte at d=256), so P0/P2/P3 (work reduction + occupancy) dominate. **Decode is bandwidth-bound** (~1 FLOP/byte GEMV), so D0 (parallelism to saturate BW) and **D1 (halve the KV bytes read)** dominate; conditional rescale (P3) does NOT apply to decode (it rescales once at combine, not per block — `fa_v1_kernel.md` topic-2 decode finding).

---

## PREFILL: concrete edits to `flash_attn.cu`

### P0 — Move O from smem to registers (highest single win)

**Current:** `sO[16][HEAD_DIM]` f32 in smem (`flash_attn.cu:153`), rescaled by a strided 32-lane loop every KV block (`flash_attn.cu:255-258`), and the PV result is added back into smem (`flash_attn.cu:285-286`). That is a `16×256=4096`-element smem read-modify-write **per KV tile** — the single biggest cost in the correctness-first design (its header admits "register O is a follow-up", `flash_attn.cu:49`).

**Edit:** delete `sO` from the smem layout (`flash_attn.cu:153`) and hold O in registers as `float O_acc[HEAD_DIM/8][4]` per the target (`fa_v1_kernel.md:190-192`). Rescale becomes a register loop (`fa_v1_kernel.md:361-365`); PV accumulates directly into `O_acc` via `mma_m16n8k16(O_acc[ocol], …)` (`fa_v1_kernel.md:383`); epilogue writes registers→global with the deferred `1/l` normalize (`fa_v1_kernel.md:398-413`), replacing `flash_attn.cu:292-299`. This frees `16*256*4 = 16 KB` of smem (helps occupancy for P2).
**Load-bearing risk:** the C-fragment lane map. Each thread owns rows `lane/4` and `lane/4+8`, col-pair `(lane%4)*2`. The current code already encodes this in `CTile::get_i/get_j` (`flash_attn.cu:72-73`) — reuse those accessors verbatim for the register epilogue so the mapping is identical to the validated `qkpv_test`.

### P1 — GQA K/V reuse

**Current:** `grid.y = n_head` (`flash_attn.cu:140`), so each of 16 heads re-loads its KV head's K/V from HBM independently — 4× redundant since 4 heads share one `kv_head` (`flash_attn.cu:144`).

**Edit:** change launch to `grid.y = n_head_kv` and wrap the body in `for (gq = 0; gq < gqa_ratio; ++gq) { head = kv_head*gqa_ratio + gq; … }` (`fa_v1_kernel.md:184-186`). Stage K/V to smem **once per KV tile**, loop the `gq` query heads inside reusing the same `sK`/`sV`. Q is re-staged per `gq` (cheap: `16×256`). Launcher change at `fa_prefill` config: `grid_dim: (n_q_tiles, n_head_kv, 1)` (`fa_v1_kernel.md:615-618`). On the 5090 this is the largest *bandwidth* win for prefill since KV dominates Q traffic 4:1.

### P2 — Multi-warp CTA (4 warps, BLOCK_Q=64)

**Current:** `block=(32,1,1)`, one warp per CTA, 16 q-rows (`flash_attn.cu:120,141`). A 4-warp-capable SM runs at ~25% warp occupancy.

**Edit:** `NTHREADS=128`, `BLOCK_Q=64`, `WARP_Q=16` (`fa_v1_kernel.md:63-68`). Each warp owns its own 16 q-rows of the 64-row tile; all 4 warps share staged `sK`/`sV` (synergizes with P1 — one K/V load now feeds 4 warps × 4 gq heads). Replace `__syncwarp()` with `__syncthreads()` at the staging barriers (`flash_attn.cu:166,187,216,251,259,289`) since staging is now block-cooperative. This is the change that turns the kernel from "correct" into "fills the SM".

### P3 — FA-4 conditional ("slack") rescale

**Current:** every KV block unconditionally computes `alpha = exp2(...)` and rescales all of O (`flash_attn.cu:234,255-258`).

**Edit:** compute `delta = max(m_i, rmax) - m_i`; only bump `m` and rescale O when `delta > RESCALE_TAU` (=8.0, `log2(256)`), made **warp-uniform** via `bool need = __any_sync(0xffffffff, delta > tau)` (`fa_v1_kernel.md:321-331,75`). When `!need`, `corr=1.0` and the O-rescale loop is skipped. Because the final epilogue does `O/l`, deferred rescales stay exact. The `__any_sync` is mandatory: divergent lanes using different `m` in `exp2` silently corrupts P (`fa_v1_kernel.md:423`).
**Validation trap:** run with `RESCALE_TAU=0` (forces classic every-block) vs `8` — outputs must match `<1e-5` (`fa_v1_kernel.md:701`).

### P4 — cp.async 2-stage K double-buffer (FP16 path only)

**Current:** synchronous f32→bf16 staging (`flash_attn.cu:180-186`); cp.async helpers don't exist yet in this file.

**Edit:** add `cp_async_16`/`cp_commit`/`cp_wait<N>` helpers (`fa_v1_kernel.md:135-140`). Allocate **two** K smem buffers (`Ks + 2*BLOCK_KV*D_CHUNK`, `fa_v1_kernel.md:177`). At top of the KV loop `cp.async` the *next* K chunk into the alternate buffer, `commit_group`, compute QK on the *current* buffer, then `wait_group<1>` — overlapping the 40-50 cycle LSU round-trip with the HMMA/MUFU pipes (`fa_v1_kernel.md:20-21,387-389`). **Gate this on `FA_KV_FP16`**: cp.async cannot convert dtype, so the f32 Stage-1 path keeps synchronous staging; only a 16-bit KV cache benefits. **Do not** attempt 3-stage — at d=256 it blows the 99 KB smem cap and FA-3 itself found it slower (`fa_v1_kernel.md:31`).

### P5 — Smem XOR swizzle

**Edit:** wrap every smem tile index in `swz<STRIDE>()` (`fa_v1_kernel.md:100-105`) at the Q/K/V store sites (`flash_attn.cu:162,184-185`) AND the matching `ld_A`/`ld_A_trans` load sites (`flash_attn.cu:202-203,272-273`) — the swizzle must be applied symmetrically on store and load or it corrupts data. Pure address math; lowest-risk perf item, do it last so a bug here doesn't mask P0-P3 validation.

---

## DECODE: concrete edits to `flash_attn.cu`

### D0 — Split-K / flash-decoding

**Current:** `fa_decode_f32` already has the `split` parameter, `t_lo/t_hi` partitioning (`flash_attn.cu:341-344`), partial-output + `(m,l)` meta writes (`flash_attn.cu:396-398`), and a working `fa_decode_combine_f32` (`flash_attn.cu:403-426`). The kernel-side split-K is **present**; what's missing is the **launcher** choosing `split_k` to fill the 82 SMs.

**Edit:** in the Rust `fa_decode_view` launcher set `split_k = ceil(82/n_head)` when `t_kv >= 2048`, else 1 (`fa_v1_kernel.md:636`), and dispatch the combine pass only when `split_k>1` (`fa_v1_kernel.md:642-673`). With `n_head=16` this is ~6 splits → ~96 blocks, saturating the GPU. This is the dominant decode-latency win at long context and needs **no kernel change** beyond what's already at `flash_attn.cu:325-426`.
**Note** the current decode uses `block=(head_dim,1,1)=256` threads with a per-key block-reduce (`flash_attn.cu:372-384`). The target uses 4 warps with lanes owning 8 dims and warp-shuffle reduction (`fa_v1_kernel.md:490-518`), which is faster (no `__syncthreads` per key). Porting the warp-stripe decode is a secondary decode optimization after D0/D1.

### D1 — Inline KV dequant ⚠ SHARED EDIT WITH KV-QUANT WORKFLOW

This is the **single highest-impact decode change** (decode is BW-bound; quantized KV halves/quarters the bytes read per key) **and it is the same kernel region the KV-quant workflow must edit.** Flag it as a coordination point — do not let the two workflows edit `flash_attn.cu` decode independently.

**The shared region** is explicitly marked in the kernel as a hook:
- **K-dot dequant:** `flash_attn.cu:362-368` ("q8_0-K / q5_1-V HOOK") — the f32 `kt[tid]` load at `flash_attn.cu:369-370` becomes a per-thread dequant (q8_0: `int8 * d_K`; the design's target form is dp4a over int8 with Q pre-quantized to q8_1, `fa_v1_kernel.md:492-493`).
- **V-gather dequant:** `flash_attn.cu:390-391` — the `vt[tid]` f32 load becomes an affine dequant (q5_1: `val = d*q + m`, `fa_v1_kernel.md:511`).

**Coordination rule:** the KV-quant workflow owns the **cache storage format** (how K/V quant blocks are laid out in `cache.rs` and the KvLayer dtype), while this FA workflow owns the **dequant-in-dot arithmetic**. They meet exactly at `flash_attn.cu:362-391` (decode) and the corresponding prefill staging at `flash_attn.cu:180-186`. **Land the cache-format change first**, then this FA edit consumes it — otherwise the dequant reads garbage. The online-softmax math (`m_i`/`l_i`/`acc`) is **unchanged** by dtype (`flash_attn.cu:366-367` notes "keep m_i/l_i in f32 regardless of KV dtype"); only the load+multiply changes. Whoever touches this region second must rebase, not overwrite.

The prefill staging loop (`flash_attn.cu:180-186`) has the **same shared dependency**: if the KV cache becomes quantized, the f32 `K[...]`/`V[...]` reads there must also dequant before the `__float2bfloat16` store. That makes `flash_attn.cu:180-186` a **third shared edit site** between the FA tile-tuning work (P4 cp.async restructures this loop) and the KV-quant workflow (changes the load dtype). Sequence: KV-quant lands cache format → FA P4 restructures staging → both validate together.

---

## Suggested landing sequence

1. **P0 + P2** together (register O only makes sense with the multi-warp tile; they share the lane-map refactor) — validate vs `sdpa_naive_f32` at `tol<1e-3` per `fa_v1_kernel.md:690`.
2. **P1** (GQA reuse) — validate `n_head_kv ∈ {1, n_head}` traps (`fa_v1_kernel.md:702`).
3. **D0** (decode split-K launcher) — validate `split_k ∈ {1,4}` agree (`fa_v1_kernel.md:697`).
4. **P3** (conditional rescale) — validate `tau=0` vs `tau=8` match `<1e-5`.
5. **P5** (swizzle), then **P4** (cp.async, FP16 path) — perf-only, gate after correctness.
6. **D1 / prefill-staging dequant** — **coordinate with KV-quant workflow**, land cache format first.

## Files touched
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` — all P0-P5, D0-D1 kernel edits (lines cited inline above).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — `fa_prefill` launcher (`grid.y=n_head_kv` for P1; 99 KB smem opt-in via `set_attribute` for P2/P4) and `fa_decode_view` (`split_k` selection for D0); target signatures at `fa_v1_kernel.md:599-675`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/bin/kernel_check.rs` — validation grid (`fa_v1_kernel.md:692-707`).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/cache.rs` — **KV-quant workflow's file**; the shared dependency for D1 and prefill-staging dequant. FA workflow consumes its dtype, does not define it.

**Single highest-risk detail across all edits** (per `fa_v1_kernel.md:719`): the prefill mma C-fragment→register index mapping (`r0=lane/4`, `c0=(lane%4)*2`, P repack `kf=nb>>1, half=nb&1`). A wrong lane map corrupts P@V silently. The current `CTile::get_i/get_j` accessors (`flash_attn.cu:72-73`) are already validated — reuse them rather than re-deriving the indices, and keep the Step-2 llama.cpp cross-check (`fa_v1_kernel.md:705`) as the independent oracle.
