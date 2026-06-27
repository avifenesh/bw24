# PREFILL LEVERS — RANKED (pp512: 2090 tok/s measured → toward llama.cpp 6139, currently 0.34x)

**Scope:** prefill pp512 only. 9B-NVFP4, `BW24_GEMM=1 BW24_FP4=1`. RTX 5090 Laptop sm_120 / GB203 (82 SMs, 48 warps/SM max, 65536 regs/SM, ~99KB opt-in smem/SM, 847 GB/s achieved, warp-level `mma.sync` + `cp.async` + `ldmatrix` only — NO wgmma/tcgen05).

**Measured prefill profile (nsys, T=512, FP4 on — TRUSTED):**
GEMMs = 58% (nvfp4_fp4 22.5% + q4_K 19.4% + q5_K 16.1% + q8_0 4.4%) · SSM/attn = 27% (ssm_conv1d_silu 10.9% + fa_prefill 10.4% + gdn_scan 6%).

**Hard rule for this round:** every projected % below is a **projection, NOT a measurement**. The decode workflow this session proved old-doc bound-claims wrong (doc said "bandwidth-bound", ncu said latency-bound). Therefore **the implement phase MUST run the named ncu probe FIRST and gate the build on the measured bound before touching code.** Do NOT sum best-cases (see §Ceiling).

---

## Verdict summary

| Rank | Lever | Corrected gain (projected) | Argmax risk | Feasibility | Decision |
|------|-------|----------------------------|-------------|-------------|----------|
| **1** | `fp4-gemm-occupancy` (carveout 1-liner) | +0% to +5% pp512 (poss. ~0/neg) | none (math-identical) | ~1 line, trivially revertible | **KEEP — #1 next step** (cheapest probe, biggest single GEMM slice at 22.5%) |
| **2** | `fa-prefill-regO` (multi-warp + GQA-reuse + reg-O) | +3% to +6% pp512 | **high (algebra-adjacent)** | medium — proven in-repo decode pattern to port | **KEEP — #2**, larger ceiling but riskier/bigger diff |
| **3** | `gdn-scan-fuse` (smem-stage + feeder fusion) | +0% to +1.5% pp512 | low (float-reorder in A2 fusion only) | medium | **KEEP (low priority)** — third-order; only if ncu shows HBM-read-bound |
| — | `conv1d-smem-int8gemm` | +0% to +3% (already captured) | none | — | **DROP — LIKELY-REVERT** (see below) |

---

## DROPPED: `conv1d-smem-int8gemm` (LIKELY-REVERT)

Dropped per the lever's own self-rating and verified read-only against the code:

- **Part 1 (conv1d smem-halo + float4):** The float4-store premise is **already false**. `cu/hybrid.cu:40-44` maps consecutive `threadIdx.x → consecutive t → consecutive yc[t]`, so **the store is already fully coalesced** (the strided-write pathology only existed in the OLD 1-thread/channel-serial kernel, which was replaced by the 2D grid this session for the +7% already banked). smem-halo only relocates L1-resident redundant tap reads (`d_conv=4`, taps already in `wreg[]` registers per line 37-39) — it removes L1 hits, not DRAM traffic. Near-zero by construction; real chance of net-negative from added staging overhead.
- **Part 2 (int8 GEMM Fix B, barrier batching):** Fix A (pre-decode) is **already shipped** (`qmatvec_gemm.cu:455-503`, `USE_PREDECODE` path at 470-475). The only un-shipped lever, Fix B, requires deeper `NSTAGE` for batch depth — **the same occupancy direction that already reverted** (`NSTAGE=4` regressed 1287→1220; tile redesign reverted task #20). Current measured **2090 already ≈ the plan's full post-Fix-A+B target ~2100**, so the headroom is already captured/exhausted. The 3.55 barrier/issue figure **predates Fix A** and is stale.

Neither part dents the 2.9x gap; both carry revert precedent. Do not invest.

---

## RANK 1 — `fp4-gemm-occupancy` (carveout one-liner) — #1 NEXT STEP

**Why #1:** Largest single GEMM slice (nvfp4_fp4 = 22.5% of prefill), and the change is a **~1-line launch-attribute** that is **bit-identical** to the output (changes zero arithmetic, load order, repack, mma, or accumulator depth — distinct from the two prior tile/read-volume reverts). Cheapest possible probe-then-decide.

### Verified code facts (read-only)
- `src/lib.rs:386-401` `fp4_gemm_launch`: `block_dim:(32,4,1)` (4 warps/CTA), **`shared_mem_bytes:0` with NO carveout set** → kernel uses **static** smem and the **default 48KB carveout pool**.
- `cu/qmatvec_gemm.cu:968` `#define FP4_NS 2`. Committed static smem per CTA = sWq(4096) + sWsc(512) + sAq(8192) + sAsc(1024) + sWraw(8192) ≈ **22016 B ≈ 21.5KB**. `floor(48/21.5)=2` CTAs/SM — **exactly reproduces the observed 2-CTA limit**.
- The in-code comment at `qmatvec_gemm.cu:963-967` saying "36KB/CTA" refers to the **reverted FP4_NS=3** case, NOT the committed FP4_NS=2 — it is **stale** and must be reconciled.
- `set_attribute(...)` API is **confirmed live** in this cudarc 0.19 build (already used at `lib.rs:1051`).

### The two competing bounds (THIS is the make-or-break)
- **Optimistic premise (OLD ncu):** "occupancy-bound, 2 CTAs, Mem-SoL 77%" → raising to 4 CTAs helps.
- **Pessimistic premise (the author's OWN fresh in-code comment, lines 963-967):** kernel is **"MIO/smem-throughput-bound (Mem 77% / Compute 44%, top stall = MIO queue), NOT cp.async-latency-bound."** A throughput/SoL bound is **not** fixed by more resident CTAs — more co-resident CTAs add more concurrent consumers of the same saturated MIO/smem pipe → SM throughput stays **flat** (the exact int8 flat-throughput-across-occupancy pattern) or **regresses** via contention.

The two premises **contradict**, and the fresher (in-code) one points to near-zero. This is why the headline "occupancy-bound 2x" is **INFLATED**.

### Amdahl ceiling
22.5% slice → a perfect 2x kernel caps total at 22.5%·(1−1/2) = **+11.25%**, unreachable here because the kernel is throughput- not occupancy-bound. **Realistic: +0% to +5% total pp512 (2090 → ~2090–2195), with a real chance of ~0% or slight regression.**

### Exact change (apply ONLY after the ncu probe confirms occupancy-bound)
- `src/lib.rs:389` (in `fp4_gemm_launch`, after `let f = self.func("qmatvec_gemm_nvfp4_fp4");`): add
  `f.set_attribute(A::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)?;` (100 = prefer max smem; raises pool 48KB→~99KB so `floor(99/21.5)=4` CTAs/SM). Keep `shared_mem_bytes:0` (still static). Bring the `use ...CUfunction_attribute_enum as A;` into scope (mirror `lib.rs:1050`). **Verify the exact const name compiles** (cudarc may expose it as `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`, value 8).
- **Fallback if the carveout const is unavailable in the binding:** convert `sWraw`/rings to dynamic `extern __shared__` and call `set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 22016)` + `shared_mem_bytes:22016` — same effect, more invasive. Prefer the one-liner first.
- `cu/qmatvec_gemm.cu:963-967`: **update the stale comment** (22016 B / 2-CTA-from-48KB-default, not 36KB) — documentation-only, do AFTER measuring.
- **DO NOT** touch BN, FP4_NS, accumulator depth, repack, or `sWraw` layout this pass — those map to the two prior reverts.

### Per-kernel + argmax gate
- Per-kernel: `nvfp4_fp4` GEMM time / SM throughput / achieved occupancy via ncu before & after.
- Argmax: **none required beyond the standing gate** — carveout is math-identical, so the 268/271/1178 output gate is untouched. Still run `kernel_check` + pp512 as a smoke test.

### ncu metrics to capture FIRST (gate the build on these)
1. **CONFIRM** current FP4_NS=2 runs at **2 CTAs/SM** and limiter == **"Block Limit Shared Mem"** (Occupancy section). If limiter is registers/warps instead → carveout does nothing; abort.
2. **CONFIRM** Static Shared Memory Per Block ≈ **21.5KB** (NOT 36KB) — the whole premise rests on this reconciliation.
3. **AFTER carveout:** CONFIRM CTAs/SM rises **2→4** (achieved occupancy ~doubles) **AND pp512 / kernel time actually improves**. If SM throughput stays FLAT (MIO-queue-bound, the fresh-comment prediction) or regresses → **revert the 1-liner**. This is the decisive measurement.
4. **RESOLVE** occupancy- vs MIO/smem-bandwidth-bound — the OLD "occupancy-bound" verdict must be re-validated on the CURRENT build (decode lesson: old bound-claims can be wrong, here the optimistic one).
5. ptxas `-Xptxas -v` register count to confirm 4 CTAs is register-feasible once smem stops being the limiter.

### Honest cumulative after Rank 1
**2090 → ~2090–2195 tok/s (best case ~1.05x; expected ~1.0–1.03x). Still ~0.34–0.36x of llama.**

---

## RANK 2 — `fa-prefill-regO` (multi-warp + GQA-shared K/V + register-O)

**Why #2 (not #1):** larger Amdahl ceiling (fa_prefill = 10.4%) and it attacks a **genuinely different, near-certain bound** than the int8 revert — but it is a **bigger, algebra-adjacent diff** with **high argmax risk** if scope creeps, so it ranks below the zero-risk 1-liner.

### Verified code facts (read-only)
- `cu/flash_attn.cu:310-477` `fa_prefill_f32`. `lib.rs:1052-1055`: `block_dim:(32,1,1)` = **ONE warp/CTA**, `grid:((t+15)/16, n_head, 1)` = one head per `blockIdx.y` (re-stages each kv_head `gqa` times → ~4x K/V over-read).
- smem (lib.rs:1048 + flash_attn.cu:326-333): sQ(8192) + sK(32768) + sV(32768) + sP(2048) + sO(16384) + sS(4096) + sM/sL(128) = **96384 B ≈ 94.1KB/CTA**. At 94KB vs ~99KB cap → **exactly 1 CTA = 1 warp = 2.1% of the 48-warp SM.** With 1 warp there is **nothing to hide** the `__syncwarp` barriers / ldmatrix scoreboard / staging loads — opposite of the int8 case (16-way ILP in a 4-warp baseline hid latency, so SPLITTING hurt). **Adding parallelism is the correct direction here; the int8 revert does NOT transfer.**
- **Strongest de-risker:** the exact proposed pattern is **already shipped in the same file** for decode — `fa_decode_vec_q` (`flash_attn.cu:760-798`, launched `lib.rs:1101`): `block:(32, gqa)`, one CTA per kv_head, bf16 sK/sV staged ONCE and broadcast to all gqa warps, register accumulator. P1/P2 are a **structural port of proven, kernel_check-passing code**, not a novel redesign.
- Hot-path confirmed: `fa_prefill` feeds attn in the DEFAULT 9B-NVFP4 prefill path (`hybrid_forward.rs:150`), not just MTP.

### The bound to confirm + the BK subtlety
- Load-bearing premise: **latency-exposed (barriers + long-scoreboard), NOT mma-compute-bound** at 1 warp. Structurally near-certain but MUST be ncu-confirmed (if the QK/PV mma chain is compute-saturated via the deep `K_STEP` unroll, P1/P2 give less).
- **P0 alone canNOT raise occupancy:** removing sO (16KB → 78KB) still leaves sK+sV=64KB dominating → still 1 CTA. P0 is an instruction/smem-traffic win only (~1-2%).
- **The real win needs P1+P2 TOGETHER, and to break the 1-CTA wall needs BK 64→32** (sK+sV 64KB→32KB → 43KB → 2 CTAs). Without the BK shrink, even 4 warps stays 1 CTA/SM (≈17% occupancy) — half-measured.

### Amdahl ceiling
10.4% slice → even a 3x kernel caps total at 10.4%·(2/3) = **+6.9%**. **Realistic P1+P2: +3% to +6% pp512 (2090 → ~2150–2215). P0-alone ~+1-2% (NOT the doc's "2-3x", which the occupancy math does not support).**

### Exact changes (apply after ncu; P3 rescale MUST stay OUT)
- `lib.rs:1053`: `grid_dim ((t+BLOCK_Q-1)/BLOCK_Q, n_head_kv, 1)`, `block_dim (32, NWARPS, 1)` with `NWARPS=4`. One CTA per kv_head; its warps cover BLOCK_Q query rows and/or the GQA Q-heads of that kv_head.
- `flash_attn.cu:316-321`: `head=blockIdx.y` → `kv_head=blockIdx.y`; loop GQA Q-heads (`head=kv_head*gqa+gq`) inside, re-staging only sQ (cheap 8KB) per gq while sK/sV (64KB) stay resident. Mirror `fa_decode_vec_q:765`.
- `flash_attn.cu:330-331,432-435,456-464,470-476`: move `sO[16][256]` f32 (P0) into a per-lane **register** array `O_acc[HEAD_DIM/8][4]` reusing `CTile::get_i/get_j` (`flash_attn.cu:72-73`) for the lane map; the strided smem rescale at 432-435 becomes `O_acc *= alpha`; the `sO +=` at 462-463 becomes register `+=`; epilogue 470-476 writes regs→global. Frees 16KB.
- `flash_attn.cu:61`: `BK 64→32` to drop sK+sV 64KB→32KB → 2 CTAs/SM (the only way past 1 CTA given the 64KB tile wall). **MUST re-tune against the causal early-out at `flash_attn.cu:354`.**
- `flash_attn.cu:343,364,393,428,436,466`: every `__syncwarp()` → `__syncthreads()` once staging is block-cooperative across NWARPS warps.
- `lib.rs:1048-1049,1051`: recompute shmem after removing sO and after the BK change; keep the `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in.

### Per-kernel + argmax gate
- **argmax risk = HIGH-adjacent but contained:** P1 (GQA grid) and P2 (multi-warp) do **NOT** change per-(head,q-row) online-softmax accumulation order — they only relocate row/warp ownership and whether K/V is re-staged; the recurrence per row is byte-identical. P0 reuses `CTile::get_i/get_j` verbatim → PV mma FMA order bit-identical. **The ONLY numerics-changing item is P3 conditional rescale (tau=8) — SCOPE P3 OUT this pass.** Risk becomes real only if P3 creeps in.
- **Mandatory gate:** `kernel_check` rel-err **<1e-3 vs sdpa_naive** AND end-to-end pp512 + the 268/271/1178 argmax gate. The int8 tile redesign looked right on paper and regressed — treat projections as unconfirmed until pp512 measured.

### ncu metrics to capture FIRST (gate the build)
1. **CONFIRM latency-exposed, not mma-bound** at 1 warp/SM: Achieved Occupancy (~2.1% predicted), Warp Cycles Per Issued Instruction, stall reasons (expect high **Barrier** from `__syncwarp` + **Long Scoreboard** from ldmatrix/global staging + **No Eligible**). **This is the load-bearing premise — if mma-bound, P1/P2 won't help and the int8 revert pattern repeats.**
2. CONFIRM occupancy limiter == **Shared Memory** (not registers); predicted 1 CTA/SM from 94KB.
3. CONFIRM the O-rescale + sO-accumulate (432-435, 462-463) is a **measurable fraction** of kernel time before investing in P0 (if <5%, P0 not worth the lane-map risk).
4. MEASURE GQA K/V over-read: DRAM read bytes for K/V vs theoretical (expect ~4x) — confirms P1 upside.

### Honest cumulative after Rank 2 (on top of Rank 1)
**Best case ~2195 + ~6% ≈ ~2300 tok/s. Expected ~2150–2280. Still ~0.37–0.38x of llama.** (NOT a sum of independent best-cases — see §Ceiling.)

---

## RANK 3 — `gdn-scan-fuse` (smem-stage + feeder fusion) — LOW PRIORITY

**Why kept but low:** third-order (gdn_scan = 6%), and the code structure argues the bound may be the **wrong one** for the proposed Option (A).

### Verified code facts (read-only)
- `cu/hybrid.cu:59-116` `gdn_scan_kernel`: a genuine **512-step serial recurrence** (`s_shard[r]=g*s_shard[r]+k_reg[r]*delta_col`, line 100) with **two `warp_reduce_sum` per step** (lines 93, 103) — i.e. 2× `__shfl` reduction-latency chains × 512 iters, strictly serialized. smem-staging does NOTHING to this chain; it only relocates q_t/k_t/v_t reads from HBM to smem.
- `lib.rs:1324`: `COLS_PER_BLOCK=4` (NOT 8). So intra-CTA redundancy is **~4x** (4 warps re-read the same q_t/k_t), dedupable by smem; the full 32x is spread across **8 separate CTAs per head** (32 cols / 4 per CTA) which smem **cannot** dedupe, and L2/L1 likely already serves most of those identical-row hits. Realistic dedup ≪ the implied 32x.
- `grid.x = H = num_v = 32` (hybrid_forward.rs:227) → 32 CTAs over 82 SMs is **under-subscribed**; needs column-tiling.

### Amdahl + realistic
6% slice → perfect 2x = **+3.2% ceiling**. **Realistic +0% to +1.5%, gated entirely on an ncu HBM-read-bound check that the code structure suggests will FAIL** (it looks reduction/`__shfl`-latency-bound — the SAME trap that bit the int8 GEMM and decode matvec). If latency-bound, only the deprioritized WY-form (Option B) helps, and that is the argmax-risky one.

### Changes (Option A — low-revert, neutral-safe; only if ncu shows HBM-read-bound)
- `cu/hybrid.cu:59-116`: add `extern __shared__ float smem[]`; at top of `for(t)`, CTA warps cooperatively block-stride load q_t/k_t/v_t[0..128] + g_val/beta_val into smem ONCE, `__syncthreads`, then replace per-warp HBM reads at 80-87/95 with smem reads. CTA owns a column-tile of ONE head.
- `lib.rs:1324-1329`: CTA spans a column-tile of one head; `shared_mem_bytes = 3*S_v*4` (1536B); e.g. `block (32,8)`, `grid (H,1,S_v/8)`.
- **A2 fusion (the larger AND safer perf lever):** fold `qkv_to_gdn_repack` gather (`hybrid.cu:254`, `kh=vh%num_k`) + the two `l2_norm` passes (hybrid_forward.rs:210-212) into the gdn prologue so `q_g/k_g/v_g/q_l2/k_l2` (~75MB) are never materialized — removes 5 tensors + 3 feeder launches.

### Per-kernel + argmax gate
- Option A alone is **argmax-safe** (bit-identical staged f32, no reorder of `warp_reduce` or accumulation → `o=attn_col*scale` at line 104 unchanged).
- **A2 fusion is the hazard:** it MUST replicate the L2 sumsq-over-128-row + rsqrt bit-exactly via `warp_reduce_sum`; any reorder breaks the gate. Argmax risk = low-float-reorder, **understated** in the lever framing — validate L2 fusion against the gate explicitly.

### ncu metrics to capture FIRST (gate the build)
1. **IS gdn_scan HBM-bound or recurrence/`__shfl`-latency-bound?** High Memory SoL + long-scoreboard → A wins. Bound on serial recurrence (`__shfl` latency, barrier/MIO, No-Eligible from the 512-deep chain) → A gains ~nothing. **Do NOT build A until this confirms HBM-read-bound.**
2. L2/L1 hit-rate + `dram__bytes` for gdn_scan — does cache already absorb the 32x q_t/k_t re-reads?
3. Per-kernel split: what fraction of the 6% is repack + 2× l2_norm feeders vs the scan itself (A2 may be the bigger, safer win).
4. Occupancy if a CTA owns a whole head (32 CTAs / 82 SMs under-subscribed → confirms column-tiling need).

### Honest cumulative after Rank 3
**Adds at most ~+1.5%. Stack expected ~2200–2320 tok/s, ~0.36–0.38x of llama.**

---

## #1 NEXT PREFILL STEP (once the crate compiles)

**`fp4-gemm-occupancy` carveout probe-then-1-liner.** Rationale: biggest single GEMM slice (22.5%), zero numeric risk, ~1-line diff, trivially revertible — the lowest-cost / highest-information-per-effort move.

**FIRST ncu measurement (before writing any code):** profile the CURRENT `qmatvec_gemm_nvfp4_fp4` (FP4_NS=2) and read the **Occupancy section**:
- Is the limiter **"Block Limit Shared Mem"** at **2 CTAs/SM**? (else carveout is inert — stop)
- Is **Static Shared Memory Per Block ≈ 21.5KB** (NOT 36KB)? (reconciles the stale comment)
- Capture the dominant **stall reason / SoL**: if it's already **MIO-queue / smem-throughput** (Mem 77% / Compute 44% per the in-code comment) rather than occupancy, the carveout will leave SM throughput FLAT — **do not build it**, and pivot effort to Rank 2.

If the probe confirms occupancy-bound, apply the 1-line `set_attribute(...PREFERRED_SHARED_MEMORY_CARVEOUT, 100)`, re-profile for 2→4 CTAs, and gate on a **measured pp512 improvement** (revert if flat/negative).

---

## HONEST CEILING — does this stack reach 6139?

**No. Not remotely.** Do NOT sum best-cases — these levers share the same HW and the GEMM cluster dominates everything else.

- **Amdahl caps, individually:** fp4 +11.25% (theoretical, unreachable), fa +6.9%, gdn +3.2%, conv1d ~0. The slices overlap in time and in resource (smem/MIO), so the realistic **combined** stack is **roughly +5% to +12%**, NOT the arithmetic sum.
- **Best realistic stack ≈ 2090 → ~2200–2340 tok/s ≈ 0.36–0.38x of llama** (from 0.34x). That **closes only ~1/8 of the 2.9x gap.** These four levers **do not** approach 6139.

**Why the gap survives:** 58% of prefill is GEMMs, and the residual ~2.9x is a **compute (mma-issue) ceiling**, not a latency/barrier/occupancy problem the above levers address. bw24's int8 GEMM tops out near 219 TFLOP/s int8 peak; the FP4 GEMM is hitting only **~5.9% of the 762 TFLOP/s block-FP4 peak**; llama's MMQ runs near its peak. You cannot reach 6139 by tuning occupancy/barriers on kernels that are already near their numeric-format compute ceiling — you must **raise the format/throughput ceiling itself**.

**The regime-change required beyond these levers:**
1. **Stage-C block-scale FP8 GEMM (381 TFLOP/s)** for the q4_K/q5_K/q8_0 cluster (~40% of prefill), or push the FP4 GEMM (762 TFLOP/s peak) from ~5.9% toward a respectable fraction of peak. This is where the 2.9x actually lives. The FP4 kernel's real ceiling is the MIO/smem-throughput bound flagged in its own comment — closing THAT (smem-traffic reduction / better B-operand reuse / a real tile-throughput redesign), not occupancy, is the FP4 path. **Both need fresh ncu to pick the right sub-lever** (the FP4 "Pass-5 BN128→64 redesign" is untried and higher-risk given the int8 tile-redesign revert).
2. **A prefill GEMM that actually saturates the tensor pipe** — i.e. the compute-issue ceiling the int8-GEMM ledger points to — likely a larger-M batched-GEMM formulation that amortizes weight-decode and barriers across many more tokens than the current BN=128 tile. This is a redesign, not a tune, and is **out of scope for the four levers ranked here**.

**Bottom line:** Rank 1 → 3 are worth doing (cheap, mostly safe, each ncu-gated), but they are **incremental polish on a latency/occupancy regime**. Crossing toward 6139 requires the **format/throughput regime-change (Stage-C FP8 and/or a genuine FP4 throughput redesign)** on the 58% GEMM cluster — measured, not assumed, against fresh ncu of the CURRENT kernels.
