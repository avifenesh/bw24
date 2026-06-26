# GEMM Tuning Plan (cp.async + ldmatrix + swizzle)

I now have all load-bearing facts grounded. Current smem is tiny (7.7KB kernel1, 9.2KB kernel2), single-buffered, scalar byte-assembly fragment loads, two `__syncthreads()` per K-step, and no cp.async. The findings' "5.6% matches exactly" and "Turing" claims are physically wrong and I'll correct them. Here is the plan.

---

# qmatvec_gemm.cu Kernel-Rewrite Plan: 5.6% → 40%+ of sm_120 int8 peak

Target file: `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec_gemm.cu` (539 lines, two kernels: `qmatvec_gemm_kernel<QT>` lines 161-280 for Q8_0/Q4_K/Q5_K, `qmatvec_gemm_kernel2<QT>` lines 406-525 for Q6_K/NVFP4). Launch: `lib.rs:551-563` (`qmatvec_gemm`), `lib.rs:572-594` (`qmatvec_gemm_raw`). Gate: `kernel_check.rs:360-447`.

## 0. Honest correction of two findings claims before we build on them

The findings repeatedly call sm_120 "Turing" and assert cp.async is a "Turing add". **This is wrong and it changes the plan's ceiling.** sm_120 is consumer Blackwell (RTX 50-series), CUDA 13.1 per `build.rs:7`, compiled `arch=compute_120a` (`build.rs:17`). It has real `cp.async.cg.shared.global` async-copy hardware (Ampere+ class), bypassing the register file on global→smem — not the SM_75 emulation the findings imply. The upside (cp.async genuinely overlaps loads) is real; the "Turing" framing understates it.

Second: the finding "21µs wasted ⇒ 6% loss, observed 5.6% matches exactly" (mmq topic, last bullet) is **reverse-engineered numerology** — it back-fits one cause (bank conflicts) to the whole 5.6%. The real 5.6% is dominated by several independent stalls (scalar fragment assembly, no load/compute overlap, 2 syncs/K-step), not bank conflicts alone. Do not trust that single-cause attribution; the ranked gains below are built from the actual code structure, not that formula.

## 1. RANKED changes by expected speedup (honest, no double-counting)

The four attack **different stalls**, so gains compound — but each is bounded by what fraction of K-step latency it removes, and by Amdahl once the others land. Current per-K-step cost breakdown (lines 190-266) is roughly: global load + decode into smem (~40%, fully exposed, no overlap), scalar fragment byte-assembly (~25%, lines 226-251), the mma itself (~15%), scale fold (~10%, lines 257-263), 2× `__syncthreads` (~10%, lines 221+265).

| # | Change | Attacks | Honest standalone gain | Why bounded | Cite |
|---|--------|---------|----------------------|-------------|------|
| **1** | **cp.async ring buffer (2→3 stage)** — overlap next K-step's global→smem load behind current mma | Exposed load latency (~40% of K-step, the single biggest waste; current kernel is fully serial: load → sync → compute → sync) | **2.2–2.8×** | Cannot hide more than the load fraction; once compute-bound, returns drop. Bounded by smem capacity for stages (we have headroom, §below) | `qmatvec_gemm.cu:189-221, 265` |
| **2** | **ldmatrix fragment load** — replace scalar byte-assembly (`afrag` 226-236, `bfrag` 244-251) with `ldmatrix.x4.b16`/`.x2.b16` reinterpreted for s8, reusing the proven `flash_attn.cu:90-96 ld_A` helper | Fragment-build instruction overhead (~25% of K-step): currently 4 byte-loads + 3 shifts + 3 ORs per .b32, ×4 A-regs + ×2 B-regs per n-tile, ×8 n-tiles | **1.3–1.5×** (after cp.async lands; ~1.6× standalone) | Only collapses the fragment-assembly fraction; the mma and scale fold are untouched. Smaller once cp.async has already removed the dominant stall | `qmatvec_gemm.cu:226-236, 244-251` |
| **3** | **Swizzled smem layout** — pad `sW[BM][BK]`→`[BM][BK+pad]`, `sA[BN][BK]`→`[BN][BK+pad]` (or XOR-swizzle) so ldmatrix is conflict-free | smem bank-conflict replays **on the ldmatrix from #2** (enabler, not standalone) | **1.1–1.25×**, but ONLY meaningful with #2 present | This is the multiplier that makes #2's ldmatrix actually conflict-free. Without ldmatrix, the current scalar 4-byte reads barely conflict (32B rows = 8 banks, mostly fine), so swizzle alone ≈ 1.0× | `qmatvec_gemm.cu:174-175, 228-233, 248` |
| **4** | **One `__syncthreads`/K-step + scale fold hoist** — fuse decode into the cp.async pipeline so only one barrier remains; precompute `sWd*da`-style products | 2 barriers/K-step (~10%) + redundant per-(ci) float mults in scale fold (lines 257-263) | **1.05–1.15×** | Small absolute; partly subsumed once cp.async pipelining already reorders barriers. Real value is correctness-preserving cleanup that unblocks #1 | `qmatvec_gemm.cu:221, 265, 257-263` |

**No double-counting**: #1 hides *time the SM spends waiting on DRAM*; #2 removes *instructions the SM issues to build fragments*; #3 removes *smem-bank replay cycles incurred by #2's instruction*; #4 removes *barrier stalls*. They touch four disjoint cost buckets. #3's gain is explicitly conditioned on #2 (swizzle is pointless without ldmatrix consuming it) — that's why it's ranked below #2 and quoted as a multiplier, not an addend.

## 2. Concrete new K-loop structure

Apply to `qmatvec_gemm_kernel<QT>` (lines 161-280). Same structure ports to `kernel2` (406-525) with two weight buffers (`sWlo`/`sWhi`) and two mmas — see §5.

**New smem (replaces lines 174-179).** 3-stage ring + swizzle pad. Current usage is only 7.7KB (verified), so we have vast headroom under sm_120's ~100KB/CTA:
```
// pad BK 32 -> 36 (int8) so ldmatrix m8n8 rows land on distinct banks (K%8 != 0 alignment)
#define BKP 36
#define NSTAGE 3
__shared__ int8_t sW[NSTAGE][BM][BKP];   // 3*64*36  = 6.75 KB
__shared__ int8_t sA[NSTAGE][BN][BKP];   // 3*128*36 = 13.5 KB
__shared__ float  sWd[NSTAGE][BM], sWb[NSTAGE][BM];
__shared__ float  sAd[NSTAGE][BN], sAsum[NSTAGE][BN];   // ~3*1.5KB
// total ~25 KB/CTA — still >=3 CTAs/SM
```

**Helper to add near line 58** (port of `flash_attn.cu:90-96`, s8 reinterpret — proven on sm_120 per mma_validate.cu):
```
__device__ __forceinline__ void ld_A_s8(int (&t)[4], const int8_t* xs0, int stride_bytes){
    const uint32_t* xs = (const uint32_t*)xs0 + (lane%16)*(stride_bytes/4) + (lane/16)*4;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(xs);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(t[0]),"=r"(t[1]),"=r"(t[2]),"=r"(t[3]):"r"(addr));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* g){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(s),"l"(g));
}
```

**New K-loop (replaces lines 189-266).** Pseudocode with line anchors:
```
// --- PROLOGUE: async-fill stages 0..NSTAGE-2 (replaces nothing; new before line 190) ---
for (s = 0; s < NSTAGE-1; s++)
    issue_load_stage(s, /*g=*/s);      // decode weights + copy activations into sW[s],sA[s]
                                       //   weights NOT cp.async (need decode); activations
                                       //   sA via cp_async16 straight from aq (it's already int8)
asm volatile("cp.async.commit_group;");

float facc[BN/8][4] = {0};             // unchanged accumulator (line 183-187)

for (int g = 0; g < nblk; g++) {       // was line 190
    int cur = g % NSTAGE, nxt = (g + NSTAGE-1) % NSTAGE;

    // (A) prefetch g+NSTAGE-1 BEFORE consuming g  -> overlaps DRAM with this iter's mma
    if (g + NSTAGE-1 < nblk) {
        issue_load_stage(nxt, g + NSTAGE-1);     // decode-to-smem (weights) + cp_async16 (acts)
        asm volatile("cp.async.commit_group;");
    }
    asm volatile("cp.async.wait_group %0;" :: "n"(NSTAGE-2));  // ensure `cur` ready
    __syncthreads();                              // SINGLE barrier/K-step (was 2: lines 221+265)

    // (B) build A fragment via ldmatrix (REPLACES scalar lines 226-236)
    int afrag[4];
    ld_A_s8(afrag, &sW[cur][warp*WARP_M][0], BKP);   // BKP=36 -> conflict-free (change #3)

    // (C) per n-tile: ldmatrix B (REPLACES scalar 244-251) + mma + scale fold
    #pragma unroll
    for (int nt = 0; nt < BN/8; nt++) {
        int bfrag[2];
        ld_B_s8(bfrag, &sA[cur][nt*8][0], BKP);      // x2.b16 col-major, two 16-k halves
        int dacc[4] = {0,0,0,0};
        mma_s8_m16n8k32(dacc, afrag, bfrag);          // UNCHANGED mma, lines 52-58
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {              // scale fold, was lines 257-263
            int rr = warp*WARP_M + lane/4 + (ci>>1)*8;
            int nn = nt*8 + (lane%4)*2 + (ci&1);
            float da = sAd[cur][nn];
            facc[nt][ci] += sWd[cur][rr]*da*(float)dacc[ci] + sWb[cur][rr]*da*sAsum[cur][nn];
        }
    }
    // NO second __syncthreads here (line 265 removed; ring buffer makes it unnecessary —
    // next iter's wait_group + the single top barrier serialize correctly per-stage)
}
// write-out UNCHANGED (lines 269-279)
```
where `issue_load_stage(s,g)` is the existing decode/load bodies (lines 193-220) retargeted to `sW[s]/sA[s]/...[s]`, with the activation copy (lines 211-213) replaced by a `cp_async16` of the 32 int8 bytes (one 16B + one 16B, or a 32B cp.async if the row is 16B-aligned — `aq + t*in_f + g*32`, `in_f%32==0` so it is). Weight decode stays synchronous-into-smem (it needs ALU), but is now overlapped with the *previous* stage's mma.

**`__byte_perm` note for `ld_A_s8` reinterpret**: the `.b16` ldmatrix loads 8×bf16 = 16 bytes/lane; those 16 bytes ARE the 16 int8 A-operands in the exact m16n8k32 layout the manual code at 234-235 builds — so `afrag[0..3]` consume identically by `mma_s8_m16n8k32` with zero repacking. This is why the swap is bit-equivalent (validated below).

## 3. Realistic cumulative: 634 → ? pp512

Multiplicative, with honest Amdahl (each later gain shrinks because the earlier ones already removed their bucket):

- Start: **634** pp512 (5.6% of 219 TFLOP/s peak)
- ×2.5 (cp.async, #1): **~1585**
- ×1.4 (ldmatrix, #2): **~2220**
- ×1.18 (swizzle enabling #2, #3): **~2620**
- ×1.10 (sync/fold, #4): **~2880**

**Cumulative ≈ 2700–3200 pp512, ≈ 24–28% of int8 peak.** This **does NOT beat 6240** — it lands at roughly **0.45–0.51×** llama.cpp. This matches the project's own honest estimate in `research/BW24-BUILD-MAP.md:121` ("first tuned lands ~3500–5500 pp512, not llama's 6240") and `GEMM-PLAN.md:116`. I am slightly *below* that prior estimate because I am not counting occupancy/register gains the findings hand-waved.

**What's left to close 3000 → 6240** (out of scope for this pass, but the honest gap):
1. **Tile size + register accumulator pressure** — BN=128 with `facc[16][4]`=64 regs/thread caps occupancy. The findings' "occupancy via register reduction" (qmatvec topic, change 5) is real but speculative; a proper BM/BN sweep (e.g. BM=128, warp-specialized) is the next ×1.4–1.8.
2. **K-dimension multi-block per mma** — currently BK=32 = one mma K-step. Issuing 2-4 mmas back-to-back per smem fill (deeper inner unroll) raises mma:load ratio toward compute-bound. ×1.3–1.5.
3. **Weight decode off critical path** — Q4_K/Q6_K decode ALU (lines 81-125, 320-345) still runs inline. Pre-decoding to an int8 staging buffer (the findings' "fused decode + async load") removes it. ×1.1–1.3 for the non-Q8_0 dtypes.

Stacking those plausibly reaches ~5500–6500, i.e. *then* it's in striking distance of 6240. **Be honest: this single rewrite pass gets to ~half of llama.cpp, not parity.** The four changes here are the highest-ROI, lowest-risk first step; parity needs the tile/occupancy/decode-staging follow-on.

## 4. Validation gate (per change, gated before the next lands)

The gate already exists at `kernel_check.rs:360-447`: it runs `qmatvec_gemm_raw` vs the dp4a fast path for Q8_0/Q4_K/Q6_K/Q5_K/NVFP4 at T∈{16,64,128,512} with **rel < 1e-3** (lines 401, 426, 443). The argmax check (tokens 268/271/1178) is the end-to-end gate via `run_gen`. Per change:

- **After #2 ldmatrix (and #3 swizzle, landed together since #3 enables #2):** rerun `cargo run --bin kernel_check -- <gguf>`. Because ldmatrix.x4.b16 loads the *exact same 16 bytes* the scalar path assembled (§2 note), the s32 mma input is **bit-identical** → expect `rel` unchanged from current (≈1e-7 to ≤1e-3, NOT degraded). Any `rel` jump means the per-lane address (`(lane%16)*stride + (lane/16)*4`, `flash_attn.cu:91`) or the BKP stride is wrong. **Gate: all `GEMM ... rel < 1e-3 OK` lines hold, no new FAIL.**
- **After #1 cp.async:** async copy changes *timing only*, not values (same bytes land in smem). `rel` must be **identical bit-pattern** to pre-#1. A `rel` change ⇒ a missing `wait_group`/`commit_group` race (stage consumed before its cp.async completed). **Gate: rel unchanged + the round-trip lines (561-585) still OK.**
- **After #4 barrier/fold:** removing the second `__syncthreads` is the riskiest correctness change — verify no warp reads stage `cur` while another is still filling `nxt`. **Gate: rel unchanged AND run the full forward → argmax must produce 268/271/1178** (the authoritative end-to-end gate, per `BW24-BUILD-MAP.md:129`). If argmax drifts, the barrier removal raced.
- **Per change, also confirm pp512 monotonically rises** in `run_gen` (the commit `8d1c0b7` PREFILL timing). A change that doesn't move pp512 means it didn't bind the bottleneck it targeted — investigate before stacking the next.

## 5. Keeping the 5 dtypes + kernel2 (Q6_K/NVFP4) working

- **kernel1 (Q8_0/Q4_K/Q5_K, lines 161-280):** `issue_load_stage` wraps the existing `decode_block<QT>` call (line 198) unchanged — the template specializations (lines 149-151) are untouched, so all three dtypes decode identically into the staged `sW[s]`. Only the *destination* changes (`sW[r][k]` → `sW[s][r][k]`).
- **kernel2 (Q6_K/NVFP4, lines 406-525):** ports the SAME pipeline but with **two** weight ring buffers `sWlo[NSTAGE][BM][BKP]`, `sWhi[NSTAGE][BM][BKP]` (lines 419-420) and **two** ldmatrix loads + **two** mmas per n-tile (lines 501-502 stay, fed by `ld_A_s8(aflo,...)` and `ld_A_s8(afhi,...)`). The two-sub-scale fold (line 508) is unchanged. `decode_q6_k_2` (320-345) and `decode_nvfp4_2` (373-398) bodies are untouched — only retargeted to `sWlo[s]/sWhi[s]`. NVFP4's per-tensor macro-scale stays in `scale_inplace` (`lib.rs:564`), outside the kernel, so it is unaffected.
- **smem budget for kernel2:** 3-stage doubles weight buffers → `3*(2*64*36 + 128*36)` int8 ≈ 27KB + scales ≈ 5KB ≈ **32KB/CTA**, still ≥2 CTAs/SM. If occupancy drops too far, fall kernel2 back to **NSTAGE=2** (it's the less hot path; Q6_K/NVFP4 are attn_v/lm_head/MoE, not every matmul).
- **Launch config unchanged** (`lib.rs:554-559, 584-588`): `grid=(out_f/BM, ceil(T/BN),1)`, `block=(32,4,1)`, `shared_mem_bytes=0` works as long as static `__shared__` ≤ 48KB; the 32KB kernel2 worst case is under the static cap, so **no dynamic-smem opt-in needed** and lib.rs needs no change for the smem. If a later tile-size bump exceeds 48KB, that's when `cudaFuncSetAttribute`/`shared_mem_bytes` wiring gets added — not in this pass.
- **build.rs:** no change — same single `cu/qmatvec_gemm.cu` → `BW24_GEMM_FATBIN` (build.rs:11), `arch=compute_120a` (build.rs:17) already supports cp.async + ldmatrix.

**One-line summary:** four disjoint-bucket changes (cp.async pipeline, ldmatrix loads, swizzle to make ldmatrix conflict-free, single-barrier+fold) take 634 → ~2700–3200 pp512 (≈24–28% of peak, ~0.5× llama.cpp's 6240), gated bit-equivalent (rel<1e-3) + argmax 268/271/1178 per change; reaching 6240 needs a follow-on tile/occupancy/decode-staging pass this plan deliberately scopes out.