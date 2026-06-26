I found another bug. Now I have all findings verified against the actual code and authoritative references. Let me compile the complete adversarial review.

# Adversarial Review: bw24 FlashAttention sm_120 — VERDICT: FIX-FIRST (multiple correctness-fatal bugs)

I verified every claim against the on-box ground truth: the oracle `sdpa_naive_f32` (`cu/kernels.cu:99`), the launcher style (`src/lib.rs:234`), the decode call site (`decode.rs:127`), `build.rs`, and the authoritative PTX MMA/ldmatrix layout in `/home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/mma.cuh`. The design is algorithmically literate and the deferred-feature triage (no wgmma/tcgen05/TMEM/setmaxnreg) is correct for sm_120. But it will **not** match the oracle as written. Bugs below, by severity.

---

## CRITICAL (correctness-fatal or won't-compile-correctly)

### C1. `ldmatrix` addressing is fundamentally wrong — every lane must supply its own row address
The design calls `ldmatrix_x4(Q_rmem[dc][ks], &Qs[swz(qrow_base*HEAD_DIM + d_base)])` — a single per-**warp** base pointer. But `ldmatrix.sync.aligned.m8n8.x4` consumes **32 lane-private addresses**, one per row of the matrices being loaded. The authoritative llama.cpp loader (`mma.cuh:834`) proves it:
```
const int * xs = xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J/2);
```
Each lane adds `(lane%16)*stride + (lane/16)*(J/2)`. The design's helpers (`ldmatrix_x4`/`x2`/`x2_trans`) take a `const void* smem` and apply `__cvta_generic_to_shared` to *the same address on every lane*. That loads garbage (all lanes read row 0's neighborhood). **Every** ldmatrix call site (Q load, K load, V-trans load) is affected. This alone guarantees wrong output. **Fix:** fold the per-lane `(lane%16)*stride + (lane/16)*8` offset into the pointer inside each loader (matching mma.cuh), and pass the tile base + warp/n-block offset separately.

### C2. Register pressure is non-viable — `O_acc` alone is 128 f32/thread
`float O_acc[HEAD_DIM/8][4]` = 32×4 = **128 f32 registers per thread**, held live across the entire KV loop. Add `Q_rmem` (64×u32, also live for the whole loop), plus `S_frag` (16), `P_rmem` (8), `m_i/l_i/new_m/corr/psum` and ldmatrix temporaries. That is **>200 registers/thread before spills**, against the 255 hard cap. With `__launch_bounds__(128)` the compiler must fit 128 threads → it will spill `O_acc`/`Q_rmem` to local memory catastrophically, or fail occupancy entirely. The "Q stays in registers for the whole KV loop" decision (64 regs) compounds this. This is the dominant viability problem and is **not** flagged in the design's own risk list. **Fix:** reduce `BLOCK_Q` to 16 or 32 (one m16 row-group/warp is already the case, so this mostly cuts O_acc only if you also shrink the per-thread output footprint), or keep `O_acc` for one D-chunk at a time and re-stream — but that conflicts with online softmax needing all of O resident. Realistically: process `HEAD_DIM` output in 2 passes is impossible with single-pass softmax; the honest fix is smaller `BLOCK_Q` won't help O_acc (it's per-thread, head_dim-driven). You must accept ~128 O regs and drop `Q_rmem` from registers (re-`ldmatrix` Q each KV tile from smem). Measure actual `nvcc --ptxas-options=-v` register count before claiming GO.

### C3. PV GEMM A/B operand roles are inverted / V-transpose layout is wrong
For `mma.sync.m16n8k16.row.col`, operand A is m16×k16 **row-major**, operand B is n8×k16 (i.e. the k×n matrix read **column-major**). In PV: P is `[m=16 q-rows, k=BLOCK_KV]` (A, fine), V must be supplied as `[n=D_CHUNK, k=BLOCK_KV]` — V^T. The design loads V with `ldmatrix.x2.trans` from `Vs[swz((kf*16)*D_CHUNK + nb8*8)]`, but:
- `ldmatrix.x2.trans` returns a **16×8 transposed** fragment, and the llama.cpp trans loader (`mma.cuh:892`) reorders the output registers (`xi[0],xi[2],xi[1],xi[3]`) — the design's `ldmatrix_x2_trans` writes `{r[0],r[1]}` in naive order, so even the register interleave is wrong.
- The B operand of m16n8k16 needs **k=16** contraction (2×u32 = 4 bf16 per lane). The design loads only `x2` (2 regs) per `kf` and iterates `kf` over `BLOCK_KV/16=2` — that's k=16 per step, plausible, but the addressing `(kf*16)*D_CHUNK + nb8*8` indexes V as `[key][d]` row-major, then asks `.trans` to flip it. Whether that yields B = V^T in the exact lane layout the accumulator expects is **unverified and almost certainly off** given C1. This is the author's self-flagged "single highest-risk detail" and it is not derived correctly here. **Fix:** mirror `load_ldmatrix_trans` from mma.cuh exactly (register reorder + per-lane address), and validate PV in isolation against a CPU `P@V` before integrating.

### C4. P-repack into the A-operand is incorrectly indexed
```
int kf = nb >> 1, half = nb & 1;
P_rmem[kf][half*2 + 0] = pack_bf16x2(p00, p01);  // row0 pair
P_rmem[kf][half*2 + 1] = pack_bf16x2(p10, p11);  // row1 pair
```
The claim "m16n8 accumulator layout == m16k16 A-operand left half" is **false as written**. The C/D `.f32` accumulator for m16n8k16 has each lane holding `{r0c0, r0c1, r1c0, r1c1}` where `(r0,r1)=(lane/4, lane/4+8)`, `(c0,c1)=2*(lane%4), +1` (confirmed via mma.cuh `get_i/get_j` and standard PTX). The bf16 **A-operand** for m16n8k16 has a *different* lane→element map (4 u32 = 8 bf16 per lane spanning k0..15 for rows `lane/4`, `lane/4+8`). The S→P values a lane owns (`S_frag[nb]` for 4 column-blocks nb=0..3, i.e. cols 0,1,8,9,16,17,24,25 for this lane) do **not** line up with the 8 contiguous-k bf16 the A-operand expects. The "free repack" assumes S's column index == A's k index; it does not, because S's columns are *keys* (n of QK^T) and A's k is *also* keys (k of PV) — these are the same axis, but the *physical lane layout* of QK^T's n-output differs from PV's A k-input. A cross-lane shuffle (or smem round-trip) is required to repack P. **This is silent corruption.** **Fix:** verify the layout identity numerically; if it fails (it will), stage P to smem and `ldmatrix` it back, like every production FA kernel does for the S→P→PV transition.

### C5. K-operand for QK^T uses `ldmatrix.x2` but m16n8k16 B-operand needs the col layout
GEMM0 does `mma_m16n8k16(S_frag[nb], Q_rmem[dc][ks], K_rmem)` with `K_rmem` from `ldmatrix_x2`. For QK^T = Q @ K^T with `.row.col`, K^T means K is the B operand read column-major (B = `[n=keys, k=d]`). K in smem is `[key][d]` row-major — that **is** the natural `.col` layout for B (n-major), good. But `x2` gives k=16 only if the 2 regs are the right 4 bf16 spanning d0..15 for the 8 keys this lane's n-block needs. Combined with C1 (no per-lane address) this is wrong. Lower severity than C1–C4 only because it's subsumed by them, but must be re-derived.

### C6. Decode `KQ_MAX_OFFSET` is a unit mismatch (ln vs log2)
`KQ_MAX_OFFSET = 3*ln2 = 2.079` is a **natural-log** quantity, but it is added to `s_l2` which is in the **log2 domain** (`s * scale * log2e`): `m_new = max(m_run, s_l2 + OFFSET)`. In log2 units, "8× headroom" is `log2(8) = 3.0`, not 2.079. As written it applies `2^2.079 ≈ 4.2×` headroom, not 8×, and more importantly it's a domain error copied from a natural-exp kernel. It won't break correctness (the offset cancels in the final normalize as long as it's applied consistently to *all* keys — which it is, since every key adds the same constant to `m_new` candidate... actually it only enters via the running max, and `p = exp2(s_l2 - m_new)` uses the offset-inflated `m_new`, so it uniformly shrinks all p by `2^OFFSET` and cancels in `p/l`). So it's **not** fatal, but it is wrong-intent and the magnitude is off. **Fix:** use `log2(8)=3.0f` or drop it (f32 accum doesn't need the underflow guard the f16 VKQ path did).

---

## HIGH

### H1. cp.async / 2-stage pipeline is described but the f32 path doesn't use it, and the K double-buffer smem is mislabeled
The smem layout allocates `2 * BLOCK_KV * D_CHUNK` for "2 K pipeline buffers," but the f32 KV-loop calls `stage_K(kv0, dc, Ks)` writing to the **single** base `Ks` with a `__syncthreads()` before and after — no double-buffering, no `cp.async`, no `commit/wait`. So the 2× K smem is **allocated but unused** on the only (f32) path that runs. Not a correctness bug, but the "2-stage pipeline included" claim is false for Stage-1, and the wasted 8 KB pushes you toward the 99 KB attribute call for no benefit. **Fix:** drop the 2nd K buffer on the f32 path (saves 8 KB, may keep you ≤48 KB and avoid the `set_attribute` entirely).

### H2. The `set_attribute` for >48 KB dynamic smem is commented out — kernel will fail to launch
57.3 KB (verified 56.0 KB) dynamic smem exceeds the 48 KB default opt-in ceiling. The Rust launcher has `// f.set_attribute(...MaxDynamicSharedMemorySize...)` **commented out**. As written, `b.launch(cfg)` with `shared_mem_bytes=57344` returns `CUDA_ERROR_INVALID_VALUE` at launch. **Fix:** uncomment and actually call it (and confirm cudarc 0.19.8 exposes `CudaFunctionAttribute::MaxDynamicSharedMemorySize` on `CudaFunction` — verify the exact API; it may be `func.set_attribute` or via the module).

### H3. `__syncthreads()` inside `for (gq...)` and inside the `dc` loop creates a re-stage race / perf cliff, and Q smem is overwritten while still needed
- Q is loaded to `Qs` once per `gq`, then `ldmatrix`'d into `Q_rmem`. But within the KV loop, `stage_K`/`stage_V` write to `Ks`/`Vs` which are *separate* buffers — OK. However the epilogue `__syncthreads()` "before reusing Qs for the next gqa head" is correct intent, but `Q_rmem` is in registers, so re-staging Qs for `gq+1` while `gq`'s PV is still reading `Vs` is fine. The real issue: the `dc`-loop `__syncthreads(); stage_K(...); __syncthreads();` **inside** the per-`nb`/`ks` mma means K for chunk `dc=1` overwrites `Ks` that `dc=0`'s mma already consumed — that's fine, but there's **no guard that all warps finished `dc=0`'s ldmatrix before `dc=1`'s stage_K overwrites `Ks`**. The leading `__syncthreads()` before `stage_K` provides it. Acceptable, but the double `__syncthreads()` per chunk per KV tile × N_DCHUNK × 2 (K and V) = 8 barriers per KV tile is a severe perf cost and partly defeats the FA purpose. **Fix:** restructure so K and V for a tile are staged once.

### H4. `swz` swizzle template is likely a no-op or wrong for these strides
```
template<int STRIDE_ELEMS> int swz(int index){
  int row=(index/STRIDE_ELEMS)&7;
  int bits=row/(64/STRIDE_ELEMS>0?(64/STRIDE_ELEMS):1);
  return index ^ (bits<<3);
}
```
For `STRIDE_ELEMS=HEAD_DIM=256`: `64/256=0 → guard makes divisor 1 → bits=row` (0..7), `index ^ (row<<3)`. For `STRIDE_ELEMS=D_CHUNK=128`: `64/128=0 → divisor 1 → bits=row`. This XORs the row index (0..7) into bits 3–5 of the column. That is a *plausible* conflict-reducing pattern but it is **not** the standard 16-byte-granule swizzle that matches `ldmatrix`'s access pattern, and crucially the **store** side (`Qs[swz(...)]=...`) and the **load** side (`ldmatrix(&Qs[swz(...)])`) must apply the *identical* permutation — but `ldmatrix` reads 8 contiguous elements per row internally and the per-lane address is `base + (lane%16)*stride`, so swizzling only the base while ldmatrix adds an un-swizzled `(lane%16)*stride` **breaks the 1:1 store/load correspondence**. Swizzle + ldmatrix must be co-designed; as written they're inconsistent. **Fix:** either drop swizzle for bring-up (correctness first; eat the bank conflicts) or implement the exact XOR scheme ldmatrix expects with the per-lane offset inside the swizzle.

### H5. Decode reduction smem is malformed
```
__shared__ float m_sh[DEC_WARPS], l_sh[DEC_WARPS], v_sh[DEC_WARPS][HEAD_DIM];
if (lane < HEAD_DIM / WARP_SIZE * WARP_SIZE) { /* layout */ }   // empty stub
```
`v_sh[4][256]` f32 = 4 KB static smem — fine. But the stub `if(...){ }` is dead, and the reduction writes `v_sh[warp][lane + i*32]` for `i<8` covering all 256 dims per warp — OK. The combine then reads `v_sh[w][lane + i*32]` in warp 0 — but warp 0's lanes only cover lane∈[0,32), `i∈[0,8)` → dims `{lane, lane+32, ... lane+224}`, all 256 covered across 32 lanes. **Correct**, but fragile and the dead stub signals untested code. Lower-confidence HIGH; verify the partial-numerator semantics in split-K (H6).

### H6. Split-K combine double-applies the softmax max / numerator semantics inconsistent
In the `split_k>1` branch, each partition writes `out[i]` = `Σ_w exp2(m_sh[w]-M_local) * v_sh[...]` — i.e. numerator **rescaled to the partition-local max `M_local`**, and stores `M_local`, `L_local` to meta. The combine kernel then does `a=exp2(mp-M_global)`, `num += a*O_partial`, `den += a*lp`. For this to be exact, `O_partial` must be the numerator **at the partition's own max** and `lp` the denom **at that same max** — then `a` rescales both to global max. That algebra is correct **only if** `out[i]` is exactly `Σ p_k V_k` with `p_k=exp2(s-M_local)`. But the within-block reduction already combined 4 warps with `a_w=exp2(m_sh[w]-M_local)` — so `out` is at `M_local`, and `L = Σ a_w l_sh[w]` is also at `M_local`. Consistent. **However**, the single-block (`split_k==1`) path normalizes by `L` inline, while the partial path writes un-normalized `out` — the two paths must agree, and the test grid must check `split_k∈{1,4}` agree (the design says so). Medium-confidence: the math is recoverable but the "numerator based against M" comment is ambiguous and the meta stores `{M,L}` while combine reads `[*2+0]=M, +1=L` — matches. Keep but **test split_k=1 vs 4 vs oracle explicitly** (the design's plan does include this — good).

---

## MEDIUM

### M1. Decode is non-causal but reads `split_k` with no causal arg — relies on "decode query attends all keys"
Correct for the real call (`decode.rs:127` passes `t=1`, oracle computes `q_pos=T_kv-1` → attends all). But `fa_decode_f32` drops the `causal` parameter entirely vs the oracle's signature. Fine for decode-only use, but the launcher `fa_decode_view` signature **drops `causal`** that `sdpa_naive_view` has (`decode.rs:127` passes `true`). Call-site swap must drop the arg — flagged so it compiles.

### M2. Launcher arg-count audit
`fa_prefill` Rust builder passes 11 args (q,k,v,o,hd,nh,nhkv,ti,tkvi,scale,cz) — matches the kernel's 11-param signature. ✓
`fa_decode_view` split_k==1 passes a `dummy` for `O_meta` (12 args incl. sk) — matches kernel's 12 params. ✓ But allocating `alloc_zeros::<f32>(1)` **every decode step** (in the hot loop) adds an allocation per token. **Fix:** cache the dummy, or use split_k path's null handling.

### M3. `fa_exp2` on masked `-1e30f` scores
After mask, `S_frag=-1e30f`; then `S_frag *= scale_log2` was applied **before** masking? No — order is: multiply by `scale_log2` first, **then** set `=NEG_INF`. Good (mask overwrites). Then `p = exp2(NEG_INF - new_m)` = `exp2(-1e30)` → 0. ✓. But if an entire row is masked (possible when `T_kv<T` for early query rows... actually `T_kv>=T` always here), `l_i=0 → inv=0 → O=0`. Oracle would `1/sum` with sum from `exp(0-mx)` of at least the diagonal, so never all-masked for valid rows. Guarded. ✓ Low risk, noted.

### M4. `RESCALE_TAU=8` conditional rescale is exact **only** with the deferred final normalize — verify the `corr` applies to BOTH l and O
When `need==false`, `new_m=m_i` (old max kept), `corr=1`, and `p=exp2(S - m_i)` may be **>1** (since the true max could exceed `m_i` by up to `tau=8` → `p` up to `2^8=256`). That's intended (deferred), and `l_i += psum`, `O += P@V` with the same un-rescaled P → consistent, final `O/l` exact. **The math checks out** *provided* `corr` multiplies `l_i` and **every** `O_acc` lane (it does: the `for i<HEAD_DIM/8` loop). One subtlety: `__any_sync` makes `need` warp-uniform, but the per-row `new_m`/`corr` are computed per-`rr` from `rmax[rr]` which is already warp-reduced (uniform across the 4 lanes sharing a row) — consistent. ✓ This part is correct. Good design.

### M5. `build.rs` integration
Adding `("cu/flash_attn.cu","BW24_FLASH_FATBIN")` to the array works, but `lib.rs` must (a) add `const FLASH_FATBIN_PATH`, (b) load a 4th module, (c) add it to the `func()` `.or_else` chain. The design mentions this in prose but the lib.rs snippet doesn't show the module load. Don't fold into `kernels.cu` — it's `#include <cuda_bf16.h>` heavy and would slow the oracle TU. Keep separate. Verify `nvcc -arch=compute_120a` accepts `cp.async.bulk`/`ldmatrix` (design says verified on-box — trust but the build must actually compile this TU; it currently does not exist).

---

## LOW / NITS

- **L1.** `pack_bf16x2` uses `__floats2bfloat162_rn` then `*reinterpret_cast<uint32_t*>` of a local — works but UB-adjacent; use `__nv_bfloat162` → `__builtin_bit_cast` or `union`.
- **L2.** `fa_exp2_poly` and `cp_async_*` helpers are dead code on the f32 path — fine (documented), but will warn `-Wunused`.
- **L3.** `NEG_INF=-1e30f` then `*scale_log2 (=0.0902)` would give `-9e28` *if applied before mask* — but mask overwrites, so moot. Still, ordering is load-bearing; comment it.
- **L4.** Tolerance gate `1e-3` for bf16-input MMA vs f32 oracle is reasonable, but with the `2^8` deferred-rescale dynamic range, bf16 mantissa (8 bits) on P values up to 256 loses precision — `1e-3` may be optimistic at `T_kv=4096`. Validate empirically; be ready to lower `RESCALE_TAU` for accuracy.

---

## VERDICT: **FIX-FIRST**

Do **not** compile-and-ship this as a correctness replacement. Blocking issues: **C1** (ldmatrix per-lane addressing — wrong on every load), **C3/C4** (PV operand roles + P-repack — silent corruption, the author's own flagged top risk, and it is in fact wrong), **C2** (128+ register O_acc makes the kernel spill or fail occupancy at `__launch_bounds__(128)`), **H2** (commented-out smem opt-in → launch failure). C6/H1/H4 are wrong-but-recoverable.

The **algorithmic** core is sound and correctly scoped for sm_120: online-softmax recurrence (M4) is exact, causal mask matches the oracle (`q_pos=(T_kv-T)+qt`, `t>q_pos`) node-for-node, GQA `kv_head=head/(n_head/n_head_kv)` matches, conditional rescale is provably exact, and the deferred-HW list (no wgmma/tcgen05/TMEM/setmaxnreg) is correct. The decode kernel is closer to viable than prefill (its main bug C6 is non-fatal).

**Minimum path to GO:** (1) rewrite the three ldmatrix helpers to inject per-lane `(lane%16)*stride + (lane/16)*(J/2)` addressing copied verbatim from `mma.cuh:834/892`; (2) stage P→smem→ldmatrix for the PV A-operand instead of the bogus free-repack, OR numerically prove the layout identity first; (3) mirror `load_ldmatrix_trans`'s register reorder for V; (4) `nvcc --ptxas-options=-v` to get the real register count and shrink footprint until occupancy ≥1 block/SM; (5) uncomment the `set_attribute` smem call; (6) bring up GEMM0 and GEMM1 **in isolation** against CPU `Q@K^T` and `P@V` before wiring the full kernel; only then run the full diff-vs-oracle grid. The validation plan itself (Step 1 grid + targeted traps + llama.cpp cross-check) is solid and should stay — it's exactly what will catch C3/C4.

Files that must change beyond the design's list: `src/lib.rs` (add `FLASH_FATBIN_PATH` const + 4th module load + `func()` chain entry — the design's prose says this but the snippet omits it).