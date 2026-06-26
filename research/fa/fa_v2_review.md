Confirmed: prefill needs 94.1 KB (96384 bytes), the static cap is 48 KB (49152), and the opt-in cap is 99 KB (101376). So `set_attribute(...MAX_DYNAMIC_SHARED_SIZE_BYTES...)` is mandatory and 94.1 KB fits under the 99 KB opt-in cap. The Rust launcher formula matches the kernel layout exactly. The harness already proves this works via `cudaFuncSetAttribute`.

I have everything I need. All claims independently verified on-box.

---

# FlashAttention v2 Re-Review — VERDICT: **GO**

Independently verified on-box (RTX 5090 Laptop, sm_120, CC 12.0; nvcc 13.1; not the summary's word — I recompiled and reran). The on-disk `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu` IS the fixed version (uses `__cvta_generic_to_shared`+`"r"`, no `ld_B`, 16x8-x4 K-loader split). The summary's header comment still references `/tmp/qkpv_test.cu` and says "VALIDATED in pv_test" cosmetically, but the code body is correct.

## On-box evidence (reproduced)
- `nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o /tmp/fa_validate research/fa/fa_validate.cu` → clean (only unused-var warnings).
- `/tmp/fa_validate` → **ALL 11 PASS** (7 prefill + 4 decode). Prefill maxabs 3.0e-4..3.4e-3 (< 1e-2 bf16 budget); decode maxabs ~1e-7 (f32-exact).
- `compute-sanitizer --tool memcheck` → **ERROR SUMMARY: 0 errors**.
- ptxas `-v`: `fa_prefill_f32` = **58 regs, 0 spill, 0 stack**; `fa_decode_f32` = 40 regs / 0 spill; `fa_decode_combine_f32` = 40 / 0.
- Prefill dyn smem = **96384 B (94.1 KB)**; device `sharedMemPerBlock`=49152 (static cap), `sharedMemPerBlockOptin`=101376 (99 KB). So 94.1 KB > 48 KB ⇒ H2 set_attribute mandatory; 94.1 KB < 99 KB ⇒ fits.

## The 8 checks + 6 v1 bugs

1. **ldmatrix per-lane address matches mma.cuh exactly?** YES. `flash_attn.cu:92,101` `(threadIdx.x%16)*stride_pairs + (threadIdx.x/16)*4` + `(uint32_t)__cvta_generic_to_shared` + `"r"` is byte-identical to the proven `mma_validate.cu:56,66`. **C1 → FIXED.**

2. **Register footprint < 255, launch_bounds 128 viable?** YES — measured **58 regs** prefill, 40 decode (no `__launch_bounds__` needed; far under 255). C2's win is real: O lives in `sO` (smem), not registers. **C2 → FIXED.**

3. **PV V-transpose uses correct trans loader + reorder?** YES. `ld_A_trans` (`:99-105`) uses `ldmatrix...x4.trans.b16` with the `{x0,x2,x1,x3}` output reorder, fed via `{Bt.x[0],Bt.x[2]}`/`{Bt.x[1],Bt.x[3]}` (`:274-275`) — identical to proven `pv_test`. **C3 → FIXED.**

4. **P→A uses smem round-trip (not false free-repack)?** YES. P is written to `sP` bf16 (`:239,243`), `__syncwarp()`, then re-`ld_A`'d for PV (`:272`). Genuine round-trip, no movmatrix. **C4 → FIXED.**

5. **Causal + GQA + deferred-normalize exact?** YES, and oracle-matched. GQA `kv_head=head/(n_head/n_head_kv)` (`:144`) == oracle `kernels.cu:106`. Causal `(k0+j) > q_pos`, `q_pos=q_pos0+r=(T_kv-T)+q_base+r` (`:169,222,227`) == oracle `t>q_pos`, `q_pos=(T_kv-T)+qt` (`kernels.cu:112,120`). Deferred normalize `O=sO/l_i` (`:296-297`). The ragged `T=7,T_kv=7` and `T_kv=100/200` padding cases pass, confirming `nq`/`nk` zero-padding and the alpha-broadcast scratch (`sS[r*BK+0]`, written for all 16 row-lanes, read for `r<nq`) are correct. **C-causal/GQA → FIXED.**

6. **Decode offset log2 domain?** YES. `exp2f(x*LOG2E)` throughout (`:234,238,388-389,420`); NO 2.079 bias anywhere (grep-confirmed). Online-softmax recurrence + log-sum-exp combine are self-normalizing; decode is f32-exact (~1e-7). **C6 → FIXED.**

7. **set_attribute called when smem>48KB?** YES — required (94.1 KB) and correct. Harness proves it via `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 96384)`; the Rust `fa_prefill` launcher's smem formula (96384) matches the kernel layout exactly and calls `f.set_attribute(...CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES...)`. **FIXED.**

8. **Launcher args match?** YES. `fa_prefill`/`fa_decode` arg order and types match the kernel signatures; grid/block (prefill `(ceil(T/16),n_head)`×32; decode split-K `(n_head,n_splits)`×256 + combine `(n_head)`×head_dim) match the validated harness launches.

**All 6 v1 bugs: C1 FIXED, C2 FIXED, C3 FIXED, C4 FIXED, C5 FIXED (16x8-x4 K-loader + `{B0,B2}/{B1,B3}` split, no fragile `ld_B`), C6 FIXED.** No residual layout bug detected — the oracle match across square/ragged-q/ragged-kv/multi-tile/long-context shapes is the proof.

## Integration steps (build.rs + lib.rs are ALREADY wired — verify, don't redo)
Already present and confirmed on-disk:
- `build.rs:10` — `("cu/flash_attn.cu","BW24_FLASH_FATBIN")` in the gencode loop.
- `lib.rs:20` `FLASH_FATBIN_PATH`; `:33` `flash: Arc<CudaModule>`; `:42` loaded in `new`; `:52` chained in `func()`.

Remaining to integrate (NOT yet on-disk — `grep` shows only `sdpa_naive`/`sdpa_naive_view` in lib.rs, no `fa_prefill`/`fa_decode`):
1. Add `use cudarc::driver::sys::CUfunction_attribute_enum;` and the two methods `fa_prefill(...)` / `fa_decode(...)` from the deliverable into `impl Engine` in `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs`.
2. Swap call sites: `sdpa_naive` → `fa_prefill` (prefill); `sdpa_naive_view` → `fa_decode` (decode hot path; `fa_decode`'s `arg()` accepts `&CudaView` for resident-KV K/V — keep `sdpa_naive` as fallback oracle).
3. `cargo build -p bw24-engine` (build.rs recompiles the fatbin) and run the engine's own attention path to confirm end-to-end.

## Validation gate (run after every change to flash_attn.cu — this is the only correctness proof)
```
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o /tmp/fa_validate \
  /home/avifenesh/projects/bw24/research/fa/fa_validate.cu && /tmp/fa_validate
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/fa_validate
```
Gate = "ALL PASS" + "0 errors". Because the lane maps are layout-fragile, ANY edit to the smem layout, strides, the `{B0,B2}/{B1,B3}` split, or the C-tile write-back must re-pass this gate before integration — passing oracle diff is the sole proof, not code inspection.

Files: kernel `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu`; primitives proof `/home/avifenesh/projects/bw24/research/fa/mma_validate.cu`; e2e harness `/home/avifenesh/projects/bw24/research/fa/fa_validate.cu`; integration target `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs`.

Minor non-blocking nit: the file header (`:3`, `:19`) still cites `/tmp/qkpv_test.cu` and "VALIDATED in pv_test" as if verbatim — the actual proof file is `research/fa/mma_validate.cu`. Cosmetic comment drift only; code is correct. Worth a one-line comment fix on the next edit.