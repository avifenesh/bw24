My review is complete. All five dtypes verified byte-for-byte against ggml source (CPU dequant + GPU deq oracle + dp4a kernels numerically), the 512-entry IQ3_S grid is identical, and the full merged CUDA compiles clean to sm_120a with all 9 kernel symbols present. Here is my verdict.

---

# Adversarial review verdict: bw24 Q5_K/Q3_K/IQ4_XS/IQ3_S/NVFP4 dequant + MMVQ

**Method:** I did not trust the brief's "verified byte-for-byte" claim. I read the actual ggml source at `/home/avifenesh/projects/llama.cpp/ggml/src/{ggml-quants.c, ggml-common.h, ggml-impl.h}`, diffed the 512-entry IQ3_S grid programmatically, re-implemented every proposed CPU port, GPU deq oracle, and dp4a kernel in standalone C, and ran each against the ggml reference over 300-500 random superblocks. I then compiled the full merged `qmatvec.cu` to a real sm_120a fatbin with nvcc 13.1.

## Tables (the "assume one is wrong" check) — ALL CLEAN
- `IQ3S_GRID[512]`: `diff` vs ggml lines 1043-1106 = **byte-for-byte identical, 512/512**.
- `KVALUES_IQ4NL[16]`, `KVALUES_MXFP4[16]`: match ggml-common.h:1110-1118 exactly.
- `kmask_iq2xs = {1,2,4,8,16,32,64,128}` = `(1<<j)` — proposed sign logic matches.
- `ue4m3_to_f32` (and CUDA `ue4m3_to_f32_d`): identical to `ggml_ue4m3_to_fp32` incl. `*0.5`, NaN codes 0/0x7F→0.

## Numerical verification vs ggml (mismatch count / samples)
| dtype | CPU dequant port | GPU `deq_*` (Stage-A) | dp4a kernel |
|---|---|---|---|
| Q5_K | **0 / 128000** | **0 / 76800** | 0 mismatch (2 f64-rounding diffs, ~8e-8 rel — within 3e-2 gate) |
| Q3_K | **0 / 128000** | **0 / 51200** | **0 / 2400** (maxerr 3.4e-13) |
| IQ4_XS | **0 / 128000** | **0** | **0** |
| IQ3_S | **0 / 128000** | **0 / 128000** | n/a (Stage-A only, correct) |
| NVFP4 | **0 / 32000** | **0 / 64000** | **0 / 2000** (maxerr 3.6e-12) |

The Q3_K aux-scale "in-place aliasing" trap (ggml overwrites `aux[2]/aux[3]` then reads `aux[0]/aux[1]`) is correctly avoided by the proposed fresh-variable `n0..n3` rename — verified equivalent. The Q3_K `m_bit` running across both 128-halves without reset (8 total shifts) matches ggml. The Q3_K dp4a lo/hi-16 split to two scale indices (`is_lo`/`is_hi`) is correct.

## Compile (sm_120a, nvcc 13.1)
Full merged `qmatvec.cu` (grid constant + codebooks + 5 new `deq_*` + enum 3..7 + `deq()` switch + 4 dp4a kernels) compiles to a 111 KB fatbin with **zero errors and zero warnings** (`-Wall`). All 9 kernel symbols present incl. `qmatvec_{q5_K,q3_K,nvfp4,iq4_XS}_dp4a`. The `q4k_scale_min` reuse in `deq_q5_k` resolves (it's defined above), and the `iq3s_grid_d` forward-decl/late-define pattern is fine.

## Bugs found

**B1 — NVFP4 dp4a silently requires `in_f % 64 == 0` (MEDIUM, latent).** The kernel does `sblk = g >> 1` over 32-blocks, assuming each 64-element `block_nvfp4` = exactly 2 activation blocks. If a real NVFP4 tensor has `in_f` divisible by 32 but not 64, the last block reads a partial superblock → wrong logits, no crash. Q5_K/Q3_K/IQ4_XS are immune (256-aligned via `nsb>>3`). Fix: assert/guarantee `in_f % 64 == 0` for NVFP4 at load (`block_and_type_size` already says block=64, so GGUF guarantees it — but add a debug assert in the launcher to make it loud).

**B2 — dead/confusing first `db` in GPU `deq_iq3_s` (LOW, cosmetic).** The first `float db = d * (1.0f + 2.0f * ((e < 8) ? ... : 0));` is immediately overwritten by the clean `db = d*(1+2*sc_nib)` two lines later. It's harmless (verified 0/128000), but it's dead code that invites a future editor to "fix" the wrong line. Delete the first `db` assignment; keep only the `sc_nib`-based one.

**B3 — IQ3_S dispatch must NOT hit a fast arm (MEDIUM, must-confirm-on-wire).** There is no `qmatvec_iq3_s_dp4a`; IQ3_S must fall to the catch-all `qmatvec` Stage-A. The proposed dispatch ordering is correct as written, but if anyone later adds a `*qtype == QT_IQ3_S` fast guard without a kernel, `func()` will `panic!("kernel ... not in any fatbin")`. Add a code comment at the catch-all arm naming IQ3_S explicitly.

**B4 — test harness mismatch noted (NOT a code bug).** The brief's `q3k_hmask_offset` unit test comment claims `sc≈33` from `scales=0x21`; the actual unpacked scale depends on the full 6-bit aux dance, so the comment is misleading but the test only asserts `is_finite()`, so it passes regardless. Tighten or drop the misleading comment.

## GO / FIX-FIRST per dtype
- **Q5_K — GO.** dp4a + Stage-A both exact. Wire `qmatvec_q5_K_fast` + dispatch.
- **Q3_K — GO.** dp4a + Stage-A both exact (the trickiest aux/m_bit math is correct).
- **IQ4_XS — GO.** Stage-A exact; optional dp4a exact. `aqb`/`aqb+16` int* casts are 4-byte aligned (g*32 + 16). Keep dp4a behind `BW24_IQ_FAST` as proposed.
- **IQ3_S — GO** (Stage-A f32 only, as the brief recommends). Exact. Apply B2 cleanup.
- **NVFP4 — GO with B1 guard.** Both paths exact; add the `in_f % 64 == 0` assert in `qmatvec_nvfp4_fast` before shipping.

## Validation I recommend before merge (matches existing harness)
Extend `kernel_check.rs` exactly like its current Q4_K/Q6_K fast-vs-Stage-A loop (`rel < 3e-2`), adding cases for Q5_K, Q3_K, IQ4_XS, NVFP4 against real tensors (Q5_K: 9B-NVFP4 `blk.0.ffn_down.weight`; NVFP4: `blk.0.attn_q.weight`; Q3_K/IQ4_XS/IQ3_S: 35B-MoE expert tensors). For IQ3_S, gate Stage-A vs a CPU-oracle matvec (`dequantize(IQ3_S,...)` dotted with the host activation), `rel < 3e-2` — since no dp4a path exists. The CPU dequant unit tests (q5k/q3k/nvfp4) can be added as proposed; tighten the B4 comment.

## Files (absolute) and net assessment
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/dequant.rs` — 5 dequant fns + 3 tables + `ue4m3_to_f32` + 8 dispatch arms: **correct as written.**
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec.cu` — grid constant + codebooks + 5 `deq_*` + enum + 4 dp4a kernels: **correct; apply B2; compiles clean on sm_120a.**
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — 5 `QT_*` consts + 4 launchers + `qmatvec_dp4a_named` + dispatch: **correct; add B1 assert in NVFP4 launcher + B3 comment.**
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs` — import + 5 qtype map arms: **correct.**
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/lib.rs` — **no change needed** (block sizes already correct: Q5_K=176, Q3_K=110, IQ4_XS=136, IQ3_S=110, NVFP4=(64,36); confirmed against ggml static_asserts).

**Bottom line: the deliverable is correct.** No wrong table, no wrong layout, no wrong dp4a packing — all five dtypes match ggml byte-for-byte and the CUDA compiles clean. Only B1 (NVFP4 64-alignment guard) is a must-fix-before-wire safety assert; B2/B3 are cleanups. GO on all five.