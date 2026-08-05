# FP8 DECODE-v1 — verdict: **native e4m3 residency becomes the default for per-tensor-scale FP8-ST**

Lane: `lane/fp8-decode-v1` (off train `7ac05f54`) · 2026-08-05 · RTX 5090 Laptop (sm_120a, 24463 MiB)

Charter: make per-tensor-scale FP8-ST checkpoints decode **natively on e4m3 weights** instead of
paying the ARM B′ Q8_0-slab re-encode tax. The estimate this lane was funded on is
`research/fp8v3-gate-20260805/VERDICT.md` Q2: native e4m3 GEMV **+6.00pp weighted** on the 27B m=1
shape sheet, against a **+6.25pp** byte ceiling (34 B per 32 weights vs 1 B per weight), i.e. 96% of
theory — with the explicit finding that *"no kernel authoring is required… the remaining work is
dispatch/residency."*

Every raw run is in this directory. `RESULTS.jsonl` carries one row per measured fact.

---

## 0. Salvage report

The worktree was resumed, not recreated. `git status --porcelain` came back **empty**: the prior
agent had committed everything into `e65ef35b` (*"e4m3 launch-fusion twins"*). There was no dirty
`qmatvec.cu` / `kernel_check.rs` / `lib.rs` work to salvage or discard — the whole of that slice was
already in the tree and it builds. Nothing was thrown away at resume.

What `e65ef35b` claims, and what this lane verified before measuring it: native e4m3 residency used
to **un-fuse** the m=1 decode trunk, because the fusion doors (`matmul_q8_fused2/_x/_t`,
`matmul_q8_fused3/_t`, `matmul_pre_dual_noscale`) were dtype-locked to Q8_0. Read at
`lib.rs:5927` (`matmul_pre_dual_noscale` now carries an F8-E4M3 arm before the NVFP4 gate, folding
the per-tensor scales into the SwiGLU epilogue), `lib.rs:6706` (`e4m3_fused_params` rejects
non-`QT_F8_E4M3`, `row_bytes != in_f`, and split-plane mirrors), and `lib.rs:5546`
(`uses_q8_1_fast` admits `QT_F8_E4M3`, so the doors can actually open). Confirmed by construction,
then confirmed by measurement in §3.

---

## 1. The instruments this lane could not conclude without

Two were missing, and both were added before any number was taken (`26ece2f2`).

**Residency census** (`model.rs`, `residency_census_report()`). This arm is a *residency* change, so
the primary question is what each 2D matmul weight actually **became**. The pre-existing evidence for
the arm was a load-probe pair (38.89 vs 38.17 tok/s, identical text) that is logically incapable of
separating *"the arm ran and was flat"* from *"the arm never engaged on this checkpoint"*. A tally at
the single `load_from_source` seam answers it directly and costs nothing when unread.

**Teacher-forced exactness** (`run_gen.rs`). The `MEMRA_FORCE_TOKENS_FILE` seam already existed but
computed `greedy` and then **threw it away** — it forced the tape and measured nothing. It now counts
argmax disagreements, records the first disagreeing position, and accumulates the tape's NLL. This is
the correct question for this change: e4m3 in-kernel dequant and the Q8_0 re-encode are *different
arithmetic*, so bit-identity between the containers is the wrong bar (see §4).

---

## 2. The checkpoint: the 27B on this rig IS this arm's class

`/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4` (modelopt 0.45.0, `MIXED_PRECISION`):

| | count | scale class |
|---|---|---|
| `F8_E4M3` | **208** | **scalar `weight_scale` on all 208** — per-tensor, exactly this arm's class |
| NVFP4 | 193 | block scales + `weight_scale_2` (MLP gate/up/down) |
| block-128 F8 | **0** | — |

The 208 cover the entire attention + linear-attn stack (`linear_attn.in_proj_qkv/in_proj_z/out_proj`
×48 each, `self_attn.q/k/v/o_proj` ×16 each). So the OWNER RULE — *verdicts anchor on 27B shapes
only* — is satisfied natively: every number below is a 27B number on the deployment-target rig.

**A documented lane assumption is REFUTED.** `fp8v3-gate/VERDICT.md` states that 27B FP8 e2e *"does
not fit the 24 GB card"* and names the 96 GB pod as the only honest e2e ground. That is true of
`MEMRA_PP_FP8`, whose stash keeps e4m3 **on top of** the Q8_0 slab (dual residency — the arm that
needs a budget and OOM'd at 8.7% coverage). It is **not** true of native residency, which **replaces**
the slab. Measured 2D-weight totals: **16.79 GiB slab → 16.37 GiB native**, both inside 24 GB. The
27B decode claim below is therefore made on the target rig, not extrapolated from the pod.

The 96 GB pod remains the right proving ground for **27B FP8 prefill at long context and for
multi-model / larger-batch serving**, where the 24 GB card genuinely runs out — not for this
single-stream decode + pp512 claim.

---

## 3. Decode: **+2.58pp**, and the mechanism split

Interleaved A,B pairs in **one** `flock` hold, N=5, one frozen binary
(`0448713165a45528515ce89824337837`), pp512 prompt, `MEMRA_NGEN=128`, thermal 59→74 °C /
1687–1807 MHz after the r1 ramp.

| arm | tok/s (5 runs) | median |
|---|---|---|
| A — Q8_0 slab | 38.53, 38.60, 38.76, 38.76, 38.75 | **38.75** |
| B — native e4m3 + fused trunk | 39.75, 39.74, 39.87, 39.99, 39.71 | **39.75** |

**+2.58pp (1.02581x)**, per-pair +2.48…+3.17pp, and the distributions **do not overlap**:
min(B) = 39.71 > max(A) = 38.76. Both arms greedy-stable ×5.

Mechanism split via the rollback seam, arm C = `MEMRA_ST_E4M3=1 MEMRA_E4M3_DUAL=0` (native bytes,
launch fusion off): **39.32 tok/s**.

* bytes (native residency) → **+1.47pp**
* launch fusion (`e65ef35b`) → **+1.09pp**
* they **compose** to +2.58pp

That composition is precisely what `e65ef35b` was written to guarantee, and it is the reason the
lane order mattered: without the fused twins, native residency would have shipped its bytes win with
a fusion loss subtracted from it. Arm C's greedy output is identical to B's, so the seam is a perf
seam, never a numeric one.

**Why +2.58pp e2e and not the GEMV sheet's +6.00pp:** the sheet measures the m=1 GEMV in isolation
on the F8 projections. In the real 27B forward those 208 tensors are 6.88 of 16.37 GiB of resident
2D weight — the 193 NVFP4 MLP tensors (9.86 GiB) are untouched by this arm, as is all attention/KV
work. An Amdahl share of ~42% of weight bytes against a +6pp kernel-level win lands in exactly this
range. The sheet is not contradicted; it is being correctly diluted.

---

## 4. Prefill: **+3.25pp** — the half that would otherwise have shipped unmeasured

The flip changes **prefill** dispatch too: `try_fp8_gemm` takes `QT_F8_E4M3` *unconditionally* at
m ≥ `GEMM_M_THRESHOLD` (16) on the resident bytes. Defaulting on decode evidence alone would have
left half the dispatch change unmeasured, so prefill was measured on the same prompt with the ST
arm's existing `MEMRA_PP_ONLY` timer, N=3 interleaved in one hold, one binary
(`69d88f319d8abac91153958b39e7145f`).

| arm | per-run medians (pp512 tok/s) | median |
|---|---|---|
| A — `MEMRA_ST_E4M3=0` | 1468.1, 1358.7, 1466.8 | **1466.8** |
| B — `MEMRA_ST_E4M3=1` | 1514.5, 1516.8, 1514.5 | **1514.5** |

**+3.25pp**, non-overlapping (min B 1514.5 > max A 1468.1). One pair reads +11.64pp; that is an
**A-arm outlier, not a B-arm gain** — B r2 is within 0.15% of B r1/r3 while A r2 sits 7.4% below A
r1/r3 — so the headline is the median of per-arm medians, and the honest range excluding that dip is
+3.16…+3.25pp. At rep 0 (least thermally loaded pass) the same comparison is 1508.3 vs 1592.1 =
+5.56pp: same sign, larger margin at high clocks.

### A second, independent coverage witness

The `fp8_mmq_ledger()` **hook-entry** counter increments *before* any env gate, so it counts every
prefill GEMM that reached the FP8 hook:

* arm A: 1984 entries / 4 passes = **496 per pass**
* arm B: 1152 / 4 = **288 per pass**
* difference = **exactly 208** = the census's `F8_E4M3` tensor count

Under A those 208 projections surface as Q8_0 and still walk through the hook on their way to the
Q8_0 floor; under B they are `QT_F8_E4M3` and `try_fp8_gemm` claims them before the hook is reached.
A ledger delta equal to the census count is an arithmetic confirmation, by a completely different
mechanism than the census, that prefill dispatch moved for **exactly** the intended 208 tensors —
neither more nor fewer.

`fp8-mmq dispatches: 0` appears in every log in this lane and is **not** a refusal: `gate_off ==
entries` in all arms, i.e. `MEMRA_FP8_MMQ` is simply default-off, and that ledger counts the
*block-128* MMQ path, of which this checkpoint has zero tensors.

---

## 5. VRAM: measured, not assumed

| | tensors | MiB |
|---|---|---|
| the 208 F8 tensors as a Q8_0 slab | 208 | 7310.000 |
| the same 208 as native e4m3 | 208 | 6880.000 |
| **saved** | | **430.000 (−5.88%)** |

Measured byte ratio **1.06250**; theoretical 34/32 = **1.06250**. Matching theory to five decimal
places is itself the proof of the property that mattered: there is **one** resident copy and **no**
duplicate. Full 2D-weight census:

| arm | Q8_0 | NVFP4 | F8_E4M3 | total |
|---|---|---|---|---|
| A slab | 304 t / 7333.906 MiB | 193 t / 9862.031 MiB | — | 497 t / 17195.938 MiB |
| B native | 96 t / 23.906 MiB | 193 t / 9862.031 MiB | 208 t / 6880.000 MiB | 497 t / 16765.938 MiB |

Same 497 tensors both ways — nothing was dropped or double-counted. This is the *"dual residency vs
per-layer dequant"* question answered by measurement: native residency is a **one-copy** design and
comes out 430 MiB **smaller** than the slab it replaces, which is why it needs no VRAM budget at all
while `MEMRA_PP_FP8` does.

---

## 6. Exactness — branch (b), because the arithmetic differs

e4m3 in-kernel dequant is not the same arithmetic as a Q8_0 re-encode, so *bit-identity between the
containers is the wrong question.* Protocol (v2's, reused): take the **slab arm's own greedy tape**
(128 ids), teacher-force **both** arms on it so inputs are identical at every position, then count
argmax disagreements and score the tape's NLL under each arm.

| arm | disagreements | first at | mean NLL | total NLL |
|---|---|---|---|---|
| **control** — slab on its own tape | **0 / 128** | — | 0.222145 | 28.4346 |
| native e4m3 on the slab's tape | **2 / 128 (1.56%)** | 23 | **0.220117** | 28.1750 |

The control is what validates the instrument: a nonzero control would have meant the *harness*, not
the arm, was the source of any difference. Result: 2 near-tie flips, and native e4m3's NLL on the
reference's **own** token sequence is **lower** (−0.91%) — the native arm models the slab arm's
output slightly *better* than the slab arm does. That is the expected direction, since e4m3 is the
checkpoint's own precision and the Q8_0 re-encode is the lossy extra hop. **No quality regression.**

---

## 7. Gates

| gate | arm | result |
|---|---|---|
| `kernel-check` | instruments in tree (pre-flip binary) | **ALL GREEN** — incl. E4M3-MMVQ m=1/2/5/9 (m1-bits true), E4M3-BATCHED b2/b4/b8 bit-bad=0, **24** E4M3-FUSED2/3 + FUSED2/3-T cells bit-bad=0 |
| `run-spec` K=1..8 | `MEMRA_ST_E4M3=1`, 27B ST + embedded MTP | **8/8 PASS, 0 FAIL**; acceptance 84.6–92.8%; best **2.73x** at K=6 (105.64 vs 38.71 tok/s) |
| prefill/decode argmax | both arms, every run | **MATCH**, logit maxdiff 0.000e0 |
| `serve-st-gate` | `MEMRA_ST_E4M3=1` exported to **both** arms, 27B | **0 failed** (5/5) — incl. **CLI-vs-server greedy ids IDENTICAL** (64 ids) and default-spec == tokenwise serve oracle 1494/1494 chars at 400 tok |

`serve-st-gate.sh` was read first to confirm it propagates the exported environment to the CLI arm
*and* to every `start_server` call, rather than assuming it — otherwise the server side would have
been gated in the wrong config.

---

## 8. Decision: **DEFAULT ON**, one rollback seam

Per flags doctrine (*winners are defaults; naked commands run the tuned path*), `MEMRA_ST_E4M3` is
**flipped default-ON**; `MEMRA_ST_E4M3=0` is the documented rollback seam to the Q8_0 slab. No new
flag was added. `MEMRA_E4M3_DUAL` (the fusion seam from `e65ef35b`, previously undocumented) was
added to the rollback-seam table in `docs/FLAGS.md §catalog`.

**Scope of the flip — per-tensor scalar-scale class only.** `find_fp8_native` returns `blk: Some(grid)`
for the block-128 class and `None` for per-row, and the resident arm additionally requires
`f8.blk.is_none()`, so both of those classes still take the Q8_0 re-encode exactly as before. GGUF is
untouched: `TensorSource::find_fp8_native` is `None` for GGUF sources, so memra's primary runtime and
delivery format sees no dispatch change whatsoever.

### A landmine the flip had to defuse first

`model.rs` guarded ARM B′ (`MEMRA_FP8_BLK_GPU`, the byte-identical GPU block-128 dequant) with
`&& !st_e4m3_enabled()`. That condition was written when `MEMRA_ST_E4M3` was default **off** and only
meant *"the native arm above already claimed this tensor."* Flipping the default naively would have
made it true on every run and **silently disabled ARM B′ for the entire block-128 class** — precisely
the silent-slow-path landmine the flags doctrine exists to prevent. The cross-gate was removed: the
two arms are already disjoint by construction (the native arm returns only when `f8.blk.is_none()`;
ARM B′ runs only when `f8.blk` is `Some`), so a tensor reaching ARM B′ was never eligible for native
residency and needs no cross-gate at all.

### Post-flip re-gates

The flip changes what a **naked** command does, so the gates were re-run on the post-flip binary,
including two receipts that only make sense after a flip: a **naked-default engagement census** (no
`MEMRA_ST_E4M3` in the environment at all — must show the 208 tensors resident as `F8_E4M3`) and a
**rollback-seam census** (`MEMRA_ST_E4M3=0` — must show the Q8_0 slab back). Results in §9.

---

## 9. Post-flip gate results — all four green

One `flock` hold, post-flip binaries pinned (`run-gen` `69d88f319d8abac91153958b39e7145f`,
`kernel-check` `764c7ad3f5a950de76460a9aecfa0612`, `run-spec` `19b00527053989c3f7306e5ad1b1bc06`,
`memra-server` `0eeb860e255fc14d095d1ded4dc468c4`), md5s re-checked after the battery and unchanged.

| gate | env | result |
|---|---|---|
| `kernel-check` | naked | **ALL GREEN**, rc=0 |
| naked-default engagement census | **nothing set** | **`F8_E4M3: 208 t / 6880.000 MiB`** — identical to arm B; argmax MATCH, maxdiff 0.000e0 |
| rollback-seam census | `MEMRA_ST_E4M3=0` | **`Q8_0: 304 t / 7333.906 MiB`** — identical to the pre-flip slab census; argmax MATCH |
| `run-spec` K=1..8 | naked | **8/8 PASS, 0 FAIL** (`=== SELF-CONSISTENCY PASS ===`) |
| `serve-smoke` (GGUF 9B NVFP4 + draft) | naked | **0 failed** (16/16) — incl. spec == plain greedy serving text, the 4-arm sampled-truncation matrix (bangs=0), and session-affinity resume exactness |

**The naked-default census is the receipt that mattered most.** A default flip is only real if a
command with *nothing in its environment* takes the new path, and the earlier `MEMRA_PP_ONLY` arm D
could not show that — that harness `return`s before the census print, so it could only offer the
hook-entry count (288/pass, which did equal arm B exactly). The full-path census closes it directly:
208 tensors resident as native e4m3 with an empty environment, byte-for-byte equal to the explicitly
flagged arm.

**The rollback seam is equally receipted, and in the same shape:** `MEMRA_ST_E4M3=0` reproduces the
pre-flip slab census to the milli-MiB (304 t / 7333.906 MiB), and both arms emit the *same* argmax ids
(prefill 1178, decode 1178, logit maxdiff 0.000e0) — the seam moves residency and speed, not numerics.

**`serve-smoke` is the GGUF no-regression receipt.** The scope argument in §8 says GGUF is untouched
because `TensorSource::find_fp8_native` is `None` for GGUF sources. That is a code argument; this is
the measurement of it — the whole serving battery, on the 9B NVFP4 **GGUF** plus its draft, under the
flipped default, 0 failed. memra's primary runtime and delivery format is confirmed unaffected rather
than merely argued to be.

### Re-gated once more on the final shipped bytes

Three doc/comment commits landed after the flip (`96ac93c4`, `6ca8fc00`), which relinks every binary
— so the battery above no longer described the bytes the branch actually ships. Comment-only changes
*cannot* alter behavior, but "cannot" is an argument, and this repo's evidence discipline wants a
measurement. Re-run on the final build (`kernel-check` `5dc1c4fc7b050414796979b9c1f65478`, `run-gen`
`b34b87f8e7b3a14d5e180163035dd98e`, `run-spec` `1363f7770d191c364df1c33da4885c60`): **kernel-check
ALL GREEN**, and the naked census is **identical** — `F8_E4M3: 208 t / 6880.000 MiB`, argmax MATCH,
maxdiff 0.000e0. Logs `kernel-check-final.log`, `census-final-naked.log`, `BINARY-md5-final.txt`.

Post-flip `run-spec` used a shorter window than the pre-flip run (31-token generate vs 199), so its
acceptance rates (55–94%) and speedups (0.92x–2.05x) are **not comparable** to the pre-flip sheet's
(84.6–92.8%, up to 2.73x) — short windows amortize the draft prime over fewer tokens and K=8
overshoots on a 32-token budget. The gate being asserted here is **self-consistency**, which is
K-for-K identical to `generate` in all 8 arms; the speed comparison stays with the pre-flip N-matched
run. Both are in the tree.

---

## 10. Method notes (what was discarded, and why)

Two batteries' worth of runs were **thrown away** rather than reported, both for binary hygiene —
recorded here because a discarded run is part of the evidence trail:

1. The first decode A/B was discarded because `run-gen` was rebuilt **mid-loop** to add the
   teacher-forced instrument, which would have split arms A and B across two different binaries.
2. The first prefill A/B was discarded because it was launched against the pre-flip binary with an
   **unflagged** A arm and was still in flight when the flip relinked `run-gen` underneath it — its
   "A slab" arm would silently have become the e4m3 arm. (Its A r1 read 1468.3 tok/s, which the
   re-run reproduced at 1468.1 under an explicit `MEMRA_ST_E4M3=0`, confirming what that arm had
   been — but the run was still deleted, not reported.)

`BINARY-md5.txt` pins the md5 per battery and states the trap explicitly: **post-flip, "no env" means
native e4m3**, so every arm must name its config.

An earlier harness bug is also worth recording: `MEMRA_NGEN=128 ${P:+MEMRA_PROMPT_FILE=$P} timeout …`
made the shell try to *execute* `MEMRA_PROMPT_FILE=…` and every run returned rc=127. It was caught
only by reading the raw log instead of trusting a summary line — the repo law *"never let a pipe
swallow error output"* earning its place. Fixed with explicit `env VAR=val …`.

---

## 11. What this lane did NOT establish

* **Block-128 FP8** still pays the Q8_0 re-encode. It needs either a per-block-dequant mmvq twin or
  ARM A's lossy fold. Unchanged by this lane, and deliberately so.
* **Per-row (per-channel) F8** is likewise untouched: no e4m3 kernel consumes a scale vector.
* **Long-context 27B FP8 prefill and multi-model serving** remain the 96 GB pod's jurisdiction; the
  claims here are single-stream decode and pp512 on the 24 GB target rig.
* **Qwen 3.8 (~08-10)**: if its FP8 release is per-tensor scalar scale, it lands on this default with
  no work. If it is block-128 (as 3.6-FP8 was: `weight_block_size [128,128]`), it does **not** — that
  is the gap the next lane has to close, and `docs/qwen38-bringup-runbook.md` already treats the
  scale class as a leg-A branch point.
