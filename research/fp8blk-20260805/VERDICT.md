# lane/fp8-blk128-decode — VERDICT

**Question.** Qwen3.6-FP8 ships `weight_block_size [128,128]`, and Qwen3.8 (the ~08-10 target) likely
will too. `lane/fp8-decode-v1` shipped checkpoint-native e4m3 residency as the default, but only for
the **per-tensor scalar** scale class; the **block-128** class still fell back to the ARM B' Q8_0-slab
re-encode. If 3.8-FP8 is block-128 like 3.6, the flagship model class would miss the entire native
win on day one. This lane builds the per-block twin and decides its defaults.

**Verdict: BOTH defaults ON for the block-128 class.** Native e4m3 residency (`MEMRA_ST_E4M3_BLK`,
shipped 0ce5293b) and the per-block MMQ prefill route (`MEMRA_FP8_MMQ`'s native-resident source,
flipped 1d00d2b6). Naked commands get the fast path; three rollback seams exist and are measured.

Rig: local RTX 5090 **Laptop** (24463 MiB, ~896 GB/s GDDR7 — not the desktop's ~1.7 TB/s).
Artifact: `/data/ai-ml/hf-models/qwen36-27b-blk128fp8` — 208 block-128 F8_E4M3 projections
(6880 MiB) plus the source's NVFP4 MLP planes byte-identical. Verdicts anchor on 27B shapes; the
1.7B synth appears only in bring-up/instrument rows.

## 1. What the class gets

| quantity | floor (Q8_0 slab) | shipped default | delta |
|---|---|---|---|
| decode, 27B, N=5 interleaved | 40.87–40.90 tok/s | 41.54–41.58 | **+1.69%**, distributions disjoint |
| pp512, N=6 adjacent alternating pairs | 1540.6 median | 1553.0 median | **+0.83%**, 6/6 pairwise, DISJOINT |
| weight VRAM | 7333.906 MiB (304 Q8_0 tensors) | 23.906 MiB Q8_0 + 6880.000 MiB F8_E4M3_BLK | **−430 MiB** |
| bytes/weight | 1.0625 | 1.0 | ratio 1.06250 = theory ⇒ single residency, no duplicate copy |

Decode is the mechanism win (native bytes = 1.0 B/w through a GEMV that re-reads the whole weight
every token). Prefill is the win that had to be *recovered*: see §3.

## 2. Exactness — branch (b), and why bit-identity is the wrong bar

Two different arithmetics ship here, and they need different bars:

* **Decode + the dequant fallback** are bit-exact against the floor by construction.
  `fp8_blk_dequant_q8_0` is gate-proven byte-identical to the host dequant+re-encode (kernel-check
  `fp8-blk-gpu`, 5 cells `bad=0` incl. the ragged `[5x128]`/`[6x160]` tails).
* **The per-block MMQ prefill route** is `kind::f8f6f4` MMA with a per-`[128×128]` f32 scale — *not*
  the Q8_0 re-encode's arithmetic. Demanding bit-identity of it would be demanding it be a different
  kernel. So the bars are argmax agreement, teacher-forced disagreement counting, and NLL.

Measured on `prime_cache` — **the class that actually dispatches this kernel** — with a dispatch
ledger on every arm and an A==B control:

| arm | disagree/511 | mean_nll | max_abs | rms_rel | bitdiff | argmax | top10 |
|---|---|---|---|---|---|---|---|
| A slab (floor) | 257 (50.29%) | 2.787267 | 0 | 0 | 0/248320 | 365 | ref |
| B dequant fallback | 257 (50.29%) | 2.787267 | 0 | 0 | 0/248320 | 365 | same |
| C = the shipped route | 259 (50.68%) | **2.764722** | 3.613e-1 | 2.545e-2 | 248319/248320 | **365** | **same** |

* **A == B bit-for-bit is the control that makes C's row readable.** The dequant arm's prefill is the
  floor's arithmetic by construction, so an instrument that could not see zero where zero is would be
  worthless. It sees zero.
* C's NLL on the prompt's own continuation is **lower** than the floor's (−0.81%), argmax and the
  full top-10 order are unchanged, and rms_rel is 2.5e-2 on a vector whose rms is 2.7155 — the same
  drift class lane/fp8-mmq-v2 measured. +2/511 against a 50.29% baseline is the near-tie class.
* **No tape asymmetry, by construction.** `MEMRA_PP_NLL`'s tape is the PROMPT: position *i*'s logits
  score prompt token *i+1*. Both arms are scored on the same externally-given sequence, so neither
  can win by reproducing its own output. (The decode battery's tape *was* one arm's greedy output,
  which is why that one needed the `tfrev-*` reverse-tape control — and got it: each arm scores best
  on its own tape, 2/128 @ 0.205029 vs 0/128 @ 0.198189, symmetric ⇒ the NLL ordering there is a tape
  artifact, not a quality claim.)

Gates, all on the naked default: kernel-check **ALL GREEN** (5 `fp8-blk-gpu` cells + 49
E4M3-BLK-MMVQ lines, EXACT arms bit-identical incl. `m1-bits=true`, RAND rms_rel 1.5–1.8e-6, 254/254
legal e4m3 codes exercised); run-spec **K=1..8 8/8 PASS**; `serve-st-gate` **0 failed** (CLI-vs-server
greedy streams identical, 64 ids; spec == tokenwise oracle 824/824 chars); `serve-smoke` **0 failed**.

### 2b. The GGUF classes are untouched — measured, not assumed

The flip's gate change lives in `try_fp8_blk_mmq`, which hangs off the **shared** `matmul` /
`matmul_pre` dispatch, and the stash note landed in shared `model.rs` load. GGUF models traverse both,
so "block-128 only" is a claim that needed a probe, not a code read. `fast-gate --tier 1` on the
flipped binary (`fastgate-postflip.log`, `fastgate-logs/`, md5 pin `BINARY-md5-fastgate.txt`):

```
tier 0: GREEN (417s)   full kernel-check GREEN (404s, scope=all, 10 config pins)
tier 1: 0 fail (937s)  g12 q9 q35 k27 g26 o35 o9 q35slru  golden token-identical
                       chunkinv chunkinvc samp             self-gating green
                       q35spec                             golden token-identical
                       g31spec                             stream agreement 32/32
```

Eight golden-token argmax probes across Q4_0/Q8_0/NVFP4 GGUF + SLRU-MoE + both spec paths, all
token-identical to goldens minted before this lane. The flip is confined to the class it was built
for.

**Harness trap worth naming:** the first attempt wrapped fast-gate in `flock /tmp/gpu5090.lock`, and
fast-gate takes that *same* lock internally per GPU step (`flock -w 7200`). `flock(2)` is not
reentrant across a fork, so the inner acquire blocked on the outer holder — a silent 14-minute hang
with `kernel-check.log` at **0 bytes** and no error line anywhere. It looks exactly like a slow
kernel-check. Scripts that self-lock must be invoked *naked*; the correct check is `pgrep` +
`ps -o stat` showing the inner `flock` in `S`, not a growing log.

## 3. The prefill story — a regression, then a route change, and why v2's sheet had the wrong sign

Native residency is a *decode* mechanism, and it must not be paid for in prefill. Letting a 512-token
chunk reach the warp-per-row GEMV would be a ~500x weight-traffic blowup, so prefill needs its own
arm. Three successive answers:

1. **Dequant-per-call to a transient Q8_0 slab** (not a second resident slab — that would give back
   the whole 1.0-vs-1.0625 win). Exact, but pp512 **−13.7%**.
2. **Vectorize that dequant kernel** (68712bba): 66.5 → 27.9 ms/pass, pp512 −13.7% → **−5.8%**. And
   then the arithmetic said stop: the route makes prefill move the weight **three times** (read
   6.88 GB e4m3, write 7.31 GB Q8_0, MMQ reads that 7.31 GB back). 14.19 GB extra = 15.8 ms at
   896 GB/s against a ~332 ms pp512 ⇒ **~4.5pp of the −5.8% is structural**, unremovable by any
   kernel tuning. Only deleting the dequant could remove it.
3. **Route through the per-block MMQ tile** (6b741068 + 1d00d2b6): it eats the resident e4m3 bytes and
   grid directly, so neither extra pass happens. **+0.83%** over the floor.

**Why the same tile that lane/fp8-mmq-v2 measured at 0.85–1.09x wins here.** v2's denominator was the
Q8_0 MMQ with the slab **already resident** — tile vs tile, and the tile lost 0–15% at m=512. On this
class the floor must *also create* that slab on every prefill call (27.9 ms/pass), so the tile only
has to not be 27.9 ms worse than a kernel it trails by at most ~15% on a subset of shapes. Same tile,
opposite sign, because the question changed. That is why the route had to be **measured** rather than
inferred from v2's sheet in either direction.

## 4. The flip is per operand source — the landmine that design avoids

`MEMRA_FP8_MMQ` gated two different things through one function, and flipping that function would
have shipped a VRAM regression. `fp8_mmq_enabled()` is **also** what admits the load-time e4m3
**stash** in `model.rs` — a second e4m3 copy uploaded next to a resident Q8_0 slab, spending
`MEMRA_PP_FP8_BUDGET_MB` — for every Q8_0 tensor with an fp8 sibling. A naive default flip would have
started duplicating weights that no kernel on this path reads, giving back the 430 MiB this lane
exists to free, and then some.

So the gate splits by source, checked *inside* the operand match:

| source | gate | default | why |
|---|---|---|---|
| load-time e4m3 **stash** (duplicate copy) | `fp8_mmq_enabled()` (`=1`) | **off** | floor's slab already resident ⇒ 0.85–1.09x cannot pay for a duplicate weight copy |
| checkpoint-native `QT_F8_E4M3_BLK` `blk` grid | `fp8_blk_mmq_native_enabled()` (`=0` reverts) | **on** | floor must build the slab every call ⇒ +0.83%, disjoint |

Seams, narrowest first: `MEMRA_FP8_MMQ=0` (prefill route only, keeps native decode residency) →
`MEMRA_ST_E4M3_BLK=0` (this class's residency) → `MEMRA_ST_E4M3=0` (all native e4m3 residency, the
doctrine's one seam). All three measured, not assumed.

**The dequant arm is not dead code.** Every tile precondition — `in_f % 16`, grid dims vs shape,
per-tensor scale == 1.0, the cached one-time e4m3-NaN scan (hardware reads magnitude 0x7F as NaN, the
host/ARM B' convention as 0.0) — refuses by falling through to it. A checkpoint the tile cannot take
keeps exact prefill on the floor's own bits instead of losing the class.

## 5. Post-flip re-measurement, and the honest wobble

A flip whose kernel silently fails to dispatch reproduces the pre-flip numbers *either way*, so the
naked default was re-measured with a ledger on every arm (`postflip-*`, `BINARY-md5-postflip.txt`):

```
N naked (NEW DEFAULT)  1553.9 / 1517.4 / 1549.7   hits 1040   gate_off=0
S MEMRA_FP8_MMQ=0      1449.1 / 1440.0 / 1447.9   hits 0      gate_off=2080
A MEMRA_ST_E4M3_BLK=0  1377.9 / 1539.2 / 1539.4   hits 0      no_operand=2480
```

S reproduces the pre-flip dequant arm's 1449.1 to the decimal *and* dispatches zero ⇒ the seam is
real, not nominal. N reproduces the pre-flip MMQ arm's 1553.3 with 1040 dispatches (208 projections ×
5 passes) and reproduces its exactness row in every digit. A reproduces the floor's 1540.5 in reps
2–3. Decode across the same battery: N 41.63/41.56/41.55 vs rollback 40.87/40.85/40.85 — the flip is
prefill-only and left decode exactly where it was.

**But that run's margin did not separate**: N median 1549.7 vs A 1539.4 = +0.67%, N r2 (1517.4) below
A r3 (1539.4), and A r1 = 1377.9 a 10.5% outlier below its own other two reps. Suspected cause was
position drift, not kernels — A always ran *third* in the rep, minutes of load+prefill after N. So the
margin got a dedicated two-arm sweep with N and A **adjacent** and the order **alternating**
(`pairsweep.sh`, N=6):

```
N: medians [1548.7, 1552.3, 1552.5, 1553.4, 1556.9, 1553.6]  median 1553.0  min 1548.7
A: medians [1541.2, 1538.0, 1540.0, 1540.6, 1540.6, 1540.7]  median 1540.6  max 1541.2
ratio 1.0080   min(N) > max(A) -> DISJOINT   6/6 pairwise wins
pairwise deltas [7.5, 14.3, 12.5, 12.8, 16.3, 12.9]   mean +12.72 tok/s = +0.83%
```

Position drift confirmed, and the pre-flip +0.83% reproduces exactly. Note this is *not* a claim that
the 3-arm structure is invalid — it is that a third arm sitting between the two being compared breaks
the interleaving law for that pair, and the fix is to interleave the pair.

## 6. Qwen 3.8 readiness

If 3.8-FP8 ships `weight_block_size [128,128]` like 3.6, **day-one needs no flags**: naked is the fast
path on this class — +1.69% decode, −430 MiB, +0.83% pp512, every gate green. `qwen38-bringup-runbook.md`
§3b step 2 previously told bring-up to keep `MEMRA_FP8_MMQ` OFF (citing the stash denominator, which
is the wrong denominator for a natively-resident checkpoint); following that instruction would have
cost the target model the whole win. It now says naked is the fast path and lists the three seams.

Still true and unchanged: `MEMRA_FP8_FOLD` stays off (lossy — greedy MISMATCH at pos 20), and
`MEMRA_FP8_MMQ=1`'s extra meaning (the duplicate-copy stash) buys nothing on a natively-resident
checkpoint.

Two things a new checkpoint could change, both of which fail *safe* rather than silently: an e4m3 NaN
code anywhere in a tensor (refuses the tile, falls through to exact dequant, counted in the ledger),
and a projection whose `in_f % 16 != 0` (same). Both are per-tensor and both print.

## 7. Receipts

Raw per-run logs for every number above live in this directory. Load-bearing ones:

* decode + census A/B: `ab27b-{A-slab,B-blk}-r{1..5}.log`, `census-27b-{naked,rollback}.log`, `battery.sh`
* prefill route 3-arm: `ppmmq-{A-slab,B-blkdeq,C-blkmmq}-r{1..3}.log`; anatomy `nsys-blk-pp{,-after}.log`, `ncu-dequant-before.log`
* exactness: `mmq-exactness.sh`, `ppx-{A-slab,B-blkdeq,C-blkmmq}.{log,f32}`, `ppx-driver.log`; decode-tape controls `tf-{A-slab,B-blk}.log` + `tfrev-{A-slab,B-blk}.log`
* post-flip: `postflip.sh`, `postflip-{pp,dec,ppx}-*.log`, `postflip-ppx-cmp.txt`, `postflip-runspec-K{1..8}.log`, `postflip-driver.log`
* margin: `pairsweep.sh`, `pair-{N,A}-p{1..6}.log`, `pairsweep-summary.txt`, `pairsweep-driver.log`
* gates: `kernel-check-blkcells.log`, `runspec-blk-K{1..8}.log`, `serve-st-gate-blk27b.log`, `serve-smoke.log`
* GGUF no-regression: `fastgate-postflip.log`, `fastgate-logs/{kernel-check,probe-*}.log` (13 probes), `BINARY-md5-fastgate.txt`
* shape sheet: `gemv-shape-sheet-27b.{log,jsonl}` (blk_over_e4m3 0.962–1.008, ~1.05–1.08x vs Q8_0, up to 786 GB/s)
* binary pins + GPU state: `BINARY-md5-*.txt`, `*-gpustate.txt`

**Negative receipts, kept deliberately.** `tf-C-blkmmq.log` and `mmq-decode-census.log` claim to
measure the MMQ arm and do not — both show `dispatches: 0 (hook entries=0 ...)` and tf-C's numbers
came out byte-identical to tf-B's. Cause, traced rather than guessed: `run-gen`'s verify-prefill gate
calls `decode_step_t` → `matmul_decode_exact`, which has **no GEMM/MMQ arm at all, by design** — the
decode-parity law requires every token row take the exact m=1 MMVQ program, so the prefill hooks live
only on `matmul`/`matmul_pre`. `entries=0` there is *correct behavior*, not a wiring bug, and adding a
hook would have violated the law. `MEMRA_PREFILL_LOGITS` is therefore structurally **blind** to any
prefill-GEMM arm; its comment claimed the opposite and is corrected in place. Kept because the trap is
worth naming: the instrument was the bug, and the two harness bugs found in `postflip.sh` (a
`gen-only` grep that never matches the ST branch's wording; a heredoc that attached to `tee` instead
of `python`, so a comparator silently never ran while the log looked populated) are the same family.

## 8. Commits

| commit | slice |
|---|---|
| `382dae48` | `qmatvec_e4m3_blk_mmvq` + the `QT_F8_E4M3_BLK` residency arm |
| `e4171248` | block-128 GEMV shape sheet (+6.13pp weighted, 98% of the byte ceiling) |
| `0ce5293b` | residency survives the hybrid V-head reorder — 144 of the 27B's 208 projections |
| `68712bba` | vectorized the block-128 → Q8_0 dequant kernel (2.38x) + the 2 ragged cells |
| `925673c7` | 27B e2e battery: decode +1.69%, −430 MiB, prefill −13.7% |
| `4aa7101f` | nsys prefill anatomy + the dequant-fix A/B (66.5 → 27.9 ms/pass) |
| `6b741068` | route block-128 prefill through the per-block MMQ tile (+0.83%), still flag-gated |
| `0aa69f95` | the MMQ arm's REAL exactness cells — the first pass measured the wrong arm |
| `20a2f5dd` | serve gates + the ragged-cell kernel-check receipt that was missing |
| `1d00d2b6` | **the default flip**, per operand source |
| `c0006e27` | post-flip battery on the naked default |
| `b5c0e995` | the margin settled on an adjacent-pair sweep (DISJOINT, 6/6, +0.83%) + this verdict |
| `31c16fec` | full kernel-check on the flipped binary — ALL GREEN |
| `dfcad385` | GGUF no-regression: fast-gate tier 0 GREEN + tier 1 13/13, 0 fail |
