# Chunk-order invariance — root cause, fix, gate (2026-08-05)

Lane: `lane/chunk-invariance` off `restructure/public-split` (train c025ac5b).
Rig: local RTX 5090 Laptop (24 GB, sm_120a), shared with two other lanes — every
measurement batched per `flock /tmp/gpu5090.lock` hold.
Model: `qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (qwen hybrid, GDN linear-attn
+ full-attn hd128, NVFP4) — the family the original finding was measured on.
Raw: `logs/` (every run tee'd before parsing), rows in `RESULTS.jsonl`, scripts
`run-lane.sh` + `run-phase2.sh`, reproducer arm `concat-prime-probe chunkinv`.

## The finding under test

`research/session-affinity-20260805/RESULTS.md` and `docs/SERVING.md` recorded: varying ONLY
`MEMRA_PRIME_CHUNK` changes greedy output on 97- and 149-token prompts with zero cache reuse.
Reproduced here at engine level (no server), 4/4 arms — see phase A.

## Root cause: NOT reduction order. A numeric-CLASS edge at the first chunk boundary.

The repo's recorded explanation was **"a different split changes the reduction order in the
prefill GEMMs"** (`docs/SERVING.md`). **That is refuted.** Three receipts:

**Phase B — the prefill GEMM is m-INVARIANT.** `concat-prime-probe gemm` feeds the same
activation rows at m=32 and at m=33..80 and bit-compares rows `[0,32)`. Both the layer-0 `wq`
(quantized, MMQ/f16 lane) and the `output` head are **BIT-IDENTICAL at every m**. Growing the
batch does not move an existing row's value, so "chunk size changes the GEMM reduction order"
cannot be the mechanism. (`logs/B-gemm-wq.log`, `logs/B-gemm-head.log`)

**Phase A — the divergence is a STEP at the boundary, not a band.** Per-row maxdiff of the
prefill hidden stack, chunk 64 vs 2048 on the 149-token prompt:

```
rows[0,64)   maxdiff = 0.000e0      <- BIT-IDENTICAL before the boundary
rows[64,149) maxdiff = 6.909e0      <- O(1) immediately after it
per-8-row: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 | 6.9 3.6 4.8 5.3 4.9 5.4 4.3 6.5 3.5 3.5 4.4
```

`first_div_pos` **equals the chunk size exactly** in all four baseline rows (64→64, 32→32, on
both prompts). A reduction-order effect would perturb every row a little; this perturbs
*nothing* before the boundary and everything after it by O(1) — the signature of crossing a
different arithmetic path, not of reordering the same sum.

**The actual mechanism** (`hybrid_forward.rs:1399`, `full_attn_prime_fa_dispatch`): the
attention path is selected by `base_len == 0`, i.e. *"is this the first chunk?"*

- chunk 0 (`base_len == 0`) → `fa_prefill` over the **f32** K/V of this batch;
- every later chunk → `fa_prefill_view_ws` over the **q8_0/q5_1 quantized KV cache**.

So `MEMRA_PRIME_CHUNK` decides *at which token position the prefill stops reading f32 K/V and
starts reading dequantized cache*. Rows before the first boundary are computed identically in
both configs (hence bit-identity); rows after it are computed in a lossier class, and the
q8_0/q5_1 quantization error is exactly the O(1) logit perturbation measured. A near-tie
argmax then flips, and the greedy stream diverges (step 16-47 in the receipts).

**Phase F — the two other suspects are eliminated.** `MEMRA_PRIME_DEQW=0` (the *other*
quantized-cache FA kernel, inline dequant instead of dequant-once workspace) reproduces the
divergence with the *identical* maxdiff (3.909e-1 / 5.041e-1) and the same `first_div_pos`, so
it is not that kernel's workspace. `MEMRA_GDN_CHUNKED=0` (sequential GDN scan, removing the WY
chunk segmentation entirely) **still diverges** (4.845e-1 / 5.687e-1), so the GDN state carry
is not the cause either — notable because the GDN scan is what the vLLM #38561 analogue
(mamba chunk boundaries) would have blamed.

**Exact first diverging op:** the first `fa_prefill_view_ws` call — layer 0's full-attn
mixer on chunk 1 — reading `cache.kv[il]` (q8_0 K / q5_1 V) where the chunk-0 path read f32
`k`/`v` directly. First diverging tensor: that call's `attn` output for query row
`chunk_size` (absolute position = the first boundary).

## Fix: landed, behind `MEMRA_PRIME_INVARIANT=1`, and it is perf-free

The blueprint (vLLM #38561) says: pin split points to a fixed grain independent of the runtime
chunk size. Implemented as `MEMRA_PRIME_INVARIANT=1` + `MEMRA_PRIME_GRAIN` (default 4096):
under the door, segmentation stops tracking `MEMRA_PRIME_CHUNK` entirely, so a given prompt
length primes through the same boundary set — and therefore the same arithmetic — on every rig.

**Phase C — it works.** Same prompts, chunk 2048/64/32, door ON at grain 32: prefill logits
**BIT-IDENTICAL** (`logit_maxdiff = 0.000e0`, `first_div_pos = -1`), greedy streams identical
for all 48 steps, on both prompts. 4/4 previously-diverging arms now exact.

**Phase E — the mechanism costs nothing.** Both arms run the *same* effective segmentation
(grain 64 vs chunk 64, with the door additionally handed a deliberately-wrong
`MEMRA_PRIME_CHUNK=4096` it must ignore), so the delta is the door's own overhead. N=5
interleaved (`off,on` alternating — never one arm then the other), median of 3 timed
`prime_cache` reps per point via `run-gen MEMRA_PP_ONLY`, thermal regime 62 C/180 MHz cold
ramping to 77 C/1687 MHz:

| prefill | off (chunk-steered) | on (grain-pinned) | delta |
|---|---|---|---|
| 4881 tok (pp6257-class) | 4557.8 tok/s | 4555.7 tok/s | **-0.05%** |
| 400 tok (pp512-class) | 4038.3 tok/s | 4045.1 tok/s | **+0.17%** |

Both inside run-to-run noise. **Under the 2% bar, so invariance is a default candidate on
mechanism cost.**

**Phase G — the door is a no-op on today's default path.** At the default grain (4096) any
prompt shorter than 4096 primes monolithically, exactly as the current default config does.
Door-off vs door-on: identical argmax *and* identical margin to 6 decimals on both prompts
(271/0.483452, 4558/1.227142). Enabling the door does not change today's shipped output.

**Phase H — the grain is also perf-free, on this model.** Pure segmentation cost under the
door at 4881 tok, N=5 interleaved: grain 4096 → 4544.5, 2048 → 4546.5, 512 → 4545.6,
64 → 4546.4 tok/s (all within +0.04%). This prime is not segmentation-bound here.

## Why this is NOT flipped to default yet — the honest limit

The measured evidence covers **one model (9B NVFP4 hybrid), one prompt scale (≤4881 tok), one
rig (24 GB laptop 5090)**. `MEMRA_PRIME_CHUNK` exists for a regime this lane did not measure:
long-context primes on large models where per-layer transients scale with the chunk and OOM is
the binding constraint (the knob's origin, `5c716c06` — "16k/32k+ now run on 24GB"). Under the
door `MEMRA_PRIME_CHUNK` no longer bounds the transient footprint — `MEMRA_PRIME_GRAIN` does —
so flipping the default is a **policy change with an OOM surface**, and per CLAUDE.md a
default flip needs the 27B/long-ctx correctness+memory+throughput gates on the target rig,
which are not in this lane's receipts.

Second, exactness would still be scoped, not absolute: the door makes output invariant to
`MEMRA_PRIME_CHUNK`, but `MEMRA_PRIME_GRAIN` is then itself a numeric knob (phase D1: two
grains give different text — as designed, same class as `MEMRA_KV_K`/`MEMRA_GDN_CHUNK`). The
win is real but bounded: **it converts an accidental dependence on a documented *memory* knob
into an explicit dependence on a declared *numeric* knob.** That is the whole value, and it is
worth stating in exactly those terms rather than as "we won back one canonical output".

A cheaper stronger fix exists and is NOT taken here: make chunk 0 also read the quantized
cache (delete the `base_len == 0` f32 special case), which would make output invariant to the
chunk size *without* a grain knob at all — every row in the same class. That trades prefill
quality/speed on the *unchunked* path (today's fast default for short prompts) and so needs
its own arm; filed as the follow-up, not smuggled in here.

## Exactness battery (9B, on-box, this branch)

| gate | result |
|---|---|
| `kernel-check` | **ALL GREEN: kernels match CPU reference** |
| `run-gen` argmax | **MATCH** — prefill 760 / decode 760, maxdiff 9.425e-1; batched-prime 760 vs tokenwise 760, maxdiff 9.361e-1 |
| `run-spec` K=1..8 | **SELF-CONSISTENCY PASS** at every K (acceptance 88.2 / 90.9 / 81.5 / 84.4 / 57.5 / 69.0 / 61.2 / 55.4%) |
| `chunk-invariance-gate.sh` (default, door off) | **PASS** — CHUNK-DEPENDENT on both pinned prompts |
| `chunk-invariance-gate.sh --canary` | **PASS** — injected door flip broke the assertion (teeth proven) |
| `chunk-invariance-gate.sh --expect-invariant` | **PASS** — bit-identical, `first_div_pos=-1` |

Raw: `logs/BATT-*.log`. The default path is unchanged by this lane (door off, trace off), which
is why the argmax/spec goldens are untouched.

## Two self-audit finds worth recording

1. **The first canary had no teeth.** `--canary` flipped only the *expected label* and re-ran
   the identical configuration, so it passed exactly when the default gate passed — perfectly
   correlated, proving nothing. A canary must change the WORLD, not the label; it now toggles
   the invariance door and requires the assertion to fail. (An even earlier shape was worse: a
   single `--chunks` value has nothing to compare and always reported CHUNK-INVARIANT.) Fixed
   in `2aa29179`, trap documented in the script header.
2. **The expect-variant assertion accepted a partial disappearance.** One CHUNK-DEPENDENT
   verdict among N prompts read as "variant" and passed, which would have hidden exactly the
   silent behavior change the gate exists to catch. It now requires divergence on every pinned
   prompt.

Also corrected: the in-code comment landed in `f75ab3c6` asserted "the prefill GEMM is
m-DEPENDENT" as leak (1) — the very thing phase B refutes. Leaving a disproven mechanism in a
comment is how the wrong cause reached `docs/SERVING.md` in the first place (`43656383`).

## Verdict

- **Root cause: found and it contradicts the documented one.** Not GEMM reduction order
  (refuted, phase B) and not GDN chunk state (refuted, phase F2), but the `base_len == 0`
  f32-vs-quantized-KV attention class switch at the first chunk boundary.
- **Fix: landed as an opt-in door** (`MEMRA_PRIME_INVARIANT=1`), delivers bit-identity 4/4,
  mechanism cost **-0.05% / +0.17%** (N=5 interleaved) — under the 2% default-candidate bar,
  but held opt-in pending long-ctx/27B OOM+perf gates it cannot claim from this rig.
- **The repo's scoped exactness claim stands** ("tokens never depend on batchmates"), and
  `docs/SERVING.md`'s mechanism paragraph is corrected by this lane.
- **Gate added with teeth:** `tools/chunk-invariance-gate.sh`, fast-gate ids
  `chunkinv` / `chunkinvc`, routed from the `hybrid_forward.rs` map row. It asserts the
  behavior in whichever direction is the documented contract and fails if it silently changes
  in *either* direction; the canary arm flips the expectation and is required to diverge, so
  the gate is proven able to fail (vLLM #40372's pattern).
