# f8f4 default-flip — evidence matrix

**Date:** 2026-08-06 · **Branch:** `lane/f8f4-flip` · **Base:** `7c0df07b` (the w4a8-prefill merge)
**Rig:** local RTX 5090 Laptop (sm_120a, GB203, 82 SM), clocks LOCKED 1860/1860, nvcc 13.1.115
**Lock:** `flock /tmp/gpu5090.lock` (the box convention — the brief's `/tmp/memra-5090.lock` does
not exist on this rig; using the real lock is what actually serializes against other lanes).

**Mandate:** gather the multi-model acceptance evidence that gates flipping the f8f4 block_scale
prefill form to DEFAULT — the 1.2153x prefill win merged at `7c0df07b`.

---

## 0. The flags, read off the code (not guessed)

| flag | kind | what it does | site |
|---|---|---|---|
| `MEMRA_MMQ_F8F4=1` | runtime | selects the **route**: e4m3 weight containers × **e4m3 activations** instead of NVFP4→int8 × q8_1 int8 acts | `crates/memra-engine/src/mmq_ffi.rs:1129` |
| `MEMRA_MMQ_F8F4_PLAIN=1` | **build-time** | rollback seam for the **form**: restores the 1.00x plain `kind::f8f6f4` MMA | `crates/memra-engine/build.rs:204` |

The distinction is the whole lane. `7c0df07b` changed the **form** (bit-exact, 0/128 elements
differ — `w4a8-prefill-20260806` slice 4). The **route** is what a default flip would turn on, and
the route changes activation quantization. So "the win is bit-exact" and "the flip needs acceptance
evidence" are both true, about different things.

## 1. SCOPING — the seam is reachable on 2 of 12 gate models, and it is a property of the ARTIFACT

The brief's matrix asks for "q27-NVFP4, 31B, 12B, 9B — every model with a fast-gate row that runs
prefill through MMQ." Read against the dispatch, that set is **not** what the flag can touch.

The f8f4 seam sits inside `qmatvec_mmq_nvfp4_w4a8_scaled`, which is reached from exactly one arm of
`qmatvec_mmq` (`mmq_ffi.rs:568`):

```rust
q if q == crate::QT_NVFP4 && use_w4a8 => {
    self.qmatvec_mmq_nvfp4_w4a8(bytes, x, m, in_f, out_f, *scale, *rp)
}
```

`MEMRA_MMQ_F8F4` is read *inside* that call. A weight that is not `QT_NVFP4` never reaches the
branch, so on a model with no NVFP4 tensors the flag is a **no-op by construction** — there is no
dispatch site to change, and argmax/acceptance are invariant for structural reasons, not measured
ones.

GGUF header scan of every fast-gate model (`tools/scope_nvfp4.py`, raw:
`logs/scope-nvfp4.log`) — NVFP4 is ggml type 40:

| gate probe | model | NVFP4 tensors | f8f4 reachable? |
|---|---|---|---|
| `k27` | Qwen3.6-27B-NVFP4-Q4_K_M-mtp | **504 / 866** | **YES** |
| `q9` | Qwen3.5-9B-NVFP4-MTP | **114 / 668** | **YES** |
| `g12`, `g12d` | gemma-4-12b-it-qat-q4_0 | 0 | no |
| `g31` | gemma-4-31B_q4_0-it | 0 | no |
| `e4b` | gemma-4-E4B_q4_0-it | 0 | no |
| `g26` | gemma-4-26B_q4_0-it | 0 | no |
| `q35`, `q35slru`, `q35spec` | Qwen3.6-35B-A3B-UD-IQ4_XS | 0 | no |
| `aw35` | Qwen-AgentWorld-35B-A3B-UD-IQ4_XS | 0 | no |
| `o9` | ornith-1.0-9b-Q8_0 | 0 | no |
| `o35` | ornith-1.0-35b-Q4_K_M | 0 | no |
| `kat` | KAT-Coder-V2.5-Dev-IQ4_XS | 0 | no |

**The brief's 31B and 12B rows are not measurable arms of this question** — those are Q4_0 gemma
QAT artifacts with zero NVFP4 tensors. Running them would produce a green cell that means "the flag
did nothing," which is not acceptance evidence. They are recorded here as the **structural control**
(and one is run below to *demonstrate* the invariance rather than assume it).

So the honest measurable population is **k27 (q27-NVFP4) and q9** — i.e. the exact two models the
2026-07-10 flip battery already ran. That is the first thing this lane learned, and it changes what
"multi-model evidence" can mean here: **the served NVFP4 population is 2, and it already has a
recorded split verdict.**

## 2. THE PRIOR — this flip was already run, and it already concluded NO

`docs/FLAGS.md:148` and `research/tune-data/rig5090.jsonl:286` (tag `f8f4-flip-decision`,
2026-07-10) are a **pre-registered, owner-recorded verdict on this exact question**:

> **NO GLOBAL FLIP.** pp1845 +3.9..6.3% and prime(TTFT) -4.0..-5.6% on ALL models, but e2e spec is
> model-signed: **9B GGUF -3.5% (acc 68.1→63.9), 9B ST -6.1% (74.0→65.7)**, 27B GGUF -0.3%
> (68.3→69.4), **27B ST +7.2% (acc 68.2→76.2)**. All gates PASS both arms.

and the LAW it produced, now in CLAUDE.md as the prefill-KV acceptance law:

> the PREFILL numeric config is part of the ACCEPTANCE config — prefill writes the prompt
> KV/hidden lineage the draft head reads, so an argmax-clean prefill kernel swap still moves
> acceptance by ±8pp with the sign model-dependent.

The corroborating row is `rig5090.jsonl:292` (`k32-imma-closed`): a **completely different** numeric
config (int8 per-32 recode, ~10x cleaner than the e4m3 fold, "near-int8 class") moved 9B acceptance
**68.1 → 57.2 (-11pp)** and 27B by +0.8pp. Its lesson:

> Two independent numeric configs (e4m3 fold, int8 per-32 recode) BOTH shift 9B acceptance by
> ~-8..-11pp → the 9B draft head is hypersensitive to ANY prompt-KV lineage change, not to a
> specific format.

**This is the load-bearing fact for the flip decision.** The failure mode the flip bar guards is
not hypothetical and not unmeasured — it is a measured, twice-reproduced, mechanism-explained
property of one of the two reachable models. And what `7c0df07b` changed (the MMA *form*) is
provably bit-exact against the *plain f8f4 arm* — meaning it **cannot repair** the acceptance dip,
because the dip belongs to the route (e4m3 acts), which the form swap leaves byte-identical.

That is the flip's structural trap, stated precisely:

- the **form** swap is bit-exact ⇒ it makes the route **1.2153x faster** and changes **no** numerics
- ⇒ the acceptance dip measured on 2026-07-10 for the **route** transfers **unchanged**
- ⇒ a faster route does not become an acceptance-neutral route

## 3. Evidence collected by this lane

Protocol, common to every cell: local 5090, clocks locked 1860/1860, `flock /tmp/gpu5090.lock`,
one binary per comparison (`1dd58cc8`-era `target/release`), arms interleaved, every raw log
committed under `logs/`. No `.nsys-rep` produced anywhere in this lane.

**Clock honesty note.** On this laptop 5090 the locked 1860 is a **ceiling, not a floor** —
sustained prefill/spec load power-caps into the 1550-1850 range and the two arms do not always
see matched clock windows. Every table below therefore states which metric is clock-invariant.
Acceptance is a **ratio over a fixed token budget** and is clock-invariant (proven below: it is
bit-stable across 5 reps at 1552-1852 MHz). Wall-clock tok/s is not, so tok/s is only quoted
where it came from the idle-guarded `ab_clean.sh` harness.

### Gate 1 — run-gen argmax A/B, per model (OFF = int8 default, ON = `MEMRA_MMQ_F8F4=1`)

Raw: `logs/argmax-*.log`, comparator `tools/cmp_arms.py`, output `logs/cmp-argmax.log`. Two
distinct checks — run-gen's **internal** MATCH lines (prefill-vs-decode, batched-prime-vs-tokenwise,
which must hold *within* an arm) and the **cross-arm** `tokens:` comparison (the greedy identity
the flip bar asks about).

| model | arm | internal argmax | vs pinned golden | cross-arm greedy |
|---|---|---|---|---|
| k27 (q27-NVFP4), ngen=24 | OFF | MATCH / MATCH | MATCH (20/20) | **DIVERGE at generated index 22** (OFF=271, ON=11) |
| | ON | MATCH / MATCH | MATCH (20/20) | |
| q9, ngen=64 | OFF | MATCH / MATCH | MATCH (20/20) | **DIVERGE at generated index 38** (OFF=20340, ON=5861) |
| | ON | MATCH / MATCH | MATCH (20/20) | |

Reading, precisely: **each arm is internally self-consistent and each arm still reproduces the
pinned fast-gate golden**, so neither arm is "broken" and the existing gates pass in both — which
is exactly what the 2026-07-10 battery also found ("All gates PASS both arms"). What the arms do
*not* do is agree with each other. That is expected (e4m3 activations are a different numeric
class than q8_1 int8; the bit-exactness proven at `7c0df07b` is between the two MMA *forms*, not
between the two *routes*) and it is the reason the flip cannot be justified on argmax alone: the
golden pins would survive the flip while the model's actual output text changes.

Note also that the goldens are only 20 tokens, and both divergences land **past** token 20
(index 22 and 38). A 20-token golden battery is structurally incapable of seeing this change —
`--refresh-goldens` after a flip would silently re-pin the new arm's tokens.

### Gate 2 (DECISIVE) — spec acceptance A/B, `run-spec` K=1..8, interleaved per K

Harness `tools/accept_ab.sh`, parser `tools/parse_accept.py`, raw `logs/accept-*.log`, tables
`logs/table-*.txt`, rows in `RESULTS.jsonl`. Prompt: `research/e2e/prompts/p2-code-medium.txt`
(a real code prompt — README rule: synthetic sequences badly under-state acceptance), ngen=128.
Each invocation **is** the K=1..8 self-consistency gate as well as the telemetry: run-spec asserts
greedy identity to plain `generate` and exits non-zero on FAIL.

**q9 (Qwen3.5-9B-NVFP4-MTP, bare MTP head) — no dip:**

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| OFF | 80.3 | 64.3 | 52.0 | 45.6 | 36.9 | 32.9 | 28.2 | 24.7 | 45.6 |
| ON | 80.3 | 65.2 | 53.3 | 44.1 | 40.5 | 36.2 | 31.0 | 27.1 | 47.2 |
| **Δpp** | +0.0 | +0.9 | +1.3 | **−1.5** | +3.6 | +3.3 | +2.8 | +2.4 | **+1.6** |

**q27 (Qwen3.6-27B-NVFP4-Q4_K_M-mtp) — down at every single K:**

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| OFF | 91.0 | 81.6 | 76.1 | 67.1 | 63.2 | 52.7 | 47.1 | 41.2 | 65.0 |
| ON | 86.8 | 79.6 | 72.5 | 65.7 | 61.9 | 51.6 | 46.2 | 40.4 | 63.1 |
| **Δpp** | **−4.2** | −2.0 | **−3.6** | −1.4 | −1.3 | −1.1 | −0.9 | −0.8 | **−1.9** |

Self-consistency **PASS in all 32 cells, both arms, rc=0** — so the K=1..8 exactness gate is green
on the ON arm and gives no reason to block. The block is the acceptance column.

`K=3` is q27's **serve K** (`rig5090.jsonl:289` NV-27B standing line; the `MEMRA_SPEC_PMIN` row in
`docs/FLAGS.md` calls K=3 "the 5090's serve optimum"). So the largest interior loss sits on the
configuration this model is actually served under.

**Sign reversal vs 2026-07-10.** That battery recorded 9B **negative** (−3.5/−6.1% e2e, acc
68.1→63.9 and 74.0→65.7) and 27B ST **positive** (+7.2%, acc 68.2→76.2). This lane measures the
9B bare head **positive** (+1.6pp) and the 27B GGUF **negative** (−1.9pp). Both readings agree on
the LAW; they disagree on *which model pays*. That disagreement is itself the finding: the sign is
not a stable per-model property, so "adopt per-model" cannot be discharged once and trusted — and
a global default flip is a bet on a sign that has already flipped once under regime change.

### Gate 2b — is the quoted cell real? N=5 determinism check

A cell that goes in a verdict has to repeat. `tools/repeat_k.sh`, K=3, N=5, interleaved OFF/ON,
raw `logs/repeat-q27-k3.log`:

| arm | acceptance, all 5 reps | distinct values | spec tok/s | clocks |
|---|---|---|---|---|
| OFF | 89/117 = 76.1% ×5 | **{89/117}** — deterministic | median 90.51 | 1620-1852 MHz, 54-77 C |
| ON | 87/120 = 72.5% ×5 | **{87/120}** — deterministic | median 88.30 | 1552-1635 MHz, 66-77 C |

**−3.6pp reproduces 5/5, bit-stable.** sc=PASS and rc=0 in 10/10 runs. Two things follow:

1. Acceptance is deterministic *and* clock-invariant here — identical fractions at 1552 MHz and
   at 1852 MHz. So the gate-2 tables are readable despite the power-capping.
2. The **drafted count itself moves** (117 → 120). The ON arm is not merely getting the same
   proposals verified worse; it is making **different proposals**. That is the prefill-KV
   acceptance law's mechanism visible directly in the counters: a changed prompt-KV lineage
   changes what the draft head reads, hence what it proposes.

The `−2.4%` spec-tok/s median in that table is **clock-confounded** (the arms ran in different
clock windows) and is deliberately not quoted as the arm's cost.

### Gate 4 — per-model prefill delta (the win, restated per model)

Harness `tools/ab_clean.sh` — the w4a8 lane's per-arm **idle guard** (require util < 15% and ≤1
compute app before *every* run), pp512, one binary, 3 rounds × `MEMRA_PP_REPS=5` = N=15/arm.

| model | OFF median | ON median | ratio | ranges |
|---|---|---|---|---|
| q27 (from `w4a8-prefill-20260806`, N=15) | 1395.5 | 1696.0 | **1.2153x** | do not overlap |
| **q9 (this lane, N=15)** | 4438.1 [4435.5..4441.5] | 5097.2 [5008.8..5100.4] | **1.1485x** | **do not overlap** |

So the merged win is **not** a q27 artifact — it reproduces on the other reachable model at
1.1485x. Raw `logs/ab-q9.log`, table `logs/table-ab-q9.txt`.

The idle guard earned its keep mid-run: another lane's `./target/debug/kernel-check` took the GPU
(2 compute apps) and the guard **parked rounds 2-3 until it cleared** rather than publishing
contaminated rounds — the exact failure the w4a8 lane had to discard a whole battery over.

