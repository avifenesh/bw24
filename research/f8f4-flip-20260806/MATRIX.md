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

### Gate 1 — run-gen argmax A/B, per model (OFF = int8 default, ON = `MEMRA_MMQ_F8F4=1`)

Raw: `logs/argmax-*.log`. Two distinct checks per arm — run-gen's **internal** MATCH lines
(prefill-vs-decode, batched-prime-vs-tokenwise, which must hold *within* an arm) and the
**cross-arm** `tokens:` comparison (the greedy identity the flip bar asks about).

