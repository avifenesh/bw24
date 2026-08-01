# residency-cap: why memra spilled the Ornith-35B expert bank that fits — and the fix (2026-08-01/02)

Lane `lane/residency-cap` (from `restructure/public-split`, 9c697367). Rig: **RTX 5090 Laptop,
24463 MiB** (the mission brief said "32GB"; nvidia-smi total is 24463 MiB = 23.9 GiB — all
envelope math below uses the real card, consistent with the ornith-bar lane's "24.5GiB").
Driver 595.71.05, platform_profile `performance`. Every GPU run under `flock /tmp/gpu5090.lock`
(one co-lane shares the rig; the co-resident `llama-server --embedding` (332 MiB) is allowlisted
and untouched — it is inside every "used/peak" figure below). llama.cpp arm: local fork build
`bb090d1f1`, same binary as the ornith-bar lane. Model:
`/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf` (same bytes as
ornith-bar; cross-lane token-sha agreement below is the identity check).

## 1. The budget math: why a bank that fits was spilled

The decision is `build_dev_exps` (`crates/memra-engine/src/hybrid.rs`), made once at the first
MoE layer. Old default, captured verbatim (`smoke-default.log`):

```
[moe] resident-experts decision: per-layer 522MB x 40 layers = 20.9GB vs budget 19.1GB -> SLRU cache
```

Two compounding causes, both measured:

1. **Misprojection.** Projected bank = first-MoE-layer bytes x n_layer. Ornith-35B is a UD-style
   mixed quant: blk.0's `ffn_down_exps` is q6_K (220.2 MB) while the 40-layer mean is 185.6 MB
   (`o35b-tensor-dump.txt`, ornith-bar lane). Projection 522,190,848 x 40 = **20.89 GB** vs the
   exact bank **19,503,513,600 B = 19.50 GB** (+7.1% phantom bytes). The error is not even
   safely conservative: on Qwen3.6-35B the same projection *under*-estimates (15.3 vs 15.60 GB
   exact) because its blk.0 is lighter than average.
2. **The 0.80 x free budget.** Free at decision ≈ 23.93 GB → budget 19.1 GB, i.e. ~4.8 GB of a
   24 GB card reserved as headroom. Measured non-weight need beside the fully-resident model at
   board shape is ~1.7 GB (peak 21394 MiB − weights 21.156 GB − baseline); even serve c=8 peaks
   23220/24463 MiB. The 20%-of-free reserve is calibrated for nothing on this card.

Even the exact 19.50 GB bank fails the old 19.1 GB budget; the misprojected 20.89 GB fails it
comfortably. Meanwhile the whole file is 21.156 GB (experts 19.50 + trunk/embd/output 1.65)
against 23.93 GB free — llama holds the same bytes resident on this card. That is the entire
0.72x story: a budget default, not an engine limit.

## 2. Ornith-35B residency sweep — interleaved x5, same session

Board shape (pp512.txt, NGEN=128), arms interleaved slru→resident→llama per rep (rep loop
outside), N=5 medians, temps 62–72 °C across the cell, busy-proc gate before every arm.
`resident` = `MEMRA_MOE_RESIDENT_GB=21` (clears the old 20.9 GB projected check on the
pre-patch binary; everything else naked). Raw: `residency-sweep.jsonl`,
`{slru,resident,llama}-rep{1..5}.{log,vram}`, console `sweep-console.log`. The llama rows in
the jsonl tail are re-parsed from the raw per-rep logs (llama-bench `-o json` stdout is mixed
with stderr load-noise in the same log; rows carry `"note":"reparsed-from-raw-log"`).

| metric | slru (old default) | resident | llama (plain best) |
|---|---|---|---|
| decode tok/s (N=5 med) | 138.19 [137.7–138.6] | **207.77** [205.6–208.2] | 191.28 [190.0–191.6] |
| prefill tok/s (N=5 med) | 495.2 | **1079.2** [1076–1086] | 3937.2 [3244–3971] |
| run-gen argmax | MATCH 5/5 | MATCH 5/5 | — |
| generated tokens sha | 61c48f62f98e5dbc 5/5 | 61c48f62f98e5dbc 5/5 | — |
| peak VRAM MiB | 19986 | 21394–21900 | 20822 |

- **resident/slru decode = 1.504x (+50.4%), prefill 2.18x (+118%)** — the lever was priced at
  +39% decode; it over-delivers, and the SLRU path was throttling prefill too.
- **resident/llama decode = 1.086x** (ranges cleanly separated) — residency alone moves the
  0.72x decode ratio past the priced ~1.0x parity.
- Bit-exactness: all ten memra runs (both arms) and all post-patch runs emit the same token
  stream (sha 61c48f62f98e5dbc), which is also the ornith-bar lane's KQRP-sweep sha —
  cross-lane, cross-config identity on the same bytes.

## 3. What shipped: RESIDENT-IF-FITS with measured headroom

`build_dev_exps` policy change (config/policy-level, one function + one call site — no kernel or
dispatch surgery):

- **Exact accounting**: expert bank = sum of `blk.*._exps.` tensor bytes from the GGUF header
  (zero-copy metadata walk; dense layers naturally contribute 0, fixing the n_layer overcount
  class too). Non-GGUF sources keep the old first-layer upper bound (ST spill profiles load
  tiered and never reach this decision).
- **Honest headroom**: default expert budget = free − (file's non-expert bytes) − reserve.
  Reserve default 2.0 GB = measured ~1.7 GB need (CUDA ctx + KV + workspace at board shape)
  plus margin; serve c=8 validated below. New env `MEMRA_MOE_RESIDENT_HEADROOM_GB`
  (machine-specific VRAM-budget class, per flags doctrine). `MEMRA_MOE_RESIDENT_GB` keeps its
  absolute-override semantics; `MEMRA_MOE_RESIDENT=0` keeps forcing SLRU (rollback seam,
  re-verified post-patch: `post-rollback-seam.log`, 137.09 tok/s = the spill rate).
- Winner is the default: naked Ornith-35B now decides
  `experts 19.50GB + trunk 1.65GB vs free 23.93GB (expert budget 20.27GB) -> RESIDENT`.

Post-patch verification (naked, N=3, `post-default-rep{1..3}.log`): decode 209.79 / 201.47 /
204.30 (median 204.30), prefill 1091.5 / 1084.2 / 1081.1, argmax MATCH 3/3, token sha
61c48f62f98e5dbc 3/3, peak 21394 MiB — matches the resident sweep arm.

Gates on the patched build: `kernel-check` **ALL GREEN** (`kernel-check-post.log`); `run-spec`
self-consistency **PASS at K=1,2,4,8** with the adopted own-trim drafter, RESIDENT decision in
every run, zero fail lines (`post-spec-k{1,2,4,8}.log`); spec K=2 decode at board shape:
282.31 tok/s.

## 4. Supported-model guard: Qwen3.6-35B-A3B UD-IQ4_XS

Does it spill on this card? **No — RESIDENT before and after** (`qwen-decision-lines.log`):

- pre: `per-layer 373MB x 41 layers = 15.3GB vs budget 19.1GB -> RESIDENT`
- post: `experts 15.60GB + trunk 2.60GB vs free 23.93GB (expert budget 19.33GB) -> RESIDENT`

So no residency sweep is owed; the guard is that the policy change is a no-op for it. Verified
(N=3 pre / N=3 post, `qwen-{pre,post}-rep*.log`): identical decision, identical peak VRAM
(18154 MiB), prefill 2219–2377 pre vs 2363–2370 post (within noise), argmax MATCH 6/6.

**Board-row decode guard** (row: 178.2): valid 128-token cells measured 189.47 / 191.59 /
191.38 (N=3, median **191.38**) — above the board row and above main's 2026-07-30 raw cell
(186.0/186.2). No regression. These cells run `MEMRA_PRIME_TOKENWISE=1` because of the
pre-existing branch finding below (decode gen-only rate is prime-mode-independent).

### Branch finding (pre-existing on restructure/public-split, exposed by this guard):

Naked on this branch, Qwen3.6-35B greedy-emits `"\n"` + EOS after 2 tokens on the pp512 prompt
(`qwen-pre-rep*.log`, `[stop: Eos]`, tokens `[198, 248046]`), while main (85ab7b96 binary,
`BW24_*` env) generates 128 tokens starting `[365, 33682, 18, 17, ...]`. Bisection receipts:
prompt token ids are IDENTICAL across binaries; the run-gen argmax gate MATCHes (argmax=365)
in the failing run itself; `MEMRA_FAST=0` (oracle) reproduces main's stream; and
**`MEMRA_PRIME_TOKENWISE=1` alone restores main's exact stream on the fast path** — the flip
lives in the batched/concat prime path's last-position logits (near-tie first token), not in
decode and not in this lane's change (probed on the pre-patch binary; Ornith is unaffected —
its stream is cross-lane sha-identical). Ornith-35B `[stop: MaxNew]` throughout. This is a
public-split correctness item to run down separately: the run-gen argmax gate does not cover
the batched-prime last-position logits that seed generation.

## 5. Serve check — c=8, new default, no OOM

One short serve run (`serve-c8-server.log`, `serve-c8.jsonl`): memra-server naked (RESIDENT
decision in-log, decode chunk cap 8), `load-serve.py` c=8, 32 requests x 128 tokens:
**32/32 OK, 0 errors, agg 299.07 tok/s, p50 3.43 s, no OOM lines, peak 23220/24463 MiB**
(1.2 GiB headroom at serving ctx). The #43 greedy c1-vs-c16 batch-isolation caveat stands as
documented there — out of scope here, not touched.

## 6. The re-priced #44 gap (Ornith-35B vs llama, same session)

- **Decode: closed and inverted.** 0.72x → **1.086x plain** (N=5, same-session); with the
  adopted drafter K=2, 282.3 tok/s ≈ 1.48x llama's plain decode.
- **e2e (board shape 512+128):** memra 1.082 s (prime 0.472 + decode 0.610) vs llama 0.799 s
  (512/3937 + 128/191.3) = **0.74x** (was 0.41x). With spec K=2: ~0.93 s → 0.86x.
- **Prefill is now the whole remaining gap:** 1079 vs 3937 = 0.274x (was 0.125x). Against the
  #44 targets (pp512 ≥ 2134 p1 / 4563 p2 / 5753 p3), the needed multiplier drops from
  4.3–11.6x to **2.0–5.3x**, and the decode side of the bar is already banked. The Q4_K expert
  prefill lane (#44 lever 4) is the binding piece; residency (lever 3) is done.

## Files

`run-residency-sweep.sh`, `run-qwen-check.sh`, `residency-sweep.jsonl` (N stated per row),
`qwen-check.jsonl`, `token-hashes.log`, `qwen-decision-lines.log`, per-run `*.log`/`*.vram`
(peak sampler, 1 s), `smoke-{default,resident}.log`, `post-default-rep{1..3}.log`,
`post-rollback-seam.log`, `post-spec-k{1,2,4,8}.log`, `kernel-check-post.log`,
`qwen-post-tokenwise-rep{1..3}.log`, `serve-c8-server.log`, `serve-c8.jsonl`,
`serve-c8-per-request.jsonl`, `serve-c8.vram`, `sweep-console.log`.
