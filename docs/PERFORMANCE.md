# memra performance — mechanisms, history, receipts

The README carries the current boards (generated from
[`research/tune-data/current-board.json`](../research/tune-data/current-board.json) by
[`tools/update-perf-board.py`](../tools/update-perf-board.py)). This document carries the
depth behind them: how each cell moved, what was refuted along the way, and where the raw
runs live. The measurement protocol is [`research/benchmarks.md`](../research/benchmarks.md);
competitor engines run at their swept best
([`docs/COMPETITOR-SETUP.md`](COMPETITOR-SETUP.md)); the H100 lane's append-only evidence
ledger — every promoted config, every mechanism refutation — is
[`ARCHITECTURE-H100.md`](../ARCHITECTURE-H100.md).

Every published median states its N and thermal regime; every perf claim is a same-session
interleaved pair (cross-run and cross-day comparisons are clock-drift-invalid, including the
competitor denominator).

## Standing (2026-08-02)

Ten supported models on the 5090, all fully gated; MTP-spec cells run 1.06–2.3x llama.cpp
(one Gemma near-parity cell, 0.98x, still open), plain cells sit at the DRAM wall or above.
Newest in: **Ornith-1.0-35B** — AUTO-KQUANT (k-quant expert banks join the grouped-f16
prefill lane by default: board-2048 prefill 3.14x) stacked on resident-if-fits residency,
and its own-gen drafter cleared the deployment bar on every prompt class (best-vs-best e2e
1.31/1.14/1.12x, [`research/q4k-expert-prefill-20260802/`](../research/q4k-expert-prefill-20260802/)).
On the H100, a full per-model board against vLLM 0.26: **no end-to-end losses** — seven wins
of seven (1.02–1.81x); decode wins 7 of 7. Multi-user serving measured on 3×H100: 1,477 tok/s
managed fleet, chaos-tested. Trimmed MTP drafter heads are published ready-to-use at
[huggingface.co/Avifenesh/memra-bench](https://huggingface.co/Avifenesh/memra-bench).

## The serving exactness arc — a defect the discipline caught

The batched F32 prefill router GEMM was m-dependent, so under cross-request prefill batching
a MoE request's own expert routing could change with its co-arrivals — a real serving defect.
The fix routes prefill through the decode path's m-invariant router/gate kernels
(`MEMRA_ROUTER_PREFILL_EXACT`, default ON), and a bit-identical batched twin then recovered
70% of the prefill it cost. The serving contract — greedy serving is isolated-identical under
concurrent load; a request's tokens never depend on its co-arrivals — is now explicit and
gated, not assumed. Receipts:
[`research/concat-prime-exact-20260802/`](../research/concat-prime-exact-20260802/),
[`research/fast-router-20260802/`](../research/fast-router-20260802/).

The H100 35B cell tells the arc in one row: it slipped from 218/1.02x to dead-even when the
concat-prime fix landed (m-invariant prefill router + shexp gates, −13% Hopper expert
prefill) so a session's routing no longer depends on its co-arrivals — then a bit-identical
batched twin of the exact kernel recovered 82% of that cost (8136 pp2048,
kernel-check-pinned mism=0; `MEMRA_ROUTER_BATCH=0` is its perf-only rollback seam) and the
row reached 217/1.01x, ahead again with the contract held — then jumped to 226/1.05x when
direct-from-quant tile loaders reached ~100% of the expert bank (IQ4_XS/IQ3_S added to
Q4_K/Q6_K) and the single-kernel grouped GEMM flipped past cublas as the Hopper default
(round 55: mode-2 prime 13164 vs cublas 8627 tok/s, +52.6% interleaved x5, bit-identical
tiles by construction; expert prefill 8136 → 13258 on the row). Decode never moved; the
dense 27B row was never on the path (re-cell bit-stable). Receipts:
[`research/router-fix-recells-20260802/`](../research/router-fix-recells-20260802/),
[`research/fast-router-20260802/`](../research/fast-router-20260802/),
[`research/q35-recell-final-20260802/`](../research/q35-recell-final-20260802/),
[`research/h100-flip-full-20260802/`](../research/h100-flip-full-20260802/).

## RTX 5090 vs llama.cpp

### Depth behavior

Depth is part of the contract: at 6.3k-token context every plain-decode lead holds
(1.02–1.07x). The depth rows live in `current-board.json` (`plain_decode_depth`), from the
same validity-gated cold-start rebaseline as the 512-context rows.

### Prefill, root-caused

5090 prefill trails llama.cpp (0.59–0.78x), root-caused: llama benches NVFP4 prefill at
W4A4 (FP4 activations), a numeric class memra's exactness gates reject. Output quality
outranks the prefill column.

### Speculative protocol notes

The three spec columns are three prompt classes: short code / medium code (greedy) / long
agentic (temp 0.7, distribution-exact rejection sampling). One asterisk: the 35B short-code
llama bar (251.4) is an EOS-suppressed continuation (the raw short prompt EOSes at 1 token)
and is not a clean win basis. Every spec row uses one trimmed draft built by the standard
regime ([`docs/DRAFT-REGIME.md`](DRAFT-REGIME.md)); prebuilt drafts live in the
[bench repo](https://huggingface.co/Avifenesh/memra-bench), or build your own:

```bash
./target/release/frspec-owngen model.gguf ranks.gguf 32768        # ranks from the model's OWN generations
tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf  # extract + trim + quantize
```

### Gemma-4 (QAT Q4_0)

Same protocol, own campaign log
([`research/gemma4-bringup/rig5090-gemma4.jsonl`](../research/gemma4-bringup/rig5090-gemma4.jsonl));
cells re-paired best-vs-best with llama's own draft depth swept per cell. Highlights
(full tables and the per-cell archaeology live in the campaign log):

| Cell | memra | llama.cpp | Ratio |
|---|---|---|---|
| 12B MTP spec, 1.7k ctx (K=4 + own-gen trim) | 269.3 | 175.1 | **1.54x** |
| 12B MTP spec, chat (K=4 + own-gen trim) | 240.9 | 172.4 | **1.40x** |
| 31B MTP spec, chat (K=5 + own-gen trim) | 124.3 | 99.0 | **1.26x** |
| 31B MTP spec, 1.7k (K=6 + FR trim) | 97.3 | 83.9 | **1.16x** |
| 26B MTP spec, short ctx (K=4 + own-gen trim) | 328.1 | 298.0 | **1.10x** |
| E4B plain, short | 199.9 | 181.0 | **1.10x** |
| 26B plain, 4.9k ctx | 162.6 | 142.0 | 1.14x |
| plain decode elsewhere | — | — | 1.00–1.07x |

What buys the margins: an FP8 (e4m3) KV cache, occupancy-tuned attention tiles, wide-load
Q4_0 expert dots, FR-Spec drafter trims (150 MB → 18 MB at unchanged acceptance), a
per-model adaptive-K floor (+20% on 12B/31B spec at unchanged exactness), and an in-round
draft-confidence cut. One near-parity cell remains (26B 1.7k spec, 0.98x). Gemma plain
margins are thin where both engines sit at the DRAM wall (1.02–1.06x). Exact prompts and
llama.cpp's swept-best flags: [`docs/COMPETITOR-SETUP.md`](COMPETITOR-SETUP.md).

### Ornith-1.0-9B (Q8_0)

Supported model #8, and the first community post-train to clear the deployment bar
(best-vs-best e2e ≥ 1.1x on every prompt class). Each engine at its measured best on the
same GGUF: memra runs the regime drafter at K=3 (donor-block own-gen trim — the published
model ships no MTP head, [`docs/DRAFT-REGIME.md`](DRAFT-REGIME.md)); llama.cpp's best is
plain — its draftless speculative doors are structurally broken on this arch (M-RoPE
position faults, screened in the receipts). Interleaved N=3 medians, same session
([`research/ornith-bar-20260802/`](../research/ornith-bar-20260802/)):

| Prompt class | memra e2e, 256 tok (spec K=3) | llama.cpp e2e (plain best) | Ratio |
|---|---:|---:|---:|
| code short | 1.395 s | 3.084 s | **2.21x** |
| code medium (1.8k ctx) | 2.050 s | 3.429 s | **1.67x** |
| agentic long (6.3k ctx) | 2.984 s | 4.376 s | **1.47x** |

Spec-vs-own-plain: 2.16/1.77/1.70x at 47-61% acceptance
([`research/ornith-drafters-20260801/`](../research/ornith-drafters-20260801/)); build the
drafter with the standard two commands in [`docs/DRAFT-REGIME.md`](DRAFT-REGIME.md)
(donor-block variant — donor pairs and receipts documented there).

### Ornith-1.0-35B (Q4_K_M MoE)

Supported model #9. A three-lane arc took it over the bar: resident-if-fits residency
(the 19.5 GB expert bank fits 24 GB — +50% decode over the old spill default), the
adopted own-gen donor-block drafter (K=2, 60-68% acceptance), and grouped-f16 expert
prefill (the Q4_K/Q6_K bank rides the grouped-f16 lane by default — AUTO-KQUANT then,
the mode-2 default now, same dispatch for this bank: board-2048 prefill 3.14x, pp512 +54%,
decode flat). Best-vs-best per class (memra = spec K=2; llama.cpp = plain, its measured
best on this arch); interleaved N=3, same session
([`research/q4k-expert-prefill-20260802/`](../research/q4k-expert-prefill-20260802/)):

| Prompt class | memra e2e, 256 tok (spec K=2) | llama.cpp e2e (plain best) | Ratio |
|---|---:|---:|---:|
| code short (27 tok) | 1.033 s | 1.357 s | **1.31x** |
| code medium (1.8k ctx) | 1.618 s | 1.837 s | **1.14x** |
| agentic long (6.3k ctx) | 2.738 s | 3.053 s | **1.12x** |

The honest residual: short-prompt prefill (pp512 0.42x llama) — the per-pass k-quant → f16
dequant cost amortizes with prompt length (0.91x at 2048, a 1.06x memra win at 6.3k); the
no-dequant kill (Q4_K/Q6_K expert MMQ tile loaders) is priced, not built. Plain decode runs
1.086x.

## H100 vs vLLM — flip history

The last e2e losses fell in two days. The two MoE expert-prefill cells closed in round 49:
the 35B when the grouped f16 expert lane reached full dequant coverage and became the
Hopper default (expert prefill +53%) — then +63% more at round 55 when direct-from-quant
tile loaders covered the whole bank and the in-house grouped kernel flipped past cublas —
and the 27B via K-quant f16 prefill mirrors (+54% pp2048) plus split-plane decode mirrors
(+16% decode, bit-identical). The 26B cell flipped from dead-even to a win in round 50,
when a cross-box interleaved arbitration caught that same f16g default *regressing* the
gemma expert class −6 to −8% and narrowed it per-model — the board is tuned per dispatch
class, not per flag. The last two decode losses fell to the shexp fused dot and a
router-default re-arbitration on real prompts.

### Refuted: a Q8_0-exact int8 GEMM on Hopper

Per-cell prefill still trails vLLM on the dense models — the int8-GEMM dtype edge: a
Q8_0-exact int8 GEMM is mechanism-refuted on Hopper (per-32-block rescale costs 5.4x naive
/ 17x pipelined; ptxas serializes cross-bank GMMA register reads), so crossing it means
w8a8-class numerics that change model outputs — an accuracy-bar decision with measured
receipts, not an engineering unknown. The e2e board stays loss-free because decode wins
7 of 7 and dominates the p2048/g512 shape. Full refutation ledger:
[`ARCHITECTURE-H100.md`](../ARCHITECTURE-H100.md).

### Shipped on the sm_90a build

FA3-class prefill attention (TMA swizzled ring + wgmma, 4.8x the mma kernel), fused wgmma
GDN chunk kernels with varlen twins, f16 prefill mirrors on the cuBLASLt lane for the
Q8_0/Q4_0/Q4_K/Q5_K/Q6_K classes, K-quant split-plane decode mirrors (bit-identical layout
v2, 27B decode +15-16%), the grouped f16 expert lane (one single-kernel grouped GEMM over
the routed experts with direct-from-quant Q4_K/Q6_K/IQ4_XS/IQ3_S tile loaders and a 3-stage
deep tail — Hopper default for the qwen/silu MoE class, 35B expert prefill +53% at round 49
and +63% again at round 55 when it flipped past cublas; per-model off for the gemma class
after the round-50 regression arbitration), a batched serving decode tick (z-batched
attention + KV append, device-side sampling, lean logits: 654-659 tok/s/replica, +25-36%),
an MTP speculative serving fast lane (1.82x plain serving at c=1 on the 27B), cross-request
prefill batching, per-session CUDA-graph decode with kernel-class segment recapture, and
the Hopper wgmma toolkit
([`cu/wgmma_common.cuh`](../crates/memra-engine/cu/wgmma_common.cuh)) — canonical
core-matrix pairings probed for bf16/tf32/s8: one byte-geometry, three MMA kinds.

- Evidence ledger (every verdict and refutation): [`ARCHITECTURE-H100.md`](../ARCHITECTURE-H100.md)
- Flags + promoted defaults: [`docs/FLAGS.md` §7](FLAGS.md)
- One-command battery: `tools/validate-h100.sh <model.gguf> [--quick]`

## Serving performance

Batched decode runs one z-batched tick across sequences, chunked 16-wide on models with a
bit-exact 16-batch kernel class: +18.8% at c=16 on a 5090 single replica (same-mirror
interleaved N=4); 654-659 tok/s/replica on the H100 serving tick (+25-36%). Greedy serving
is isolated-identical under concurrent load at defaults (m-invariant prefill router/gate
kernels, serve-gate c=1-vs-c=16 byte identity). Multi-GPU boxes serve as a replica fleet —
supervisor + admission proxy + load harness, measured at 1,477 tok/s managed on 3×H100,
chaos-tested: kill a replica mid-load and the breaker + supervisor recover in seconds with
only in-flight requests lost. Runbook: [`docs/SERVING.md`](SERVING.md).

## Bring-up notes

- **KAT-Coder-V2.5** — onboarded with zero code change (native `qwen35moe` arch strings;
  argmax + chat gates green — [receipts](../research/onboard-ornith-20260801/)). Decode
  anomaly RESOLVED: its IQ4_XS trunk was riding the f32 oracle path; default dp4a admission
  moved decode +81% to llama parity (1.016x) and flipped its drafter net-positive on
  code-short (1.25x e2e @K=2). The bar-binding gap is now prefill alone (0.169x — the
  IQ4_XS-trunk MMQ port is priced, [receipts](../research/kat-anomaly-20260802/)).
- **Qwen-AgentWorld-35B-A3B** — same-stack by construction (header bytes verified),
  unbenched.
- **Hy3 Layer103.5 overlay** — VRAM→RAM→dual-NVMe spill, correctness-gated end-to-end;
  serves at a 5.13 tok/s N=3 median, tuning toward 10
  ([`docs/HY3-SPILL.md`](HY3-SPILL.md)).
- **MiniMax-M3 REAP50** — safetensors spill; loads + generates, router tuning open.
