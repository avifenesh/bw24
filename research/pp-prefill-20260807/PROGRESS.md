# lane/pp-prefill-serve — the PP-2 PREFILL serving bill

**Mission:** Step-3.7-Flash over PP-2 prefills at **90.9 tok/s** on pp4096
(`research/step-sku-20260807/raw/capacity-20260807T075551Z.log`, N=5, spread 0.12%) because the
pp door is eager-decode only. At the 89.5:1 prompt-heavy traffic ratio that caps the SKU at ~$2/day;
every 1K tok/s of sustained prefill ≈ $18/day. Target: multi-thousand tok/s class.

Box: 2x RTX PRO 6000 Blackwell Server 96GB (`18.195.123.14`), PP-2
(`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`), artifact `~/step37/models/step-3.7-flash/`
(IQ4_XS 3 shards + MTP Q8_0). Every GPU window under `flock /tmp/memra-gpu.lock` (shared with
pp2spec + tick-seg lanes). Raw receipts land in `raw/` here; `.nsys-rep` stays in `/tmp` on the
box, never in git.

---

## Increment 0 — structural reading (code facts, established before the profile)

Read first: `research/step37-p2-20260806/PROGRESS.md` (bring-up), `research/pp2-batch-20260806/RESULTS.md`
(the paid decode side of this bill), `research/pp2-hardening-20260806/PROGRESS.md` (P2P verdict,
refusal audit), `research/step35-chunkfix-20260807/PROGRESS.md` (seq_end chunk-invariance law).

### Fact 1 — the prime path has NO pp arm and FAILS OPEN over a sharded split

`prime_cache` / `prime_chunk` (`crates/memra-engine/src/hybrid_forward.rs:407-860`) contain zero
`pp::` references. The 2026-08-06 hardening audit fixed the four fail-open doors
(`decode_step_batch`, `decode_step_dc`, graph capture, spec verify — all now call
`pp::refuse_unsplit_if_remote`) but **prime was never in that audit**. Under
`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1` with the sharded loader (the Step SKU's only placement):

- every prime chunk walks all 45 layers on the PRIMARY engine's stream (dev0);
- layers 22-44's trunk weights (attn, norms, shexp, router) live on dev1 → peer-read per GEMM.
  At m=4096 the weight read amortizes over the chunk (~2 GB of stage-1 trunk per chunk ≈ ~36 ms
  at the measured 56 GB/s P2P — real but not the 45 s), unlike decode where it was the 28x cliff;
- layers 22-44's KV (allocated on dev1 by `pp::new_cache`) is peer-WRITTEN by the append and
  peer-read by the FA view kernels;
- **dev1 contributes zero FLOPs to prefill.** The second card is a KV closet.

### Fact 2 — step35's MoE prefill rides the PER-TOKEN sequential expert loop

`moe_ffn_inner` dispatch (`hybrid_forward.rs:2005-2560`) for step35 at prefill t:

| arm | gate | step35 verdict |
|---|---|---|
| `moe_ffn_pairs` (one launch per proj for ALL pairs) | `cfg.sigmoid_router().is_none()` | **DENIED** (sigmoid router, and clamp layers 43/44 have no fused clamp form) |
| `moe_ffn_dev` (device router, zero-DtoH) | `dev_ok` requires `sigmoid_router().is_none()` | **DENIED** (softmax-only device router — the M3 74602-vs-92 lesson) |
| `moe_ffn_grouped` (A2: group tokens by expert, m=m_e GEMMs) | `MEMRA_MOE_GROUPED=1` env | **OFF by default** |
| per-token sequential loop | fallback | **THIS RUNS** |

The sequential loop per token per MoE layer: `quantize_q8_1_view` + 8 experts x
(`moe_cached_gemm_q8` gate + up + `ffn_act_lim` + `quantize_q8_1` + down + `axpy_into`)
≈ ~49 kernel launches per token-layer at m=1, **unless** all 24 of the token's blocks are
SLRU-resident, in which case `moe_gdec_token_q8` folds it to ~3 launches — but the model does NOT
go resident on this box (101.07 GB experts vs 94.96 GB budget → SLRU, boot receipt), and a token
needs ALL 8 experts x 3 projections resident to take gdec: at the measured ~96% per-block
steady-state hit rate, P(all 24) ≈ 0.96^24 ≈ 0.37, so most tokens fall through to the m=1 loop
with H2D staging of misses.

Launch-count arithmetic at pp4096: 42 MoE layers x 4096 tokens x O(3..49) launches per token-layer
= **O(0.5M..8M) kernel launches per prime**, every expert GEMM at m=1 with zero operand reuse.
45.06 s / 4096 tokens / 45 layers ≈ **244 us per token-layer** — the decode shape, at prefill.
This is why 90.9 tok/s prefill sits within 3x of the 34 tok/s decode: for the dominant MoE cost,
prefill IS per-token decode today.

By contrast `moe_ffn_grouped` runs ~288 active experts x 3 GEMMs at m_e ≈ 114 (avg over
4096x8/288) per layer ≈ ~1K launches/layer of real GEMMs, reading each expert block once per
layer-chunk instead of once per token-hit.

### Fact 3 — the chunk-invariance laws that bind any prefill change

- step35 kernel selection MUST key on the request's `seq_end`, never a chunk-local `t_kv`
  (`step35-chunkfix`: `P ≡ 0` by construction; `chunkinv35` is the gate; `tickinv` may still be
  red on this branch — the tick-seg lane owns it, coordinate via gate status only).
- `moe_ffn_grouped` as written routes via `e.matmul(&m.gate_inp, z, t)` — the cuBLASLt router GEMM
  is **m-DEPENDENT** (research/concat-prime-exact-20260802), so chunk size would steer router
  logits → expert selection → chunk-dependent text: the exact class chunkinv exists to kill.
  Any grouped adoption must route selection through the m-invariant `router_gemv`
  (`router_prefill_exact_on()`, already the sequential path's default) so selection is
  bit-identical to the sequential arm and chunk-size-invariant by construction.
- Sequential-vs-grouped expert math differ in class (q8 dp4a vs f32 dequant qmatvec — the
  documented ~3.4e-4 t>1 mismatch; `MEMRA_MOE_Q8=0` restores byte-identity). A dispatch-class
  change on the served prefill path needs the full exactness battery + before/after numbers,
  and must be uniform across all chunks of a request.

### Fact 4 — what the decode side of this bill already proved (reusable)

- The `[B, n_embd]` boundary transfer is ~free (pp2-batch: split costs 0.5-1.5% at B=4-16;
  transport alone 0.986-0.997x). At prefill the payload is `[chunk_t, n_embd]` f32 = 64 MB at
  4096x4096 — at 56 GB/s uni ≈ 1.1 ms per boundary crossing, trivially hidden by chunk compute.
- `PpNRt` already has per-stage engines/streams/contexts, grow-only boundary slots (`tx`/`rx`
  take a payload element count — the batched arm already sends `b_n * n_embd`), and the
  slot first-use ordering + publish_to laws are receipted. A chunked prime split reuses all of it.
- SGLang #33666 law: per-stage resources budget on the stage's OWN layer slice; TRT-LLM #16170
  law: drain sends before blocking in compute, missing-relay = loud error.

## Increment 1 — anatomy profile (RUNNING)

`anatomy-pp4096.sh` (committed here) launched detached on the box 2026-08-07T11:41Z, queued
behind a co-tenant chunkinv window per flock discipline. nsys 2026.1.3 installed
(`/opt/nvidia/nsight-systems/2026.1.3`). Two arms in one lock hold:
1. nsys-traced ppprime (1 warmup + 1 rep), `--trace=cuda --sample=none`, `.nsys-rep` in /tmp;
2. untraced control (2 reps) — the nsys-overhead check against the 90.9 baseline.

Extraction: `cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum` CSVs → `raw/`.

**Pre-registered predictions (written before the profile is read):**
- P1: GPU kernel time is dominated by m=1-class expert kernels (`qmatvec`/`moe_*_q8`/dp4a
  family), not by MMQ prefill GEMMs and not by `cudaMemcpyPeerAsync`.
- P2: a large fraction of the 45 s is NOT covered by GPU kernel time at all (launch/host gaps —
  the per-token loop's ~5-6 us/launch shape), visible as cuLaunchKernel dominating the API sum.
- P3: H2D memcpy (SLRU staging of expert misses) is material (multi-GB) but not the majority.
- P4: dev1's kernel time ≈ 0 (KV appends/reads only — no compute split exists).

If P1/P2 hold, the biggest single lever is the MoE prefill dispatch shape (grouped/batched expert
GEMMs for the sigmoid-router arch), with the PP chunk pipeline as the second multiplier on top —
and the increment order below gets re-scoped accordingly per the stop-and-report clause.
