# lane/cx-prime-batch — step35 cross-request batched prime

Branch base: `13f5ddb8` (train tip after Lever B and step35 batched decode).
Preferred rig: box2 `13.59.112.147`, 2x RTX PRO 6000 Blackwell Server Edition, PP-2
`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`.

## Read conclusions

- `worker.rs` already forms concurrent fresh-prompt batches and calls
  `prime_cache_batch`; step35 reaches that call, the engine refuses, and the worker
  restores the queues and serves serial primes. The receipt is the existing
  `[prime-batch] failed ... single primes serve` line.
- The generic concat attention core is not valid for step35. It assumes scalar model
  geometry, while step35 requires per-layer query-head count, partial/dual-base RoPE,
  a 512-token 3:1 SWA pattern, and a separate head-wise gate.
- `prime_chunk_ppn` defines the PP contract the new path must preserve: stage-scoped
  layer ranges and engines, `fence_stages_behind` at entry, per-stage position buffers,
  KV appends on the layer-owning device, boundary transport of only the materialized
  residual, last-stage epilogue, and `publish_to` before returning device outputs.
- The chosen mechanism is a dedicated step35 concat-prime range walk. Weight-streaming
  norms/projections/FFN run at `m=sum(T)`; each sequence's attention core and KV append
  retain its own positions, cache, and absolute request `seq_end`. Under PP-2, each
  range runs on its owning stage.

## Increment 1 — exact gate, registered RED

`tools/step35-prime-batch-gate.sh` runs `prime-batch-gate` over PP-2 with two uneven
prompts of 520 and 537 tokens, so both cross the SWA window. It compares serial vs
batched last-row logits, `h_seed`, full hidden stacks, and four teacher-forced decode
logit vectors bit-for-bit. The decode replay reads the primed KV, so wrong per-stage
cache writes cannot hide behind matching returned logits.

Two liveness counters prevent a vacuous pass: one must prove the dedicated step35
batch path ran, and one must prove its PP stage split ran. At the base commit the engine
refuses step35 before either advances, so the naked gate is RED by construction.

Box2 red receipt: `raw/primebatch35-naked-20260808T101939Z.log`. The two serial PP-2
reference primes completed, then the batch call returned the named
`prime_cache_batch: step35 has no batched prime core` error. Both cards were 0 MiB at
entry and exit.
