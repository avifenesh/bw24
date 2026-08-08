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

## Increment 2 — dedicated step35 concat prime, exact + PP-2 green

Mechanism: `step35_prime_cache_batch` is a step35-specific range walker rather than an
admission into the generic concat attention core. It concatenates the prompt rows for
embedding, norms, Q/K/V projections, and FFN work. Each sequence is then split back out
for the existing `step35_attn_pre_wo` core and `wo`, preserving the layer's query-head
count, partial/dual-base RoPE, request-level `seq_end`, SWA view/fence arithmetic,
head-wise gate, and isolated KV append. The output head remains one-row per request,
matching the serial prime's arithmetic.

The PP-N wrapper mirrors `prime_chunk_ppn`: fence stage streams behind the caller,
allocate positions and execute each layer range on its owning stage engine, transport
only the concatenated residual at each boundary, run the epilogue on the head stage,
and publish its device-resident outputs back to the caller. Interactive serving already
drains every eligible fresh prompt in full before `prime_cache_batch`; carried step35
dark-lane batches refuse and keep their existing single-chunk fallback because that path
would need per-request queued-after metadata.

Box2 receipts:

| gate | result | evidence |
|---|---:|---|
| `pbatch35` B=2, T=520/537, PP-2 | GREEN | `raw/primebatch35-naked-20260808T103724Z.log` |
| serial vs batch logits | 0 / 257,792 differing f32 | same |
| serial vs batch `h_seed` | 0 / 8,192 differing f32 | same |
| serial vs batch hidden stacks | 0 / 4,329,472 differing f32 | same |
| 4 teacher-forced decode steps/sequence | 0 differing logits; streams match | same |
| dedicated batch / PP split liveness | 1 / 1 | same |
| `pbatch35c` (`MEMRA_STEP35_PRIME_BATCH=0`) | CANARY OK, named refusal | `raw/primebatch35-canary-20260808T103938Z.log` |

The naked run's entry snapshot was not idle (`87537 / 56081 MiB` in use); it is a
correctness receipt only and will not be used for performance. It exited with both cards
at 0 MiB. The canary run entered and exited at 0 MiB on both cards.
