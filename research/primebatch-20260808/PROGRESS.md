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

## Increment 3 — batch the remaining attention weight streams

The exact-first implementation deliberately kept the step35 gate projection and `wo`
per sequence. Its clean paired N=5 baseline showed that this was correct but left nearly
linear work: at T=520, B=2 moved 3993.8 -> 3933.5 ms (+1.5%) and B=4 moved
7981.0 -> 7829.8 ms (+1.9%). Raw alternating pairs and thermal snapshots:
`raw/primebench35-paired-20260808T105411Z.log`.

At fixed layer, `attn_gate` and `wo` have the same geometry for every request. The path
now projects Q/K/V/gate together at `m=sum(T)`, keeps Q/K normalization, partial RoPE,
SWA/cache attention, and head-gate application per sequence, concatenates the gated
attention rows, and runs one `wo` at `m=sum(T)`. This removes two repeated weight streams
without mixing request state.

The promotion is still byte-exact: `raw/primebatch35-naked-20260808T110701Z.log` repeats
the complete PP-2 gate with 0 differing logits, `h_seed`, hidden, or teacher-forced
decode bits, and both liveness counters at 1. Both cards entered and exited at 0 MiB.

## Final report

### Mechanism chosen

A dedicated step35 concat-prime range walker, not the generic batch attention core:

- batched at `m=sum(T)`: embedding, layer norms, Q/K/V/gate projections, `wo`, residual
  and post-attention norm, dense/MoE FFN, and output norm;
- per request: Q/K head normalization, partial/dual-base RoPE, SWA/global attention
  view, KV append/length publication, and head-gate application;
- per PP stage: the owning engine executes only its `[lo, hi)` layers, with
  `fence_stages_behind`, residual transport, stage-owned KV writes, last-stage
  epilogue, and `publish_to`;
- fresh complete prompts only. Carried step35 dark-lane batches still refuse and use
  the existing single-chunk fallback because they need per-request queued-after
  metadata.

### Final gate table

| gate | final verdict | receipt |
|---|---|---|
| registered RED at base | named step35 batch refusal after two serial PP-2 references | `raw/primebatch35-naked-20260808T101939Z.log` |
| `pbatch35`, B=2 T=520/537, 4 decode steps, PP-2 | **PASS**: 0 differing logits, `h_seed`, 4,329,472 hidden f32 values, or teacher-forced decode logits; batch/split liveness 1/1 | `raw/primebatch35-naked-20260808T110701Z.log` |
| `pbatch35c` | **CANARY OK**: `MEMRA_STEP35_PRIME_BATCH=0` restores the refusal | `raw/primebatch35-canary-20260808T103938Z.log` |
| `kernel-check` | **ALL GREEN**, 0 FAIL lines | `raw/box1/kernel-check-20260808T115107Z.log` |
| `run-gen`, PP-2, 64 tokens | **MATCH x2**: prefill/decode 6776; batched-prime/tokenwise 6776 | `raw/box1/run-gen-20260808T115107Z.log` |
| `run-spec`, PP-2 + drafter | **SELF-CONSISTENCY PASS K=1..8** | `raw/box1/run-spec-20260808T115107Z.log` |
| `tickinv35`, final code | **TICK-INVARIANT**: all five budgets and three LCP split points exact, including the 77-call arm | `raw/box1/tickinv35-final-summary-20260808T115107Z.log` |
| `tickinv35c` teeth | **CANARY OK**: every segmented arm differs under the legacy call-local seam | `raw/tickinv35c-summary-20260808T104728Z.log` |
| real `memra-server`, c=2/c=4 | **PASS**: 6/6 responses byte-identical to c=1; successful `[prime-batch] B=2` and `B=4`; no fallback | `raw/box1/serve-primebatch-gate-20260808T115711Z.log`, `raw/box1/serve-primebatch-server-20260808T115711Z.log` |

The final box1 battery entered and exited with both cards at 0 MiB. The server gate did
the same. Coordination moved the final runner/server checks to box1 after the lane inbox
reserved box2 for the serving trial.

### Concurrent-prime wall time

Box2, PP-2 dev01, T=520 per prompt, N=5 paired runs per cell, alternating arm order,
2370/2272 MHz fixed clocks, 38-42 C, both cards 0 MiB before/after:

| prompts | serial primes median | one concat prime median | delta | pair wins |
|---:|---:|---:|---:|---:|
| 2 | 3995.951 ms | 3898.753 ms | **+2.5%** | 5/5 |
| 4 | 7972.470 ms | 7791.082 ms | **+2.3%** | 5/5 |

Raw pairs: `raw/primebench35-final-20260808T110945Z.log`. This is a modest but
consistent win at the correctness geometry (past the 512-token SWA window); the lane
removes the host-level single-prime fallback but does not claim sublinear total compute
at T=520. The short real-server smoke additionally logged B=2/68 tokens in 340.8 ms and
B=4/136 tokens in 591.5 ms, but those are single runs and are not used as performance
claims.

The original final timing log was printed in full to the local command transcript. Box2
stopped accepting SSH during its serving handoff before rsync, so the committed copy is
a verbatim reconstruction from that captured output; the raw five pairs, thermal
snapshots, exactness rows, exit codes, and cleanup lines are preserved.

### Commits

| commit | increment |
|---|---|
| `f4d459c4` | exact PP-2 sibling gate registered red |
| `a41b4ada` | dedicated fresh-prompt step35 batch + PP stage walk |
| `62d78b27` | alternating N=5 raw-pair benchmark output |
| `7d36a61c` | batch step35 gate projection and `wo`, exactness re-gated |
