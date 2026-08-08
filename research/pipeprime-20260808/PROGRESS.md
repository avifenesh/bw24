# pipeprime — chunk-pipelined PP-2 prime

Branch: `lane/cx-pipeline-prime`  
Base: `2fc86b14` (Lever B walker + step35 batched decode merged)

## Mission

Overlap stage 0 of prime chunk N+1 with stage 1 of chunk N. The serial Lever-B split is
the arithmetic oracle and rollback path:

- serial split baseline: pp4096 **266.1 tok/s** (N=5, 22+23 layers);
- pipeline seam: `MEMRA_PRIME_PIPE=0`;
- exactness bar: every returned f32 bit and the primed-cache continuation must match the
  serial split;
- target class: balanced-stage overlap, approximately 400-500 tok/s if scheduling and
  transport costs remain hidden.

## Increment 0 — structural reading and ordering verdict

### Existing walker

`prime_chunk_ppn` already provides the stage-local pieces needed by a chunk pipeline:

- stage 0 owns embed, its `pos_d`, layers `[0, fence[1])`, and its KV/recurrent writes;
- stage 1 waits the boundary publication, owns its `pos_d`, remaining layers and KV
  writes, then runs output norm + head;
- boundary transport is `PpNRt::tx/rx`, with two persistent grow-only slots available
  under `MEMRA_PP_OVERLAP=1`;
- `fence_stages_behind` runs at walker entry and `publish_to` orders the last-stage
  device results before the caller consumes them;
- per-device prime slabs and stage-owned caches are already in place.

The current `prime_cache` chunk loop is still strictly serial: a complete
`prime_chunk_ppn` returns before the next chunk starts. It also relies on `cache.pos` as
the current chunk base and advances it in the per-chunk epilogue, so pipelining must make
each chunk's base explicit rather than letting stage 0 of N+1 observe stage 1 of N's
later host-side position update.

### Old deferred-pipeline flake

The 2026-08-02 deferred decode arm recorded a low-rate cross-device flake and a much
higher same-device flake. The later #87 failure supplied the allocator mechanism:
stage-allocated buffers were freed on a stage stream while the primary stream still had
queued reads, then reused and overwritten by later stage work. `PpNRt::fence_stages_behind`
now orders every stage stream behind the caller before new stage allocations, and the
Lever-B walker invokes it at entry. That closes the old allocator reverse-publication
mechanism for the serial walker and for caller-visible buffers crossing chunk calls.

Chunk pipelining still needs the separate boundary-slot anti-reuse edge. The existing
transport already expresses it:

1. stage 0 TX waits the selected slot's prior `ev_rx`;
2. stage 1 RX waits `ev_tx`, copies the slot into stage-owned work, then records `ev_rx`;
3. only after that copy may the same slot be overwritten by a later chunk.

Therefore the old flake mechanism is **dead under the new entry fences**, and no new
global caller/stage fence is required. The pipeline must preserve the existing
write-after-read event chain. If stage 1 is changed to consume the persistent slot
directly instead of the copied work buffer, `ev_rx` must move after the stage-1 layer
range; recording it at RX would reopen the exact anti-reverse-publication hole.

### Implementation shape

The minimum safe shape is a PP-2-specific chunk scheduler:

- double-buffer boundary slots by enabling slot alternation for this path;
- enqueue stage 0 chunk N, publish its boundary slot;
- enqueue stage 1 chunk N after that slot's `ev_tx`;
- before draining stage 1 N, enqueue stage 0 chunk N+1 into the other slot;
- on slot reuse, rely on TX's wait for that slot's prior `ev_rx`;
- keep stage-local chunk bases/positions explicit and preserve per-stage stream order for
  KV writes;
- drain each stage-1 result in original chunk order, copying its hidden stack and retaining
  only the final chunk's logits/h_seed exactly as the serial loop does.

## Gate plan

| Gate | Required verdict |
|---|---|
| `ppsplit` serial split vs pipelined | bit-identical plus pipeline-overlap counter advances |
| `chunkinv35` / `tickinv35` + canaries | pass / teeth |
| `kernel-check` | ALL GREEN |
| `run-gen` over PP-2 | argmax MATCH |
| `run-spec` K=1..8 | pinned acceptance, all pass |
| pp4096 soak | at least 200 pipelined primes, zero divergence/fault |
| perf | pp512/2048/4096 serial-vs-pipeline N=5 interleaved, one flock hold |
| serve | 4k TTFT receipt |

Raw logs will live under `research/pipeprime-20260808/raw/`; every claimed median will
state N and thermal regime.
