# LMCache — Implementation Map

**What it is:** LMCache is a KV-cache *reuse and offload* layer that sits between an inference
server (vLLM / SGLang) and the GPU KV pool. Its job is **cross-request** KV management: hash a
request's token prefix, look it up in a multi-tier store (GPU → CPU → disk), and stream matched KV
back into the paged GPU pool *before/while* the forward pass runs — so a prompt prefix that another
request already computed never gets recomputed. It is wired in as a `KVConnector` (the "vllm
connector") and as an `LMRadixCache` (the SGLang connector).

**Why most of it is irrelevant to bw24:** LMCache's entire value proposition is *amortizing prefill
across multiple requests sharing a prefix*. bw24 is a **single-stream, batch=1, sm_120 GGUF engine**:
one request at a time, no shared prefixes, no disaggregated prefill/decode split, no tensor
parallelism. So the **caching/lookup/eviction machinery has zero single-stream value** — there is no
"prior request" to reuse. What *is* portable is the **mechanical substrate** LMCache built to make
tiering fast: the async load/store CUDA-stream discipline, the layer-wise H2D overlap pattern, the
paged slot-mapping layout, and the "spill KV to CPU/disk" idea (useful only if a single context
exceeds VRAM). None of LMCache's hot path is a GEMM/attention kernel — it is metadata + `cudaMemcpy`
+ stream sync. There are **no wgmma/tcgen05/MMA kernels** here at all.

Source roots:
- vLLM connector: `/home/avifenesh/.cache/uv/archive-v0/EExmZo2-cib0C2ZTLlwRb/vllm/distributed/kv_transfer/kv_connector/v1/` (`lmcache_connector.py`, `lmcache_mp_connector.py`, `lmcache_integration/{vllm_v1_adapter.py,multi_process_adapter.py,utils.py}`)
- SGLang connector: `/home/avifenesh/.cache/uv/archive-v0/tYSYwgZB1kk0A6opiTDTz/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py`

---

## What bw24 could take from LMCache

Ranked by portability × value for a single-stream sm_120 GGUF engine. None of these are kernels —
they are **memory-movement and lifetime-management patterns** that any engine spilling KV off-GPU
or pipelining loads needs.

1. **Async load/store CUDA-stream discipline (deferred sync).** `start_load_kv()` queues H2D copies
   onto a dedicated stream *before* the forward pass; `wait_for_layer_load(layer)` blocks only on the
   layer about to be consumed; `wait_for_save()` drains store stream at forward-context exit. This is
   the cleanest "overlap I/O with compute" template and maps 1:1 onto a Rust+CUDA single-stream
   engine that wants to prefetch a long context from CPU/disk without stalling.
   `lmcache_connector.py:136-197`.
2. **Layer-wise transfer counter for per-layer H2D overlap.** `LayerTransferCounter.wait_until(layer_id)`
   is called by the KV pool *as each layer finishes*, translating into `load_kv_layerwise(layer_id)` on
   a load stream. The result: layer N's KV streams in while layer N-1 computes. Directly portable as a
   "prefetch layer i+1's weights/KV while layer i runs" hook. `lmc_radix_cache.py:37-61, 172-180`.
3. **Paged slot-mapping layout with `-1` sentinel for already-resident tokens.** `slot_mapping =
   cat([-1 * cached_len, fresh_slots])`; `slot = block_id * block_size + block_offset`. The `-1`
   sentinel cleanly marks "don't copy, already on GPU" inside one flat int64 tensor. A good layout for
   bw24's own KV indexing whenever it does partial loads/quantized-KV gathers.
   `lmc_radix_cache.py:127-212`.
4. **In-flight store barrier before eviction (use-after-free guard).** `evict()` calls
   `store_stream.synchronize()` and releases lock-refs on `_in_flight_nodes` *before* freeing slots, so
   an async store can never race a free. This is the exact correctness pattern bw24 needs if it ever
   spills KV asynchronously: never free a GPU slot until the copy off it completed.
   `lmc_radix_cache.py:244-267`.
5. **Chunk-aligned KV granularity (256-token chunks over `block_size` blocks).** KV is keyed/moved in
   fixed chunks (`chunk_size=256`, `blocks_in_chunk = chunk_size / block_size`); partial trailing
   chunks are discarded until full. Fixed-size transfer units make CPU/disk offload alloc and copy
   sizes predictable — useful sizing wisdom even for single-stream spill.
   `multi_process_adapter.py:28-39`; `vllm_v1_adapter.py:299-338`.
6. **GPU→CPU→disk tiering shape & H2D recovery path** *(only if a single bw24 context exceeds VRAM)*.
   `VLLMPagedMemGPUConnectorV2` is the non-blending, non-layerwise connector that simply pages KV
   between the GPU pool and a CPU interim buffer (`cudaMemcpy`), with disk behind a queue. If bw24
   ever serves contexts larger than fits in 32 GB, this "spill cold KV blocks to CPU, fault them back
   on attention" structure is the reference. `vllm_v1_adapter.py:500-541`.

---

## DEAD for bw24

These are multi-request server machinery or hardware-specific paths with **zero** single-stream value.

- **Token-hash prefix matching / cross-request reuse** — the entire premise (reuse another request's
  KV) is meaningless at batch=1 with no prior request. `multi_process_adapter.py:28-39, 192-259`.
- **Disaggregated prefill→decode KV transfer (1P1D via ZMQ)** — `DisaggSpec`, prefill-worker→decode-worker
  KV shipping. Requires a multi-process serving pipeline bw24 does not have. `vllm_v1_adapter.py:89-96, 136-139`.
- **TP-group KV broadcast for MLA** — `tpg.broadcast()` to sync KV across tensor-parallel ranks;
  `world_size > 1` only. Single RTX 5090 = `world_size 1`. `lmcache_mp_connector.py:85-106, 542-548`.
- **Multimodal token hashing** — `hex_hash_to_int16`, `apply_mm_hashes_to_token_ids`: builds cache keys
  for image/MM placeholders so MM prefixes hit across requests. Cross-request + multimodal; bw24 is
  text GGUF, single-stream. `utils.py:66-89`.
- **`discard_partial_chunks` / `SaveSpec.skip_leading_tokens`** — manages "which chunks were already
  saved by a prior batch" so the next request can reuse them. No prior save exists at batch=1.
  `vllm_v1_adapter.py:299-338`.
- **ZMQ disk-offload server + cross-instance store** — `ZMQOffloadServer`, `LookupClientFactory`
  sync/async lookup clients: a network/IPC cache service for many clients. Single-process bw24 needs
  none of it. `vllm_v1_adapter.py:638-642, 632-637`.
- **CacheBlend blending connector (`VLLMBufferLayerwiseGPUConnector`, `enable_blending`)** — recomputes
  attention to *blend* partially-matched cached KV with fresh KV across non-contiguous prefixes; needs
  extra intermediate GPU buffers and only pays off when stitching multiple cached fragments. Dead for
  single contiguous single-stream prefill. `vllm_v1_adapter.py:513-522, 622-630`.
- No wgmma/tcgen05/MMA tensor-core kernels and no AMX paths exist in LMCache to mark dead — its hot
  path is pure metadata + stream-ordered `cudaMemcpy`. (Noted for completeness.)

---

## Subsystem: KV-tiering (KV-cache reuse / offload across requests)

| Technique | How implemented (mechanism) | Kernel / layout / instruction | source file:line | sm120_fit (single-stream value) |
|---|---|---|---|---|
| Token-hash prefix matching for cross-request KV reuse | Hash block-hashes or token-id tuples; stripe by chunk (`blocks_in_chunk = chunk_size/block_size`), take the **last** block hash per chunk as the representative key; `LookupClientFactory` creates sync/async clients; scheduler `maybe_submit_lookup_request()` queues a lookup, worker `check_lookup_result()` returns `num_chunks*chunk_size` matched tokens. | Layout: token_ids grouped into 256-token chunks; key = chunk-representative block hash or `tuple(token_ids)`; `islice(block_hashes, blocks_in_chunk-1, None, blocks_in_chunk)`. No kernel (CPU dict lookup). | `multi_process_adapter.py:28-39` (striding_block_hashes), `:192-259` (maybe_submit_lookup_request key creation) | **RUNS but ZERO value** — async CPU metadata op, no kernel. Reuse only helps when a *prior request* cached a prefix; batch=1 single-stream has none. |
| GPU/CPU/disk tiered offload with LRU eviction | `LMCacheEngine` manages 3 tiers: GPU paged pool (`VLLMPagedMemGPUConnectorV2` / layerwise connector), optional CPU interim buffer (`enable_pd`), disk via `ZMQOffloadServer`. Miss → `load_spec` triggers async retrieve from CPU/disk; under pressure `evict(EvictParams(num_tokens=uncached_len))` frees GPU slots for prefetched chunks. | Layout: KV shape `(num_layers, 1or2 for MLA, chunk=256, num_kv_heads, head_size)`; paged `slot = block_id*block_size + block_offset`; CPU mirror same layout; disk via ZMQ `RequestType.RETRIEVE/STORE`. Movement = `cudaMemcpy` H2D/D2H, no compute kernel. | `vllm_v1_adapter.py:500-541` (GPU connector selection), `:638-642` (ZMQOffloadServer) | **RUNS; value only if context > VRAM** — H2D/D2H spill works at batch=1, but disk tier is multi-request oriented. The GPU↔CPU paging *shape* is the portable bit. |
| Layer-wise async transfer with stream pipelining (CacheBlend) | `use_layerwise + enable_blending` → `LMCBlenderBuilder` wraps engine+connector; per-layer async load/store on dedicated `load_stream`/`store_stream`; `LayerTransferCounter.wait_until(layer_id)` after each layer → `load_kv_layerwise(layer_id)`. SGLang registers the counter with the KV pool to drive per-layer loads during forward. | Layout: per-layer KV to `save_kv_layer(layer_name, kv_layer, attn_metadata)`; slot mapping from block_ids + `arange(0, block_size)`; ordering by `load_stream.synchronize()` / `store_stream.synchronize()`. No tensor-core; pure H2D + stream sync. | `vllm_v1_adapter.py:622-630` (blender), `:513-522` (layerwise connector); `lmc_radix_cache.py:37-61` (LayerTransferCounter), `:172-180` (async load via stream) | **Pattern RUNS / blending DEAD** — the *layer-wise overlap* (per-layer prefetch while prior layers compute) is highly portable. The **blending** path (extra GPU buffers, multi-fragment stitch) is dead for single-stream. |
| Asynchronous load/store streams with deferred synchronization | `start_load_kv()` queues async loads before forward; `wait_for_layer_load(layer_name)` blocks only on the layer about to run; `save_kv_layer()` queues async store; `wait_for_save()` drains at forward-context exit. Deferred sync overlaps CPU/disk I/O with compute and prevents premature paged-buffer overwrite. | CUDA stream events: `stream.synchronize()` + stream waits. `LoadMetadata(token_ids, slot_mapping, offset)`; `slot_mapping` = flat int64 physical slot addresses. No kernel. | `lmcache_connector.py:136-197` (start_load_kv / wait_for_layer_load / save_kv_layer / wait_for_save); `lmc_radix_cache.py:172-180` (load_stream context) | **RUNS — highest-value substrate.** Pure CUDA stream ops; batch=1 gets full async overlap. This is the template for bw24 KV/weight prefetch. |
| Multimodal token hashing for cache key generation | `extract_mm_features()` reads mm_features / legacy mm_hashes+mm_positions; `apply_mm_hashes_to_token_ids()` overwrites placeholder token-id slices in-place; `hex_hash_to_int16()` packs hash → 16-bit key; result becomes part of the LMCache lookup key. | int64 token_ids tensor with 16-bit MM hashes embedded at placeholder ranges; `token_ids[start:end] = hex_to_int16(hash)`. CPU tensor op. | `utils.py:66-89` (hex_hash_to_int16, apply_mm_hashes_to_token_ids), `:344-352` (applied in ReqMeta.from_request_tracker) | **DEAD** — cross-request + multimodal cache keying; bw24 is text GGUF single-stream. |
| Chunk-aligned partial save with `discard_partial_chunks` | `SaveSpec(skip_leading_tokens, can_save)`; `num_tokens_to_save = (len // chunk * chunk)` unless last prefill or `discard_partial_chunks`; skip save when already-saved < chunk boundary, decode phase w/o `save_decode_cache`, or `skip_save`. Only full chunks persisted during prefill; trailing partial chunk deferred. | `SaveSpec` in `ReqMeta`; chunk boundary `cdiv(num_saved+1, chunk)*chunk`; token_ids truncated to `num_tokens_to_save` before slot_mapping. CPU bookkeeping. | `vllm_v1_adapter.py:299-338` (SaveSpec logic), `:662-667` (discard_partial_chunks config) | **DEAD** — manages "already saved by prior batch"; no prior save at batch=1. (Chunk-granularity *sizing* idea is folded into "take" #5.) |
| Disaggregated prefill KV transfer via ZMQ + async prefetch | `DisaggSpec(receiver_id, receiver_host, receiver_init_port, receiver_alloc_port)`; on request finish, if a receiver is set, async-ship generated KV from prefill worker → decode worker via ZMQ before blocks freed; scheduler polls for completion. | KV blocks serialized via CUDA IPC / ZMQ queues; `BlockStored` events carry block_hashes, parent_block_hash, token_ids, block_size, medium(GPU/CPU/disk). | `vllm_v1_adapter.py:89-96` (DisaggSpec), `:99` (tmp_disagg_tracker), `:136-139` (disagg_spec in RequestTracker) | **DEAD** — 1P1D disaggregated serving; no prefill/decode split process in single-stream bw24. |
| Cross-instance KV broadcast (TP-group broadcast for MLA) | `extract_world_size_and_kv_rank()` adjusts rank/world_size for MLA (`kv_world_size = world_size // tp_size`); `tpg.broadcast()` / `tpg.broadcast_object()` sync KV across TP ranks (MLA shares KV); `ParallelStrategy` carries use_mla, kv_world_size, kv_worker_id. | NCCL collective broadcast over TP group (delegated to torch.distributed). No local kernel. | `lmcache_mp_connector.py:85-106` (extract_world_size_and_kv_rank / ParallelStrategy), `:542-548` (tpg broadcast in engine init) | **DEAD** — tensor-parallel (`world_size > 1`); single RTX 5090 = world_size 1. |
| Stateful request lookup with in-flight store tracking (eviction barrier) | `LMRadixCache` keeps `_in_flight_nodes` under `_node_lock`; `cache_finished_req()` stores via `store_stream` and appends node; `evict()` calls `store_stream.synchronize()`, decrements lock-refs on all in-flight nodes, then runs base eviction — preventing free of a slot whose async store is still pending. | Radix `TreeNode` (parent/children, key/value); lock-ref counting; `StoreMetadata(last_node, token_ids, kv_indices, offset)`; eviction guarded by `store_stream.synchronize()`. CPU tree + stream sync. | `lmc_radix_cache.py:118-125` (reset), `:244-254` (store + in_flight tracking), `:256-267` (evict with store sync) | **RUNS — correctness pattern portable.** The radix tree is cross-request (dead), but the **store-before-free barrier** is exactly what bw24 needs for any async KV spill (take #4). |
| Prefix cache-miss recovery via chunked LMCache prefetch | `match_prefix()` runs base match, detects uncached suffix; if `uncached_len>0` and suffix page-aligned, allocates `token_slots`, builds `slot_mapping = cat([-1*cached_len, fresh_slots])`, calls `start_load_kv(LoadMetadata(...))` async; on `num_retrieved>0` materializes a new `TreeNode` child and extends `device_indices`. | `token_slots` = physical GPU slot IDs from pool; `slot_mapping` `-1` sentinel marks resident tokens; `LoadMetadata(token_ids, slot_mapping, offset)`. H2D copy, no kernel. | `lmc_radix_cache.py:127-212` (match_prefix with LMCache recovery) | **RUNS / value only cross-request** — the *recovery* is reuse-driven (dead at batch=1), but the **`-1`-sentinel slot_mapping layout** is portable (take #3). |

### Notes on layout / chunking constants
- **Chunk size:** default `chunk_size = 256` tokens; `blocks_in_chunk = chunk_size / block_size`. Cache key = hash of the **last** block in the chunk (each block hash already encodes its prefix). `multi_process_adapter.py:28-39`.
- **Paged slot math:** `slot = block_id * block_size + block_offset`, `block_offsets = arange(0, block_size)`. Flat int64 `slot_mapping`; `-1` = "already resident, skip copy". `lmc_radix_cache.py:127-212`.
- **KV tensor shape (per chunk):** `(num_layers, {1 if MLA else 2}, chunk_size=256, num_kv_heads, head_size)`. CPU buffer mirrors the GPU paged layout 1:1, so spill is a straight `cudaMemcpy` with no re-layout.
- **Connector selection matrix** (`vllm_v1_adapter.py:500-541`): `use_layerwise & enable_blending` → `VLLMBufferLayerwiseGPUConnector` (DEAD: blending); `use_layerwise & !blending` → `VLLMPagedMemLayerwiseGPUConnector` (per-layer overlap, PORTABLE pattern); else → `VLLMPagedMemGPUConnectorV2` (plain paged spill, the one closest to a bw24 use). Layerwise + MLA is explicitly unsupported (`raise ValueError`, `:507-508`).
