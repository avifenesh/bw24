# Buffered io_uring spill backend design

Status: io_uring remains a deferred, implementation-ready option. Commit `66394bf` implements the
required comparison baseline: blocking `pread` plus a bounded worker-thread positioned-read path.
On G7e, the depth-8 worker backend completed the first five frozen calibration requests in
240.2775 s versus 688.6587 s for mmap (2.866x faster), with identical response payloads and 395
byte-identical routing rows. The exact run record is
[`evidence/spill-worker-ab-g7e-20260710.md`](evidence/spill-worker-ab-g7e-20260710.md). Implement
this ring only if it improves end-to-end wall time over that worker baseline.

## Backend order and capability evidence

The common contract is fixed regardless of submission mechanism:

```text
GPU SLRU -> bounded hot page/host cache -> pinned read buffers -> caller-thread H2D -> GPU slot
```

Use these promotion rungs:

1. done: blocking buffered `pread` for byte/lifetime/fallback proof;
2. done and G7e-gated: worker-thread buffered `pread` for real disk/compute overlap;
3. worker-thread `O_DIRECT` through the same buffers;
4. buffered and direct io_uring against the winning worker baseline;
5. mapped pinned-host zero-copy only for cold one-shot experts.

All expert offsets and lengths in the five staged Hy3 artifacts are 4 KiB aligned. G7e `/scratch`
ext4 reports 512-byte memory and offset alignment via `STATX_DIOALIGN`, so direct I/O can be tested
without repacking those artifacts. Native/general GGUF still needs a per-file capability check plus
aligned over-read or an aligned sidecar.

cuFile/GDS is outside this ladder. Both machines currently report compatibility mode; the local
GeForce target is not a supported direct-storage deployment, and G7e has not passed a direct-mode
capability probe. A compatibility-mode cuFile result would only benchmark another CPU bounce path.

## Decision and scope

The implemented non-mmap baseline uses exact `FileExt::read_at` calls and a bounded pool of
CUDA-pinned buffers. `BW24_SPILL_IO=pread` performs the demand read on the caller thread;
`BW24_SPILL_IO=worker` transfers buffer ownership through bounded CPU read workers so known-next
reads can overlap GPU compute. Only the caller thread submits H2D and publishes GPU-cache
residency. Failed initialization, read errors, short reads, and a busy worker ring retain the
validated mmap extent as the byte oracle and fallback.

The proposed next backend is **buffered** `io_uring` `READ_FIXED` into the same class of bounded
CUDA-pinned buffer ring, followed by caller-owned H2D into a reserved `MoeSlotCache` slot. It would
not use `O_DIRECT` initially:

- buffered reads remain the portability baseline and accept arbitrary native-GGUF extents;
- they preserve the mmap/page-cache backend as a cheap correctness fallback;
- the five current Hy3 artifacts are already 4 KiB aligned, so `O_DIRECT` can be tested through the
  same buffers before deciding whether io_uring is justified.

The implemented runtime remains opt-in: `BW24_SPILL_IO=mmap|pread|worker`, default `mmap`.
`BW24_SPILL_PREAD_DEPTH` controls both pinned-buffer and worker count (default 2; the G7e result used
8), and `BW24_SPILL_STATS=1` prints cumulative snapshots. A future `BW24_SPILL_IO=uring` mode would
attempt the capability gate below and fall back to mmap if initialization fails.

The current research artifacts contain 237 expert files (one file per layer/projection) and expert
blocks no larger than 3,538,944 bytes. A bounded sparse registered-file table of 256 or 512 entries
therefore covers the current artifacts without a hot-path registered-file LRU.

## Implemented source and worker seams

The prerequisite seams are now implemented: `HostBuf::Mmap` retains the opened file plus its map,
`ExpertSource` distinguishes memory from disk extents, and `MoeSlotCache` accepts source-aware
dispatch/prefetch calls. `expert_bytes()` remains the byte oracle and compatibility API.

The resulting source contract is:

```rust
// bw24-gguf/src/source.rs
pub struct DiskExtent {
    pub map: Arc<Mmap>,       // permanent mmap fallback
    pub file: Arc<File>,      // retained for registered-file I/O
    pub offset: u64,
    pub len: usize,
}

pub trait TensorSource {
    fn find_expert_disk(&self, name: &str) -> Option<DiskExtent> { None }
    // find_expert_mmap remains during migration.
}

// bw24-engine/src/model.rs
pub enum HostBuf {
    // existing arms...
    Mmap {
        map: Arc<Mmap>,
        file: Arc<File>,
        off: usize,
        len: usize,
    },
}

pub enum ExpertSource<'a> {
    Memory {
        bytes: &'a [u8],
        keepalive: Option<ExpertKeepalive>,
    },
    Disk {
        file: &'a Arc<File>,
        offset: u64,
        len: usize,
        fallback: &'a [u8],
        keepalive: ExpertKeepalive,
    },
}

impl HostExps {
    pub fn expert_source(&self, expert: usize) -> ExpertSource<'_>;
}
```

`Hy3RepackSource` retains `Arc<File>` beside every mapped file, and `GgufFile` retains its opened
handle after `Mmap::map`. The source path remains immutable for the model lifetime. The loader
continues validating every manifest extent against file length. Cache call sites use
`expert_source()`; non-cache staging and correctness oracles keep using `expert_bytes()`.

The worker pool moves each pinned allocation by ownership to a CPU reader and back through the
completion channel. Canceled/stale tickets cannot recycle a buffer until the read returns, and a
buffer submitted to H2D is not reusable until its CUDA event completes. Per-forward worker scopes
cancel unused lookahead, grouped prefill queues the first and then known-next expert projections,
and ring saturation skips speculative work without moving CUDA submission off the caller thread.

## Proposed io_uring module and API

Add `crates/bw24-engine/src/spill_uring.rs`, Linux-only, behind a target-specific dependency on the
`io-uring` crate. Do not add the dependency before a matched worker/direct-I/O study justifies this
phase.

```rust
pub struct UringSpill {
    ring: io_uring::IoUring,
    buffers: Vec<FixedBuffer>,
    files: HashMap<FileKey, u32>, // registered-file index
    requests: HashMap<u64, ReadRequest>,
    next_request: u64,
    copy_stream: Arc<CudaStream>,
}

pub struct ReadTicket {
    request_id: u64,
    buffer: u16,
}

impl UringSpill {
    pub fn try_new(e: &Engine, max_block: usize, depth: usize)
        -> Result<Self, UringUnavailable>;
    pub fn register_file(&mut self, file: Arc<File>) -> Result<u32, UringUnavailable>;
    pub fn submit(&mut self, id: BlockId, extent: DiskExtentRef<'_>)
        -> Result<Option<ReadTicket>, SpillIoError>;
    pub fn poll(&mut self, e: &Engine, gpu_slots: &mut [CudaSlice<u8>])
        -> Result<Vec<PollEvent>, SpillIoError>;
    pub fn wait_read(&mut self, request_id: u64, e: &Engine,
                     gpu_slots: &mut [CudaSlice<u8>]) -> Result<(), SpillIoError>;
    pub fn h2d_event(&self, ticket: ReadTicket) -> Option<&CudaEvent>;
    pub fn detach_h2d(&mut self, ticket: ReadTicket);
    pub fn cancel(&mut self, request_id: u64) -> Result<(), SpillIoError>;
    pub fn shutdown(&mut self);
}
```

`MoeSlotCache` would own `Option<UringSpill>`. This keeps io_uring, fixed buffers, pending block state,
and GPU slot reservations behind the existing cache mutex on the single GPU-worker path. No async
runtime or background CUDA thread is required. Poll completions at cache entry, after submissions,
and immediately before a pending block is consumed.

Current knobs and proposed additions:

```text
BW24_SPILL_IO=mmap|pread|worker default mmap (implemented)
BW24_SPILL_PREAD_DEPTH=2      pinned buffers and worker count; G7e A/B used 8
BW24_SPILL_STATS=1            cumulative read/fallback/wait/ring-full snapshots
BW24_SPILL_IO=uring           future selector after promotion
BW24_SPILL_URING_DEPTH=8       number of fixed buffers and maximum disk reads in flight
BW24_SPILL_URING_SQ=32         SQ/CQ entries, power of two and >= 2*depth
```

Each fixed buffer has capacity `max_moe_block()` rounded up to the host page size. The actual
`READ_FIXED` and H2D lengths remain the exact expert extent; padding is never copied or decoded.

## Capability gate and registration

`try_new` is all-or-nothing and runs before the first io_uring prefetch:

1. Require Linux and `BW24_SPILL_IO=uring`.
2. Create `IoUring`, then probe `ReadFixed::CODE` and `AsyncCancel::CODE`.
3. Allocate `depth` long-lived `PinnedHostSlice<u8>` buffers with the CUDA context.
4. Build one `iovec` per raw pinned pointer and call `register_buffers` once.
5. Create a bounded sparse registered-file table. On first use of a `(device,inode)`, fill one empty
   slot with `register_files_update`, retain its `Arc<File>`, and never replace that slot while the
   model is live. `READ_FIXED` uses the registered index. Table-full falls back to mmap.
6. Run a one-page sacrificial `READ_FIXED` and verify its CQE length and bytes before enabling.

Registered buffers must remain at stable addresses until the ring is drained and destroyed. A
registration failure (`ENOMEM`, `EPERM`, `EFAULT`, or `EOPNOTSUPP`), disabled io_uring, unsupported
opcode, file registration failure, or CUDA pinned-allocation failure logs one diagnostic and returns
`UringUnavailable`; the runtime uses mmap thereafter. Never raise `RLIMIT_MEMLOCK` inside bw24.

The current G7e research host reports Linux 6.17, io_uring enabled, unlimited memlock, ext4 scratch,
and 237 artifact files. The local target reports Linux 7.0 and io_uring enabled but an 8 MiB
memlock limit, so the registration attempt itself—not kernel version assumptions—is authoritative.

## Fixed-buffer state machine

One owner controls every transition. A buffer is reusable only in `Free`.

```text
Free
  -> Reading { request, block, gpu_slot, expected, orphaned: false }
  -> ReadReady { block, gpu_slot, len }                    (ephemeral)
  -> H2d { block, gpu_slot, ready_event, consumed: false }
  -> Free                                                  (ready_event complete)

Reading -> Cancelling { request, block, gpu_slot }         (cancel submitted)
Cancelling -> Free                                         (target read CQE observed)
H2d -> H2d { consumed: true }                              (compute wait inserted)
H2d -> Free                                                (ready_event complete)
Any read error/short read -> Failed -> Free                (no H2D is issued)
```

The request table outlives a buffer when necessary: after an unconsumed H2D event completes, the
buffer returns to `Free` and the request becomes `H2dComplete { block, gpu_slot }`. A later dispatch
can consume that request without a CUDA wait. Failed/orphaned transitions emit `PollEvent` back to
`MoeSlotCache`, which alone restores reserved GPU slots to its `free` list.

Important ownership rules:

- A read CQE does **not** release the pinned buffer. The buffer remains owned through H2D and is
  released only after `CudaEvent::is_complete()`.
- Inserting `compute_stream.wait(ready_event)` makes the GPU slot safe to consume, but does not make
  the pinned source safe to overwrite. Event completion does.
- A short read is an error. Do not H2D partially overwritten bytes. Fall back to the validated mmap
  extent for that block and increment a degradation counter.
- `user_data` is a monotonic request id, never a raw pointer. CQEs are resolved through the request
  table, so late completions cannot target a recycled Rust object.

## Cache integration and race rules

Extend `PendingBlock` rather than adding a second pending table:

```rust
enum PendingTransfer {
    HostH2d { ready: CudaEvent },
    Disk { ticket: ReadTicket },
}

struct PendingBlock {
    slot: usize,
    transfer: PendingTransfer,
}
```

Disk prefetch follows this order:

1. Under the cache lock, reject ids already in `table` or `pending`.
2. Reserve a free/evicted GPU slot and remove it from both SLRU queues.
3. Record the current compute-stream point and enqueue the existing copy-stream wait. This protects
   prior users of the evicted slot before any future H2D.
4. Acquire a `Free` pinned buffer and submit one registered-file `READ_FIXED` for the exact extent.
5. Insert `pending[id]` only after successful SQ submission. On SQ-full/no-buffer, restore the GPU
   slot and let the mmap path handle the miss.
6. On read completion, enqueue H2D on `copy_stream`, record `ready_event`, and retain the buffer.
7. On dispatch of the same id, wait for its read if necessary, insert a compute-stream wait for the
   H2D event, then move the GPU slot into `table`/probation. Later hits are ordered after that wait on
   the same compute stream.

This preserves current race invariants:

- pending GPU slots are absent from SLRU and cannot be evicted;
- a second prefetch for the same `BlockId` is suppressed;
- a synchronous dispatch never races the same pending disk request—it consumes it or degrades it to
  mmap first;
- the table is never populated before the compute-stream wait is inserted;
- `dev_rows` cannot include a pending block because `per_layer` increments only on consumption.

If the fixed-buffer ring is full, do not block speculative prefetch. Return `false`. Only demand
dispatch may call `submit_and_wait` for its already-pending request.

## Cancellation, errors, and shutdown

Cancellation is completion-based. Submit `AsyncCancel` by target request id, but never free its
buffer on the cancel CQE: cancel and target CQEs can arrive in either order, and disk I/O may finish
successfully despite cancellation. Free or advance the buffer only when the original read CQE is
observed (`-ECANCELED`, successful length, or another terminal error).

Use a monotonically increasing prefetch epoch per MoE layer/forward. On normal completion, cancel
unconsumed requests from that epoch. Request abort/error calls `abort_epoch`; H2Ds already queued are
marked orphaned and reaped after their CUDA event, while reads transition to `Cancelling`. A GPU slot
reserved by an orphan returns to `free` only after its read/H2D can no longer touch it.

`UringSpill::shutdown` must:

1. submit cancellation for every `Reading` request;
2. drain both target and cancellation CQEs;
3. synchronize `copy_stream` so no H2D still reads a pinned buffer;
4. unregister/drop the io_uring instance;
5. drop the CUDA-pinned buffers and retained files.

The spill manager stores an `Arc<CudaStream>` so its `Drop` can perform the same conservative drain
if explicit shutdown was skipped. Drop errors are diagnostics only; buffers must never be freed
while kernel I/O or CUDA H2D still owns them.

## Fallback policy

Fallback is intentionally simple:

- initialization failure: disable io_uring process-wide and use mmap;
- no free fixed buffer/SQE during speculative prefetch: skip it, use normal demand dispatch later;
- read CQE error or short read: remove pending reservation, record the error, and dispatch from the
  same extent's mmap slice;
- repeated runtime errors (default threshold 1): trip a circuit breaker, drain the ring, and use
  mmap for all later blocks.

The mmap mapping and `fallback: &[u8]` remain alive for the model lifetime even when io_uring is
enabled. Therefore backend failure never changes bytes or model behavior.

## Tests and measurement gate

GPU-free unit tests:

- state-machine transition table, including invalid/double-release transitions;
- unique request ids and stale/late CQE rejection;
- read success, short read, `-EIO`, and `-ECANCELED` using temporary files;
- unaligned buffered file offset/length reads into a registered buffer;
- ring-full returns `WouldBlock` without losing the GPU slot;
- duplicate `BlockId` prefetch suppression;
- cancel-CQE-before-target-CQE and target-CQE-before-cancel-CQE orderings;
- teardown drains requests before buffer drop;
- forced registration failure selects mmap.

GPU integration gates:

- fixed-buffer H2D bytes equal the mmap slice for every qtype and odd manifest offset;
- ring buffer is not reused before `CudaEvent::is_complete()`;
- pending slot never appears resident before `compute_wait`;
- forced cancellation/error produces the same argmax as mmap-only;
- full kernel-check, run-gen argmax, and run-spec battery on the target rig.

Performance promotion requires matched cold-cache runs against mmap windows `1/4/8/16`, recording:
prefill wall time, major faults, `folio_wait` samples, per-device read MiB/s, read queue depth,
read-to-H2D latency, H2D latency, ring-full count, mmap fallback count, cache hit rate, and GPU duty.
The worker baseline already passed a five-request frozen-input G7e direction gate; it still requires
the full calibration run and an RTX 5090 deployment gate. Keep io_uring only if it improves
end-to-end wall time over worker without reducing steady-state cache behavior.

## Implementation sequence

1. Done: retain files in `GgufFile`/`Hy3RepackSource`; add `DiskExtent` and `expert_source()` with tests.
2. Done: add blocking and worker `PreadPool` modes, bounded pinned-buffer ownership, exact reads,
   caller-thread H2D, cancellation, telemetry, and mmap fallback.
3. Done: pass the live depth-8 ignored CUDA test and the frozen first-five G7e mmap/worker A/B.
4. Next only after the direct-I/O decision: add Linux-only `spill_uring.rs` plus fixed-buffer
   state-machine and capability/readback tests; no cache integration yet.
5. Integrate disk tickets into `MoeSlotCache::prefetch/dispatch` behind future
   `BW24_SPILL_IO=uring`, then add epoch cancellation and shutdown gates.
6. Run a matched G7e A/B against worker, then repeat any winner on the local RTX 5090 target.

Do not combine this phase with SIMD/kernel changes. It is a storage-to-pinned-to-HBM pipeline
experiment and must be measured separately before interaction tuning.
