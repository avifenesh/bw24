# Buffered io_uring spill backend design

Status: implementation-ready design, intentionally not implemented. Land only if the rolling
`MADV_WILLNEED` window still leaves material `folio_wait` time in a matched cold-cache A/B.

## Decision and scope

The second backend is **buffered** `io_uring` `READ_FIXED` into a bounded CUDA-pinned buffer ring,
followed by the existing copy-stream H2D into a reserved `MoeSlotCache` slot. It does not use
`O_DIRECT` initially:

- buffered reads accept the artifact's exact, unaligned expert offsets and lengths;
- they preserve the mmap/page-cache backend as a cheap correctness fallback;
- `O_DIRECT` would require aligned offsets, lengths, and padded artifact layout, which is a separate
  experiment after buffered io_uring proves useful.

The runtime remains opt-in. `BW24_SPILL_IO=mmap` is the default. `BW24_SPILL_IO=uring` attempts the
capability gate below and falls back to mmap for the whole process if initialization fails.

The current research artifacts contain 237 expert files (one file per layer/projection) and expert
blocks no larger than 3,538,944 bytes. A bounded sparse registered-file table of 256 or 512 entries
therefore covers the current artifacts without a hot-path registered-file LRU.

## Current seams that must change

`HostBuf::Mmap` currently retains only `Arc<Mmap>`, offset, and length. The file descriptor used to
create the map has already been closed. `MoeSlotCache::{dispatch,prefetch}` receive only `&[u8]`, so
they cannot distinguish a pinned source from a disk extent.

Keep `expert_bytes()` unchanged as the oracle/fallback API, and add a source-aware twin:

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
        file: Option<Arc<File>>,
        off: usize,
        len: usize,
    },
}

pub enum ExpertSource<'a> {
    Memory(&'a [u8]),
    Disk {
        file: &'a Arc<File>,
        offset: u64,
        len: usize,
        fallback: &'a [u8],
    },
}

impl HostExps {
    pub fn expert_source(&self, expert: usize) -> ExpertSource<'_>;
}
```

`Hy3RepackSource` should retain `Arc<File>` beside every mapped file. `GgufFile` should retain its
opened `Arc<File>` instead of closing it after `Mmap::map`. The source path remains immutable for the
model lifetime. The loader continues validating every manifest extent against file length.

Change only the cache call sites to use `expert_source()`. Non-cache staging and all correctness
oracles keep using `expert_bytes()` until the backend is proven.

## New module and API

Add `crates/bw24-engine/src/spill_uring.rs`, Linux-only, behind a target-specific dependency on the
`io-uring` crate. Do not add the dependency before the mmap-window A/B justifies this phase.

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

`MoeSlotCache` owns `Option<UringSpill>`. This keeps io_uring, fixed buffers, pending block state,
and GPU slot reservations behind the existing cache mutex on the single GPU-worker path. No async
runtime or background CUDA thread is required. Poll completions at cache entry, after submissions,
and immediately before a pending block is consumed.

Recommended initial knobs:

```text
BW24_SPILL_IO=mmap|uring       default mmap
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
Keep io_uring only if it improves end-to-end wall time without reducing steady-state cache behavior.

## Implementation sequence

1. Retain files in `GgufFile`/`Hy3RepackSource`; add `DiskExtent` and `expert_source()` with tests.
2. Add Linux-only `spill_uring.rs` plus fixed-buffer state-machine tests; no cache integration yet.
3. Add the capability gate and byte-for-byte temporary-file/readback probe.
4. Integrate disk tickets into `MoeSlotCache::prefetch/dispatch` behind `BW24_SPILL_IO=uring`.
5. Add epoch cancellation and shutdown gates.
6. Run cold-cache A/B on G7e, then repeat any winner on the local RTX 5090 target.

Do not combine this phase with SIMD/kernel changes. It is a storage-to-pinned-to-HBM pipeline
experiment and must be measured separately before interaction tuning.
