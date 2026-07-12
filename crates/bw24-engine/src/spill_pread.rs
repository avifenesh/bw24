//! Opt-in explicit positioned-read backends for mmap-backed MoE experts.
//!
//! `BW24_SPILL_IO=pread` keeps the blocking correctness proof. `worker` moves free pinned buffers
//! through a bounded CPU worker pool so known-next reads overlap GPU compute. Only the CUDA owner
//! thread submits H2D or publishes cache residency. mmap remains the default and byte oracle.

use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use cudarc::driver::{CudaEvent, CudaStream, PinnedHostSlice};

use crate::Engine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpillIoMode {
    Mmap,
    Pread,
    Worker,
}

fn parse_spill_io(value: Option<&str>) -> Result<SpillIoMode, &'static str> {
    match value.unwrap_or("mmap") {
        "mmap" => Ok(SpillIoMode::Mmap),
        "pread" => Ok(SpillIoMode::Pread),
        "worker" => Ok(SpillIoMode::Worker),
        _ => Err("expected mmap, pread, or worker"),
    }
}

pub(crate) fn configured_mode() -> SpillIoMode {
    static MODE: std::sync::OnceLock<SpillIoMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("BW24_SPILL_IO").ok();
        match parse_spill_io(raw.as_deref()) {
            Ok(mode) => mode,
            Err(reason) => {
                eprintln!(
                    "[spill-pread] invalid BW24_SPILL_IO={:?} ({reason}); using mmap",
                    raw.as_deref().unwrap_or("")
                );
                SpillIoMode::Mmap
            }
        }
    })
}

pub(crate) fn worker_enabled() -> bool { configured_mode() == SpillIoMode::Worker }

fn parse_depth(value: Option<&str>) -> Result<usize, &'static str> {
    let depth = value.unwrap_or("2").parse::<usize>().map_err(|_| "expected an integer")?;
    if (1..=64).contains(&depth) { Ok(depth) } else { Err("expected 1..=64") }
}

fn configured_depth() -> usize {
    let raw = std::env::var("BW24_SPILL_PREAD_DEPTH").ok();
    match parse_depth(raw.as_deref()) {
        Ok(depth) => depth,
        Err(reason) => {
            eprintln!(
                "[spill-pread] invalid BW24_SPILL_PREAD_DEPTH={:?} ({reason}); using 2",
                raw.as_deref().unwrap_or("")
            );
            2
        }
    }
}

pub(crate) fn pread_exact_at(file: &File, mut dst: &mut [u8], mut offset: u64)
                                  -> io::Result<()> {
    while !dst.is_empty() {
        match file.read_at(dst, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short positioned read at offset {offset}: {} bytes remain", dst.len()),
                ));
            }
            Ok(n) => {
                offset = offset.checked_add(n as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "positioned-read offset overflow")
                })?;
                dst = &mut dst[n..];
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferPhase {
    Free,
    Reading,
    Canceled,
    Ready,
    Failed,
    H2d,
}

impl BufferPhase {
    fn begin_read(&mut self) -> bool {
        if *self != Self::Free { return false; }
        *self = Self::Reading;
        true
    }

    /// Complete a CPU read. Returns true when the ticket was canceled while I/O was in flight and
    /// the restored allocation became immediately reusable instead of exposing Ready/Failed.
    fn finish_read(&mut self, success: bool) -> bool {
        match *self {
            Self::Reading => {
                *self = if success { Self::Ready } else { Self::Failed };
                false
            }
            Self::Canceled => {
                *self = Self::Free;
                true
            }
            _ => panic!("completed read was not in flight"),
        }
    }

    /// Cancel a ticket. An active worker still owns the allocation, so Reading must wait for its
    /// completion; Ready/Failed can return to Free immediately. Returns true if already reusable.
    fn cancel_read(&mut self) -> bool {
        match *self {
            Self::Reading => {
                *self = Self::Canceled;
                false
            }
            Self::Ready | Self::Failed => {
                *self = Self::Free;
                true
            }
            _ => false,
        }
    }

    fn begin_h2d(&mut self) {
        assert_eq!(*self, Self::Ready, "pinned buffer H2D without a completed read");
        *self = Self::H2d;
    }

    fn abort_read(&mut self) {
        assert!(matches!(*self, Self::Reading | Self::Canceled | Self::Ready | Self::Failed),
                "only a read-owned buffer can abort a read");
        *self = Self::Free;
    }

    fn finish_h2d(&mut self, complete: bool) -> bool {
        if *self != Self::H2d || !complete { return false; }
        *self = Self::Free;
        true
    }
}

struct PinnedBuffer {
    // Option lets the failure path intentionally leak the allocation instead of freeing host
    // memory while an H2D may still reference it. That path is CUDA-context failure only.
    data: Option<PinnedHostSlice<u8>>,
    phase: BufferPhase,
    ready: Option<Arc<CudaEvent>>,
    ticket: Option<ReadTicket>,
    error: Option<WorkerReadError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ReadTicket(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerAdmission {
    Demand,
    Speculative,
}

impl WorkerAdmission {
    fn free_buffer_floor(self) -> usize {
        match self {
            Self::Demand => 0,
            Self::Speculative => 1,
        }
    }
}

#[derive(Debug)]
struct WorkerReadError { message: String, short: bool }

impl WorkerReadError {
    fn from_io(err: io::Error) -> Self {
        Self { short: err.kind() == io::ErrorKind::UnexpectedEof, message: err.to_string() }
    }

    fn other(err: impl std::fmt::Display) -> Self {
        Self { message: err.to_string(), short: false }
    }

    fn into_io(self) -> io::Error {
        let kind = if self.short { io::ErrorKind::UnexpectedEof } else { io::ErrorKind::Other };
        io::Error::new(kind, self.message)
    }
}

struct ReadRequest {
    ticket: ReadTicket,
    index: usize,
    file: Arc<File>,
    offset: u64,
    len: usize,
    data: PinnedHostSlice<u8>,
}

struct ReadCompletion {
    ticket: ReadTicket,
    index: usize,
    len: usize,
    data: PinnedHostSlice<u8>,
    result: Result<(), WorkerReadError>,
}

struct WorkerPool {
    requests: Option<mpsc::SyncSender<ReadRequest>>,
    completions: mpsc::Receiver<ReadCompletion>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl WorkerPool {
    fn try_new(depth: usize) -> io::Result<Self> {
        let (request_tx, request_rx) = mpsc::sync_channel::<ReadRequest>(depth);
        let (completion_tx, completion_rx) = mpsc::channel();
        let request_rx = Arc::new(Mutex::new(request_rx));
        let mut threads: Vec<thread::JoinHandle<()>> = Vec::with_capacity(depth);
        for worker in 0..depth {
            let requests = request_rx.clone();
            let completions = completion_tx.clone();
            let handle = match thread::Builder::new().name(format!("bw24-spill-{worker}"))
                .spawn(move || read_worker(requests, completions)) {
                Ok(handle) => handle,
                Err(err) => {
                    drop(request_tx);
                    for handle in threads { let _ = handle.join(); }
                    return Err(err);
                }
            };
            threads.push(handle);
        }
        drop(completion_tx);
        Ok(Self { requests: Some(request_tx), completions: completion_rx, threads })
    }

    fn shutdown(&mut self) -> bool {
        self.requests.take();
        let mut clean = true;
        for handle in self.threads.drain(..) {
            if handle.join().is_err() {
                clean = false;
                eprintln!("[spill-worker] disk worker panicked during shutdown");
            }
        }
        clean
    }
}

fn read_worker(requests: Arc<Mutex<mpsc::Receiver<ReadRequest>>>,
               completions: mpsc::Sender<ReadCompletion>) {
    loop {
        let request = {
            let receiver = match requests.lock() { Ok(receiver) => receiver, Err(_) => return };
            match receiver.recv() { Ok(request) => request, Err(_) => return }
        };
        let ReadRequest { ticket, index, file, offset, len, mut data } = request;
        let result = match data.as_mut_slice() {
            Ok(dst) => pread_exact_at(file.as_ref(), &mut dst[..len], offset)
                .map_err(WorkerReadError::from_io),
            Err(err) => Err(WorkerReadError::other(err)),
        };
        // The receiver outlives and is drained after every worker joins. No CUDA operation can
        // reference this allocation until the caller receives the completion and submits H2D.
        if completions.send(ReadCompletion { ticket, index, len, data, result }).is_err() {
            return;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PreadStats {
    pub reads: u64,
    pub bytes: u64,
    pub read_errors: u64,
    pub short_reads: u64,
    pub fallbacks: u64,
    pub buffer_waits: u64,
    pub ring_full: u64,
}

pub(crate) struct PreadPool {
    buffers: Vec<PinnedBuffer>,
    capacity: usize,
    stream: Arc<CudaStream>,
    stats: PreadStats,
    mode: SpillIoMode,
    workers: Option<WorkerPool>,
    next_ticket: u64,
    tickets: HashSet<ReadTicket>,
}

impl PreadPool {
    pub(crate) fn try_new(e: &Engine, capacity: usize, mode: SpillIoMode)
                              -> Result<Self, Box<dyn std::error::Error>> {
        debug_assert_ne!(mode, SpillIoMode::Mmap);
        let requested_depth = configured_depth();
        let mut buffers = Vec::with_capacity(requested_depth);
        for _ in 0..requested_depth {
            match unsafe { e.ctx().alloc_pinned::<u8>(capacity) } {
                Ok(data) => buffers.push(PinnedBuffer {
                    data: Some(data), phase: BufferPhase::Free, ready: None,
                    ticket: None, error: None,
                }),
                Err(err) if buffers.is_empty() => return Err(err.into()),
                Err(err) => {
                    eprintln!(
                        "[spill-pread] pinned depth {requested_depth} unavailable ({err}); \
                         continuing with depth {}",
                        buffers.len()
                    );
                    break;
                }
            }
        }
        let depth = buffers.len();
        let workers = if mode == SpillIoMode::Worker {
            Some(WorkerPool::try_new(depth)?)
        } else { None };
        let description = match mode {
            SpillIoMode::Pread => "blocking demand pread",
            SpillIoMode::Worker => "bounded worker prefetch",
            SpillIoMode::Mmap => unreachable!(),
        };
        if mode == SpillIoMode::Worker && depth < 6 {
            eprintln!("[spill-worker] WARNING: effective depth {depth} < 6; grouped current+next \
                       overlap is degraded; continuing with projection concurrency and mmap fallback");
        }
        eprintln!(
            "[spill-pread] enabled: depth={depth} buffer_bytes={capacity} total_pinned_bytes={} \
             ({description}, caller-thread H2D, mmap error fallback)",
            depth.saturating_mul(capacity)
        );
        Ok(Self {
            buffers, capacity, stream: e.stream().clone(), stats: PreadStats::default(), mode,
            workers, next_ticket: 1, tickets: HashSet::new(),
        })
    }

    pub(crate) fn is_worker(&self) -> bool { self.mode == SpillIoMode::Worker }

    fn reap_completed(&mut self) {
        self.poll_worker_completions();
        for buffer in &mut self.buffers {
            if buffer.phase != BufferPhase::H2d { continue; }
            let complete = buffer.ready.as_ref().is_some_and(|event| event.is_complete());
            if buffer.phase.finish_h2d(complete) {
                buffer.ready = None;
            }
        }
    }

    fn finish_worker_completion(&mut self, completion: ReadCompletion) {
        let ReadCompletion { ticket, index, len, data, result } = completion;
        let Some(buffer) = self.buffers.get_mut(index) else {
            eprintln!("[spill-worker] ignoring completion for invalid buffer {index}");
            return;
        };
        if !matches!(buffer.phase, BufferPhase::Reading | BufferPhase::Canceled)
            || buffer.ticket != Some(ticket)
            || buffer.data.is_some() {
            eprintln!("[spill-worker] ignoring stale completion for ticket {:?}", ticket);
            return;
        }
        buffer.data = Some(data);
        let success = result.is_ok();
        let short = result.as_ref().err().is_some_and(|err| err.short);
        let canceled = buffer.phase.finish_read(success);
        if success {
            self.stats.reads += 1;
            self.stats.bytes += len as u64;
        } else {
            self.stats.read_errors += 1;
            if short { self.stats.short_reads += 1; }
        }
        if canceled {
            buffer.ticket = None;
            buffer.error = None;
        } else if let Err(err) = result {
            buffer.error = Some(err);
        }
    }

    fn poll_worker_completions(&mut self) {
        loop {
            let completion = match self.workers.as_ref()
                .and_then(|workers| workers.completions.try_recv().ok()) {
                Some(completion) => completion,
                None => break,
            };
            self.finish_worker_completion(completion);
        }
    }

    fn wait_for_one(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(index) = self.buffers.iter().position(|buffer| buffer.phase == BufferPhase::H2d)
        else {
            return Err(io::Error::other("pread buffer pool has no reusable or in-flight buffer").into());
        };
        self.stats.buffer_waits += 1;
        self.buffers[index].ready.as_ref()
            .ok_or_else(|| io::Error::other("H2D buffer is missing its completion event"))?
            .synchronize()?;
        assert!(self.buffers[index].phase.finish_h2d(true));
        self.buffers[index].ready = None;
        Ok(())
    }

    /// Read one exact demand extent on the caller thread. A full pool waits for one H2D event.
    pub(crate) fn read(
        &mut self,
        file: &File,
        offset: u64,
        len: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if len > self.capacity {
            self.stats.read_errors += 1;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expert extent {len} exceeds pread buffer capacity {}", self.capacity),
            ).into());
        }
        self.reap_completed();
        if !self.buffers.iter().any(|buffer| buffer.phase == BufferPhase::Free) {
            self.wait_for_one()?;
        }
        let index = self.buffers.iter().position(|buffer| buffer.phase == BufferPhase::Free)
            .expect("wait_for_one must make one pinned buffer reusable");
        assert!(self.buffers[index].phase.begin_read());
        let result = {
            let dst = match self.buffers[index].data.as_mut()
                .expect("live pread buffer must retain its pinned allocation")
                .as_mut_slice() {
                    Ok(dst) => dst,
                    Err(err) => {
                        self.buffers[index].phase.abort_read();
                        self.stats.read_errors += 1;
                        return Err(err.into());
                    }
                };
            pread_exact_at(file, &mut dst[..len], offset)
        };
        match result {
            Ok(()) => {
                assert!(!self.buffers[index].phase.finish_read(true));
                self.stats.reads += 1;
                self.stats.bytes += len as u64;
                Ok(index)
            }
            Err(err) => {
                self.buffers[index].phase.abort_read();
                self.stats.read_errors += 1;
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    self.stats.short_reads += 1;
                }
                Err(err.into())
            }
        }
    }

    /// Submit a demand extent without blocking the CUDA owner. `None` means every pinned buffer is
    /// busy and the mmap fallback must handle the miss.
    pub(crate) fn submit_worker(&mut self, file: Arc<File>, offset: u64, len: usize)
                                -> Result<Option<ReadTicket>, Box<dyn std::error::Error>> {
        self.submit_worker_with_admission(file, offset, len, WorkerAdmission::Demand)
    }

    /// Submit a known-future extent while reserving one pinned buffer for a demand miss.
    pub(crate) fn submit_worker_speculative(&mut self, file: Arc<File>, offset: u64, len: usize)
                                            -> Result<Option<ReadTicket>, Box<dyn std::error::Error>> {
        self.submit_worker_with_admission(file, offset, len, WorkerAdmission::Speculative)
    }

    fn submit_worker_with_admission(&mut self, file: Arc<File>, offset: u64, len: usize,
                                    admission: WorkerAdmission)
                                    -> Result<Option<ReadTicket>, Box<dyn std::error::Error>> {
        if self.mode != SpillIoMode::Worker {
            return Err(io::Error::other("worker read submitted to blocking pread backend").into());
        }
        if len > self.capacity {
            self.stats.read_errors += 1;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expert extent {len} exceeds pread buffer capacity {}", self.capacity),
            ).into());
        }
        self.reap_completed();
        let free_count = self.buffers.iter()
            .filter(|buffer| buffer.phase == BufferPhase::Free)
            .count();
        if free_count <= admission.free_buffer_floor() {
            self.stats.ring_full += 1;
            return Ok(None);
        }
        let Some(index) = self.buffers.iter()
            .position(|buffer| buffer.phase == BufferPhase::Free) else {
                unreachable!("positive free-buffer count must have a free buffer");
            };
        let ticket = ReadTicket(self.next_ticket);
        self.next_ticket = self.next_ticket.checked_add(1)
            .ok_or_else(|| io::Error::other("spill read ticket overflow"))?;
        let buffer = &mut self.buffers[index];
        assert!(buffer.phase.begin_read());
        buffer.ticket = Some(ticket);
        let data = buffer.data.take().expect("free worker buffer must retain pinned allocation");
        let request = ReadRequest { ticket, index, file, offset, len, data };
        let sender = self.workers.as_ref().and_then(|workers| workers.requests.as_ref())
            .ok_or_else(|| io::Error::other("spill worker pool is shut down"))?;
        match sender.try_send(request) {
            Ok(()) => {
                self.tickets.insert(ticket);
                Ok(Some(ticket))
            }
            Err(mpsc::TrySendError::Full(request)) => {
                self.stats.ring_full += 1;
                buffer.data = Some(request.data);
                buffer.ticket = None;
                buffer.phase.abort_read();
                Ok(None)
            }
            Err(mpsc::TrySendError::Disconnected(request)) => {
                buffer.data = Some(request.data);
                buffer.ticket = None;
                buffer.phase.abort_read();
                Err(io::Error::other("spill worker pool disconnected").into())
            }
        }
    }

    /// Wait for one exact worker ticket and return its ready pinned buffer. Other completions are
    /// harvested while waiting, so later demand reads consume them without a channel roundtrip.
    pub(crate) fn wait_worker(&mut self, ticket: ReadTicket)
                              -> Result<usize, Box<dyn std::error::Error>> {
        if !self.tickets.contains(&ticket) {
            return Err(io::Error::other("unknown spill worker ticket").into());
        }
        self.poll_worker_completions();
        let mut waited = false;
        loop {
            let index = self.buffers.iter().position(|buffer| buffer.ticket == Some(ticket))
                .ok_or_else(|| io::Error::other("spill worker ticket lost its buffer"))?;
            match self.buffers[index].phase {
                BufferPhase::Ready => {
                    self.tickets.remove(&ticket);
                    return Ok(index);
                }
                BufferPhase::Failed => {
                    self.tickets.remove(&ticket);
                    let buffer = &mut self.buffers[index];
                    buffer.ticket = None;
                    let err = buffer.error.take()
                        .unwrap_or_else(|| WorkerReadError::other("spill worker read failed"));
                    buffer.phase.abort_read();
                    return Err(err.into_io().into());
                }
                BufferPhase::Reading => {}
                _ => return Err(io::Error::other("spill worker ticket has invalid state").into()),
            }
            if !waited {
                self.stats.buffer_waits += 1;
                waited = true;
            }
            let completion = self.workers.as_ref()
                .ok_or_else(|| io::Error::other("spill worker pool is unavailable"))?
                .completions.recv()
                .map_err(|_| io::Error::other("spill worker completion channel disconnected"))?;
            self.finish_worker_completion(completion);
        }
    }

    /// Abandon an unused speculative ticket without ever reusing a buffer that a worker still
    /// mutates. Completed reads free immediately; in-flight reads transition to Canceled and their
    /// completion restores the owned allocation to Free.
    pub(crate) fn cancel_worker(&mut self, ticket: ReadTicket) -> bool {
        self.poll_worker_completions();
        if !self.tickets.remove(&ticket) { return false; }
        let Some(index) = self.buffers.iter().position(|buffer| buffer.ticket == Some(ticket))
        else { return false };
        let buffer = &mut self.buffers[index];
        let reusable = buffer.phase.cancel_read();
        debug_assert!(reusable || buffer.phase == BufferPhase::Canceled);
        if reusable {
            buffer.ticket = None;
            buffer.error = None;
        }
        true
    }

    pub(crate) fn bytes(&self, index: usize, len: usize)
                              -> Result<&[u8], Box<dyn std::error::Error>> {
        if self.buffers[index].phase != BufferPhase::Ready {
            return Err(io::Error::other("pread buffer is not ready for H2D").into());
        }
        Ok(&self.buffers[index].data.as_ref()
            .expect("live pread buffer must retain its pinned allocation")
            .as_slice()?[..len])
    }

    pub(crate) fn mark_h2d(&mut self, index: usize, ready: Arc<CudaEvent>) {
        self.buffers[index].phase.begin_h2d();
        self.buffers[index].ticket = None;
        self.buffers[index].ready = Some(ready);
    }

    /// Conservatively retain a buffer when CUDA submission/event recording could not prove a
    /// completion point. Only a later whole-stream synchronization may release it.
    pub(crate) fn mark_unknown_h2d(&mut self, index: usize) {
        self.buffers[index].phase.begin_h2d();
        self.buffers[index].ticket = None;
        self.buffers[index].ready = None;
    }

    pub(crate) fn abort_read(&mut self, index: usize) {
        self.buffers[index].ticket = None;
        self.buffers[index].error = None;
        self.buffers[index].phase.abort_read();
    }

    pub(crate) fn note_fallback(&mut self) -> u64 {
        self.stats.fallbacks += 1;
        self.stats.fallbacks
    }

    pub(crate) fn stats(&self) -> PreadStats { self.stats }

    /// Drain the retained compute stream before pinned allocations can be freed. If CUDA cannot
    /// prove completion, leak the bounded pool: leaking is preferable to a DMA use-after-free.
    pub(crate) fn drain(&mut self) -> bool {
        let worker_clean = if let Some(workers) = self.workers.as_mut() {
            workers.shutdown()
        } else { true };
        self.poll_worker_completions();
        match self.stream.synchronize() {
            Ok(()) => {
                for buffer in &mut self.buffers {
                    buffer.ready = None;
                    buffer.ticket = None;
                    buffer.error = None;
                    if buffer.phase == BufferPhase::H2d {
                        assert!(buffer.phase.finish_h2d(true));
                    } else if matches!(buffer.phase,
                        BufferPhase::Reading | BufferPhase::Canceled
                            | BufferPhase::Ready | BufferPhase::Failed) {
                        buffer.phase.abort_read();
                    }
                }
                self.tickets.clear();
                worker_clean
            }
            Err(err) => {
                eprintln!(
                    "[spill-pread] CUDA stream drain failed ({err}); leaking pinned buffers for safety"
                );
                for buffer in &mut self.buffers {
                    if let Some(data) = buffer.data.take() {
                        std::mem::forget(data);
                    }
                }
                false
            }
        }
    }
}

impl Drop for PreadPool {
    fn drop(&mut self) {
        let _ = self.drain();
        if self.stats.reads != 0 || self.stats.fallbacks != 0 || self.stats.ring_full != 0 {
            eprintln!(
                "[spill-pread] reads={} bytes={} errors={} short_reads={} fallbacks={} \
                 buffer_waits={} ring_full={}",
                self.stats.reads,
                self.stats.bytes,
                self.stats.read_errors,
                self.stats.short_reads,
                self.stats.fallbacks,
                self.stats.buffer_waits,
                self.stats.ring_full,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferPhase, parse_depth, parse_spill_io, pread_exact_at, SpillIoMode};

    fn temp_file(name: &str, bytes: &[u8]) -> (std::path::PathBuf, std::fs::File) {
        let path = std::env::temp_dir().join(format!(
            "bw24-pread-{name}-{}", std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        (path, file)
    }

    #[test]
    fn mode_and_depth_defaults_are_conservative() {
        assert_eq!(parse_spill_io(None), Ok(SpillIoMode::Mmap));
        assert_eq!(parse_spill_io(Some("pread")), Ok(SpillIoMode::Pread));
        assert_eq!(parse_spill_io(Some("worker")), Ok(SpillIoMode::Worker));
        assert!(parse_spill_io(Some("uring")).is_err());
        assert_eq!(parse_depth(None), Ok(2));
        assert_eq!(parse_depth(Some("8")), Ok(8));
        assert!(parse_depth(Some("0")).is_err());
    }

    #[test]
    fn exact_and_unaligned_positioned_reads_match_file_bytes() {
        let bytes: Vec<u8> = (0..97u8).collect();
        let (path, file) = temp_file("unaligned", &bytes);
        let mut exact = vec![0u8; bytes.len()];
        pread_exact_at(&file, &mut exact, 0).unwrap();
        assert_eq!(exact, bytes);
        let mut unaligned = vec![0u8; 37];
        pread_exact_at(&file, &mut unaligned, 3).unwrap();
        assert_eq!(unaligned, bytes[3..40]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn short_positioned_read_is_an_error() {
        let (path, file) = temp_file("short", &[1, 2, 3, 4, 5]);
        let mut dst = [0u8; 4];
        let err = pread_exact_at(&file, &mut dst, 3).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn buffer_state_rejects_early_reuse() {
        let mut phase = BufferPhase::Free;
        assert!(phase.begin_read());
        assert!(!phase.begin_read());
        assert!(!phase.finish_read(true));
        phase.begin_h2d();
        assert!(!phase.begin_read());
        assert!(!phase.finish_h2d(false));
        assert_eq!(phase, BufferPhase::H2d);
        assert!(phase.finish_h2d(true));
        assert_eq!(phase, BufferPhase::Free);
        assert!(phase.begin_read());
        phase.abort_read();
        assert_eq!(phase, BufferPhase::Free);
        assert!(phase.begin_read());
        assert!(!phase.finish_read(false));
        phase.abort_read();
        assert_eq!(phase, BufferPhase::Free);
        assert!(phase.begin_read());
        assert!(!phase.cancel_read());
        assert_eq!(phase, BufferPhase::Canceled);
        assert!(phase.finish_read(true));
        assert_eq!(phase, BufferPhase::Free);
        assert!(phase.begin_read());
        assert!(!phase.finish_read(true));
        assert!(phase.cancel_read());
        assert_eq!(phase, BufferPhase::Free);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read() {
        let engine = crate::Engine::new(0).unwrap();
        let bytes: Vec<u8> = (0..97u8).collect();
        let (path, file) = temp_file("worker", &bytes);
        let file = std::sync::Arc::new(file);
        let mut pool = super::PreadPool::try_new(&engine, 64, SpillIoMode::Worker).unwrap();
        assert!(pool.buffers.len() >= 2, "worker reservation test requires depth >= 2");

        let first = pool.submit_worker(file.clone(), 3, 37).unwrap().unwrap();
        let second = pool.submit_worker(file.clone(), 51, 29).unwrap().unwrap();
        let second_buffer = pool.wait_worker(second).unwrap();
        assert_eq!(pool.bytes(second_buffer, 29).unwrap(), &bytes[51..80]);
        pool.abort_read(second_buffer);
        let first_buffer = pool.wait_worker(first).unwrap();
        assert_eq!(pool.bytes(first_buffer, 37).unwrap(), &bytes[3..40]);
        pool.abort_read(first_buffer);

        let speculative: Vec<_> = (1..pool.buffers.len())
            .map(|_| pool.submit_worker_speculative(file.clone(), 30, 17).unwrap().unwrap())
            .collect();
        assert!(pool.submit_worker_speculative(file.clone(), 0, 8).unwrap().is_none());
        let demand = pool.submit_worker(file.clone(), 11, 13).unwrap().unwrap();
        assert!(pool.submit_worker(file.clone(), 0, 8).unwrap().is_none());
        assert_eq!(pool.stats().ring_full, 2);
        let demand_buffer = pool.wait_worker(demand).unwrap();
        assert_eq!(pool.bytes(demand_buffer, 13).unwrap(), &bytes[11..24]);
        pool.abort_read(demand_buffer);
        for ticket in speculative {
            let buffer = pool.wait_worker(ticket).unwrap();
            assert_eq!(pool.bytes(buffer, 17).unwrap(), &bytes[30..47]);
            pool.abort_read(buffer);
        }

        let canceled = pool.submit_worker(file.clone(), 7, 41).unwrap().unwrap();
        let blockers: Vec<_> = (1..pool.buffers.len())
            .map(|_| pool.submit_worker(file.clone(), 30, 17).unwrap().unwrap())
            .collect();
        let ring_full_before = pool.stats().ring_full;
        assert!(pool.submit_worker(file.clone(), 0, 8).unwrap().is_none());
        assert_eq!(pool.stats().ring_full, ring_full_before + 1);
        assert!(pool.cancel_worker(canceled));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let after_cancel = loop {
            if let Some(ticket) = pool.submit_worker(file.clone(), 11, 13).unwrap() {
                break ticket;
            }
            assert!(std::time::Instant::now() < deadline, "canceled buffer was not reclaimed");
            std::thread::yield_now();
        };
        let after_cancel_buffer = pool.wait_worker(after_cancel).unwrap();
        assert_eq!(pool.bytes(after_cancel_buffer, 13).unwrap(), &bytes[11..24]);
        pool.abort_read(after_cancel_buffer);
        for blocker in blockers {
            let blocker_buffer = pool.wait_worker(blocker).unwrap();
            assert_eq!(pool.bytes(blocker_buffer, 17).unwrap(), &bytes[30..47]);
            pool.abort_read(blocker_buffer);
        }

        let short = pool.submit_worker(file.clone(), 93, 8).unwrap().unwrap();
        let err = pool.wait_worker(short).unwrap_err();
        assert_eq!(err.downcast_ref::<std::io::Error>().unwrap().kind(),
                   std::io::ErrorKind::UnexpectedEof);
        let reused = pool.submit_worker(file, 0, 8).unwrap().unwrap();
        let reused_buffer = pool.wait_worker(reused).unwrap();
        assert_eq!(pool.bytes(reused_buffer, 8).unwrap(), &bytes[..8]);
        pool.abort_read(reused_buffer);
        std::fs::remove_file(path).ok();
    }
}
