//! Opt-in explicit positioned-read proof backend for mmap-backed MoE experts.
//!
//! `BW24_SPILL_IO=pread` reads one exact expert extent into a bounded CUDA-pinned buffer before the
//! cache enqueues H2D. The disk read itself is deliberately blocking: this module proves the storage
//! direction without adding a background runtime. mmap remains the default and the byte oracle used
//! on every read error or short read.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaStream, PinnedHostSlice};

use crate::Engine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpillIoMode {
    Mmap,
    Pread,
}

fn parse_spill_io(value: Option<&str>) -> Result<SpillIoMode, &'static str> {
    match value.unwrap_or("mmap") {
        "mmap" => Ok(SpillIoMode::Mmap),
        "pread" => Ok(SpillIoMode::Pread),
        _ => Err("expected mmap or pread"),
    }
}

pub(crate) fn enabled() -> bool {
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
    }) == SpillIoMode::Pread
}

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
    Filling,
    H2d,
}

impl BufferPhase {
    fn begin_read(&mut self) -> bool {
        if *self != Self::Free { return false; }
        *self = Self::Filling;
        true
    }

    fn begin_h2d(&mut self) {
        assert_eq!(*self, Self::Filling, "pinned buffer H2D without a completed read");
        *self = Self::H2d;
    }

    fn abort_read(&mut self) {
        assert_eq!(*self, Self::Filling, "only a filling buffer can abort a read");
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
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PreadStats {
    pub reads: u64,
    pub bytes: u64,
    pub read_errors: u64,
    pub short_reads: u64,
    pub fallbacks: u64,
    pub buffer_waits: u64,
}

pub(crate) struct PreadPool {
    buffers: Vec<PinnedBuffer>,
    capacity: usize,
    stream: Arc<CudaStream>,
    stats: PreadStats,
}

impl PreadPool {
    pub(crate) fn try_new(e: &Engine, capacity: usize)
                              -> Result<Self, Box<dyn std::error::Error>> {
        let requested_depth = configured_depth();
        let mut buffers = Vec::with_capacity(requested_depth);
        for _ in 0..requested_depth {
            match unsafe { e.ctx().alloc_pinned::<u8>(capacity) } {
                Ok(data) => buffers.push(PinnedBuffer {
                    data: Some(data), phase: BufferPhase::Free, ready: None,
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
        eprintln!(
            "[spill-pread] enabled: depth={depth} buffer_bytes={capacity} total_pinned_bytes={} \
             (blocking demand pread, compute-stream H2D, mmap error fallback)",
            depth.saturating_mul(capacity)
        );
        Ok(Self {
            buffers, capacity, stream: e.stream().clone(), stats: PreadStats::default(),
        })
    }

    fn reap_completed(&mut self) {
        for buffer in &mut self.buffers {
            if buffer.phase != BufferPhase::H2d { continue; }
            let complete = buffer.ready.as_ref().is_some_and(|event| event.is_complete());
            if buffer.phase.finish_h2d(complete) {
                buffer.ready = None;
            }
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

    /// Read one exact demand extent. A full pool waits for one H2D event; disk lookahead is not
    /// implemented here because a FileExt call on the GPU worker would delay current compute.
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

    pub(crate) fn bytes(&self, index: usize, len: usize)
                              -> Result<&[u8], Box<dyn std::error::Error>> {
        if self.buffers[index].phase != BufferPhase::Filling {
            return Err(io::Error::other("pread buffer is not ready for H2D").into());
        }
        Ok(&self.buffers[index].data.as_ref()
            .expect("live pread buffer must retain its pinned allocation")
            .as_slice()?[..len])
    }

    pub(crate) fn mark_h2d(&mut self, index: usize, ready: Arc<CudaEvent>) {
        self.buffers[index].phase.begin_h2d();
        self.buffers[index].ready = Some(ready);
    }

    /// Conservatively retain a buffer when CUDA submission/event recording could not prove a
    /// completion point. Only a later whole-stream synchronization may release it.
    pub(crate) fn mark_unknown_h2d(&mut self, index: usize) {
        self.buffers[index].phase.begin_h2d();
        self.buffers[index].ready = None;
    }

    pub(crate) fn abort_read(&mut self, index: usize) {
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
        match self.stream.synchronize() {
            Ok(()) => {
                for buffer in &mut self.buffers {
                    buffer.ready = None;
                    if buffer.phase == BufferPhase::H2d {
                        assert!(buffer.phase.finish_h2d(true));
                    } else if buffer.phase == BufferPhase::Filling {
                        buffer.phase.abort_read();
                    }
                }
                true
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
        if self.stats.reads != 0 || self.stats.fallbacks != 0 {
            eprintln!(
                "[spill-pread] reads={} bytes={} errors={} short_reads={} fallbacks={} \
                 buffer_waits={}",
                self.stats.reads,
                self.stats.bytes,
                self.stats.read_errors,
                self.stats.short_reads,
                self.stats.fallbacks,
                self.stats.buffer_waits,
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
        phase.begin_h2d();
        assert!(!phase.begin_read());
        assert!(!phase.finish_h2d(false));
        assert_eq!(phase, BufferPhase::H2d);
        assert!(phase.finish_h2d(true));
        assert_eq!(phase, BufferPhase::Free);
        assert!(phase.begin_read());
        phase.abort_read();
        assert_eq!(phase, BufferPhase::Free);
    }
}
