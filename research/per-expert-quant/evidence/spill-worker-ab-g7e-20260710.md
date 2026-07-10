# G7e worker-pread calibration A/B — 2026-07-10

This is a directional storage/runtime result from the AWS G7e RTX PRO 6000 96 GB research host. It
promotes the implemented worker backend for the full calibration capture; it does not change the
default backend or replace the final RTX 5090 deployment gate.

## Frozen comparison

- Artifact: `/scratch/artifacts/plain-quant` (full 192-expert NVFP4 bank).
- Input: `/data/calibration/hy3-routing-v1/requests-first5.jsonl`, SHA-256
  `989e1b0eaa35b1bf71bf4eea79c83161fd00b6e8a964a1783fbcf1fafc452975`.
- Mmap baseline: corrected runtime commit `38a5b08`, five completed response rows.
- Worker candidate: branch commit `66394bf`, `BW24_SPILL_IO=worker`,
  `BW24_SPILL_PREAD_DEPTH=8`, and `BW24_SPILL_STATS=1`.
- Both cells used the same model alias, frozen requests, cache/grouped expert path, and corrected Hy3
  norm semantics. Mmap remained the worker cell's byte oracle and fallback.

## Wall time

| request | mmap | worker depth 8 |
|---:|---:|---:|
| 1 | 150.4062 s | 51.6213 s |
| 2 | 123.1853 s | 36.0561 s |
| 3 | 91.6610 s | 43.6020 s |
| 4 | 156.6478 s | 56.6845 s |
| 5 | 166.7584 s | 52.3136 s |
| **total** | **688.6587 s** | **240.2775 s** |

The worker cell was 2.866x faster and reduced wall time by 65.1%. Its five-request mean was
48.0555 s. Multiplying that small-sample mean by all 192 frozen requests gives about 2.56 hours;
this is an arithmetic projection, not a guaranteed full-run ETA.

## Correctness and lifetime gates

All five response payloads were identical after excluding their elapsed-time fields. The worker
trace contains 395 routing rows and has SHA-256
`f4de6c259c51b572b8508f8083b1abf6cd91c5124b135a9ad22f9b78394a137d`. The first 395 rows of the
mmap trace have the same hash. The stored mmap trace has 472 rows because capture of a partial sixth
request had already started; only its first 395 rows belong to this comparison.

The worker response file is
`/data/calibration/hy3-routing-v1/plain-quant-normfix-66394bf-worker-d8-first5.jsonl`, SHA-256
`7bc3a183b3303eb44a88f68462487c02afbf64d259ba55f049e9503aea6d3d22`. The live ignored CUDA test
`worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read` also passed at configured
depth 8 before the A/B.

The final server snapshot was:

```text
reads=199413 bytes=705711439872 errors=0 short_reads=0 fallbacks=894
buffer_waits=88450 ring_full=22184
```

These server snapshots are cumulative; the final row is the total and earlier snapshots must not be
summed. The 894 mmap fallbacks are 0.448% of completed positioned reads. `buffer_waits` and
`ring_full` measure bounded-ring pressure, not read failures; no read error or short read occurred.
The raw server log is
`/data/results/per-expert-quant/diagnostics/66394bf-worker-d8-first5.server.log`.

## Decision

Use worker depth 8 for the fresh full-bank G7e routing capture, while retaining mmap as the runtime
default and fallback and blocking `pread` as the byte/H2D oracle. Evaluate `O_DIRECT`, io_uring, and
mapped host access only against this measured worker baseline. Re-run correctness, memory, and
throughput gates on the local RTX 5090 before promoting any runtime default.
