# G7e cold spill prefetch result — 2026-07-10

This is a research-host result from the AWS G7e RTX PRO 6000 machine. It selects a setting for the
directional capability screen; it is not the final RTX 5090 deployment result.

## Frozen input

- Artifact: `plain_quant`, 161,036,107,776 expert-overlay bytes.
- Prompt: 8,865 bytes, SHA-256
  `a722a2d135c674955d8b12d717cdffa87b9a5040a9e2ae543b13c79a1ca93596`.
- Runtime commit: `0c7f2e5df72c7c3059b3fcb5bb16b9bfc796ed66`.
- Page-prefetch window: 8.
- Expert files were evicted with `POSIX_FADV_DONTNEED` before each cell.

## Result

| mmap advice | status | wall time | active read throughput | physical request size | major faults |
|---|---:|---:|---:|---:|---:|
| `random` | stopped after the failure mode was stable | not comparable | 174.87 MiB/s mean | 4.00 KiB mean | 44,774.5/s mean |
| `normal` | HTTP 200, one generated token | 202.289 s | 552.31 MiB/s mean, 632.47 MiB/s median | 123.49 KiB mean | 158.3/s mean |

The completed `normal` cell issued 925,505 physical reads for 117,034,496,000 bytes
(108.997 GiB), averaging 123.491 KiB per request. Process `read_bytes` increased by
116,996,632,576 bytes. Peak RSS was 104,718,956 KiB, almost entirely file-backed
(`RssFile` 103,797,268 KiB); anonymous RSS remained 913,496 KiB.

`MADV_NORMAL` therefore increased mean storage throughput by 3.16x and reduced the observed major
fault rate by about 283x for this frozen request. Use `BW24_MOE_MMAP_ADVICE=normal` for the compact
G7e capability screen. Keep `random` selectable until matched steady-state decode and RTX 5090
measurements are complete.

The remaining limitation is policy-controlled, unbounded page-cache growth. The next storage
backend experiment remains explicit multi-megabyte reads into a bounded CUDA-pinned buffer ring,
then asynchronous H2D. A buffered explicit-read storage probe should precede full io_uring cache
integration.

## Explicit-read storage probe

`bench_explicit_reads.py` used the artifact's exact maximum expert-block size (3,538,944 bytes),
reused one userspace buffer per worker, set `POSIX_FADV_RANDOM` to suppress implicit readahead, and
read a cold sorted 32-file subset totaling 21,743,271,936 bytes. This is a storage-side upper-bound
probe only; it does not yet include CUDA H2D or inference.

| explicit-read depth | bounded buffers | wall time | logical throughput | peak process RSS |
|---:|---:|---:|---:|---:|
| 2 | 7,077,888 bytes | 11.194 s | 1,852.39 MiB/s | 21,788 KiB |
| 8 | 28,311,552 bytes | 6.171 s | 3,359.97 MiB/s | 42,660 KiB |

The depth-8 run physically sustained about 3.36 GB/s at 128 KiB device requests and 100% NVMe
utilization. Even depth 2 delivered 3.35x the completed mmap run's mean storage rate while keeping
the explicit buffer footprint below 8 MiB. This passes the storage-side promotion gate for a
`BW24_SPILL_IO=pread` proof backend, followed by io_uring only after the source/cache/H2D contracts
are verified.
