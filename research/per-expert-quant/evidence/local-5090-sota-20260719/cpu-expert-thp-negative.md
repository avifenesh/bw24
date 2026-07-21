# CPU expert THP experiment: rejected

The transparent-hugepage cache experiment is not a deployable winner on this
target kernel and was removed.

- Control, `MADV_HUGEPAGE`, and synchronous-collapse probes produced the same
  exact output hash: `55f2fbeeb21f6d68`.
- Both control and advised allocations reported `AnonHugePages: 0 kB`.
- The advised 2 MiB VMAs were PMD-aligned and carried the `hg` flag, but every
  `madvise(..., 2097152, MADV_COLLAPSE)` call returned `EINVAL`.
- The reverted source rebuild is byte-identical to the retained production
  companion: SHA-256
  `8bf23e565f64baec94f9b40f1848d179e17e42b87784db1c1d3def5ea89fa031`.

Raw evidence:

- `cpu-expert-thp-control-smaps.log`
- `cpu-expert-thp-enabled-smaps.log`
- `cpu-expert-thp-collapse-vma-debug.log`
- `cpu-expert-thp-collapse-strace.log`
- `cpu-expert-thp-collapse-strace-output.log`

No THP flag or allocation-padding path is retained.
