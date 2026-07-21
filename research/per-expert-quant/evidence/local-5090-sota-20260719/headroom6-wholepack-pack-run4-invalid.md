# Headroom6 whole-pack run 4: invalid

Run 4 is not a performance sample.

- A concurrent `kernel-check` started during model loading and held 418--450 MiB
  of VRAM. The Hy3 run consequently resolved only a 14.45 GB / 5,312-slot HBM
  expert cache instead of the uncontended budget.
- The inference process itself remained healthy at 41.1 GiB peak memory inside
  an unlimited sibling cgroup. There is no kernel OOM, CUDA OOM, or Xid record.
- At 18:24:56 the detached outer runner was reaped with SIGTERM. Its cleanup trap
  explicitly stopped `bw24-headroom6-pack-2431659.scope`, producing exit 143
  during discarded warmup.

The next run must start only after all GPU gates finish and the outer runner must
be hosted by a persistent user-systemd service rather than a detached child of a
desktop command.
