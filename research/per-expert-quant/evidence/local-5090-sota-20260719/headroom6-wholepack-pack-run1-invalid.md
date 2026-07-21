# Headroom-6 whole-pack run 1: invalid

The `pack` arm was terminated by SIGTERM after 29.38 seconds while loading the model, before
inference and before the freeze/pack candidate executed. The raw log records exit status 143,
5.37 GiB maximum RSS, no process swaps, 36 GiB `MemAvailable` at exit, and 23.7 GiB free VRAM.
The termination was deliberate: the process was outside a CPU/memory cgroup and was stopped as
soon as the user's resource-limit correction was applied. This is setup/resource-guard evidence,
not performance evidence.

Raw log: `hy3-mtp-k1-headroom6-wholepack-pack-clean-ngen64-run1.log`.
