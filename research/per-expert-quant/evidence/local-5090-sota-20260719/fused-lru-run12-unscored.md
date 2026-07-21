# Fused LRU run 12: live headroom changed the cache capacity

The runner requested a 32 GiB CPU expert cache with a 4 GiB `MemAvailable` reserve, but the live
machine state left only 26.17 GiB available when the companion initialized. The safety cap reduced
the effective cache to 22.17 GiB. The run was stopped during discarded warmup, before any measured
generation or throughput result, because it could not form the capacity-locked comparison.

This attempt is unscored and is not evidence for or against LRU, fused CPU stages, or the spill
pipeline. The preflight P-cores were 98.98--100% idle and no competing bw24 or rsync process was
present; RAM capacity, not CPU contention, invalidated the arm.

Raw log: `hy3-mtp-k1-headroom6-wholepack-control-clean-ngen64-run12.log`.
Preflight: `clean-swap-before-fused-lru-cache32-run12.log`.
