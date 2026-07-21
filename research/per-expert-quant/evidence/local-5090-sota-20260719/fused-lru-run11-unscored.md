# Fused LRU run 11: interrupted before scoring

The capacity-locked 32 GiB LRU attempt reached the discarded warmup, filled and froze HBM
residency, and then ended before the measured 64-token generation began. The raw log contains no
`[generate]` result, process exit status, OOM report, or other terminal error. The cause is therefore
unknown and this attempt is unscored; it is not evidence for or against LRU, fused CPU stages, or
the spill pipeline.

Raw log: `hy3-mtp-k1-headroom6-wholepack-control-clean-ngen64-run11.log`.
