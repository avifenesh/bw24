# Fused least-stale run 10: stopped before scoring

The launch was stopped during discarded warmup because the live headroom cap resolved the CPU
expert cache to 30.27 GiB, while the preceding LRU control had received 33.97 GiB. Comparing those
runs would confound eviction policy with cache capacity. No throughput window completed and no
result is attributed to least-stale. The follow-up runner accepts `BW24_RUN_CACHE_GB` so both arms
can be pinned to an identical requested capacity below the live headroom ceiling.

Raw log: `hy3-mtp-k1-headroom6-wholepack-control-clean-ngen64-run10.log`.
