# CPU-cost HBM pack: rejected

The candidate ranked complete trunk experts by discarded-warmup route frequency multiplied by a
qtype-specific CPU-dot cost, while retaining every complete MTP resident. It was evaluated once at
N=16 under `CPUQuota=200%`, `CPUWeight=10`, four OpenMP workers, a 20 GiB CPU expert cache, an
8 GiB live-stack reserve, `MemoryHigh=34G`, and `MemoryMax=38G`.

| arm | plain | K=1 | acceptance | self-consistency |
|---|---:|---:|---:|---|
| paired control | 5.54 tok/s | 5.79 tok/s | 87.5% | PASS |
| preserve-all pack | 5.46 tok/s | 5.82 tok/s | 87.5% | PASS |
| CPU-cost pack | 5.49 tok/s | 5.86 tok/s | 87.5% | PASS |

The candidate kept all 28 complete MTP experts, converted 70 stranded projection blocks into 23
additional complete experts, and generated the same tokens as the paired runs. Its single-run K=1
delta was only +1.2% versus control, while plain decode fell 0.9%, CPU compute rose from 13.266 to
14.000 seconds, and CPU projection reads rose from 154.965 to 155.650 GB. Those mechanism counters
contradict the offline cost-saving hypothesis, so the small throughput delta is noise rather than a
default-worthy win.

The run itself reported zero swaps and stayed within the capped scope. The experimental runtime
mode, cost table, tests, documentation, and runner were removed after measurement. The raw log is
`hy3-mtp-k1-capped-cpu-cost-ram20-reserve8-warm32-ngen16-run1.log`; the comparison inputs are
`hy3-mtp-k1-capped-control-ram20-warm32-ngen16-run1.log` and
`hy3-mtp-k1-capped-pack-preserve-all-ram20-warm32-ngen16-run1.log`.
