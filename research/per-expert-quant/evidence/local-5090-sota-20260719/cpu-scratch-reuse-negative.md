# CPU expert scratch reuse: rejected

The second, correctness-fixed scratch-reuse candidate was measured in alternating control/candidate
order, seven repetitions per arm, 2,000 iterations per repetition, eight OpenMP threads, and no
intentional cooldown.

| routed experts | control median | scratch-reuse median | delta |
|---|---:|---:|---:|
| 2 | 0.236577 ms/token | 0.307475 ms/token | +30.0% slower |
| 4 | 0.403278 ms/token | 0.458596 ms/token | +13.7% slower |

All hashes and endpoint values matched the control. The candidate removed roughly 2-4 microseconds
of preparation per 2,008 calls, but compute variance and slowdown dominated that saving. The first
scratch attempt also segfaulted; the corrected attempt is safe but performance-negative. The current
source therefore keeps per-call runtime vectors and contains no scratch-reuse dispatch arm.

Raw evidence:

- `cpu-moe-scratch-reuse-segfault-gdb.log`
- `cpu-moe-scratch2-alternating-n2-n4-t8-reps2000.log`
