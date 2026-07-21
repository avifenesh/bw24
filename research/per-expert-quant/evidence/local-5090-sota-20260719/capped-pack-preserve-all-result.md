# Preserve-all whole-expert HBM pack: rejected

This is a paired capped development measurement: one control and one pack run, 16 generated
tokens, no intentional cooldown, `CPUQuota=200%`, `CPUWeight=10`, four CPU/IO workers, a 20 GiB
CPU expert cache, `MemoryHigh=34G`, and `MemoryMax=38G`.

| arm | plain | K=1 | acceptance | self-consistency |
|---|---:|---:|---:|---|
| control | 5.54 tok/s | 5.79 tok/s | 87.5% | PASS |
| preserve-all pack | 5.46 tok/s | 5.82 tok/s | 87.5% | PASS |

The pack changed frozen residency from 1,804 complete experts plus 70 stranded projection blocks
to 1,827 complete experts and zero stranded blocks. It retained every already-complete resident
expert (1,804/1,804), used 5,481 of 5,482 slots, and preserved identical generated tokens.

The single-run deltas are -1.4% plain and +0.5% K=1. CPU read volume fell only 0.5% (154.965 to
154.156 GB). A later clean-swap N=64 run (`hy3-mtp-k1-headroom6-wholepack-pack-clean-ngen64-run8.log`)
eliminated all fragments but measured 7.21 tok/s plain and 7.38 tok/s at K=1, below the stable
7.54/7.60 tok/s reference. The implementation and flag were removed under the winners-only policy.

Raw logs:

- `hy3-mtp-k1-capped-control-ram20-warm32-ngen16-run1.log`
- `hy3-mtp-k1-capped-pack-preserve-all-ram20-warm32-ngen16-run1.log`
