# CPU expert down-I/O pipeline: negative result

The experimental path deferred down-projection cache misses to a separate reader thread and
overlapped them with fused gate/up compute. The paired benchmark used four real Q2_K experts,
mirrored `O_DIRECT` reads, an eight-P-core compute team pinned to CPUs 0-7, 30 measured tokens per
process, eight warmups, and five independently started control/candidate pairs per I/O depth.

All control and pipeline runs produced the same output hash,
`0b35230a73c63d1f`. Median latency was:

| I/O threads | Control | Pipeline | Pipeline delta |
| ---: | ---: | ---: | ---: |
| 2 | 11.244787 ms/token | 10.879177 ms/token | -3.25% |
| 4 | 5.958384 ms/token | 6.129976 ms/token | +2.88% |
| 8 | 4.544018 ms/token | 4.967208 ms/token | +9.31% |

The production eight-reader configuration regressed latency, while the only improvement appeared
at an I/O depth whose absolute latency was more than twice as high. The implementation and flag
were therefore removed; the raw paired runs remain in
`cpu-moe-pipeline-direct-paired-n4-n30.log`.
