# Fused LRU RAM36 run 5 — invalid (model process swapped during warmup)

This run was intentionally stopped before scoring. It began after a successful clean swap reset,
but during the discarded warmup `/proc/2274217/status` reported `VmSwap: 352592 kB`. At the same
observation the system swap device had refilled to 5.2 GiB; shortly afterward it reached 8.0 GiB.

No throughput or correctness result was emitted. The follow-up runner sets the inference scope's
`memory.swap.max` to zero so desktop cold pages may still use system swap while the model process
cannot enter a swap-thrashing measurement regime.
