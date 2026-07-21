# Fused LRU RAM32 run 6 — correctness result, busy-stack performance only

Run 6 completed with self-consistency PASS and enforced `memory.swap.max=0`, but it is not a fair
optimization comparator. It launched at a global one-minute load of 5.85 while long-lived desktop
workloads were still schedulable on P cores. It scored 5.64 tok/s plain and 5.62 tok/s at K=1.

The phase counters isolate the loss to CPU execution: 44.917 s of CPU-expert compute versus
26.242 s in the earlier fused least-stale run, while positioned-read time was comparable
(26.772 s versus 25.048 s) and the model cgroup recorded zero swap. The follow-up regime moves
selected desktop workloads to E cores 8-23 and launches the entire model process through
`taskset --cpu-list 0-7`; its preflight records per-P-core idle state.

Keep run 6 as a real busy-desktop observation, not as evidence for or against fused stages, LRU,
least-stale caching, or the spill pipeline.
