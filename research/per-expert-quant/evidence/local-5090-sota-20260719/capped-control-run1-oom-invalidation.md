# Capped control run 1 invalidation

The first capped control did not reach model execution. At 2026-07-20T17:16:21+03:00, a stale
uncapped `run-spec` child from the interrupted prior launch was still alive outside the benchmark
cgroup and held 20,450 MiB of VRAM. The capped process held another 2,914 MiB. Its load then failed
with the captured `CUDA_ERROR_OUT_OF_MEMORY` line and exited 1.

The stale process was sent SIGINT and all `run-spec` processes exited. This result says nothing
about the CPU quota or verification prequeue. Future capped runs refuse to start if another
`run-gen`/`run-spec` exists or if free VRAM is below 23,000 MiB, and an interrupt trap stops the
transient systemd scope.
