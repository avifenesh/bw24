# 2026-07-22 gate session: ABORTED — non-scoring

This directory's runs are NOT evidence for any perf claim. The session ran while other agents
interleaved on the machine: `llama-bench` held up to 19,004 MiB HBM (captured in
`pair2-candidate.contam`), one candidate attempt exited status 1 under that pressure, and runs
that passed the GPU-idle sampler were still CPU/storage-contended (pair1-candidate 2.87 tok/s,
pair2-control 2.58 tok/s against the 4.4–4.6 tok/s clean band from
`../local-5090-native-next-20260721/`).

Lesson recorded: the hy3 spill measurement is a whole-machine workload — a GPU-process guard is
insufficient; CPU-core and NVMe contention corrupt arms without any foreign GPU allocation. The
gate requires an exclusive machine window (GPU + pinned P-cores + both NVMe drives).

The runner (`gate_runner.zsh`) with its contamination sampler is retained in session scratch and
will re-run unchanged in the exclusive window. pair1-control (4.47 tok/s, no contamination
captured, within the clean control band) is consistent with the 2026-07-21 baseline but is not
promoted either.
