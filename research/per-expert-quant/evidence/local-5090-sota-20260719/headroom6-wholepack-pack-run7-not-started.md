# Headroom-6 whole-pack run 7: not started

- Attempt: 2026-07-20 19:21:22 IDT.
- The clean-swap guard detected PID 2670261 (`target/release/run-gen`) before executing
  `swapoff`; no swap reset and no bw24 benchmark process were started by this attempt.
- PID 2670261 belonged to the separate, already-planned
  `bw24-mixed-q8-candidate-n3-safe.scope` comparison. It was left running.
- Raw guard output: `clean-swap-before-headroom6-wholepack-pack-run7.log`.
- Run number 8 is the next eligible uncontended whole-pack measurement.
